"""Реализация биоморфной памяти с многоуровневой организацией."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import time
import threading
import uuid
from datetime import datetime
from collections import OrderedDict

from neurograph.memory.base import IMemory, MemoryItem
from neurograph.memory.consolidation import ConsolidationOrchestrator, ConsolidationManager, TransitionLogger
from neurograph.memory.strategies import (
   TimeBasedConsolidation, ImportanceBasedConsolidation,
   EbbinghausBasedForgetting, LeastRecentlyUsedForgetting,
   AdaptiveConsolidation, MemoryPressureMonitor
)
from neurograph.core.logging import get_logger
from neurograph.core.events import subscribe, publish
from neurograph.core.errors import NeuroGraphError

logger = get_logger("memory.biomorphic")


class BiomorphicMemory(IMemory):
   """Биоморфная память с многоуровневой организацией, имитирующая работу человеческой памяти."""
   
   def __init__(self, 
                stm_capacity: int = 100,
                ltm_capacity: int = 10000,
                use_semantic_indexing: bool = True,
                auto_consolidation: bool = True,
                consolidation_interval: float = 300.0):
       """Инициализирует биоморфную память.
       
       Args:
           stm_capacity: Максимальная вместимость кратковременной памяти.
           ltm_capacity: Максимальная вместимость долговременной памяти.
           use_semantic_indexing: Использовать семантическое индексирование.
           auto_consolidation: Включить автоматическую консолидацию.
           consolidation_interval: Интервал автоматической консолидации в секундах.
       """
       # Конфигурация
       self.stm_capacity = stm_capacity
       self.ltm_capacity = ltm_capacity
       self.use_semantic_indexing = use_semantic_indexing
       
       # Кратковременная память (STM) - быстрый доступ, ограниченная вместимость
       self.stm: OrderedDict[str, MemoryItem] = OrderedDict()
       
       # Долговременная память (LTM) - большая вместимость, медленный доступ
       self.ltm: Dict[str, MemoryItem] = {}
       
       # Рабочая память (Working Memory) - активные элементы для обработки
       self.working_memory: Dict[str, MemoryItem] = {}
       self.working_memory_capacity = 7  # Магическое число Миллера ±2
       
       # Семантический индекс для быстрого поиска
       self.semantic_index: Optional[Any] = None
       if use_semantic_indexing:
           self._initialize_semantic_index()
       
       # Система консолидации
       consolidation_strategies = [
           TimeBasedConsolidation(min_age_seconds=300.0, max_stm_size=stm_capacity),
           ImportanceBasedConsolidation(importance_threshold=0.6)
       ]
       adaptive_strategy = AdaptiveConsolidation(consolidation_strategies, weights=[0.6, 0.4])
       
       forgetting_strategy = EbbinghausBasedForgetting(
           base_retention=0.1,
           decay_rate=0.693,
           access_boost=1.5
       )
       
       self.consolidation_manager = ConsolidationManager(
           consolidation_strategy=adaptive_strategy,
           forgetting_strategy=forgetting_strategy,
           auto_consolidation_interval=consolidation_interval,
           enable_background_processing=auto_consolidation
       )
       
       self.transition_logger = TransitionLogger(max_history=5000)
       self.orchestrator = ConsolidationOrchestrator(
           self.consolidation_manager,
           self.transition_logger
       )
       
       # Монитор давления памяти
       self.pressure_monitor = MemoryPressureMonitor()
       
       # Потокобезопасность
       self._lock = threading.RLock()
       
       # Статистика
       self.stats = {
           "items_added": 0,
           "items_accessed": 0,
           "items_consolidated": 0,
           "items_forgotten": 0,
           "search_queries": 0,
           "cache_hits": 0,
           "cache_misses": 0
       }
       
       # Подписываемся на события консолидации
       subscribe("memory.auto_consolidation_trigger", self._handle_auto_consolidation)
       
       logger.info(f"Инициализирована биоморфная память: STM={stm_capacity}, LTM={ltm_capacity}")
   
   def _initialize_semantic_index(self) -> None:
       """Инициализирует семантический индекс для поиска."""
       try:
           from neurograph.semgraph.index.hnsw import HNSWIndex
           # Используем стандартную размерность для векторов
           self.semantic_index = HNSWIndex(dim=384, max_elements=self.ltm_capacity)
           logger.info("Семантический индекс инициализирован")
       except ImportError:
           logger.warning("HNSW не доступен, семантический поиск отключен")
           self.use_semantic_indexing = False
           self.semantic_index = None
   
   def _handle_auto_consolidation(self, data: Dict[str, Any]) -> None:
       """Обрабатывает событие автоматической консолидации."""
       if data.get("manager_id") == id(self.consolidation_manager):
           self._perform_consolidation()
   
   def add(self, item: MemoryItem) -> str:
       """Добавляет элемент в память и возвращает его ID.
       
       Args:
           item: Элемент для добавления в память.
           
       Returns:
           Уникальный идентификатор добавленного элемента.
       """
       with self._lock:
           # Генерируем уникальный ID
           item_id = str(uuid.uuid4())
           item.id = item_id
           
           # Добавляем в STM
           self.stm[item_id] = item
           
           # Обновляем статистику
           self.stats["items_added"] += 1
           
           # Проверяем необходимость консолидации
           if len(self.stm) > self.stm_capacity * 0.8:  # Превентивная консолидация
               self._perform_consolidation()
           
           # Публикуем событие добавления
           publish("memory.item_added", {
               "item_id": item_id,
               "content_type": item.content_type,
               "memory_level": "STM",
               "timestamp": time.time()
           })
           
           logger.debug(f"Добавлен элемент в STM: {item_id}")
           return item_id
   
   def get(self, item_id: str) -> Optional[MemoryItem]:
       """Возвращает элемент памяти по ID.
       
       Args:
           item_id: Идентификатор элемента.
           
       Returns:
           Элемент памяти или None, если элемент не найден.
       """
       with self._lock:
           self.stats["items_accessed"] += 1
           
           # Сначала ищем в рабочей памяти (fastest)
           if item_id in self.working_memory:
               item = self.working_memory[item_id]
               item.access()
               self.stats["cache_hits"] += 1
               return item
           
           # Затем в STM (fast)
           if item_id in self.stm:
               item = self.stm[item_id]
               item.access()
               
               # Перемещаем в конец для LRU
               self.stm.move_to_end(item_id)
               
               # Добавляем в рабочую память
               self._add_to_working_memory(item_id, item)
               
               self.stats["cache_hits"] += 1
               return item
           
           # Наконец в LTM (slow)
           if item_id in self.ltm:
               item = self.ltm[item_id]
               item.access()
               
               # Добавляем в рабочую память для быстрого доступа
               self._add_to_working_memory(item_id, item)
               
               # Возможно, стоит переместить обратно в STM при частом доступе
               if item.access_count >= 3:
                   self._promote_to_stm(item_id, item)
               
               self.stats["cache_hits"] += 1
               return item
           
           self.stats["cache_misses"] += 1
           return None
   
   def _add_to_working_memory(self, item_id: str, item: MemoryItem) -> None:
       """Добавляет элемент в рабочую память.
       
       Args:
           item_id: Идентификатор элемента.
           item: Элемент памяти.
       """
       self.working_memory[item_id] = item
       
       # Поддерживаем ограничение размера рабочей памяти
       if len(self.working_memory) > self.working_memory_capacity:
           # Удаляем наименее недавно использованный элемент
           oldest_id = min(self.working_memory.keys(),
                         key=lambda x: self.working_memory[x].last_accessed_at)
           del self.working_memory[oldest_id]
   
   def _promote_to_stm(self, item_id: str, item: MemoryItem) -> None:
       """Продвигает элемент из LTM обратно в STM.
       
       Args:
           item_id: Идентификатор элемента.
           item: Элемент памяти.
       """
       if item_id in self.ltm and len(self.stm) < self.stm_capacity:
           # Перемещаем из LTM в STM
           del self.ltm[item_id]
           self.stm[item_id] = item
           
           # Удаляем из семантического индекса
           if self.semantic_index:
               self.semantic_index.remove_item(item_id)
           
           logger.debug(f"Элемент продвинут из LTM в STM: {item_id}")
   
   def search(self, query: Union[str, np.ndarray], limit: int = 10) -> List[Tuple[str, float]]:
        """Ищет элементы, похожие на запрос.
        
        Args:
            query: Текстовый запрос или векторное представление.
            limit: Максимальное количество результатов.
            
        Returns:
            Список кортежей (id, score), где score - мера сходства.
        """
        with self._lock:
            self.stats["search_queries"] += 1
            results = []
            
            # Если запрос - строка, нужно получить его векторное представление
            if isinstance(query, str):
                # В реальной реализации здесь будет вызов векторизатора
                query_vector = self._encode_text(query)
            else:
                query_vector = query
            
            if query_vector is None:
                return []
            
            # Ищем в семантическом индексе (LTM)
            if self.semantic_index and query_vector is not None:
                semantic_results = self.semantic_index.search(query_vector, k=limit)
                # Преобразуем все результаты в float
                semantic_results = [(item_id, float(score)) for item_id, score in semantic_results]
                results.extend(semantic_results)
            
            # Ищем в STM и рабочей памяти линейно
            stm_results = self._linear_search(query_vector, self.stm, limit)
            working_results = self._linear_search(query_vector, self.working_memory, limit)
            
            # Объединяем и сортируем результаты
            all_results = results + stm_results + working_results
            
            # Удаляем дубликаты и сортируем по релевантности
            seen = set()
            unique_results = []
            for item_id, score in sorted(all_results, key=lambda x: x[1], reverse=True):
                if item_id not in seen:
                    seen.add(item_id)
                    unique_results.append((item_id, score))
            
            return unique_results[:limit]
   
   def _encode_text(self, text: str) -> Optional[np.ndarray]:
       """Кодирует текст в векторное представление.
       
       Args:
           text: Текст для кодирования.
           
       Returns:
           Векторное представление или None.
       """
       # Заглушка - в реальной реализации будет использоваться ContextVec
       # Возвращаем случайный вектор для демонстрации
       return np.random.random(384).astype(np.float32)
   
   def _linear_search(self, query_vector: np.ndarray, 
                  items: Dict[str, MemoryItem], 
                  limit: int) -> List[Tuple[str, float]]:
        """Выполняет линейный поиск по элементам.
        
        Args:
            query_vector: Вектор запроса.
            items: Элементы для поиска.
            limit: Максимальное количество результатов.
            
        Returns:
            Список результатов поиска.
        """
        results = []
        
        for item_id, item in items.items():
            if item.embedding is not None:
                # Вычисляем косинусное сходство
                similarity = self._cosine_similarity(query_vector, item.embedding)
                # Убеждаемся, что similarity является Python float
                results.append((item_id, float(similarity)))
        
        # Сортируем по убыванию сходства
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
   
   def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
       """Вычисляет косинусное сходство между векторами.
       
       Args:
           vec1: Первый вектор.
           vec2: Второй вектор.
           
       Returns:
           Косинусное сходство.
       """
       if np.all(vec1 == 0) or np.all(vec2 == 0):
           return 0.0
       
       dot_product = np.dot(vec1, vec2)
       norm1 = np.linalg.norm(vec1)
       norm2 = np.linalg.norm(vec2)
       
       if norm1 == 0 or norm2 == 0:
           return 0.0
       
       return dot_product / (norm1 * norm2)
   
   def remove(self, item_id: str) -> bool:
       """Удаляет элемент из памяти.
       
       Args:
           item_id: Идентификатор элемента.
           
       Returns:
           True, если элемент был удален, иначе False.
       """
       with self._lock:
           removed = False
           
           # Удаляем из всех уровней памяти
           if item_id in self.working_memory:
               del self.working_memory[item_id]
               removed = True
           
           if item_id in self.stm:
               del self.stm[item_id]
               removed = True
           
           if item_id in self.ltm:
               del self.ltm[item_id]
               removed = True
               
               # Удаляем из семантического индекса
               if self.semantic_index:
                   self.semantic_index.remove_item(item_id)
           
           if removed:
               publish("memory.item_removed", {
                   "item_id": item_id,
                   "timestamp": time.time()
               })
               logger.debug(f"Элемент удален: {item_id}")
           
           return removed
   
   def clear(self) -> None:
       """Очищает память."""
       with self._lock:
           self.working_memory.clear()
           self.stm.clear()
           self.ltm.clear()
           
           if self.semantic_index:
               # Пересоздаем индекс
               self._initialize_semantic_index()
           
           # Сбрасываем статистику
           for key in self.stats:
               self.stats[key] = 0
           
           publish("memory.cleared", {"timestamp": time.time()})
           logger.info("Память очищена")
   
   def size(self) -> int:
        """Возвращает размер памяти (количество уникальных элементов).
        
        Returns:
            Количество уникальных элементов в памяти.
        """
        with self._lock:
            # Собираем все уникальные ID из всех уровней памяти
            all_ids = set()
            
            # Добавляем ID из рабочей памяти
            all_ids.update(self.working_memory.keys())
            
            # Добавляем ID из STM
            all_ids.update(self.stm.keys())
            
            # Добавляем ID из LTM
            all_ids.update(self.ltm.keys())
            
            return len(all_ids)
   
   def _perform_consolidation(self) -> None:
       """Выполняет консолидацию памяти."""
       with self._lock:
           try:
               # Выполняем консолидацию через оркестратор
               to_consolidate, to_forget = self.orchestrator.perform_consolidation(
                   self.stm, self.ltm, self.stm_capacity, self.ltm_capacity
               )
               
               # Применяем изменения
               self._apply_consolidation_changes(to_consolidate, to_forget)
               
               # Адаптируем стратегии
               stm_pressure = self.pressure_monitor.get_memory_pressure(
                   len(self.stm), self.stm_capacity
               )
               ltm_pressure = self.pressure_monitor.get_memory_pressure(
                   len(self.ltm), self.ltm_capacity
               )
               
               self.consolidation_manager.adapt_strategies(
                   stm_pressure, ltm_pressure, self.stats
               )
               
           except Exception as e:
               logger.error(f"Ошибка при консолидации памяти: {str(e)}")
   
   def _apply_consolidation_changes(self, to_consolidate: List[str], 
                                  to_forget: List[str]) -> None:
       """Применяет изменения консолидации.
       
       Args:
           to_consolidate: Элементы для перемещения в LTM.
           to_forget: Элементы для удаления.
       """
       # Консолидируем элементы из STM в LTM
       for item_id in to_consolidate:
           if item_id in self.stm:
               item = self.stm.pop(item_id)
               self.ltm[item_id] = item
               
               # Добавляем в семантический индекс
               if self.semantic_index and item.embedding is not None:
                   self.semantic_index.add_item(item_id, item.embedding)
               
               self.stats["items_consolidated"] += 1
       
       # Забываем элементы
       for item_id in to_forget:
           if item_id in self.working_memory:
               del self.working_memory[item_id]
           
           if item_id in self.stm:
               del self.stm[item_id]
           
           if item_id in self.ltm:
               del self.ltm[item_id]
               if self.semantic_index:
                   self.semantic_index.remove_item(item_id)
           
           self.stats["items_forgotten"] += 1
       
       logger.debug(f"Консолидация завершена: {len(to_consolidate)} -> LTM, "
                   f"{len(to_forget)} забыто")
   
   def get_memory_statistics(self) -> Dict[str, Any]:
       """Возвращает детальную статистику памяти.
       
       Returns:
           Словарь с статистикой.
       """
       with self._lock:
           stm_pressure = self.pressure_monitor.get_memory_pressure(
               len(self.stm), self.stm_capacity
           )
           ltm_pressure = self.pressure_monitor.get_memory_pressure(
               len(self.ltm), self.ltm_capacity
           )
           
           return {
               "memory_levels": {
                   "working_memory": {
                       "size": len(self.working_memory),
                       "capacity": self.working_memory_capacity,
                       "pressure": len(self.working_memory) / self.working_memory_capacity
                   },
                   "stm": {
                       "size": len(self.stm),
                       "capacity": self.stm_capacity,
                       "pressure": stm_pressure
                   },
                   "ltm": {
                       "size": len(self.ltm),
                       "capacity": self.ltm_capacity,
                       "pressure": ltm_pressure
                   }
               },
               "performance": self.stats,
               "consolidation": self.orchestrator.get_comprehensive_statistics(),
               "semantic_indexing": {
                   "enabled": self.use_semantic_indexing,
                   "available": self.semantic_index is not None
               },
               "total_items": self.size(),
               "memory_efficiency": {
                   "cache_hit_rate": (self.stats["cache_hits"] / 
                                    max(self.stats["items_accessed"], 1)),
                   "consolidation_rate": (self.stats["items_consolidated"] / 
                                        max(self.stats["items_added"], 1)),
                   "forgetting_rate": (self.stats["items_forgotten"] / 
                                     max(self.stats["items_added"], 1))
               }
           }
   
   def force_consolidation(self) -> Dict[str, Any]:
       """Принудительно выполняет консолидацию памяти.
       
       Returns:
           Статистика выполненной консолидации.
       """
       with self._lock:
           before_stats = {
               "stm_size": len(self.stm),
               "ltm_size": len(self.ltm),
               "working_size": len(self.working_memory)
           }
           
           self._perform_consolidation()
           
           after_stats = {
               "stm_size": len(self.stm),
               "ltm_size": len(self.ltm),
               "working_size": len(self.working_memory)
           }
           
           return {
               "before": before_stats,
               "after": after_stats,
               "consolidated": before_stats["stm_size"] - after_stats["stm_size"],
               "timestamp": time.time()
           }
   
   def get_recent_items(self, hours: float = 24.0, memory_level: Optional[str] = None) -> List[MemoryItem]:
       """Возвращает недавние элементы памяти.
       
       Args:
           hours: Количество часов назад.
           memory_level: Уровень памяти ("working", "stm", "ltm") или None для всех.
           
       Returns:
           Список недавних элементов.
       """
       with self._lock:
           cutoff_time = time.time() - (hours * 3600)
           recent_items = []
           
           # Определяем, какие уровни памяти просматривать
           levels_to_search = []
           if memory_level is None:
               levels_to_search = [
                   ("working", self.working_memory),
                   ("stm", self.stm),
                   ("ltm", self.ltm)
               ]
           elif memory_level == "working":
               levels_to_search = [("working", self.working_memory)]
           elif memory_level == "stm":
               levels_to_search = [("stm", self.stm)]
           elif memory_level == "ltm":
               levels_to_search = [("ltm", self.ltm)]
           
           # Собираем недавние элементы
           for level_name, items in levels_to_search:
               for item in items.values():
                   if item.created_at >= cutoff_time:
                       recent_items.append(item)
           
           # Сортируем по времени создания (новые сначала)
           recent_items.sort(key=lambda x: x.created_at, reverse=True)
           
           return recent_items
   
   def get_most_accessed_items(self, limit: int = 10) -> List[Tuple[MemoryItem, int]]:
       """Возвращает наиболее часто используемые элементы.
       
       Args:
           limit: Максимальное количество элементов.
           
       Returns:
           Список кортежей (элемент, количество_обращений).
       """
       with self._lock:
           all_items = []
           
           # Собираем все элементы
           for items in [self.working_memory, self.stm, self.ltm]:
               for item in items.values():
                   all_items.append((item, item.access_count))
           
           # Сортируем по количеству обращений
           all_items.sort(key=lambda x: x[1], reverse=True)
           
           return all_items[:limit]
   
   def optimize_memory(self) -> Dict[str, Any]:
       """Оптимизирует организацию памяти.
       
       Returns:
           Статистика оптимизации.
       """
       with self._lock:
           optimization_stats = {
               "actions_taken": [],
               "before": self.get_memory_statistics(),
               "timestamp": time.time()
           }
           
           # 1. Принудительная консолидация при переполнении
           if len(self.stm) > self.stm_capacity * 0.9:
               self._perform_consolidation()
               optimization_stats["actions_taken"].append("emergency_consolidation")
           
           # 2. Продвижение часто используемых элементов из LTM в STM
           if len(self.stm) < self.stm_capacity * 0.7:
               frequently_accessed = []
               for item_id, item in self.ltm.items():
                   if item.access_count >= 5:  # Часто используемые
                       frequently_accessed.append((item_id, item))
               
               # Сортируем по частоте использования
               frequently_accessed.sort(key=lambda x: x[1].access_count, reverse=True)
               
               # Продвигаем в STM
               promoted_count = 0
               for item_id, item in frequently_accessed:
                   if len(self.stm) < self.stm_capacity * 0.8:
                       self._promote_to_stm(item_id, item)
                       promoted_count += 1
                   else:
                       break
               
               if promoted_count > 0:
                   optimization_stats["actions_taken"].append(f"promoted_{promoted_count}_items")
           
           # 3. Очистка рабочей памяти от старых элементов
           current_time = time.time()
           old_working_items = []
           for item_id, item in self.working_memory.items():
               if current_time - item.last_accessed_at > 300:  # 5 минут
                   old_working_items.append(item_id)
           
           for item_id in old_working_items:
               del self.working_memory[item_id]
           
           if old_working_items:
               optimization_stats["actions_taken"].append(f"cleared_{len(old_working_items)}_working_items")
           
           # 4. Перестройка семантического индекса при необходимости
           if (self.semantic_index and 
               len(self.ltm) > 0 and 
               len(optimization_stats["actions_taken"]) > 0):
               try:
                   # Пересоздаем индекс для оптимизации
                   self._rebuild_semantic_index()
                   optimization_stats["actions_taken"].append("rebuilt_semantic_index")
               except Exception as e:
                   logger.warning(f"Не удалось перестроить семантический индекс: {str(e)}")
           
           optimization_stats["after"] = self.get_memory_statistics()
           
           logger.info(f"Оптимизация памяти завершена: {optimization_stats['actions_taken']}")
           return optimization_stats
   
   def _rebuild_semantic_index(self) -> None:
       """Перестраивает семантический индекс."""
       if not self.semantic_index:
           return
       
       # Пересоздаем индекс
       self._initialize_semantic_index()
       
       # Добавляем все элементы LTM
       for item_id, item in self.ltm.items():
           if item.embedding is not None:
               self.semantic_index.add_item(item_id, item.embedding)
       
       logger.debug("Семантический индекс перестроен")
   
   def export_memory_dump(self) -> Dict[str, Any]:
       """Экспортирует дамп памяти для анализа.
       
       Returns:
           Дамп памяти.
       """
       with self._lock:
           dump = {
               "metadata": {
                   "timestamp": time.time(),
                   "version": "1.0",
                   "memory_type": "biomorphic"
               },
               "configuration": {
                   "stm_capacity": self.stm_capacity,
                   "ltm_capacity": self.ltm_capacity,
                   "working_capacity": self.working_memory_capacity,
                   "semantic_indexing": self.use_semantic_indexing
               },
               "statistics": self.get_memory_statistics(),
               "items": {
                   "working_memory": [
                       {
                           "id": item.id,
                           "content": item.content[:100] + "..." if len(item.content) > 100 else item.content,
                           "content_type": item.content_type,
                           "created_at": item.created_at,
                           "last_accessed_at": item.last_accessed_at,
                           "access_count": item.access_count,
                           "metadata": item.metadata
                       }
                       for item in self.working_memory.values()
                   ],
                   "stm": [
                       {
                           "id": item.id,
                           "content": item.content[:100] + "..." if len(item.content) > 100 else item.content,
                           "content_type": item.content_type,
                           "created_at": item.created_at,
                           "last_accessed_at": item.last_accessed_at,
                           "access_count": item.access_count,
                           "metadata": item.metadata
                       }
                       for item in self.stm.values()
                   ],
                   "ltm": [
                       {
                           "id": item.id,
                           "content": item.content[:100] + "..." if len(item.content) > 100 else item.content,
                           "content_type": item.content_type,
                           "created_at": item.created_at,
                           "last_accessed_at": item.last_accessed_at,
                           "access_count": item.access_count,
                           "metadata": item.metadata
                       }
                       for item in self.ltm.values()
                   ]
               }
           }
           
           return dump
   
   def shutdown(self) -> None:
       """Завершает работу биоморфной памяти."""
       with self._lock:
           # Останавливаем оркестратор консолидации
           self.orchestrator.shutdown()
           
           # Выполняем финальную консолидацию
           self._perform_consolidation()
           
           # Публикуем событие завершения работы
           publish("memory.shutdown", {
               "final_stats": self.get_memory_statistics(),
               "timestamp": time.time()
           })
           
           logger.info("Биоморфная память завершила работу")
   
   def __del__(self):
       """Деструктор для корректного завершения работы."""
       try:
           self.shutdown()
       except Exception:
           pass  # Игнорируем ошибки при завершении работы