"""Стратегии консолидации и забывания для биоморфной памяти."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
import time
from datetime import datetime, timedelta
from enum import Enum

from neurograph.memory.base import MemoryItem
from neurograph.core.logging import get_logger

logger = get_logger("memory.strategies")


class ConsolidationStrategy(ABC):
    """Базовый класс для стратегий консолидации памяти."""
    
    @abstractmethod
    def should_consolidate(self, stm_items: Dict[str, MemoryItem], 
                         ltm_items: Dict[str, MemoryItem]) -> List[str]:
        """Определяет, какие элементы STM должны быть перемещены в LTM.
        
        Args:
            stm_items: Элементы кратковременной памяти.
            ltm_items: Элементы долговременной памяти.
            
        Returns:
            Список ID элементов для консолидации.
        """
        pass


class ForgettingStrategy(ABC):
    """Базовый класс для стратегий забывания."""
    
    @abstractmethod
    def should_forget(self, items: Dict[str, MemoryItem]) -> List[str]:
        """Определяет, какие элементы должны быть забыты.
        
        Args:
            items: Элементы памяти для анализа.
            
        Returns:
            Список ID элементов для удаления.
        """
        pass


class TimeBasedConsolidation(ConsolidationStrategy):
    """Консолидация на основе времени."""
    
    def __init__(self, min_age_seconds: float = 300.0, 
                 max_stm_size: int = 100):
        """Инициализирует стратегию.
        
        Args:
            min_age_seconds: Минимальный возраст элемента для консолидации.
            max_stm_size: Максимальный размер STM.
        """
        self.min_age_seconds = min_age_seconds
        self.max_stm_size = max_stm_size
    
    def should_consolidate(self, stm_items: Dict[str, MemoryItem], 
                         ltm_items: Dict[str, MemoryItem]) -> List[str]:
        """Консолидирует старые элементы или при переполнении STM."""
        current_time = time.time()
        candidates = []
        
        # Если STM переполнена, консолидируем принудительно
        if len(stm_items) > self.max_stm_size:
            # Сортируем по времени создания (старые сначала)
            sorted_items = sorted(stm_items.items(), 
                                key=lambda x: x[1].created_at)
            overflow_count = len(stm_items) - self.max_stm_size + 10  # +10 для буфера
            candidates.extend([item_id for item_id, _ in sorted_items[:overflow_count]])
        
        # Консолидируем элементы старше min_age_seconds
        for item_id, item in stm_items.items():
            if item_id not in candidates:
                age = current_time - item.created_at
                if age >= self.min_age_seconds:
                    candidates.append(item_id)
        
        logger.debug(f"Выбрано {len(candidates)} элементов для консолидации")
        return candidates


class ImportanceBasedConsolidation(ConsolidationStrategy):
    """Консолидация на основе важности элементов."""
    
    def __init__(self, importance_threshold: float = 0.7,
                 access_weight: float = 0.3,
                 recency_weight: float = 0.4,
                 content_weight: float = 0.3):
        """Инициализирует стратегию.
        
        Args:
            importance_threshold: Порог важности для консолидации.
            access_weight: Вес фактора доступа к элементу.
            recency_weight: Вес фактора свежести.
            content_weight: Вес фактора содержания.
        """
        self.importance_threshold = importance_threshold
        self.access_weight = access_weight
        self.recency_weight = recency_weight
        self.content_weight = content_weight
    
    def should_consolidate(self, stm_items: Dict[str, MemoryItem], 
                         ltm_items: Dict[str, MemoryItem]) -> List[str]:
        """Консолидирует важные элементы."""
        candidates = []
        current_time = time.time()
        
        for item_id, item in stm_items.items():
            importance = self._calculate_importance(item, current_time)
            
            if importance >= self.importance_threshold:
                candidates.append(item_id)
        
        logger.debug(f"Выбрано {len(candidates)} важных элементов для консолидации")
        return candidates
    
    def _calculate_importance(self, item: MemoryItem, current_time: float) -> float:
        """Вычисляет важность элемента памяти."""
        # Фактор доступа (нормализуем количество обращений)
        access_factor = min(item.access_count / 10.0, 1.0)
        
        # Фактор свежести (обратно пропорционален возрасту)
        age_hours = (current_time - item.created_at) / 3600.0
        recency_factor = max(0.1, 1.0 / (1.0 + age_hours))
        
        # Фактор содержания (длина текста как приблизительная мера информативности)
        content_length = len(item.content)
        content_factor = min(content_length / 1000.0, 1.0)
        
        # Взвешенная сумма факторов
        importance = (self.access_weight * access_factor + 
                     self.recency_weight * recency_factor +
                     self.content_weight * content_factor)
        
        return importance


class EbbinghausBasedForgetting(ForgettingStrategy):
    """Забывание на основе кривой забывания Эббингауза."""
    
    def __init__(self, base_retention: float = 0.1,
                 decay_rate: float = 0.693,  # ln(2) для половинного периода в 1 день
                 access_boost: float = 2.0):
        """Инициализирует стратегию.
        
        Args:
            base_retention: Базовый коэффициент удержания.
            decay_rate: Скорость забывания.
            access_boost: Усиление при каждом доступе.
        """
        self.base_retention = base_retention
        self.decay_rate = decay_rate
        self.access_boost = access_boost
    
    def should_forget(self, items: Dict[str, MemoryItem]) -> List[str]:
        """Определяет элементы для забывания по кривой Эббингауза."""
        candidates = []
        current_time = time.time()
        
        for item_id, item in items.items():
            retention_probability = self._calculate_retention(item, current_time)
            
            # Используем стохастическое забывание
            if np.random.random() > retention_probability:
                candidates.append(item_id)
        
        logger.debug(f"Выбрано {len(candidates)} элементов для забывания")
        return candidates
    
    def _calculate_retention(self, item: MemoryItem, current_time: float) -> float:
        """Вычисляет вероятность удержания элемента в памяти."""
        # Время с последнего доступа
        time_since_access = current_time - item.last_accessed_at
        
        # Базовая вероятность забывания по формуле Эббингауза
        # R(t) = base_retention * e^(-decay_rate * t)
        base_probability = self.base_retention * np.exp(-self.decay_rate * time_since_access / 86400.0)  # в днях
        
        # Усиление от повторных доступов
        access_multiplier = 1.0 + (item.access_count - 1) * self.access_boost
        
        # Финальная вероятность удержания
        retention_probability = min(1.0, base_probability * access_multiplier)
        
        return retention_probability


class LeastRecentlyUsedForgetting(ForgettingStrategy):
    """Забывание наименее недавно использованных элементов (LRU)."""
    
    def __init__(self, max_capacity: int = 1000):
        """Инициализирует стратегию.
        
        Args:
            max_capacity: Максимальная вместимость памяти.
        """
        self.max_capacity = max_capacity
    
    def should_forget(self, items: Dict[str, MemoryItem]) -> List[str]:
        """Удаляет наименее недавно использованные элементы при превышении лимита."""
        if len(items) <= self.max_capacity:
            return []
        
        # Сортируем по времени последнего доступа
        sorted_items = sorted(items.items(), 
                            key=lambda x: x[1].last_accessed_at)
        
        # Удаляем самые старые элементы
        overflow_count = len(items) - self.max_capacity
        candidates = [item_id for item_id, _ in sorted_items[:overflow_count]]
        
        logger.debug(f"Выбрано {len(candidates)} LRU элементов для забывания")
        return candidates


class SemanticClusteringConsolidation(ConsolidationStrategy):
    """Консолидация на основе семантической кластеризации."""
    
    def __init__(self, similarity_threshold: float = 0.8,
                 min_cluster_size: int = 3):
        """Инициализирует стратегию.
        
        Args:
            similarity_threshold: Порог семантического сходства.
            min_cluster_size: Минимальный размер кластера для консолидации.
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
    
    def should_consolidate(self, stm_items: Dict[str, MemoryItem], 
                         ltm_items: Dict[str, MemoryItem]) -> List[str]:
        """Консолидирует семантически связанные элементы."""
        if len(stm_items) < self.min_cluster_size:
            return []
        
        # Создаем матрицу сходства
        item_ids = list(stm_items.keys())
        embeddings = [stm_items[item_id].embedding for item_id in item_ids]
        
        if not embeddings or embeddings[0] is None:
            return []
        
        similarity_matrix = self._calculate_similarity_matrix(embeddings)
        
        # Находим кластеры
        clusters = self._find_clusters(item_ids, similarity_matrix)
        
        # Выбираем элементы из больших кластеров для консолидации
        candidates = []
        for cluster in clusters:
            if len(cluster) >= self.min_cluster_size:
                candidates.extend(cluster)
        
        logger.debug(f"Найдено {len(clusters)} кластеров, выбрано {len(candidates)} элементов")
        return candidates
    
    def _calculate_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Вычисляет матрицу косинусного сходства."""
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Косинусное сходство
                dot_product = np.dot(embeddings[i], embeddings[j])
                norm_i = np.linalg.norm(embeddings[i])
                norm_j = np.linalg.norm(embeddings[j])
                
                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (norm_i * norm_j)
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    def _find_clusters(self, item_ids: List[str], 
                      similarity_matrix: np.ndarray) -> List[List[str]]:
        """Находит кластеры элементов на основе сходства."""
        n = len(item_ids)
        visited = set()
        clusters = []
        
        for i in range(n):
            if item_ids[i] in visited:
                continue
            
            # DFS для поиска связанных элементов
            cluster = []
            stack = [i]
            
            while stack:
                current = stack.pop()
                if item_ids[current] in visited:
                    continue
                
                visited.add(item_ids[current])
                cluster.append(item_ids[current])
                
                # Добавляем похожие элементы
                for j in range(n):
                    if (item_ids[j] not in visited and 
                        similarity_matrix[current][j] >= self.similarity_threshold):
                        stack.append(j)
            
            if cluster:
                clusters.append(cluster)
        
        return clusters


class AdaptiveConsolidation(ConsolidationStrategy):
    """Адаптивная консолидация, комбинирующая несколько стратегий."""
    
    def __init__(self, strategies: List[ConsolidationStrategy],
                 weights: Optional[List[float]] = None):
        """Инициализирует адаптивную стратегию.
        
        Args:
            strategies: Список стратегий консолидации.
            weights: Веса стратегий (если None, используются равные веса).
        """
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("Количество весов должно совпадать с количеством стратегий")
    
    def should_consolidate(self, stm_items: Dict[str, MemoryItem], 
                         ltm_items: Dict[str, MemoryItem]) -> List[str]:
        """Комбинирует решения нескольких стратегий."""
        candidate_scores: Dict[str, float] = {}
        
        # Собираем кандидатов от каждой стратегии
        for strategy, weight in zip(self.strategies, self.weights):
            candidates = strategy.should_consolidate(stm_items, ltm_items)
            
            for candidate in candidates:
                candidate_scores[candidate] = candidate_scores.get(candidate, 0) + weight
        
        # Нормализуем оценки
        total_weight = sum(self.weights)
        threshold = total_weight * 0.5  # Консолидируем, если больше половины стратегий согласны
        
        final_candidates = [candidate for candidate, score in candidate_scores.items()
                          if score >= threshold]
        
        logger.debug(f"Адаптивная консолидация: выбрано {len(final_candidates)} элементов")
        return final_candidates


class MemoryPressureMonitor:
    """Монитор давления памяти для адаптации стратегий."""
    
    def __init__(self, low_pressure_threshold: float = 0.7,
                 high_pressure_threshold: float = 0.9):
        """Инициализирует монитор.
        
        Args:
            low_pressure_threshold: Порог низкого давления памяти.
            high_pressure_threshold: Порог высокого давления памяти.
        """
        self.low_pressure_threshold = low_pressure_threshold
        self.high_pressure_threshold = high_pressure_threshold
    
    def get_memory_pressure(self, current_items: int, max_capacity: int) -> float:
        """Вычисляет текущее давление памяти.
        
        Args:
            current_items: Текущее количество элементов.
            max_capacity: Максимальная вместимость.
            
        Returns:
            Давление памяти от 0.0 до 1.0.
        """
        if max_capacity <= 0:
            return 1.0
        
        return min(current_items / max_capacity, 1.0)
    
    def should_be_aggressive(self, pressure: float) -> bool:
        """Определяет, нужно ли быть агрессивным в забывании."""
        return pressure >= self.high_pressure_threshold
    
    def should_be_conservative(self, pressure: float) -> bool:
        """Определяет, нужно ли быть консервативным в забывании."""
        return pressure <= self.low_pressure_threshold