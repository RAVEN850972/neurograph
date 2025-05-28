"""Механизмы консолидации памяти и управления переходами между уровнями."""

from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
import time
import threading
from datetime import datetime
from enum import Enum
import uuid

from neurograph.memory.base import MemoryItem
from neurograph.memory.strategies import (
    ConsolidationStrategy, ForgettingStrategy, MemoryPressureMonitor,
    TimeBasedConsolidation, ImportanceBasedConsolidation,
    EbbinghausBasedForgetting, LeastRecentlyUsedForgetting
)
from neurograph.core.logging import get_logger
from neurograph.core.events import publish

logger = get_logger("memory.consolidation")


class ConsolidationEvent(Enum):
    """События консолидации памяти."""
    ITEM_CONSOLIDATED = "item_consolidated"
    ITEM_FORGOTTEN = "item_forgotten"
    CONSOLIDATION_STARTED = "consolidation_started"
    CONSOLIDATION_COMPLETED = "consolidation_completed"
    MEMORY_PRESSURE_HIGH = "memory_pressure_high"
    MEMORY_PRESSURE_LOW = "memory_pressure_low"


class ConsolidationManager:
    """Менеджер консолидации памяти между STM и LTM."""
    
    def __init__(self, 
                 consolidation_strategy: Optional[ConsolidationStrategy] = None,
                 forgetting_strategy: Optional[ForgettingStrategy] = None,
                 auto_consolidation_interval: float = 60.0,
                 enable_background_processing: bool = True):
        """Инициализирует менеджер консолидации.
        
        Args:
            consolidation_strategy: Стратегия консолидации.
            forgetting_strategy: Стратегия забывания.
            auto_consolidation_interval: Интервал автоматической консолидации в секундах.
            enable_background_processing: Включить фоновую обработку.
        """
        self.consolidation_strategy = consolidation_strategy or TimeBasedConsolidation()
        self.forgetting_strategy = forgetting_strategy or EbbinghausBasedForgetting()
        self.auto_consolidation_interval = auto_consolidation_interval
        self.enable_background_processing = enable_background_processing
        
        # Монитор давления памяти
        self.pressure_monitor = MemoryPressureMonitor()
        
        # Статистика консолидации
        self.stats = {
            "total_consolidated": 0,
            "total_forgotten": 0,
            "last_consolidation": None,
            "consolidation_count": 0
        }
        
        # Потокобезопасность
        self._lock = threading.RLock()
        
        # Фоновый поток
        self._background_thread = None
        self._background_running = False
        
        if enable_background_processing:
            self._start_background_processing()
    
    def _start_background_processing(self) -> None:
        """Запускает фоновую обработку консолидации."""
        with self._lock:
            if self._background_running:
                return
            
            self._background_running = True
            self._background_thread = threading.Thread(
                target=self._background_consolidation_loop, 
                daemon=True
            )
            self._background_thread.start()
            logger.info("Запущена фоновая консолидация памяти")
    
    def _stop_background_processing(self) -> None:
        """Останавливает фоновую обработку."""
        with self._lock:
            if not self._background_running:
                return
            
            self._background_running = False
            
            if self._background_thread and self._background_thread.is_alive():
                self._background_thread.join(timeout=1.0)
            
            logger.info("Остановлена фоновая консолидация памяти")
    
    def _background_consolidation_loop(self) -> None:
        """Цикл фоновой консолидации."""
        while self._background_running:
            try:
                time.sleep(self.auto_consolidation_interval)
                
                # Периодически вызываем внешнюю функцию консолидации
                # (в реальной системе это будет метод BiomorphicMemory)
                publish("memory.auto_consolidation_trigger", {
                    "timestamp": time.time(),
                    "manager_id": id(self)
                })
                
            except Exception as e:
                logger.error(f"Ошибка в фоновой консолидации: {str(e)}")
    
    def consolidate(self, stm_items: Dict[str, MemoryItem], 
                   ltm_items: Dict[str, MemoryItem],
                   max_stm_capacity: int,
                   max_ltm_capacity: int) -> Tuple[List[str], List[str]]:
        """Выполняет консолидацию памяти.
        
        Args:
            stm_items: Элементы кратковременной памяти.
            ltm_items: Элементы долговременной памяти.
            max_stm_capacity: Максимальная вместимость STM.
            max_ltm_capacity: Максимальная вместимость LTM.
            
        Returns:
            Кортеж (элементы_для_консолидации, элементы_для_забывания).
        """
        with self._lock:
            start_time = time.time()
            
            # Публикуем событие начала консолидации
            publish("memory.consolidation_started", {
                "stm_size": len(stm_items),
                "ltm_size": len(ltm_items),
                "timestamp": start_time
            })
            
            # Определяем давление памяти
            stm_pressure = self.pressure_monitor.get_memory_pressure(
                len(stm_items), max_stm_capacity
            )
            ltm_pressure = self.pressure_monitor.get_memory_pressure(
                len(ltm_items), max_ltm_capacity
            )
            
            # Публикуем события давления памяти
            if self.pressure_monitor.should_be_aggressive(stm_pressure):
                publish("memory.memory_pressure_high", {
                    "memory_type": "STM",
                    "pressure": stm_pressure,
                    "timestamp": start_time
                })
            
            if self.pressure_monitor.should_be_aggressive(ltm_pressure):
                publish("memory.memory_pressure_high", {
                    "memory_type": "LTM", 
                    "pressure": ltm_pressure,
                    "timestamp": start_time
                })
            
            # Определяем элементы для консолидации из STM в LTM
            consolidation_candidates = []
            if ltm_pressure < 0.95:  # Есть место в LTM
                consolidation_candidates = self.consolidation_strategy.should_consolidate(
                    stm_items, ltm_items
                )
            
            # Определяем элементы для забывания
            forgetting_candidates = []
            
            # Забываем из STM, если она переполнена
            if self.pressure_monitor.should_be_aggressive(stm_pressure):
                stm_forgetting = self.forgetting_strategy.should_forget(stm_items)
                # Не забываем элементы, которые планируем консолидировать
                stm_forgetting = [item_id for item_id in stm_forgetting 
                                if item_id not in consolidation_candidates]
                forgetting_candidates.extend(stm_forgetting)
            
            # Забываем из LTM, если она переполнена
            if self.pressure_monitor.should_be_aggressive(ltm_pressure):
                ltm_forgetting = self.forgetting_strategy.should_forget(ltm_items)
                forgetting_candidates.extend(ltm_forgetting)
            
            # Обновляем статистику
            self.stats["total_consolidated"] += len(consolidation_candidates)
            self.stats["total_forgotten"] += len(forgetting_candidates)
            self.stats["last_consolidation"] = start_time
            self.stats["consolidation_count"] += 1
            
            end_time = time.time()
            
            # Публикуем событие завершения консолидации
            publish("memory.consolidation_completed", {
                "consolidated_count": len(consolidation_candidates),
                "forgotten_count": len(forgetting_candidates),
                "duration": end_time - start_time,
                "stm_pressure": stm_pressure,
                "ltm_pressure": ltm_pressure,
                "timestamp": end_time
            })
            
            logger.debug(f"Консолидация завершена: {len(consolidation_candidates)} -> LTM, "
                        f"{len(forgetting_candidates)} забыто, давление STM: {stm_pressure:.2f}, "
                        f"давление LTM: {ltm_pressure:.2f}")
            
            return consolidation_candidates, forgetting_candidates
    
    def adapt_strategies(self, stm_pressure: float, ltm_pressure: float,
                        recent_performance: Dict[str, Any]) -> None:
        """Адаптирует стратегии на основе текущего состояния системы.
        
        Args:
            stm_pressure: Давление памяти STM.
            ltm_pressure: Давление памяти LTM.
            recent_performance: Данные о недавней производительности.
        """
        with self._lock:
            # Адаптируем параметры стратегий на основе давления памяти
            
            # Для TimeBasedConsolidation
            if isinstance(self.consolidation_strategy, TimeBasedConsolidation):
                if self.pressure_monitor.should_be_aggressive(stm_pressure):
                    # Ускоряем консолидацию при высоком давлении
                    self.consolidation_strategy.min_age_seconds *= 0.5
                elif self.pressure_monitor.should_be_conservative(stm_pressure):
                    # Замедляем консолидацию при низком давлении
                    self.consolidation_strategy.min_age_seconds *= 1.2
            
            # Для ImportanceBasedConsolidation
            if isinstance(self.consolidation_strategy, ImportanceBasedConsolidation):
                if self.pressure_monitor.should_be_aggressive(stm_pressure):
                    # Снижаем порог важности при высоком давлении
                    self.consolidation_strategy.importance_threshold *= 0.8
                elif self.pressure_monitor.should_be_conservative(stm_pressure):
                    # Повышаем порог важности при низком давлении
                    self.consolidation_strategy.importance_threshold *= 1.1
            
            # Для EbbinghausBasedForgetting
            if isinstance(self.forgetting_strategy, EbbinghausBasedForgetting):
                if self.pressure_monitor.should_be_aggressive(max(stm_pressure, ltm_pressure)):
                    # Ускоряем забывание при высоком давлении
                    self.forgetting_strategy.decay_rate *= 1.5
                elif self.pressure_monitor.should_be_conservative(max(stm_pressure, ltm_pressure)):
                    # Замедляем забывание при низком давлении
                    self.forgetting_strategy.decay_rate *= 0.8
            
            logger.debug("Стратегии консолидации адаптированы к текущим условиям")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику консолидации.
        
        Returns:
            Словарь со статистикой.
        """
        with self._lock:
            return {
                **self.stats,
                "strategy_type": type(self.consolidation_strategy).__name__,
                "forgetting_type": type(self.forgetting_strategy).__name__,
                "auto_interval": self.auto_consolidation_interval,
                "background_running": self._background_running
            }
    
    def shutdown(self) -> None:
        """Завершает работу менеджера консолидации."""
        self._stop_background_processing()
        logger.info("Менеджер консолидации завершил работу")


class MemoryTransition:
    """Представляет переход элемента памяти между уровнями."""
    
    def __init__(self, item_id: str, source_level: str, target_level: str, 
                 reason: str, timestamp: Optional[float] = None):
        """Инициализирует переход памяти.
        
        Args:
            item_id: Идентификатор элемента памяти.
            source_level: Исходный уровень памяти.
            target_level: Целевой уровень памяти.
            reason: Причина перехода.
            timestamp: Время перехода.
        """
        self.item_id = item_id
        self.source_level = source_level
        self.target_level = target_level
        self.reason = reason
        self.timestamp = timestamp or time.time()
        self.transition_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует переход в словарь."""
        return {
            "transition_id": self.transition_id,
            "item_id": self.item_id,
            "source_level": self.source_level,
            "target_level": self.target_level,
            "reason": self.reason,
            "timestamp": self.timestamp
        }


class TransitionLogger:
    """Логгер переходов памяти для анализа и отладки."""
    
    def __init__(self, max_history: int = 1000):
        """Инициализирует логгер переходов.
        
        Args:
            max_history: Максимальное количество переходов в истории.
        """
        self.max_history = max_history
        self.transitions: List[MemoryTransition] = []
        self._lock = threading.RLock()
    
    def log_transition(self, transition: MemoryTransition) -> None:
        """Логирует переход памяти.
        
        Args:
            transition: Переход для логирования.
        """
        with self._lock:
            self.transitions.append(transition)
            
            # Ограничиваем размер истории
            if len(self.transitions) > self.max_history:
                self.transitions = self.transitions[-self.max_history:]
            
            # Публикуем событие перехода
            publish("memory.transition_logged", transition.to_dict())
            
            logger.debug(f"Переход зарегистрирован: {transition.item_id} "
                        f"{transition.source_level} -> {transition.target_level} "
                        f"({transition.reason})")
    
    def get_transitions_by_item(self, item_id: str) -> List[MemoryTransition]:
        """Возвращает все переходы для конкретного элемента.
        
        Args:
            item_id: Идентификатор элемента.
            
        Returns:
            Список переходов.
        """
        with self._lock:
            return [t for t in self.transitions if t.item_id == item_id]
    
    def get_recent_transitions(self, hours: float = 24.0) -> List[MemoryTransition]:
        """Возвращает недавние переходы.
        
        Args:
            hours: Количество часов назад.
            
        Returns:
            Список недавних переходов.
        """
        with self._lock:
            cutoff_time = time.time() - (hours * 3600)
            return [t for t in self.transitions if t.timestamp >= cutoff_time]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику переходов.
        
        Returns:
            Словарь со статистикой.
        """
        with self._lock:
            if not self.transitions:
                return {"total_transitions": 0}
            
            # Подсчет переходов по типам
            consolidations = len([t for t in self.transitions 
                                if t.source_level == "STM" and t.target_level == "LTM"])
            forgettings = len([t for t in self.transitions 
                             if t.target_level == "FORGOTTEN"])
            
            # Подсчет по причинам
            reasons = {}
            for transition in self.transitions:
                reasons[transition.reason] = reasons.get(transition.reason, 0) + 1
            
            return {
                "total_transitions": len(self.transitions),
                "consolidations": consolidations,
                "forgettings": forgettings,
                "reasons": reasons,
                "oldest_transition": min(t.timestamp for t in self.transitions),
                "newest_transition": max(t.timestamp for t in self.transitions)
            }


class ConsolidationOrchestrator:
    """Оркестратор консолидации, координирующий работу всех компонентов."""
    
    def __init__(self, consolidation_manager: ConsolidationManager,
                 transition_logger: Optional[TransitionLogger] = None):
        """Инициализирует оркестратор.
        
        Args:
            consolidation_manager: Менеджер консолидации.
            transition_logger: Логгер переходов.
        """
        self.consolidation_manager = consolidation_manager
        self.transition_logger = transition_logger or TransitionLogger()
        
        # Подписываемся на события автоматической консолидации
        from neurograph.core.events import subscribe
        subscribe("memory.auto_consolidation_trigger", self._handle_auto_consolidation)
    
    def _handle_auto_consolidation(self, data: Dict[str, Any]) -> None:
        """Обрабатывает событие автоматической консолидации.
        
        Args:
            data: Данные события.
        """
        # В реальной реализации здесь будет вызов метода BiomorphicMemory
        logger.debug("Получено событие автоматической консолидации")
    
    def perform_consolidation(self, stm_items: Dict[str, MemoryItem],
                            ltm_items: Dict[str, MemoryItem],
                            max_stm_capacity: int,
                            max_ltm_capacity: int) -> Tuple[List[str], List[str]]:
        """Выполняет полную консолидацию с логированием переходов.
        
        Args:
            stm_items: Элементы кратковременной памяти.
            ltm_items: Элементы долговременной памяти.
            max_stm_capacity: Максимальная вместимость STM.
            max_ltm_capacity: Максимальная вместимость LTM.
            
        Returns:
            Кортеж (элементы_для_консолидации, элементы_для_забывания).
        """
        # Выполняем консолидацию
        to_consolidate, to_forget = self.consolidation_manager.consolidate(
            stm_items, ltm_items, max_stm_capacity, max_ltm_capacity
        )
        
        # Логируем переходы
        current_time = time.time()
        
        # Логируем консолидации STM -> LTM
        for item_id in to_consolidate:
            transition = MemoryTransition(
                item_id=item_id,
                source_level="STM",
                target_level="LTM",
                reason="consolidation",
                timestamp=current_time
            )
            self.transition_logger.log_transition(transition)
        
        # Логируем забывания
        for item_id in to_forget:
            # Определяем источник забывания
            source_level = "STM" if item_id in stm_items else "LTM"
            
            transition = MemoryTransition(
                item_id=item_id,
                source_level=source_level,
                target_level="FORGOTTEN",
                reason="forgetting",
                timestamp=current_time
            )
            self.transition_logger.log_transition(transition)
        
        return to_consolidate, to_forget
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Возвращает полную статистику консолидации.
        
        Returns:
            Словарь с полной статистикой.
        """
        consolidation_stats = self.consolidation_manager.get_statistics()
        transition_stats = self.transition_logger.get_statistics()
        
        return {
            "consolidation": consolidation_stats,
            "transitions": transition_stats,
            "orchestrator_id": id(self)
        }
    
    def shutdown(self) -> None:
        """Завершает работу оркестратора."""
        self.consolidation_manager.shutdown()
        logger.info("Оркестратор консолидации завершил работу")