"""Утилиты для сбора и мониторинга метрик."""

import time
from typing import Dict, Any, List, Callable, Optional
import threading


class MetricsCollector:
    """Коллектор метрик для мониторинга производительности."""
    
    def __init__(self):
        """Инициализирует коллектор метрик."""
        self._metrics: Dict[str, Any] = {}
        self._counters: Dict[str, int] = {}
        self._timers: Dict[str, List[float]] = {}
        self._lock = threading.RLock()  # Используем RLock вместо Lock
    
    def set_metric(self, name: str, value: Any) -> None:
        """Устанавливает значение метрики.
        
        Args:
            name: Имя метрики.
            value: Значение метрики.
        """
        with self._lock:
            self._metrics[name] = value
    
    def get_metric(self, name: str, default: Any = None) -> Any:
        """Возвращает значение метрики.
        
        Args:
            name: Имя метрики.
            default: Значение по умолчанию, если метрика не найдена.
            
        Returns:
            Значение метрики или значение по умолчанию.
        """
        with self._lock:
            return self._metrics.get(name, default)
    
    def increment_counter(self, name: str, value: int = 1) -> int:
        """Увеличивает значение счетчика.
        
        Args:
            name: Имя счетчика.
            value: Значение для увеличения.
            
        Returns:
            Новое значение счетчика.
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = 0
            self._counters[name] += value
            return self._counters[name]
    
    def get_counter(self, name: str, default: int = 0) -> int:
        """Возвращает значение счетчика.
        
        Args:
            name: Имя счетчика.
            default: Значение по умолчанию, если счетчик не найден.
            
        Returns:
            Значение счетчика или значение по умолчанию.
        """
        with self._lock:
            return self._counters.get(name, default)
    
    def reset_counter(self, name: str) -> None:
        """Сбрасывает значение счетчика.
        
        Args:
            name: Имя счетчика.
        """
        with self._lock:
            if name in self._counters:
                self._counters[name] = 0
    
    def record_time(self, name: str, time_value: float) -> None:
        """Записывает время выполнения операции.
        
        Args:
            name: Имя операции.
            time_value: Время выполнения в секундах.
        """
        with self._lock:
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(time_value)
    
    def get_average_time(self, name: str) -> Optional[float]:
        """Возвращает среднее время выполнения операции.
        
        Args:
            name: Имя операции.
            
        Returns:
            Среднее время выполнения или None, если операция не найдена.
        """
        with self._lock:
            if name not in self._timers or not self._timers[name]:
                return None
            return sum(self._timers[name]) / len(self._timers[name])
    
    def reset_timers(self, name: Optional[str] = None) -> None:
        """Сбрасывает таймеры.
        
        Args:
            name: Имя операции для сброса или None для сброса всех таймеров.
        """
        with self._lock:
            if name is None:
                self._timers = {}
            elif name in self._timers:
                self._timers[name] = []
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Возвращает все метрики.
        
        Returns:
            Словарь со всеми метриками.
        """
        with self._lock:
            result = dict(self._metrics)
            for name, value in self._counters.items():
                result[f"counter.{name}"] = value
            for name, timer_values in self._timers.items():
                if timer_values:  # Проверяем, что список не пустой
                    result[f"timer.{name}"] = sum(timer_values) / len(timer_values)
            return result


def timed(name: str, collector: Optional[MetricsCollector] = None) -> Callable:
    """Декоратор для измерения времени выполнения функции.
    
    Args:
        name: Имя операции.
        collector: Коллектор метрик.
        
    Returns:
        Декоратор.
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                if collector is not None:
                    collector.record_time(name, end_time - start_time)
        return wrapper
    return decorator


# Создаем глобальный коллектор метрик
global_metrics = MetricsCollector()