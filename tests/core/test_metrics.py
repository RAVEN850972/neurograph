"""Тесты для утилит сбора метрик."""

import time
import pytest
import threading

from neurograph.core.utils.metrics import MetricsCollector, timed, global_metrics


class TestMetricsCollector:
    """Тесты для коллектора метрик."""
    
    def test_set_get_metric(self):
        """Проверка установки и получения метрик."""
        collector = MetricsCollector()
        
        # Установка метрики
        collector.set_metric("test_metric", 42)
        
        # Получение метрики
        value = collector.get_metric("test_metric")
        assert value == 42
        
        # Несуществующая метрика
        assert collector.get_metric("non_existent") is None
        assert collector.get_metric("non_existent", "default") == "default"
    
    def test_increment_counter(self):
        """Проверка работы счетчиков."""
        collector = MetricsCollector()
        
        # Инкремент счетчика
        value = collector.increment_counter("test_counter")
        assert value == 1
        
        # Повторный инкремент
        value = collector.increment_counter("test_counter")
        assert value == 2
        
        # Инкремент с указанием значения
        value = collector.increment_counter("test_counter", 10)
        assert value == 12
        
        # Получение значения счетчика
        assert collector.get_counter("test_counter") == 12
        
        # Сброс счетчика
        collector.reset_counter("test_counter")
        assert collector.get_counter("test_counter") == 0
    
    def test_record_time(self):
        """Проверка записи времени выполнения."""
        collector = MetricsCollector()
        
        # Запись времени
        collector.record_time("test_operation", 0.1)
        collector.record_time("test_operation", 0.2)
        collector.record_time("test_operation", 0.3)
        
        # Получение среднего времени
        avg_time = collector.get_average_time("test_operation")
        assert avg_time == 0.2
        
        # Сброс таймеров
        collector.reset_timers("test_operation")
        assert collector.get_average_time("test_operation") is None
    
    def test_get_all_metrics(self):
        """Проверка получения всех метрик."""
        collector = MetricsCollector()
        
        # Установка различных метрик
        collector.set_metric("metric1", "value1")
        collector.increment_counter("counter1", 5)
        collector.record_time("timer1", 0.5)
        
        # Получение всех метрик
        all_metrics = collector.get_all_metrics()
        
        assert all_metrics["metric1"] == "value1"
        assert all_metrics["counter.counter1"] == 5
        assert all_metrics["timer.timer1"] == 0.5
    
    def test_thread_safety(self):
        """Проверка потокобезопасности."""
        collector = MetricsCollector()
        
        def worker1():
            for i in range(100):
                collector.increment_counter("shared_counter")
        
        def worker2():
            for i in range(100):
                collector.increment_counter("shared_counter")
        
        # Запускаем потоки
        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Проверяем результат
        assert collector.get_counter("shared_counter") == 200


def test_timed_decorator():
    """Проверка декоратора для измерения времени."""
    collector = MetricsCollector()
    
    @timed("test_function", collector)
    def slow_function():
        time.sleep(0.1)
        return 42
    
    # Вызываем функцию
    result = slow_function()
    
    # Проверяем результат
    assert result == 42
    
    # Проверяем, что время было записано
    avg_time = collector.get_average_time("test_function")
    assert avg_time is not None
    assert 0.05 < avg_time < 0.2  # Примерно 0.1 секунды


def test_global_metrics():
    """Проверка работы глобального коллектора метрик."""
    # Устанавливаем метрику
    global_metrics.set_metric("global_test", "value")
    
    # Проверяем, что метрика установлена
    assert global_metrics.get_metric("global_test") == "value"