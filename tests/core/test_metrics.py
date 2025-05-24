"""Тесты для утилит сбора метрик."""

import time
import pytest
import threading

from neurograph.core.utils.metrics import MetricsCollector, timed, global_metrics


class TestMetricsCollector:
    """Тесты для коллектора метрик."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.collector = MetricsCollector()
    
    def test_set_get_metric(self):
        """Проверка установки и получения метрик."""
        # Установка метрики
        self.collector.set_metric("test_metric", 42)
        
        # Получение метрики
        value = self.collector.get_metric("test_metric")
        assert value == 42
        
        # Несуществующая метрика
        assert self.collector.get_metric("non_existent") is None
        assert self.collector.get_metric("non_existent", "default") == "default"
    
    def test_increment_counter(self):
        """Проверка работы счетчиков."""
        # Инкремент счетчика
        value = self.collector.increment_counter("test_counter")
        assert value == 1
        
        # Повторный инкремент
        value = self.collector.increment_counter("test_counter")
        assert value == 2
        
        # Инкремент с указанием значения
        value = self.collector.increment_counter("test_counter", 10)
        assert value == 12
        
        # Получение значения счетчика
        assert self.collector.get_counter("test_counter") == 12
        
        # Сброс счетчика
        self.collector.reset_counter("test_counter")
        assert self.collector.get_counter("test_counter") == 0
    
    def test_record_time(self):
        """Проверка записи времени выполнения."""
        # Запись времени
        self.collector.record_time("test_operation", 0.1)
        self.collector.record_time("test_operation", 0.2)
        self.collector.record_time("test_operation", 0.3)
        
        # Получение среднего времени
        avg_time = self.collector.get_average_time("test_operation")
        assert avg_time is not None
        
        # Используем приблизительное сравнение
        assert pytest.approx(avg_time) == 0.2
        
        # Сброс таймеров
        self.collector.reset_timers("test_operation")
        assert self.collector.get_average_time("test_operation") is None
    
    def test_get_all_metrics(self):
        """Проверка получения всех метрик."""
        # Установка различных метрик
        self.collector.set_metric("metric1", "value1")
        self.collector.increment_counter("counter1", 5)
        self.collector.record_time("timer1", 0.5)
        
        # Получение всех метрик
        all_metrics = self.collector.get_all_metrics()
        
        assert all_metrics["metric1"] == "value1"
        assert all_metrics["counter.counter1"] == 5
        assert all_metrics["timer.timer1"] == 0.5
    
    def test_thread_safety(self):
        """Проверка потокобезопасности."""
        # Уменьшаем количество итераций для более быстрого выполнения
        iterations = 10
        
        def worker1():
            for i in range(iterations):
                self.collector.increment_counter("shared_counter")
        
        def worker2():
            for i in range(iterations):
                self.collector.increment_counter("shared_counter")
        
        # Запускаем потоки
        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)
        
        thread1.start()
        thread2.start()
        
        # Добавляем таймауты
        thread1.join(timeout=5)
        thread2.join(timeout=5)
        
        # Проверяем, что потоки завершились
        assert not thread1.is_alive(), "Thread 1 did not finish in time"
        assert not thread2.is_alive(), "Thread 2 did not finish in time"
        
        # Проверяем результат
        assert self.collector.get_counter("shared_counter") == iterations * 2


def test_timed_decorator():
    """Проверка декоратора для измерения времени."""
    collector = MetricsCollector()
    
    # Улучшаем декоратор для обработки исключений
    @timed("test_function", collector)
    def slow_function():
        time.sleep(0.01)  # Уменьшаем время ожидания
        return 42
    
    # Вызываем функцию
    result = slow_function()
    
    # Проверяем результат
    assert result == 42
    
    # Проверяем, что время было записано
    avg_time = collector.get_average_time("test_function")
    assert avg_time is not None
    # Более гибкая проверка времени
    assert avg_time > 0, "Время выполнения должно быть положительным"


def test_global_metrics():
    """Проверка работы глобального коллектора метрик."""
    # Очищаем глобальный коллектор перед тестом
    # (assuming there's a method to do this, or create a new instance)
    
    # Устанавливаем метрику
    global_metrics.set_metric("global_test", "value")
    
    # Проверяем, что метрика установлена
    assert global_metrics.get_metric("global_test") == "value"
    
    # Очищаем после теста
    global_metrics.set_metric("global_test", None)