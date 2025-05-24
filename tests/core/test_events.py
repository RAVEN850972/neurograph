"""Тесты для системы событий."""

import threading
import time
import pytest

from neurograph.core.events import EventBus, subscribe, unsubscribe, publish, has_subscribers


class TestEventBus:
    """Тесты для класса EventBus."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.event_bus = EventBus()
        self.received_events = []
    
    def test_subscribe_unsubscribe(self):
        """Проверка подписки и отписки от событий."""
        # Обработчик события
        def handler(data):
            self.received_events.append(data)
        
        # Подписываемся на событие
        subscription_id = self.event_bus.subscribe("test_event", handler)
        
        # Проверяем, что подписка создана
        assert self.event_bus.has_subscribers("test_event")
        
        # Отписываемся от события
        result = self.event_bus.unsubscribe("test_event", subscription_id)
        
        # Проверяем, что отписка прошла успешно
        assert result is True
        
        # Проверяем, что подписчиков больше нет
        assert not self.event_bus.has_subscribers("test_event")
    
    def test_publish_event(self):
        """Проверка публикации событий."""
        # Обработчик события
        def handler(data):
            self.received_events.append(data)
        
        # Подписываемся на событие
        self.event_bus.subscribe("test_event", handler)
        
        # Публикуем событие
        test_data = {"message": "Hello, World!"}
        self.event_bus.publish("test_event", test_data)
        
        # Проверяем, что событие получено
        assert len(self.received_events) == 1
        assert self.received_events[0] == test_data
    
    def test_multiple_subscribers(self):
        """Проверка работы с несколькими подписчиками."""
        # Обработчики событий
        def handler1(data):
            self.received_events.append(f"handler1: {data['message']}")
        
        def handler2(data):
            self.received_events.append(f"handler2: {data['message']}")
        
        # Подписываемся на события
        self.event_bus.subscribe("test_event", handler1)
        self.event_bus.subscribe("test_event", handler2)
        
        # Публикуем событие
        test_data = {"message": "Test"}
        self.event_bus.publish("test_event", test_data)
        
        # Проверяем, что событие получено обоими обработчиками
        assert len(self.received_events) == 2
        assert "handler1: Test" in self.received_events
        assert "handler2: Test" in self.received_events
    
    def test_handler_exception(self):
        """Проверка обработки исключений в обработчиках."""
        # Обработчик, вызывающий исключение
        def failing_handler(data):
            raise ValueError("Test exception")
        
        # Обычный обработчик
        def normal_handler(data):
            self.received_events.append(data)
        
        # Подписываемся на события
        self.event_bus.subscribe("test_event", failing_handler)
        self.event_bus.subscribe("test_event", normal_handler)
        
        # Публикуем событие - не должно выбросить исключение
        test_data = {"message": "Test"}
        self.event_bus.publish("test_event", test_data)
        
        # Проверяем, что нормальный обработчик получил событие
        assert len(self.received_events) == 1
        assert self.received_events[0] == test_data
    
    def test_event_in_thread(self):
        """Проверка работы с событиями в потоках."""
        # Флаг для контроля получения события
        event_received = threading.Event()
        
        # Обработчик события
        def handler(data):
            self.received_events.append(data)
            event_received.set()
        
        # Подписываемся на событие
        self.event_bus.subscribe("thread_event", handler)
        
        # Функция для публикации события в отдельном потоке
        def publish_in_thread():
            test_data = {"message": "Thread test"}
            self.event_bus.publish("thread_event", test_data)
        
        # Запускаем поток
        thread = threading.Thread(target=publish_in_thread)
        thread.start()
        
        # Ждем получения события
        event_received.wait(timeout=1.0)
        
        # Проверяем, что событие получено
        assert len(self.received_events) == 1
        assert self.received_events[0]["message"] == "Thread test"


def test_global_event_bus():
    """Проверка работы глобальной шины событий."""
    received_events = []
    
    # Обработчик события
    def handler(data):
        received_events.append(data)
    
    # Подписываемся на событие
    subscription_id = subscribe("global_event", handler)
    
    # Проверяем, что подписка создана
    assert has_subscribers("global_event")
    
    # Публикуем событие
    test_data = {"message": "Global event"}
    publish("global_event", test_data)
    
    # Проверяем, что событие получено
    assert len(received_events) == 1
    assert received_events[0] == test_data
    
    # Отписываемся от события
    result = unsubscribe("global_event", subscription_id)
    
    # Проверяем, что отписка прошла успешно
    assert result is True
    
    # Проверяем, что подписчиков больше нет
    assert not has_subscribers("global_event")