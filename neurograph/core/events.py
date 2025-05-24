# neurograph/core/events.py

from typing import Dict, List, Any, Callable, Optional
import threading
import uuid

EventHandler = Callable[[Dict[str, Any]], None]

class EventBus:
    """Шина событий для асинхронного взаимодействия между компонентами."""
    
    def __init__(self):
        """Инициализирует шину событий."""
        self._handlers: Dict[str, Dict[str, EventHandler]] = {}
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: str, handler: EventHandler) -> str:
        """Подписывает обработчик на событие.
        
        Args:
            event_type: Тип события.
            handler: Функция-обработчик события.
            
        Returns:
            Идентификатор подписки.
        """
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = {}
                
            subscription_id = str(uuid.uuid4())
            self._handlers[event_type][subscription_id] = handler
            
            return subscription_id
    
    def unsubscribe(self, event_type: str, subscription_id: str) -> bool:
        """Отписывает обработчик от события.
        
        Args:
            event_type: Тип события.
            subscription_id: Идентификатор подписки.
            
        Returns:
            True, если отписка прошла успешно, иначе False.
        """
        with self._lock:
            if event_type not in self._handlers:
                return False
                
            if subscription_id not in self._handlers[event_type]:
                return False
                
            del self._handlers[event_type][subscription_id]
            
            # Если обработчиков больше нет, удаляем запись о типе события
            if not self._handlers[event_type]:
                del self._handlers[event_type]
                
            return True
    
    def publish(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Публикует событие.
        
        Args:
            event_type: Тип события.
            data: Данные события.
        """
        if data is None:
            data = {}
            
        # Создаем копию обработчиков, чтобы избежать проблем при изменении во время итерации
        handlers = {}
        with self._lock:
            if event_type in self._handlers:
                handlers = dict(self._handlers[event_type])
        
        # Вызываем обработчики вне блока блокировки
        for handler in handlers.values():
            try:
                handler(data)
            except Exception as e:
                from neurograph.core.logging import get_logger
                logger = get_logger("events")
                logger.error(f"Ошибка при обработке события {event_type}: {str(e)}")
    
    def has_subscribers(self, event_type: str) -> bool:
        """Проверяет, есть ли подписчики на событие.
        
        Args:
            event_type: Тип события.
            
        Returns:
            True, если есть подписчики, иначе False.
        """
        with self._lock:
            return event_type in self._handlers and bool(self._handlers[event_type])

# Глобальная шина событий
global_event_bus = EventBus()

def subscribe(event_type: str, handler: EventHandler) -> str:
    """Подписывает обработчик на событие в глобальной шине.
    
    Args:
        event_type: Тип события.
        handler: Функция-обработчик события.
        
    Returns:
        Идентификатор подписки.
    """
    return global_event_bus.subscribe(event_type, handler)

def unsubscribe(event_type: str, subscription_id: str) -> bool:
    """Отписывает обработчик от события в глобальной шине.
    
    Args:
        event_type: Тип события.
        subscription_id: Идентификатор подписки.
        
    Returns:
        True, если отписка прошла успешно, иначе False.
    """
    return global_event_bus.unsubscribe(event_type, subscription_id)

# Добавим эту функцию
def has_subscribers(event_type: str) -> bool:
    """Проверяет, есть ли подписчики на событие в глобальной шине.
    
    Args:
        event_type: Тип события.
        
    Returns:
        True, если есть подписчики, иначе False.
    """
    return global_event_bus.has_subscribers(event_type)

def publish(event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Публикует событие в глобальной шине.
    
    Args:
        event_type: Тип события.
        data: Данные события.
    """
    global_event_bus.publish(event_type, data)