# NeuroGraph Core: Документация для разработчиков

## Обзор модуля

**neurograph-core** — базовый модуль системы NeuroGraph, предоставляющий фундаментальные интерфейсы, утилиты и инфраструктуру для всех остальных компонентов системы.

## Основные компоненты

### 1. Базовые интерфейсы (`neurograph.core`)

#### Component
Базовый интерфейс для всех компонентов системы.

```python
from neurograph.core import Component

class MyComponent(Component):
    def __init__(self, component_id: str):
        super().__init__(component_id)
    
    def initialize(self) -> bool:
        """Инициализация компонента"""
        return True
    
    def shutdown(self) -> bool:
        """Очистка ресурсов"""
        return True
```

#### Configurable
Интерфейс для компонентов с поддержкой конфигурации.

```python
from neurograph.core import Configurable
from typing import Dict, Any

class ConfigurableComponent(Configurable):
    def configure(self, config: Dict[str, Any]) -> bool:
        """Настройка компонента"""
        self.param1 = config.get("param1", "default")
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """Получение текущей конфигурации"""
        return {"param1": self.param1}
```

### 2. Конфигурация (`neurograph.core.config`)

#### Configuration
Система управления настройками с поддержкой вложенных параметров.

```python
from neurograph.core.config import Configuration

# Создание конфигурации
config = Configuration({
    "database": {
        "host": "localhost",
        "port": 5432
    },
    "memory": {
        "stm_capacity": 100
    }
})

# Получение значений с точечной нотацией
host = config.get("database.host")  # "localhost"
capacity = config.get("memory.stm_capacity", 50)  # 100

# Установка значений
config.set("database.host", "remote-server")

# Обновление из словаря
config.update({
    "database": {"timeout": 30},
    "new_section": {"param": "value"}
})

# Сохранение и загрузка
config.save_to_file("config.json")
loaded_config = Configuration.load_from_file("config.json")
```

#### Работа с переменными окружения
```python
from neurograph.core.config import from_env, merge_configs

# Загрузка из переменных окружения (NEUROGRAPH_DATABASE__HOST=localhost)
env_config = from_env("NEUROGRAPH_")

# Объединение конфигураций
final_config = merge_configs(default_config, env_config)
```

### 3. Система регистрации (`neurograph.core.utils.registry`)

#### Registry
Система для динамической регистрации и создания компонентов.

```python
from neurograph.core.utils.registry import Registry
from typing import Protocol

# Определение интерфейса
class IProcessor(Protocol):
    def process(self, data: str) -> str: ...

# Создание регистра
processor_registry = Registry[IProcessor]("processors")

# Регистрация через декоратор
@processor_registry.decorator("text_processor")
class TextProcessor:
    def process(self, data: str) -> str:
        return data.upper()

# Или явная регистрация
processor_registry.register("simple_processor", SimpleProcessor)

# Создание компонентов
processor = processor_registry.create("text_processor")
result = processor.process("hello")  # "HELLO"

# Получение доступных типов
available = processor_registry.get_names()  # ["text_processor", "simple_processor"]
```

### 4. Система событий (`neurograph.core.events`)

#### Глобальная шина событий
```python
from neurograph.core.events import subscribe, publish, unsubscribe

# Подписка на события
def handle_memory_event(data):
    print(f"Memory event: {data}")

subscription_id = subscribe("memory.item_added", handle_memory_event)

# Публикация событий
publish("memory.item_added", {
    "item_id": "123",
    "content_type": "text"
})

# Отписка
unsubscribe("memory.item_added", subscription_id)
```

#### Локальная шина событий
```python
from neurograph.core.events import EventBus

# Создание локальной шины
bus = EventBus()
subscription_id = bus.subscribe("local_event", my_handler)
bus.publish("local_event", {"key": "value"})
```

### 5. Логирование (`neurograph.core.logging`)

#### Настройка логирования
```python
from neurograph.core.logging import setup_logging, get_logger

# Настройка глобального логирования
setup_logging(
    level="INFO",
    log_file="app.log",
    rotation="10 MB",
    retention="1 week"
)

# Получение логгера с контекстом
logger = get_logger("my_component")
logger.info("Component initialized")

# Логгер с дополнительным контекстом
logger = get_logger("processor", user_id="123", session="abc")
logger.debug("Processing request")
```

### 6. Обработка ошибок (`neurograph.core.errors`)

#### Иерархия исключений
```python
from neurograph.core.errors import (
    NeuroGraphError, ConfigurationError, ComponentError, 
    ValidationError, GraphError, VectorError
)

# Базовое использование
try:
    # операция
    pass
except ComponentError as e:
    logger.error(f"Component error: {e.message}")
    print(f"Details: {e.details}")

# Создание собственных исключений
raise ConfigurationError(
    "Invalid parameter", 
    details={"parameter": "database.host", "value": None}
)
```

### 7. Кеширование (`neurograph.core.cache`)

#### Глобальный кеш
```python
from neurograph.core.cache import global_cache, cached

# Прямое использование кеша
global_cache.set("key1", "value1", ttl=300)  # TTL 5 минут
value = global_cache.get("key1", "default")

# Декоратор для кеширования функций
@cached(ttl=600)  # TTL 10 минут
def expensive_operation(param1: str, param2: int) -> str:
    # тяжелая операция
    return f"result_{param1}_{param2}"

result = expensive_operation("test", 42)  # выполняется
result = expensive_operation("test", 42)  # берется из кеша
```

#### Локальный кеш
```python
from neurograph.core.cache import Cache

# Создание локального кеша
cache = Cache(max_size=1000, ttl=300)
cache.set("key", "value")

# Статистика кеша
stats = cache.stats()
print(f"Cache size: {stats['size']}, hits: {stats['hit_count']}")
```

### 8. Метрики (`neurograph.core.utils.metrics`)

#### Сбор метрик
```python
from neurograph.core.utils.metrics import global_metrics, timed

# Простые метрики
global_metrics.set_metric("active_users", 42)
global_metrics.increment_counter("requests_processed")
global_metrics.record_time("query_time", 0.5)

# Декоратор для измерения времени
@timed("expensive_function")
def expensive_function():
    # операция
    pass

# Получение метрик
avg_time = global_metrics.get_average_time("query_time")
all_metrics = global_metrics.get_all_metrics()
```

### 9. Сериализация (`neurograph.core.utils.serialization`)

#### JSON сериализация
```python
from neurograph.core.utils.serialization import JSONSerializer
import numpy as np

# Сериализация сложных объектов
data = {
    "vectors": np.array([1, 2, 3]),
    "metadata": {"param": "value"}
}

# В строку
json_str = JSONSerializer.serialize(data)

# В файл
JSONSerializer.save_to_file(data, "data.json")
loaded_data = JSONSerializer.load_from_file("data.json")
```

#### Pickle сериализация
```python
from neurograph.core.utils.serialization import PickleSerializer

# Для сложных объектов Python
PickleSerializer.save_to_file(my_object, "object.pkl")
loaded_object = PickleSerializer.load_from_file("object.pkl")
```

### 10. Мониторинг ресурсов (`neurograph.core.resources`)

#### Глобальный мониторинг
```python
from neurograph.core.resources import (
    start_resource_monitoring, 
    get_resource_usage,
    stop_resource_monitoring
)

# Запуск мониторинга
start_resource_monitoring(check_interval=5.0)

# Получение статистики
usage = get_resource_usage()
print(f"CPU: {usage['cpu_percent']}%")
print(f"Memory: {usage['memory_rss']} bytes")

# Остановка мониторинга
stop_resource_monitoring()
```

#### Локальный мониторинг
```python
from neurograph.core.resources import ResourceMonitor

monitor = ResourceMonitor(check_interval=1.0)
monitor.start_monitoring()

# Получение данных
stats = monitor.get_resource_usage()
print(f"Threads: {stats['thread_count']}")
```

### 11. Система плагинов (`neurograph.core.plugins`)

#### Регистрация плагинов
```python
from neurograph.core.plugins import plugin_manager, register_plugin
from neurograph.core.component import Component

# Создание реестра для типа компонентов
my_registry = plugin_manager.create_registry("my_components", Component)

# Регистрация плагина
class MyPlugin(Component):
    def initialize(self) -> bool:
        return True
    
    def shutdown(self) -> bool:
        return True

register_plugin("my_components", "my_plugin", MyPlugin)

# Создание экземпляра
plugin = my_registry.create("my_plugin", "component_id")
```

#### Загрузка плагинов из директории
```python
from neurograph.core.plugins import load_plugins

# Загрузка всех плагинов из директории
count = load_plugins("./plugins", module_prefix="neurograph.plugins")
print(f"Loaded {count} plugins")
```

## Быстрый старт

### Создание простого компонента
```python
from neurograph.core import Component, Configurable
from neurograph.core.logging import get_logger
from neurograph.core.events import subscribe, publish
from typing import Dict, Any

class MyProcessor(Component, Configurable):
    def __init__(self, component_id: str):
        super().__init__(component_id)
        self.logger = get_logger(f"processor.{component_id}")
        self.config = {}
        
        # Подписка на события
        subscribe("input.data", self._handle_input)
    
    def configure(self, config: Dict[str, Any]) -> bool:
        self.config = config
        self.logger.info(f"Configured with: {config}")
        return True
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
    
    def initialize(self) -> bool:
        self.logger.info("Processor initialized")
        return True
    
    def shutdown(self) -> bool:
        self.logger.info("Processor shutdown")
        return True
    
    def _handle_input(self, data: Dict[str, Any]) -> None:
        result = self.process(data["content"])
        publish("output.result", {"result": result})
    
    def process(self, content: str) -> str:
        return content.upper()
```

### Использование с фабрикой
```python
from neurograph.core.utils.registry import Registry

# Создание фабрики
processor_registry = Registry[MyProcessor]("processors")
processor_registry.register("default", MyProcessor)

# Создание и настройка компонента
processor = processor_registry.create("default", "proc_1")
processor.configure({"param1": "value1"})
processor.initialize()

# Использование
publish("input.data", {"content": "hello world"})
```

## Лучшие практики

1. **Всегда наследуйтесь от базовых интерфейсов** для обеспечения совместимости
2. **Используйте type hints** для всех публичных методов
3. **Логируйте важные события** с соответствующим уровнем
4. **Обрабатывайте исключения** используя иерархию NeuroGraphError
5. **Используйте события** для слабосвязанного взаимодействия между компонентами
6. **Применяйте кеширование** для часто используемых вычислений
7. **Регистрируйте компоненты** через Registry для гибкости конфигурации

## Конфигурация по умолчанию

```python
default_config = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "cache": {
        "max_size": 1000,
        "ttl": 300
    },
    "events": {
        "max_handlers": 100
    },
    "resources": {
        "monitoring_interval": 5.0
    }
}
```

## Примеры интеграции

### Создание комплексного компонента
```python
from neurograph.core import Component, Configurable
from neurograph.core.config import Configuration
from neurograph.core.logging import get_logger
from neurograph.core.events import subscribe, publish
from neurograph.core.cache import cached
from neurograph.core.utils.metrics import global_metrics, timed
from neurograph.core.utils.registry import Registry
from typing import Dict, Any, Optional, List
import time

class AdvancedProcessor(Component, Configurable):
    """Пример полнофункционального компонента с использованием всех возможностей Core"""
    
    def __init__(self, component_id: str):
        super().__init__(component_id)
        self.logger = get_logger(f"advanced_processor.{component_id}")
        self.config = Configuration()
        self.is_initialized = False
        self._subscriptions = []
        
    def configure(self, config: Dict[str, Any]) -> bool:
        """Настройка компонента с валидацией"""
        try:
            self.config.update(config)
            
            # Валидация конфигурации
            required_params = ["processing_mode", "timeout"]
            for param in required_params:
                if not self.config.get(param):
                    raise ValueError(f"Missing required parameter: {param}")
            
            self.logger.info(f"Component configured successfully")
            global_metrics.increment_counter("components_configured")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration failed: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.to_dict()
    
    def initialize(self) -> bool:
        """Инициализация с подпиской на события и настройкой метрик"""
        try:
            # Подписка на события
            subscription_id = subscribe("data.incoming", self._handle_incoming_data)
            self._subscriptions.append(("data.incoming", subscription_id))
            
            subscription_id = subscribe("system.shutdown", self._handle_shutdown)
            self._subscriptions.append(("system.shutdown", subscription_id))
            
            # Установка начальных метрик
            global_metrics.set_metric(f"processor.{self.id}.status", "active")
            global_metrics.set_metric(f"processor.{self.id}.processed_items", 0)
            
            self.is_initialized = True
            self.logger.info("Component initialized successfully")
            
            # Публикация события инициализации
            publish("component.initialized", {
                "component_id": self.id,
                "component_type": "AdvancedProcessor",
                "timestamp": time.time()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Корректное завершение работы с очисткой ресурсов"""
        try:
            # Отписка от событий
            for event_type, subscription_id in self._subscriptions:
                from neurograph.core.events import unsubscribe
                unsubscribe(event_type, subscription_id)
            self._subscriptions.clear()
            
            # Обновление метрик
            global_metrics.set_metric(f"processor.{self.id}.status", "shutdown")
            
            self.is_initialized = False
            self.logger.info("Component shutdown completed")
            
            # Публикация события завершения
            publish("component.shutdown", {
                "component_id": self.id,
                "timestamp": time.time()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")
            return False
    
    @timed("data_processing")
    @cached(ttl=60)
    def process_data(self, data: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Основной метод обработки данных с кешированием и метриками"""
        if not self.is_initialized:
            raise RuntimeError("Component not initialized")
        
        self.logger.debug(f"Processing data: {data[:50]}...")
        
        try:
            # Симуляция обработки
            processing_mode = self.config.get("processing_mode", "default")
            timeout = self.config.get("timeout", 30)
            
            # Основная логика обработки
            result = {
                "processed_data": f"[{processing_mode}] {data.upper()}",
                "processing_time": time.time(),
                "options": options or {},
                "processor_id": self.id
            }
            
            # Обновление метрик
            current_count = global_metrics.get_metric(f"processor.{self.id}.processed_items", 0)
            global_metrics.set_metric(f"processor.{self.id}.processed_items", current_count + 1)
            global_metrics.increment_counter("total_processed_items")
            
            self.logger.debug("Data processing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            global_metrics.increment_counter("processing_errors")
            raise
    
    def _handle_incoming_data(self, event_data: Dict[str, Any]) -> None:
        """Обработчик входящих данных через события"""
        try:
            data = event_data.get("data")
            options = event_data.get("options")
            
            if data:
                result = self.process_data(data, options)
                
                # Публикация результата
                publish("data.processed", {
                    "processor_id": self.id,
                    "result": result,
                    "timestamp": time.time()
                })
                
        except Exception as e:
            self.logger.error(f"Error handling incoming data: {e}")
            publish("data.processing_error", {
                "processor_id": self.id,
                "error": str(e),
                "timestamp": time.time()
            })
    
    def _handle_shutdown(self, event_data: Dict[str, Any]) -> None:
        """Обработчик события завершения системы"""
        self.logger.info("Received shutdown signal")
        self.shutdown()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики работы компонента"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "processed_items": global_metrics.get_metric(f"processor.{self.id}.processed_items", 0),
            "average_processing_time": global_metrics.get_average_time("data_processing"),
            "config": self.get_config(),
            "subscriptions": len(self._subscriptions)
        }
```

### Создание фабрики компонентов
```python
from neurograph.core.utils.registry import Registry
from typing import Type, Dict, Any

class ProcessorFactory:
    """Фабрика для создания процессоров разных типов"""
    
    def __init__(self):
        self.registry = Registry[Component]("processors")
        self._register_default_processors()
    
    def _register_default_processors(self):
        """Регистрация процессоров по умолчанию"""
        self.registry.register("advanced", AdvancedProcessor)
        # Можно добавить другие типы процессоров
    
    def create_processor(self, processor_type: str, component_id: str, 
                        config: Optional[Dict[str, Any]] = None) -> Component:
        """Создание и настройка процессора"""
        processor = self.registry.create(processor_type, component_id)
        
        if config and isinstance(processor, Configurable):
            if not processor.configure(config):
                raise RuntimeError(f"Failed to configure processor {component_id}")
        
        if not processor.initialize():
            raise RuntimeError(f"Failed to initialize processor {component_id}")
        
        return processor
    
    def get_available_types(self) -> List[str]:
        """Получение списка доступных типов процессоров"""
        return self.registry.get_names()

# Использование фабрики
factory = ProcessorFactory()

processor = factory.create_processor(
    processor_type="advanced",
    component_id="main_processor",
    config={
        "processing_mode": "enhanced",
        "timeout": 60
    }
)
```

### Система управления компонентами
```python
from neurograph.core.events import EventBus
from neurograph.core.logging import get_logger
from neurograph.core.config import Configuration
from typing import Dict, List, Any
import threading
import time

class ComponentManager:
    """Менеджер для управления жизненным циклом компонентов"""
    
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.logger = get_logger("component_manager")
        self.event_bus = EventBus()
        self._lock = threading.RLock()
        self._running = False
        
    def register_component(self, component: Component) -> bool:
        """Регистрация компонента в менеджере"""
        with self._lock:
            if component.id in self.components:
                self.logger.warning(f"Component {component.id} already registered")
                return False
            
            self.components[component.id] = component
            self.logger.info(f"Component {component.id} registered")
            
            # Подписка на события компонента
            self.event_bus.subscribe("component.initialized", self._handle_component_event)
            self.event_bus.subscribe("component.shutdown", self._handle_component_event)
            
            return True
    
    def unregister_component(self, component_id: str) -> bool:
        """Отмена регистрации компонента"""
        with self._lock:
            if component_id not in self.components:
                return False
            
            component = self.components[component_id]
            component.shutdown()
            del self.components[component_id]
            
            self.logger.info(f"Component {component_id} unregistered")
            return True
    
    def start_all(self) -> bool:
        """Запуск всех зарегистрированных компонентов"""
        with self._lock:
            success_count = 0
            
            for component_id, component in self.components.items():
                try:
                    if component.initialize():
                        success_count += 1
                        self.logger.info(f"Component {component_id} started")
                    else:
                        self.logger.error(f"Failed to start component {component_id}")
                except Exception as e:
                    self.logger.error(f"Error starting component {component_id}: {e}")
            
            self._running = success_count > 0
            self.logger.info(f"Started {success_count}/{len(self.components)} components")
            return success_count == len(self.components)
    
    def stop_all(self) -> bool:
        """Остановка всех компонентов"""
        with self._lock:
            success_count = 0
            
            # Публикуем событие завершения системы
            publish("system.shutdown", {"timestamp": time.time()})
            
            for component_id, component in self.components.items():
                try:
                    if component.shutdown():
                        success_count += 1
                        self.logger.info(f"Component {component_id} stopped")
                    else:
                        self.logger.error(f"Failed to stop component {component_id}")
                except Exception as e:
                    self.logger.error(f"Error stopping component {component_id}: {e}")
            
            self._running = False
            self.logger.info(f"Stopped {success_count}/{len(self.components)} components")
            return success_count == len(self.components)
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Получение компонента по ID"""
        return self.components.get(component_id)
    
    def get_component_status(self) -> Dict[str, Any]:
        """Получение статуса всех компонентов"""
        with self._lock:
            status = {
                "total_components": len(self.components),
                "running": self._running,
                "components": {}
            }
            
            for component_id, component in self.components.items():
                try:
                    if hasattr(component, 'get_statistics'):
                        status["components"][component_id] = component.get_statistics()
                    else:
                        status["components"][component_id] = {"status": "unknown"}
                except Exception as e:
                    status["components"][component_id] = {"status": "error", "error": str(e)}
            
            return status
    
    def _handle_component_event(self, event_data: Dict[str, Any]) -> None:
        """Обработчик событий компонентов"""
        component_id = event_data.get("component_id")
        self.logger.debug(f"Received event for component {component_id}")
```

## Паттерны использования

### 1. Паттерн "Конфигурируемая фабрика"
```python
from neurograph.core.config import Configuration
from neurograph.core.utils.registry import Registry

class ConfigurableFactory:
    """Фабрика с поддержкой конфигурации"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.registry = Registry[Component]("configurable_components")
    
    def create_from_config(self, section: str) -> Component:
        """Создание компонента из секции конфигурации"""
        component_config = self.config.get(section, {})
        component_type = component_config.get("type")
        component_id = component_config.get("id", section)
        
        if not component_type:
            raise ValueError(f"No type specified for {section}")
        
        component = self.registry.create(component_type, component_id)
        
        if isinstance(component, Configurable):
            component.configure(component_config)
        
        return component

# Использование
config = Configuration({
    "text_processor": {
        "type": "advanced",
        "id": "main_text_processor",
        "processing_mode": "enhanced",
        "timeout": 60
    }
})

factory = ConfigurableFactory(config)
processor = factory.create_from_config("text_processor")
```

### 2. Паттерн "Наблюдатель через события"
```python
from neurograph.core.events import subscribe, publish
from typing import Callable, List

class EventObserver:
    """Наблюдатель для отслеживания событий системы"""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
        self.subscriptions: Dict[str, str] = {}
    
    def observe(self, event_type: str, handler: Callable) -> None:
        """Добавление обработчика для типа события"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
            # Подписываемся на событие только при первом обработчике
            subscription_id = subscribe(event_type, self._handle_event)
            self.subscriptions[event_type] = subscription_id
        
        self.handlers[event_type].append(handler)
    
    def _handle_event(self, event_data: Dict[str, Any]) -> None:
        """Внутренний обработчик событий"""
        # Определяем тип события из контекста или данных
        for event_type, handlers in self.handlers.items():
            for handler in handlers:
                try:
                    handler(event_data)
                except Exception as e:
                    # Логирование ошибок обработчиков
                    pass

# Использование
observer = EventObserver()

def log_component_events(data):
    print(f"Component event: {data}")

def update_metrics(data):
    global_metrics.increment_counter("component_events")

observer.observe("component.initialized", log_component_events)
observer.observe("component.initialized", update_metrics)
```

### 3. Паттерн "Middleware"
```python
from typing import Callable, Any, List
from functools import wraps

class MiddlewareManager:
    """Менеджер middleware для обработки запросов"""
    
    def __init__(self):
        self.middlewares: List[Callable] = []
    
    def add_middleware(self, middleware: Callable) -> None:
        """Добавление middleware"""
        self.middlewares.append(middleware)
    
    def process(self, data: Any, final_handler: Callable) -> Any:
        """Обработка данных через цепочку middleware"""
        def create_next(index: int):
            if index >= len(self.middlewares):
                return final_handler
            
            def next_handler(data):
                return self.middlewares[index](data, create_next(index + 1))
            
            return next_handler
        
        return create_next(0)(data)

# Middleware для логирования
def logging_middleware(data: Any, next_handler: Callable) -> Any:
    logger = get_logger("middleware.logging")
    logger.info(f"Processing: {type(data)}")
    
    start_time = time.time()
    result = next_handler(data)
    end_time = time.time()
    
    logger.info(f"Completed in {end_time - start_time:.2f}s")
    return result

# Middleware для кеширования
def caching_middleware(data: Any, next_handler: Callable) -> Any:
    cache_key = f"middleware_{hash(str(data))}"
    
    cached_result = global_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    result = next_handler(data)
    global_cache.set(cache_key, result, ttl=300)
    return result

# Использование
manager = MiddlewareManager()
manager.add_middleware(logging_middleware)
manager.add_middleware(caching_middleware)

def final_processor(data):
    return f"Processed: {data}"

result = manager.process("test data", final_processor)
```

## Отладка и диагностика

### Включение детального логирования
```python
from neurograph.core.logging import setup_logging, set_log_level

# Настройка детального логирования для отладки
setup_logging(
    level="DEBUG",
    log_file="debug.log",
    format_string="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{extra[name]}</cyan> | <level>{message}</level>"
)

# Изменение уровня логирования в runtime
set_log_level("DEBUG")
```

### Мониторинг производительности
```python
from neurograph.core.utils.metrics import global_metrics
from neurograph.core.resources import get_resource_usage
import json

def get_system_diagnostics() -> Dict[str, Any]:
    """Получение полной диагностики системы"""
    return {
        "metrics": global_metrics.get_all_metrics(),
        "resources": get_resource_usage(),
        "cache_stats": global_cache.stats(),
        "timestamp": time.time()
    }

# Периодический вывод диагностики
def print_diagnostics():
    diagnostics = get_system_diagnostics()
    print(json.dumps(diagnostics, indent=2))

# Установка таймера для периодической диагностики
import threading
timer = threading.Timer(60.0, print_diagnostics)  # каждую минуту
timer.daemon = True
timer.start()
```

## Миграция и обновления

### Версионирование конфигурации
```python
from neurograph.core.config import Configuration

class ConfigMigrator:
    """Мигратор для обновления конфигурации между версиями"""
    
    VERSION_KEY = "_config_version"
    CURRENT_VERSION = "1.2.0"
    
    def __init__(self):
        self.migrations = {
            "1.0.0": self._migrate_1_0_to_1_1,
            "1.1.0": self._migrate_1_1_to_1_2,
        }
    
    def migrate(self, config: Configuration) -> Configuration:
        """Миграция конфигурации до текущей версии"""
        version = config.get(self.VERSION_KEY, "1.0.0")
        
        while version != self.CURRENT_VERSION:
            if version not in self.migrations:
                raise ValueError(f"No migration path from version {version}")
            
            config = self.migrations[version](config)
            version = config.get(self.VERSION_KEY)
        
        return config
    
    def _migrate_1_0_to_1_1(self, config: Configuration) -> Configuration:
        """Миграция с версии 1.0.0 на 1.1.0"""
        # Переименование параметров
        if config.get("old_param"):
            config.set("new_param", config.get("old_param"))
        
        config.set(self.VERSION_KEY, "1.1.0")
        return config
    
    def _migrate_1_1_to_1_2(self, config: Configuration) -> Configuration:
        """Миграция с версии 1.1.0 на 1.2.0"""
        # Добавление новых значений по умолчанию
        config.set("cache.enabled", True)
        config.set(self.VERSION_KEY, "1.2.0")
        return config
```

## Заключение

Модуль **neurograph-core** предоставляет все необходимые инструменты для создания надежных, масштабируемых и легко поддерживаемых компонентов системы NeuroGraph. Используйте предоставленные интерфейсы, утилиты и паттерны для обеспечения единообразия и качества вашего кода.

**Ключевые принципы при работе с Core:**
- Всегда наследуйтесь от базовых интерфейсов
- Используйте систему событий для слабосвязанного взаимодействия
- Применяйте логирование и метрики для мониторинга
- Конфигурируйте компоненты через единую систему
- Используйте регистры для гибкого управления компонентами

Это обеспечит совместимость с остальными модулями системы и упростит разработку, тестирование и поддержку.