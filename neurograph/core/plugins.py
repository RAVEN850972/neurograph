# neurograph/core/plugins.py

import os
import sys
import importlib
import inspect
from typing import Dict, List, Any, Callable, Optional, Type, TypeVar, Generic
import threading

from neurograph.core.logging import get_logger
from neurograph.core.component import Component

logger = get_logger("core.plugins")

T = TypeVar('T')

class PluginRegistry(Generic[T]):
    """Реестр плагинов определенного типа."""
    
    def __init__(self, name: str, base_class: Type[T]):
        """Инициализирует реестр плагинов.
        
        Args:
            name: Имя реестра.
            base_class: Базовый класс для плагинов этого типа.
        """
        self.name = name
        self.base_class = base_class
        self._registry: Dict[str, Type[T]] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, plugin_class: Type[T]) -> None:
        """Регистрирует плагин в реестре.
        
        Args:
            name: Имя плагина.
            plugin_class: Класс плагина.
            
        Raises:
            ValueError: Если плагин с таким именем уже зарегистрирован или
                если класс не является подклассом базового класса.
        """
        with self._lock:
            if name in self._registry:
                raise ValueError(f"Плагин с именем '{name}' уже зарегистрирован в реестре '{self.name}'")
            
            if not issubclass(plugin_class, self.base_class):
                raise ValueError(f"Класс '{plugin_class.__name__}' не является подклассом '{self.base_class.__name__}'")
            
            self._registry[name] = plugin_class
            logger.info(f"Зарегистрирован плагин '{name}' в реестре '{self.name}'")
    
    def unregister(self, name: str) -> bool:
        """Отменяет регистрацию плагина.
        
        Args:
            name: Имя плагина.
            
        Returns:
            True, если плагин был удален, иначе False.
        """
        with self._lock:
            if name in self._registry:
                del self._registry[name]
                logger.info(f"Отменена регистрация плагина '{name}' в реестре '{self.name}'")
                return True
            return False
    
    def get(self, name: str) -> Optional[Type[T]]:
        """Возвращает класс плагина по имени.
        
        Args:
            name: Имя плагина.
            
        Returns:
            Класс плагина или None, если плагин не найден.
        """
        with self._lock:
            return self._registry.get(name)
    
    def create(self, name: str, *args, **kwargs) -> T:
        """Создает экземпляр плагина.
        
        Args:
            name: Имя плагина.
            *args: Позиционные аргументы для конструктора.
            **kwargs: Именованные аргументы для конструктора.
            
        Returns:
            Экземпляр плагина.
            
        Raises:
            ValueError: Если плагин с указанным именем не найден.
        """
        with self._lock:
            plugin_class = self.get(name)
            if plugin_class is None:
                raise ValueError(f"Плагин с именем '{name}' не найден в реестре '{self.name}'")
            
            return plugin_class(*args, **kwargs)
    
    def get_all_names(self) -> List[str]:
        """Возвращает имена всех зарегистрированных плагинов.
        
        Returns:
            Список имен плагинов.
        """
        with self._lock:
            return list(self._registry.keys())
    
    def get_all_classes(self) -> Dict[str, Type[T]]:
        """Возвращает словарь со всеми зарегистрированными плагинами.
        
        Returns:
            Словарь {имя: класс}.
        """
        with self._lock:
            return dict(self._registry)


class PluginManager:
    """Менеджер плагинов для управления всеми реестрами плагинов."""
    
    def __init__(self):
        """Инициализирует менеджер плагинов."""
        self._registries: Dict[str, PluginRegistry] = {}
        self._lock = threading.RLock()
    
    def create_registry(self, name: str, base_class: Type[T]) -> PluginRegistry[T]:
        """Создает новый реестр плагинов.
        
        Args:
            name: Имя реестра.
            base_class: Базовый класс для плагинов этого типа.
            
        Returns:
            Созданный реестр.
            
        Raises:
            ValueError: Если реестр с таким именем уже существует.
        """
        with self._lock:
            if name in self._registries:
                raise ValueError(f"Реестр с именем '{name}' уже существует")
            
            registry = PluginRegistry(name, base_class)
            self._registries[name] = registry
            
            return registry
    
    def get_registry(self, name: str) -> Optional[PluginRegistry]:
        """Возвращает реестр плагинов по имени.
        
        Args:
            name: Имя реестра.
            
        Returns:
            Реестр плагинов или None, если реестр не найден.
        """
        with self._lock:
            return self._registries.get(name)
    
    def load_plugins_from_directory(self, directory: str, module_prefix: str = "") -> int:
        """Загружает плагины из указанной директории.
        
        Args:
            directory: Путь к директории с плагинами.
            module_prefix: Префикс для имен модулей.
            
        Returns:
            Количество загруженных плагинов.
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            logger.warning(f"Директория с плагинами не существует: {directory}")
            return 0
        
        # Добавляем директорию в sys.path, если ее там нет
        if directory not in sys.path:
            sys.path.append(directory)
        
        count = 0
        
        # Перебираем все файлы в директории
        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = os.path.splitext(filename)[0]
                
                # Добавляем префикс к имени модуля, если указан
                if module_prefix:
                    module_name = f"{module_prefix}.{module_name}"
                
                try:
                    # Импортируем модуль
                    module = importlib.import_module(module_name)
                    
                    # Ищем в модуле классы, являющиеся подклассами базовых классов плагинов
                    for registry in self._registries.values():
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, registry.base_class) and 
                                obj is not registry.base_class and
                                obj.__module__ == module.__name__):
                                
                                # Используем имя класса в качестве имени плагина
                                plugin_name = obj.__name__
                                
                                # Регистрируем плагин
                                try:
                                    registry.register(plugin_name, obj)
                                    count += 1
                                except ValueError as e:
                                    logger.warning(f"Не удалось зарегистрировать плагин '{plugin_name}': {str(e)}")
                
                except Exception as e:
                    logger.error(f"Ошибка при загрузке модуля '{module_name}': {str(e)}")
        
        logger.info(f"Загружено {count} плагинов из директории {directory}")
        return count


# Глобальный менеджер плагинов
plugin_manager = PluginManager()

# Создаем реестры для стандартных типов плагинов
component_registry = plugin_manager.create_registry("components", Component)

def register_plugin(registry_name: str, plugin_name: str, plugin_class: Type) -> None:
    """Регистрирует плагин в указанном реестре.
    
    Args:
        registry_name: Имя реестра.
        plugin_name: Имя плагина.
        plugin_class: Класс плагина.
        
    Raises:
        ValueError: Если реестр не найден или произошла ошибка при регистрации плагина.
    """
    registry = plugin_manager.get_registry(registry_name)
    if registry is None:
        raise ValueError(f"Реестр с именем '{registry_name}' не найден")
    
    registry.register(plugin_name, plugin_class)

def load_plugins(directory: str, module_prefix: str = "") -> int:
    """Загружает плагины из указанной директории.
    
    Args:
        directory: Путь к директории с плагинами.
        module_prefix: Префикс для имен модулей.
        
    Returns:
        Количество загруженных плагинов.
    """
    return plugin_manager.load_plugins_from_directory(directory, module_prefix)