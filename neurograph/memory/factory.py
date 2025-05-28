"""Фабрика для создания экземпляров памяти."""

from typing import Dict, Any, Type, Optional, List

from neurograph.core.utils.registry import Registry
from neurograph.memory.base import IMemory
from neurograph.memory.impl.biomorphic import BiomorphicMemory
from neurograph.core.logging import get_logger

logger = get_logger("memory.factory")

# Создаем регистр для типов памяти
memory_registry = Registry[IMemory]("memory")

# Регистрируем реализации
memory_registry.register("biomorphic", BiomorphicMemory)


class MemoryFactory:
    """Фабрика для создания экземпляров памяти."""
    
    @staticmethod
    def create(memory_type: str = "biomorphic", **kwargs) -> IMemory:
        """Создает экземпляр памяти указанного типа.
        
        Args:
            memory_type: Тип памяти ("biomorphic", и т.д.).
            **kwargs: Параметры для конструктора.
            
        Returns:
            Экземпляр IMemory.
            
        Raises:
            ValueError: Если указан неизвестный тип памяти.
        """
        logger.info(f"Создание памяти типа {memory_type} с параметрами {kwargs}")
        return memory_registry.create(memory_type, **kwargs)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> IMemory:
        """Создает экземпляр памяти из конфигурации.
        
        Args:
            config: Словарь с конфигурацией.
            
        Returns:
            Экземпляр IMemory.
            
        Raises:
            ValueError: Если не указан обязательный параметр 'type'.
        """
        if "type" not in config:
            raise ValueError("В конфигурации не указан тип памяти")
        
        memory_type = config.pop("type")
        logger.info(f"Создание памяти типа {memory_type} из конфигурации")
        return MemoryFactory.create(memory_type, **config)
    
    @staticmethod
    def register_implementation(name: str, implementation: Type[IMemory]) -> None:
        """Регистрирует новую реализацию памяти.
        
        Args:
            name: Имя реализации.
            implementation: Класс реализации.
        """
        memory_registry.register(name, implementation)
        logger.info(f"Зарегистрирована новая реализация памяти: {name}")
    
    @staticmethod
    def get_available_types() -> List[str]:
        """Возвращает список доступных типов памяти.
        
        Returns:
            Список доступных типов.
        """
        return memory_registry.get_names()


def create_default_biomorphic_memory(**kwargs) -> BiomorphicMemory:
    """Создает биоморфную память с настройками по умолчанию.
    
    Args:
        **kwargs: Дополнительные параметры для переопределения значений по умолчанию.
        
    Returns:
        Настроенная биоморфная память.
    """
    default_config = {
        "stm_capacity": 100,
        "ltm_capacity": 10000,
        "use_semantic_indexing": True,
        "auto_consolidation": True,
        "consolidation_interval": 300.0
    }
    
    # Обновляем конфигурацию переданными параметрами
    default_config.update(kwargs)
    
    logger.info("Создание биоморфной памяти с настройками по умолчанию")
    return BiomorphicMemory(**default_config)


def create_lightweight_memory(**kwargs) -> BiomorphicMemory:
    """Создает облегченную версию биоморфной памяти для ограниченных ресурсов.
    
    Args:
        **kwargs: Дополнительные параметры для переопределения значений по умолчанию.
        
    Returns:
        Облегченная биоморфная память.
    """
    lightweight_config = {
        "stm_capacity": 50,
        "ltm_capacity": 1000,
        "use_semantic_indexing": False,
        "auto_consolidation": True,
        "consolidation_interval": 600.0  # Реже консолидация
    }
    
    # Обновляем конфигурацию переданными параметрами
    lightweight_config.update(kwargs)
    
    logger.info("Создание облегченной биоморфной памяти")
    return BiomorphicMemory(**lightweight_config)


def create_high_performance_memory(**kwargs) -> BiomorphicMemory:
    """Создает высокопроизводительную версию биоморфной памяти.
    
    Args:
        **kwargs: Дополнительные параметры для переопределения значений по умолчанию.
        
    Returns:
        Высокопроизводительная биоморфная память.
    """
    performance_config = {
        "stm_capacity": 200,
        "ltm_capacity": 50000,
        "use_semantic_indexing": True,
        "auto_consolidation": True,
        "consolidation_interval": 120.0  # Чаще консолидация
    }
    
    # Обновляем конфигурацию переданными параметрами
    performance_config.update(kwargs)
    
    logger.info("Создание высокопроизводительной биоморфной памяти")
    return BiomorphicMemory(**performance_config)