"""Фабрика для создания экземпляров процессоров."""

from typing import Dict, Any, Optional, Type
from neurograph.core.utils.registry import Registry
from .base import INeuroSymbolicProcessor, ProcessorError
from .impl.pattern_matching import PatternMatchingProcessor
from .impl.graph_based import GraphBasedProcessor


class ProcessorFactory:
    """Фабрика для создания экземпляров процессоров."""
    
    _registry = Registry[INeuroSymbolicProcessor]("processors")
    
    @classmethod
    def register_processor(cls, name: str, processor_class: Type[INeuroSymbolicProcessor]):
        """Регистрирует процессор в фабрике.
        
        Args:
            name: Имя процессора.
            processor_class: Класс процессора.
        """
        cls._registry.register(name, processor_class)
    
    @classmethod
    def create(cls, processor_type: str = "pattern_matching", **kwargs) -> INeuroSymbolicProcessor:
        """Создает экземпляр процессора указанного типа.
        
        Args:
            processor_type: Тип процессора.
            **kwargs: Параметры для конструктора процессора.
            
        Returns:
            Экземпляр процессора.
            
        Raises:
            ProcessorError: Если тип процессора неизвестен.
        """
        try:
            return cls._registry.create(processor_type, **kwargs)
        except Exception as e:
            raise ProcessorError(f"Не удалось создать процессор типа '{processor_type}': {e}")
    
    @classmethod
    def get_available_types(cls) -> list[str]:
        """Возвращает список доступных типов процессоров.
        
        Returns:
            Список имен доступных процессоров.
        """
        return cls._registry.get_names()
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> INeuroSymbolicProcessor:
        """Создает процессор из конфигурации.
        
        Args:
            config: Словарь с конфигурацией.
            
        Returns:
            Экземпляр процессора.
        """
        processor_type = config.get("type", "pattern_matching")
        processor_config = config.get("config", {})
        
        return cls.create(processor_type, **processor_config)


# Регистрация стандартных процессоров
ProcessorFactory.register_processor("pattern_matching", PatternMatchingProcessor)
ProcessorFactory.register_processor("graph_based", GraphBasedProcessor)


def create_default_processor(**kwargs) -> INeuroSymbolicProcessor:
    """Создает процессор с настройками по умолчанию.
    
    Args:
        **kwargs: Дополнительные параметры.
        
    Returns:
        Экземпляр процессора с базовыми настройками.
    """
    default_config = {
        "confidence_threshold": 0.5,
        "max_depth": 5,
        "enable_explanations": True,
        "cache_rules": True
    }
    default_config.update(kwargs)
    
    return ProcessorFactory.create("pattern_matching", **default_config)


def create_high_performance_processor(**kwargs) -> INeuroSymbolicProcessor:
    """Создает высокопроизводительный процессор.
    
    Args:
        **kwargs: Дополнительные параметры.
        
    Returns:
        Экземпляр процессора, оптимизированного для производительности.
    """
    performance_config = {
        "confidence_threshold": 0.3,
        "max_depth": 10,
        "enable_explanations": False,
        "cache_rules": True,
        "parallel_processing": True,
        "rule_indexing": True
    }
    performance_config.update(kwargs)
    
    return ProcessorFactory.create("pattern_matching", **performance_config)


def create_graph_processor(graph_provider, **kwargs) -> INeuroSymbolicProcessor:
    """Создает процессор, интегрированный с графом знаний.
    
    Args:
        graph_provider: Провайдер графа знаний.
        **kwargs: Дополнительные параметры.
        
    Returns:
        Экземпляр процессора, работающего с графом.
    """
    graph_config = {
        "graph_provider": graph_provider,
        "use_graph_structure": True,
        "confidence_threshold": 0.4,
        "max_depth": 7
    }
    graph_config.update(kwargs)
    
    return ProcessorFactory.create("graph_based", **graph_config)