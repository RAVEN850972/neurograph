"""Фабрика для создания экземпляров векторных представлений."""

from typing import Dict, Any, Type, Optional, List

from neurograph.core.utils.registry import Registry
from neurograph.contextvec.base import IContextVectors
from neurograph.contextvec.impl.static import StaticContextVectors
from neurograph.contextvec.impl.dynamic import DynamicContextVectors
from neurograph.core.logging import get_logger


logger = get_logger("contextvec.factory")


# Создаем регистр для векторных представлений
vectors_registry = Registry[IContextVectors]("contextvec")

# Регистрируем реализации
vectors_registry.register("static", StaticContextVectors)
vectors_registry.register("dynamic", DynamicContextVectors)


class ContextVectorsFactory:
    """Фабрика для создания экземпляров векторных представлений."""
    
    @staticmethod
    def create(vectors_type: str = "dynamic", **kwargs) -> IContextVectors:
        """Создает экземпляр векторных представлений указанного типа.
        
        Args:
            vectors_type: Тип векторных представлений ("static", "dynamic", и т.д.).
            **kwargs: Параметры для конструктора.
            
        Returns:
            Экземпляр IContextVectors.
            
        Raises:
            ValueError: Если указан неизвестный тип векторных представлений.
        """
        logger.info(f"Создание векторных представлений типа {vectors_type} с параметрами {kwargs}")
        return vectors_registry.create(vectors_type, **kwargs)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> IContextVectors:
        """Создает экземпляр векторных представлений из конфигурации.
        
        Args:
            config: Словарь с конфигурацией.
            
        Returns:
            Экземпляр IContextVectors.
            
        Raises:
            ValueError: Если не указан обязательный параметр 'type'.
        """
        if "type" not in config:
            raise ValueError("В конфигурации не указан тип векторных представлений")
        
        vectors_type = config.pop("type")
        logger.info(f"Создание векторных представлений типа {vectors_type} из конфигурации")
        return ContextVectorsFactory.create(vectors_type, **config)
    
    @staticmethod
    def register_implementation(name: str, implementation: Type[IContextVectors]) -> None:
        """Регистрирует новую реализацию векторных представлений.
        
        Args:
            name: Имя реализации.
            implementation: Класс реализации.
        """
        vectors_registry.register(name, implementation)
        logger.info(f"Зарегистрирована новая реализация векторных представлений: {name}")
    
    @staticmethod
    def get_available_types() -> List[str]:
        """Возвращает список доступных типов векторных представлений.
        
        Returns:
            Список доступных типов.
        """
        return vectors_registry.get_names()