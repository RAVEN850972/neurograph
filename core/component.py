"""Базовые классы и интерфейсы компонентов системы."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

from loguru import logger


class Component(ABC):
    """Базовый интерфейс для всех компонентов системы."""

    def __init__(self, component_id: str):
        """Инициализирует компонент.
        
        Args:
            component_id: Уникальный идентификатор компонента.
        """
        self.id = component_id
        self.logger = logger.bind(component=self.__class__.__name__, id=component_id)
        
    @abstractmethod
    def initialize(self) -> bool:
        """Инициализирует компонент после создания.
        
        Returns:
            True, если инициализация прошла успешно, иначе False.
        """
        pass
        
    @abstractmethod
    def shutdown(self) -> bool:
        """Выполняет очистку ресурсов при завершении работы компонента.
        
        Returns:
            True, если очистка прошла успешно, иначе False.
        """
        pass


class Configurable(ABC):
    """Интерфейс для компонентов, поддерживающих конфигурацию."""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """Настраивает компонент с использованием конфигурации.
        
        Args:
            config: Словарь с параметрами конфигурации.
            
        Returns:
            True, если настройка прошла успешно, иначе False.
        """
        pass
        
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Возвращает текущую конфигурацию компонента.
        
        Returns:
            Словарь с текущими параметрами конфигурации.
        """
        pass