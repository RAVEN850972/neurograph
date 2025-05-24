"""Классы исключений для системы NeuroGraph."""

from typing import Optional, Any, Dict


class NeuroGraphError(Exception):
    """Базовый класс для всех исключений NeuroGraph."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Инициализирует исключение с сообщением и деталями.
        
        Args:
            message: Сообщение об ошибке.
            details: Дополнительные детали об ошибке.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Возвращает строковое представление исключения.
        
        Returns:
            Строковое представление исключения.
        """
        if self.details:
            return f"{self.message} - {self.details}"
        return self.message


class ConfigurationError(NeuroGraphError):
    """Исключение, связанное с ошибками конфигурации."""
    pass


class ComponentError(NeuroGraphError):
    """Исключение, связанное с ошибками компонентов."""
    pass


class InitializationError(ComponentError):
    """Исключение, связанное с ошибками инициализации компонентов."""
    pass


class RegistryError(NeuroGraphError):
    """Исключение, связанное с ошибками в регистре компонентов."""
    pass


class ValidationError(NeuroGraphError):
    """Исключение, связанное с ошибками валидации данных."""
    pass


# Исключения для SemGraph
class GraphError(NeuroGraphError):
    """Базовое исключение, связанное с ошибками в графе."""
    pass


class NodeNotFoundError(GraphError):
    """Исключение, вызываемое при попытке доступа к несуществующему узлу."""
    pass


class EdgeNotFoundError(GraphError):
    """Исключение, вызываемое при попытке доступа к несуществующему ребру."""
    pass


# Исключения для ContextVec
class VectorError(NeuroGraphError):
    """Базовое исключение, связанное с ошибками в векторных представлениях."""
    pass


class InvalidVectorDimensionError(VectorError):
    """Исключение, вызываемое при попытке использования вектора неправильной размерности."""
    pass


class VectorNotFoundError(VectorError):
    """Исключение, вызываемое при попытке доступа к несуществующему вектору."""
    pass