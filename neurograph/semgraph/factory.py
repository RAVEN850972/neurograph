"""Фабрика для создания экземпляров семантического графа."""

from typing import Dict, Any, Type, Optional

from neurograph.core.utils.registry import Registry
from neurograph.semgraph.base import ISemGraph
from neurograph.semgraph.impl.memory_graph import MemoryEfficientSemGraph


# Создаем регистр для графов
graph_registry = Registry[ISemGraph]("semgraph")

# Регистрируем реализации
graph_registry.register("memory_efficient", MemoryEfficientSemGraph)


class SemGraphFactory:
    """Фабрика для создания экземпляров семантического графа."""
    
    @staticmethod
    def create(graph_type: str = "memory_efficient", **kwargs) -> ISemGraph:
        """Создает экземпляр графа указанного типа.
        
        Args:
            graph_type: Тип графа ("memory_efficient", "persistent", и т.д.).
            **kwargs: Параметры для конструктора графа.
            
        Returns:
            Экземпляр ISemGraph.
            
        Raises:
            ValueError: Если указан неизвестный тип графа.
        """
        return graph_registry.create(graph_type, **kwargs)