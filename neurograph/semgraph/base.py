"""Базовые классы и интерфейсы семантического графа."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Tuple


class Node:
    """Узел графа с атрибутами."""
    
    def __init__(self, node_id: str, **attributes):
        """Инициализирует узел с идентификатором и атрибутами.
        
        Args:
            node_id: Идентификатор узла.
            **attributes: Дополнительные атрибуты узла.
        """
        self.id = node_id
        self.attributes = attributes
        
    def __str__(self) -> str:
        return f"Node(id={self.id}, attributes={self.attributes})"


class Edge:
    """Ребро графа с типом и весом."""
    
    def __init__(self, source: str, target: str, edge_type: str = "default", 
                 weight: float = 1.0, **attributes):
        """Инициализирует ребро между узлами.
        
        Args:
            source: Идентификатор исходного узла.
            target: Идентификатор целевого узла.
            edge_type: Тип ребра.
            weight: Вес ребра.
            **attributes: Дополнительные атрибуты ребра.
        """
        self.source = source
        self.target = target
        self.edge_type = edge_type
        self.weight = weight
        self.attributes = attributes
        
    def __str__(self) -> str:
        return f"Edge(source={self.source}, target={self.target}, type={self.edge_type}, weight={self.weight})"


class ISemGraph(ABC):
    """Интерфейс семантического графа.
    
    Этот интерфейс определяет методы для работы с семантическим графом,
    включая добавление узлов и ребер, поиск и обход графа.
    """
    
    @abstractmethod
    def add_node(self, node_id: str, **attributes) -> None:
        """Добавляет узел в граф с заданными атрибутами.
        
        Args:
            node_id: Идентификатор узла.
            **attributes: Атрибуты узла.
        """
        pass
        
    @abstractmethod
    def add_edge(self, source: str, target: str, edge_type: str = "default", 
                weight: float = 1.0, **attributes) -> None:
        """Добавляет направленное ребро между узлами.
        
        Args:
            source: Идентификатор исходного узла.
            target: Идентификатор целевого узла.
            edge_type: Тип ребра.
            weight: Вес ребра.
            **attributes: Атрибуты ребра.
        """
        pass
        
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает узел с его атрибутами.
        
        Args:
            node_id: Идентификатор узла.
            
        Returns:
            Словарь атрибутов узла или None, если узел не найден.
        """
        pass
        
    @abstractmethod
    def get_edge(self, source: str, target: str, edge_type: str = "default") -> Optional[Dict[str, Any]]:
        """Возвращает ребро с его атрибутами.
        
        Args:
            source: Идентификатор исходного узла.
            target: Идентификатор целевого узла.
            edge_type: Тип ребра.
            
        Returns:
            Словарь атрибутов ребра или None, если ребро не найдено.
        """
        pass
        
    @abstractmethod
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Возвращает соседей узла.
        
        Args:
            node_id: Идентификатор узла.
            edge_type: Тип ребра для фильтрации (если None, возвращаются все соседи).
            
        Returns:
            Список идентификаторов соседних узлов.
        """
        pass
        
    @abstractmethod
    def get_edge_weight(self, source: str, target: str, edge_type: str = "default") -> Optional[float]:
        """Возвращает вес ребра.
        
        Args:
            source: Идентификатор исходного узла.
            target: Идентификатор целевого узла.
            edge_type: Тип ребра.
            
        Returns:
            Вес ребра или None, если ребро не найдено.
        """
        pass
        
    @abstractmethod
    def update_edge_weight(self, source: str, target: str, weight: float, 
                        edge_type: str = "default") -> bool:
        """Обновляет вес ребра.
        
        Args:
            source: Идентификатор исходного узла.
            target: Идентификатор целевого узла.
            weight: Новый вес ребра.
            edge_type: Тип ребра.
            
        Returns:
            True, если ребро успешно обновлено, иначе False.
        """
        pass
        
    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        """Проверяет наличие узла в графе.
        
        Args:
            node_id: Идентификатор узла.
            
        Returns:
            True, если узел существует, иначе False.
        """
        pass
        
    @abstractmethod
    def has_edge(self, source: str, target: str, edge_type: str = "default") -> bool:
        """Проверяет наличие ребра в графе.
        
        Args:
            source: Идентификатор исходного узла.
            target: Идентификатор целевого узла.
            edge_type: Тип ребра.
            
        Returns:
            True, если ребро существует, иначе False.
        """
        pass
        
    @abstractmethod
    def get_all_nodes(self) -> List[str]:
        """Возвращает список всех узлов в графе.
        
        Returns:
            Список идентификаторов всех узлов.
        """
        pass
        
    @abstractmethod
    def get_all_edges(self) -> List[Tuple[str, str, str]]:
        """Возвращает список всех ребер в графе.
        
        Returns:
            Список кортежей (source, target, edge_type).
        """
        pass