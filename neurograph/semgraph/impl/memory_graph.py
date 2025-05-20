"""Реализация графа в памяти."""

from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx

from neurograph.semgraph.base import ISemGraph


class MemoryEfficientSemGraph(ISemGraph):
    """Оптимизированная по памяти реализация семантического графа."""
    
    def __init__(self):
        """Инициализирует граф."""
        self.graph = nx.MultiDiGraph()
        
    def add_node(self, node_id: str, **attributes) -> None:
        """Добавляет узел в граф с заданными атрибутами."""
        self.graph.add_node(node_id, **attributes)
        
    def add_edge(self, source: str, target: str, edge_type: str = "default", 
               weight: float = 1.0, **attributes) -> None:
        """Добавляет направленное ребро между узлами."""
        # Добавляем узлы, если они не существуют
        if not self.has_node(source):
            self.add_node(source)
        if not self.has_node(target):
            self.add_node(target)
            
        # Добавляем ребро с атрибутами
        self.graph.add_edge(source, target, key=edge_type, weight=weight, **attributes)
        
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает узел с его атрибутами."""
        if not self.has_node(node_id):
            return None
            
        return dict(self.graph.nodes[node_id])
        
    def get_edge(self, source: str, target: str, edge_type: str = "default") -> Optional[Dict[str, Any]]:
        """Возвращает ребро с его атрибутами."""
        if not self.has_edge(source, target, edge_type):
            return None
            
        return dict(self.graph[source][target][edge_type])
        
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Возвращает соседей узла."""
        if not self.has_node(node_id):
            return []
            
        if edge_type is None:
            return list(self.graph.successors(node_id))
            
        neighbors = []
        for _, target, key in self.graph.out_edges(node_id, keys=True):
            if key == edge_type:
                neighbors.append(target)
                
        return neighbors
        
    def get_edge_weight(self, source: str, target: str, edge_type: str = "default") -> Optional[float]:
        """Возвращает вес ребра."""
        if not self.has_edge(source, target, edge_type):
            return None
            
        return self.graph[source][target][edge_type].get("weight", 1.0)
        
    def update_edge_weight(self, source: str, target: str, weight: float, 
                         edge_type: str = "default") -> bool:
        """Обновляет вес ребра."""
        if not self.has_edge(source, target, edge_type):
            return False
            
        self.graph[source][target][edge_type]["weight"] = weight
        return True
        
    def has_node(self, node_id: str) -> bool:
        """Проверяет наличие узла в графе."""
        return node_id in self.graph
        
    def has_edge(self, source: str, target: str, edge_type: str = "default") -> bool:
        """Проверяет наличие ребра в графе."""
        if not (self.has_node(source) and self.has_node(target)):
            return False
            
        return self.graph.has_edge(source, target, key=edge_type)
        
    def get_all_nodes(self) -> List[str]:
        """Возвращает список всех узлов в графе."""
        return list(self.graph.nodes)
        
    def get_all_edges(self) -> List[Tuple[str, str, str]]:
        """Возвращает список всех ребер в графе."""
        return [(u, v, k) for u, v, k in self.graph.edges(keys=True)]