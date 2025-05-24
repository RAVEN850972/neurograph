"""Реализация графа в памяти."""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
import networkx as nx

from neurograph.semgraph.base import ISemGraph
from neurograph.core.logging import get_logger

logger = get_logger("semgraph.memory_graph")

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
    
    # Добавленные методы
    def export_to_networkx(self) -> nx.MultiDiGraph:
        """Экспортирует граф в формат NetworkX.
        
        Returns:
            Граф NetworkX.
        """
        # Просто возвращаем копию нашего внутреннего графа
        return self.graph.copy()

    def import_from_networkx(self, nx_graph: nx.MultiDiGraph) -> None:
        """Импортирует граф из формата NetworkX.
        
        Args:
            nx_graph: Граф NetworkX.
        """
        self.graph = nx_graph.copy()
        logger.info(f"Импортирован граф с {len(nx_graph.nodes)} узлами и {nx_graph.number_of_edges()} ребрами")

    def serialize(self) -> Dict[str, Any]:
        """Сериализует граф в словарь.
        
        Returns:
            Словарь с данными графа.
        """
        # Сериализуем узлы
        nodes = []
        for node_id in self.graph.nodes:
            node_data = {
                "id": node_id,
                "attributes": dict(self.graph.nodes[node_id])
            }
            nodes.append(node_data)
        
        # Сериализуем ребра
        edges = []
        for source, target, key in self.graph.edges(keys=True):
            edge_data = {
                "source": source,
                "target": target,
                "edge_type": key,
                "attributes": dict(self.graph[source][target][key])
            }
            edges.append(edge_data)
        
        return {
            "nodes": nodes,
            "edges": edges
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "MemoryEfficientSemGraph":
        """Создает граф из сериализованных данных.
        
        Args:
            data: Сериализованные данные графа.
            
        Returns:
            Созданный граф.
        """
        graph = cls()
        
        # Добавляем узлы
        for node_data in data.get("nodes", []):
            node_id = node_data["id"]
            attributes = node_data.get("attributes", {})
            graph.add_node(node_id, **attributes)
        
        # Добавляем ребра
        for edge_data in data.get("edges", []):
            source = edge_data["source"]
            target = edge_data["target"]
            edge_type = edge_data["edge_type"]
            attributes = edge_data.get("attributes", {})
            
            # Получаем вес, если он есть в атрибутах
            weight = attributes.pop("weight", 1.0) if "weight" in attributes else 1.0
            
            graph.add_edge(source, target, edge_type, weight, **attributes)
        
        logger.info(f"Десериализован граф с {len(graph.get_all_nodes())} узлами и {len(graph.get_all_edges())} ребрами")
        return graph

    def to_json(self) -> str:
        """Сериализует граф в формат JSON.
        
        Returns:
            Строка JSON.
        """
        from neurograph.core.utils.serialization import JSONSerializer
        return JSONSerializer.serialize(self.serialize())

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryEfficientSemGraph":
        """Создает граф из JSON.
        
        Args:
            json_str: Строка JSON.
            
        Returns:
            Созданный граф.
        """
        from neurograph.core.utils.serialization import JSONSerializer
        data = JSONSerializer.deserialize(json_str)
        return cls.deserialize(data)

    def save(self, file_path: str) -> None:
        """Сохраняет граф в файл.
        
        Args:
            file_path: Путь к файлу.
        """
        from neurograph.core.utils.serialization import JSONSerializer
        JSONSerializer.save_to_file(self.serialize(), file_path)
        logger.info(f"Граф сохранен в файл: {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "MemoryEfficientSemGraph":
        """Загружает граф из файла.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Загруженный граф.
        """
        from neurograph.core.utils.serialization import JSONSerializer
        data = JSONSerializer.load_from_file(file_path)
        return cls.deserialize(data)

    def merge(self, other: "MemoryEfficientSemGraph") -> None:
        """Объединяет граф с другим графом.
        
        Args:
            other: Другой граф.
        """
        # Добавляем узлы из другого графа
        for node_id in other.get_all_nodes():
            node_attrs = other.get_node(node_id) or {}
            if not self.has_node(node_id):
                self.add_node(node_id, **node_attrs)
            else:
                # Обновляем атрибуты существующего узла
                for key, value in node_attrs.items():
                    self.graph.nodes[node_id][key] = value
        
        # Добавляем ребра из другого графа
        for source, target, edge_type in other.get_all_edges():
            edge_attrs = other.get_edge(source, target, edge_type) or {}
            weight = edge_attrs.pop("weight", 1.0) if "weight" in edge_attrs else 1.0
            
            if not self.has_edge(source, target, edge_type):
                self.add_edge(source, target, edge_type, weight, **edge_attrs)
            else:
                # Обновляем атрибуты существующего ребра
                self.update_edge_weight(source, target, weight, edge_type)
                for key, value in edge_attrs.items():
                    self.graph[source][target][edge_type][key] = value
        
        logger.info(f"Граф объединен с другим графом, всего {len(self.get_all_nodes())} узлов и {len(self.get_all_edges())} ребер")