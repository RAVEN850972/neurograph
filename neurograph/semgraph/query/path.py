"""Механизмы поиска путей в графе."""

from typing import Dict, List, Set, Tuple, Any, Optional
import heapq

from neurograph.semgraph.base import ISemGraph
from neurograph.core.logging import get_logger


logger = get_logger("semgraph.query.path")


class PathFinder:
    """Класс для поиска путей в графе."""
    
    def __init__(self, graph: ISemGraph):
        """Инициализирует поиск путей.
        
        Args:
            graph: Граф для поиска путей.
        """
        self.graph = graph
    
    def find_shortest_path(self, start: str, end: str, 
                           edge_types: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
        """Находит кратчайший путь между двумя узлами.
        
        Args:
            start: Начальный узел.
            end: Конечный узел.
            edge_types: Список типов ребер для рассмотрения (если None, рассматриваются все).
            
        Returns:
            Список ребер (from, to, edge_type) в пути или пустой список, если путь не найден.
        """
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return []
        
        # Если начальный и конечный узел совпадают, возвращаем пустой путь
        if start == end:
            return []
        
        # Инициализация поиска в ширину
        queue = [(start, [])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            # Получаем соседей текущего узла
            for edge in self._get_outgoing_edges(current, edge_types):
                neighbor = edge[1]
                
                if neighbor == end:
                    # Найден путь до конечного узла
                    return path + [edge]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [edge]))
        
        # Путь не найден
        return []
    
    def find_all_paths(self, start: str, end: str, max_depth: int = 3, 
                      edge_types: Optional[List[str]] = None) -> List[List[Tuple[str, str, str]]]:
        """Находит все пути между двумя узлами, ограниченные по глубине.
        
        Args:
            start: Начальный узел.
            end: Конечный узел.
            max_depth: Максимальная глубина поиска.
            edge_types: Список типов ребер для рассмотрения (если None, рассматриваются все).
            
        Returns:
            Список путей, каждый путь - список ребер (from, to, edge_type).
        """
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return []
        
        # Если начальный и конечный узел совпадают, возвращаем пустой путь
        if start == end:
            return [[]]
        
        paths = []
        self._dfs_paths(start, end, set(), [], paths, max_depth, edge_types)
        return paths
    
    def _dfs_paths(self, current: str, end: str, visited: Set[str], 
                  path: List[Tuple[str, str, str]], paths: List[List[Tuple[str, str, str]]], 
                  max_depth: int, edge_types: Optional[List[str]]) -> None:
        """Поиск в глубину для нахождения всех путей.
        
        Args:
            current: Текущий узел.
            end: Конечный узел.
            visited: Множество посещенных узлов.
            path: Текущий путь.
            paths: Список найденных путей.
            max_depth: Максимальная глубина поиска.
            edge_types: Список типов ребер для рассмотрения.
        """
        if len(path) >= max_depth:
            return
        
        visited.add(current)
        
        for edge in self._get_outgoing_edges(current, edge_types):
            neighbor = edge[1]
            
            if neighbor == end:
                # Найден путь до конечного узла
                paths.append(path + [edge])
            elif neighbor not in visited:
                self._dfs_paths(neighbor, end, visited.copy(), path + [edge], 
                              paths, max_depth, edge_types)
    
    def find_weighted_shortest_path(self, start: str, end: str,
                           edge_types: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
        """Находит кратчайший путь между двумя узлами с учетом весов ребер (алгоритм Дейкстры)."""
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return []
        
        # Если начальный и конечный узел совпадают, возвращаем пустой путь
        if start == end:
            return []
        
        # Инициализация алгоритма Дейкстры
        distances = {}
        previous = {}
        unvisited = set()
        
        # Сначала заполняем словарь всеми узлами из графа
        for node in self.graph.get_all_nodes():
            if node == start:
                distances[node] = 0
            else:
                distances[node] = float('infinity')
            previous[node] = None
            unvisited.add(node)
        
        while unvisited:
            # Находим узел с минимальным расстоянием
            current = min(unvisited, key=lambda node: distances[node])
            
            # Если достигли конечного узла или расстояние равно бесконечности
            if current == end or distances[current] == float('infinity'):
                break
            
            unvisited.remove(current)
            
            # Обрабатываем всех соседей текущего узла
            for edge in self._get_outgoing_edges(current, edge_types):
                neighbor = edge[1]
                edge_type = edge[2]
                
                # Пропускаем соседей, которые уже посещены
                if neighbor not in unvisited:
                    continue
                    
                # Получаем вес ребра
                weight = self.graph.get_edge_weight(current, neighbor, edge_type) or 1.0
                
                # Вычисляем новое расстояние
                distance = distances[current] + weight
                
                # Если найден более короткий путь
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = (current, edge_type)  # Сохраняем узел и тип ребра
        
        # Восстанавливаем путь
        if end not in previous or previous[end] is None:
            return []  # Путь не найден
        
        path = []
        current = end
        
        while current != start:
            prev_node, edge_type = previous[current]
            path.append((prev_node, current, edge_type))
            current = prev_node
        
        # Переворачиваем путь, так как восстанавливали с конца
        path.reverse()
        
        return path
    
    def _get_outgoing_edges(self, node_id: str, 
                       edge_types: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
        """Возвращает исходящие ребра из узла.
        
        Args:
            node_id: Идентификатор узла.
            edge_types: Список типов ребер для фильтрации (если None, возвращаются все).
            
        Returns:
            Список ребер (from, to, edge_type).
        """
        edges = []
        
        # Получаем всех соседей узла
        neighbors = self.graph.get_neighbors(node_id)
        
        # Получаем все ребра графа один раз
        all_edges = self.graph.get_all_edges()
        
        # Фильтруем ребра, начинающиеся из node_id и направленные в одного из соседей
        for source, target, edge_type in all_edges:
            if source == node_id and target in neighbors:
                # Если указаны типы ребер и тип текущего ребра не входит в список, пропускаем
                if edge_types is not None and edge_type not in edge_types:
                    continue
                
                edges.append((source, target, edge_type))
        
        return edges