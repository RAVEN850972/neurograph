"""Тесты для поиска путей в графе."""

import pytest
from unittest.mock import MagicMock

from neurograph.semgraph.query.path import PathFinder


class TestPathFinder:
    """Тесты для класса PathFinder."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        # Создаем мок графа
        self.graph = MagicMock()
        
        # Настраиваем поведение графа
        def has_node(node_id):
            return node_id in ["A", "B", "C", "D", "E"]
        
        def get_all_edges():
            return [
                ("A", "B", "edge1"),
                ("B", "C", "edge2"),
                ("A", "D", "edge3"),
                ("D", "E", "edge4"),
                ("C", "E", "edge5")
            ]
        
        def get_neighbors(node_id):
            neighbors = {
                "A": ["B", "D"],
                "B": ["C"],
                "C": ["E"],
                "D": ["E"],
                "E": []
            }
            return neighbors.get(node_id, [])
        
        def get_edge_weight(source, target, edge_type=None):
            weights = {
                ("A", "B"): 1.0,
                ("B", "C"): 2.0,
                ("A", "D"): 1.5,
                ("D", "E"): 1.0,
                ("C", "E"): 1.0
            }
            return weights.get((source, target), 1.0)
        
        # Настраиваем методы мока
        self.graph.has_node.side_effect = has_node
        self.graph.get_all_edges.side_effect = get_all_edges
        self.graph.get_neighbors.side_effect = get_neighbors
        self.graph.get_edge_weight.side_effect = get_edge_weight
        
        # Создаем PathFinder
        self.path_finder = PathFinder(self.graph)
    
    def test_find_shortest_path(self):
        """Проверка поиска кратчайшего пути."""
        # Поиск пути от A до E
        path = self.path_finder.find_shortest_path("A", "E")
        
        # Проверяем, что путь найден
        assert len(path) == 2
        assert path[0][0] == "A"
        assert path[1][0] == "D"
        assert path[1][1] == "E"
        
        # Поиск пути к несуществующему узлу
        path = self.path_finder.find_shortest_path("A", "Z")
        assert path == []
        
        # Поиск пути от несуществующего узла
        path = self.path_finder.find_shortest_path("Z", "E")
        assert path == []
        
        # Поиск пути до самого себя
        path = self.path_finder.find_shortest_path("A", "A")
        assert path == []
    
    def test_find_all_paths(self):
        """Проверка поиска всех путей."""
        # Поиск всех путей от A до E с максимальной глубиной 3
        paths = self.path_finder.find_all_paths("A", "E", max_depth=3)
        
        # Проверяем, что найдены все пути
        assert len(paths) == 2
        
        # Преобразуем пути в наборы ребер для упрощения проверки
        path_sets = []
        for path in paths:
            path_sets.append(set((edge[0], edge[1], edge[2]) for edge in path))
        
        # Ожидаемые пути как наборы ребер
        path1 = {("A", "D", "edge3"), ("D", "E", "edge4")}  # A -> D -> E
        path2 = {("A", "B", "edge1"), ("B", "C", "edge2"), ("C", "E", "edge5")}  # A -> B -> C -> E
        
        # Проверяем, что оба ожидаемых пути есть в результатах
        assert path1 in path_sets, "Путь A -> D -> E не найден"
        assert path2 in path_sets, "Путь A -> B -> C -> E не найден"
    
    def test_find_weighted_shortest_path(self):
        """Проверка поиска кратчайшего пути с учетом весов."""
        # Поиск пути от A до E
        path = self.path_finder.find_weighted_shortest_path("A", "E")
        
        # A -> D -> E имеет вес 1.5 + 1.0 = 2.5
        # A -> B -> C -> E имеет вес 1.0 + 2.0 + 1.0 = 4.0
        # Поэтому должен быть выбран первый путь
        
        # Проверяем, что найден правильный путь
        assert len(path) == 2
        assert path[0][0] == "A"
        assert path[0][1] == "D"
        assert path[1][0] == "D"
        assert path[1][1] == "E"