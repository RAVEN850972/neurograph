"""Тесты для расширенных функций визуализатора графа."""

import os
import tempfile
import json
import pytest

from neurograph.semgraph.impl.memory_graph import MemoryEfficientSemGraph
from neurograph.semgraph.visualization.visualizer import GraphVisualizer


class TestGraphVisualizerExtended:
    """Тесты для расширенных функций класса GraphVisualizer."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        # Создаем граф
        self.graph = MemoryEfficientSemGraph()
        
        # Добавляем узлы
        self.graph.add_node("A", type="person", name="Alice")
        self.graph.add_node("B", type="person", name="Bob")
        self.graph.add_node("C", type="place", name="Cafe")
        
        # Добавляем ребра
        self.graph.add_edge("A", "B", "knows", 0.8)
        self.graph.add_edge("A", "C", "visits", 0.6)
        self.graph.add_edge("B", "C", "visits", 0.5)
        
        # Создаем визуализатор
        self.visualizer = GraphVisualizer(self.graph)
    
    def test_save_as_graphml(self):
        """Проверка сохранения графа в формате GraphML."""
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Сохраняем граф в формате GraphML
            self.visualizer.save_as_graphml(temp_path)
            
            # Проверяем, что файл существует и не пустой
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
            # Проверяем, что файл можно загрузить как GraphML
            import networkx as nx
            loaded_graph = nx.read_graphml(temp_path)
            
            # Проверяем структуру загруженного графа
            assert len(loaded_graph.nodes) == 3
            assert len(loaded_graph.edges) == 3
            assert "A" in loaded_graph.nodes
            assert "B" in loaded_graph.nodes
            assert "C" in loaded_graph.nodes
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)
    
    def test_save_as_gexf(self):
        """Проверка сохранения графа в формате GEXF."""
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".gexf", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Сохраняем граф в формате GEXF
            self.visualizer.save_as_gexf(temp_path)
            
            # Проверяем, что файл существует и не пустой
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
            # Проверяем, что файл можно загрузить как GEXF
            import networkx as nx
            loaded_graph = nx.read_gexf(temp_path)
            
            # Проверяем структуру загруженного графа
            assert len(loaded_graph.nodes) == 3
            assert len(loaded_graph.edges) == 3
            assert "A" in loaded_graph.nodes
            assert "B" in loaded_graph.nodes
            assert "C" in loaded_graph.nodes
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)
    
    def test_export_to_cytoscape(self):
        """Проверка экспорта графа в формат Cytoscape.js."""
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Экспортируем граф в формат Cytoscape.js
            self.visualizer.export_to_cytoscape(temp_path)
            
            # Проверяем, что файл существует и не пустой
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
            # Загружаем JSON из файла
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Проверяем структуру данных
            assert "nodes" in data
            assert "edges" in data
            assert len(data["nodes"]) == 3
            assert len(data["edges"]) == 3
            
            # Проверяем содержимое узлов
            node_ids = [node["data"]["id"] for node in data["nodes"]]
            assert "A" in node_ids
            assert "B" in node_ids
            assert "C" in node_ids
            
            # Проверяем содержимое ребер
            edges = []
            for edge in data["edges"]:
                assert "source" in edge["data"]
                assert "target" in edge["data"]
                assert "edge_type" in edge["data"]
                edges.append((edge["data"]["source"], edge["data"]["target"], edge["data"]["edge_type"]))
                
            assert ("A", "B", "knows") in edges
            assert ("A", "C", "visits") in edges
            assert ("B", "C", "visits") in edges
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)