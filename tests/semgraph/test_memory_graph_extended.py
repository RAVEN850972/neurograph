"""Тесты для расширенной функциональности графа в памяти."""

import os
import tempfile
import pytest
import networkx as nx

from neurograph.semgraph.impl.memory_graph import MemoryEfficientSemGraph


class TestMemoryEfficientSemGraphExtended:
    """Тесты для расширенной функциональности класса MemoryEfficientSemGraph."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.graph = MemoryEfficientSemGraph()
        
        # Создаем тестовый граф
        self.graph.add_node("A", type="person", name="Alice")
        self.graph.add_node("B", type="person", name="Bob")
        self.graph.add_node("C", type="place", name="Cafe")
        
        self.graph.add_edge("A", "B", "knows", 0.8, since=2020)
        self.graph.add_edge("A", "C", "visits", 0.5, frequency="weekly")
        self.graph.add_edge("B", "C", "visits", 0.3, frequency="monthly")
    
    def test_export_import_networkx(self):
        """Проверка экспорта и импорта графа в формат NetworkX."""
        # Экспортируем граф
        nx_graph = self.graph.export_to_networkx()
        
        # Проверяем, что граф правильно экспортирован
        assert len(nx_graph.nodes) == 3
        assert nx_graph.number_of_edges() == 3
        assert nx_graph.nodes["A"]["type"] == "person"
        assert nx_graph.nodes["A"]["name"] == "Alice"
        assert nx_graph["A"]["B"]["knows"]["since"] == 2020
        
        # Изменяем экспортированный граф
        nx_graph.add_node("D", type="place", name="Diner")
        nx_graph.add_edge("A", "D", key="visits", weight=0.2, frequency="rarely")
        
        # Создаем новый граф и импортируем в него измененный граф
        new_graph = MemoryEfficientSemGraph()
        new_graph.import_from_networkx(nx_graph)
        
        # Проверяем, что граф правильно импортирован
        assert len(new_graph.get_all_nodes()) == 4
        assert len(new_graph.get_all_edges()) == 4
        assert new_graph.has_node("D")
        assert new_graph.has_edge("A", "D", "visits")
        assert new_graph.get_edge("A", "D", "visits")["frequency"] == "rarely"
    
    def test_serialize_deserialize(self):
        """Проверка сериализации и десериализации графа."""
        # Сериализуем граф
        data = self.graph.serialize()
        
        # Проверяем структуру сериализованных данных
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 3
        
        # Десериализуем граф
        new_graph = MemoryEfficientSemGraph.deserialize(data)
        
        # Проверяем, что граф правильно десериализован
        assert len(new_graph.get_all_nodes()) == 3
        assert len(new_graph.get_all_edges()) == 3
        assert new_graph.has_node("A")
        assert new_graph.has_edge("A", "B", "knows")
        assert new_graph.get_node("A")["type"] == "person"
        assert new_graph.get_edge("A", "B", "knows")["since"] == 2020
    
    def test_to_from_json(self):
        """Проверка сериализации и десериализации графа в формат JSON."""
        # Сериализуем граф в JSON
        json_str = self.graph.to_json()
        
        # Проверяем, что JSON строка не пустая
        assert json_str
        assert "nodes" in json_str
        assert "edges" in json_str
        
        # Десериализуем граф из JSON
        new_graph = MemoryEfficientSemGraph.from_json(json_str)
        
        # Проверяем, что граф правильно десериализован
        assert len(new_graph.get_all_nodes()) == 3
        assert len(new_graph.get_all_edges()) == 3
        assert new_graph.has_node("A")
        assert new_graph.has_edge("A", "B", "knows")
        assert new_graph.get_node("A")["type"] == "person"
        assert new_graph.get_edge("A", "B", "knows")["since"] == 2020
    
    def test_save_load(self):
        """Проверка сохранения и загрузки графа из файла."""
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Сохраняем граф в файл
            self.graph.save(temp_path)
            
            # Загружаем граф из файла
            loaded_graph = MemoryEfficientSemGraph.load(temp_path)
            
            # Проверяем, что граф правильно загружен
            assert len(loaded_graph.get_all_nodes()) == 3
            assert len(loaded_graph.get_all_edges()) == 3
            assert loaded_graph.has_node("A")
            assert loaded_graph.has_edge("A", "B", "knows")
            assert loaded_graph.get_node("A")["type"] == "person"
            assert loaded_graph.get_edge("A", "B", "knows")["since"] == 2020
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)
    
    def test_merge(self):
        """Проверка объединения графов."""
        # Создаем второй граф
        other_graph = MemoryEfficientSemGraph()
        
        # Добавляем узлы и ребра во второй граф
        other_graph.add_node("B", type="person", name="Bobby")  # Конфликт атрибута
        other_graph.add_node("D", type="place", name="Diner")   # Новый узел
        
        other_graph.add_edge("B", "C", "visits", 0.7, frequency="daily")  # Конфликт атрибутов
        other_graph.add_edge("B", "D", "visits", 0.4, frequency="weekly")  # Новое ребро
       
        # Объединяем графы
        self.graph.merge(other_graph)

        # Проверяем результат объединения

        # Проверяем узлы
        assert len(self.graph.get_all_nodes()) == 4  # A, B, C, D
        assert self.graph.has_node("D")  # Новый узел добавлен
        assert self.graph.get_node("D")["type"] == "place"

        # Проверяем, что атрибуты существующего узла B обновлены
        assert self.graph.get_node("B")["name"] == "Bobby"

        # Проверяем ребра
        assert len(self.graph.get_all_edges()) == 4  # Все существующие + 1 новое
        assert self.graph.has_edge("B", "D", "visits")  # Новое ребро добавлено

        # Проверяем, что атрибуты существующего ребра обновлены
        assert self.graph.get_edge("B", "C", "visits")["frequency"] == "daily"
        assert self.graph.get_edge_weight("B", "C", "visits") == 0.7