"""Тесты для графа с постоянным хранением на диске."""

import os
import tempfile
import time
import pytest

from neurograph.semgraph.impl.persistent_graph import PersistentSemGraph


class TestPersistentSemGraph:
    """Тесты для класса PersistentSemGraph."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        # Создаем временный файл для хранения графа
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.file_path = self.temp_file.name
        self.temp_file.close()
        
        # Создаем граф с коротким интервалом автосохранения для тестирования
        self.graph = PersistentSemGraph(self.file_path, auto_save_interval=1.0)
        
        # Добавляем тестовые данные
        self.graph.add_node("A", type="person", name="Alice")
        self.graph.add_node("B", type="person", name="Bob")
        self.graph.add_edge("A", "B", "knows", 0.8, since=2020)
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        # Закрываем граф и удаляем файл
        if hasattr(self, "graph"):
            self.graph.close()
        
        if os.path.exists(self.file_path):
            os.unlink(self.file_path)
    
    def test_basic_operations(self):
        """Проверка базовых операций с графом."""
        # Проверяем, что добавленные узлы и ребра существуют
        assert self.graph.has_node("A")
        assert self.graph.has_node("B")
        assert self.graph.has_edge("A", "B", "knows")
        
        # Проверяем атрибуты
        assert self.graph.get_node("A")["type"] == "person"
        assert self.graph.get_edge("A", "B", "knows")["since"] == 2020
        
        # Проверяем получение соседей
        assert self.graph.get_neighbors("A") == ["B"]
        
        # Проверяем вес ребра
        assert self.graph.get_edge_weight("A", "B", "knows") == 0.8
        
        # Обновляем вес ребра
        self.graph.update_edge_weight("A", "B", 0.9, "knows")
        assert self.graph.get_edge_weight("A", "B", "knows") == 0.9
    
    def test_auto_save(self):
        """Проверка автоматического сохранения."""
        # Добавляем новый узел
        self.graph.add_node("C", type="place", name="Cafe")
        
        # Ждем, чтобы сработало автосохранение (немного больше интервала)
        time.sleep(1.5)
        
        # Проверяем, что файл существует
        assert os.path.exists(self.file_path)
        
        # Создаем новый граф с тем же файлом, чтобы проверить, что данные были сохранены
        new_graph = PersistentSemGraph(self.file_path, auto_save_interval=0)  # Отключаем автосохранение
        
        # Проверяем, что данные были загружены
        assert new_graph.has_node("A")
        assert new_graph.has_node("B")
        assert new_graph.has_node("C")
        assert new_graph.has_edge("A", "B", "knows")
        
        # Закрываем новый граф
        new_graph.close()
    
    def test_manual_save_load(self):
        """Проверка ручного сохранения и загрузки."""
        # Добавляем новый узел
        self.graph.add_node("C", type="place", name="Cafe")
        
        # Сохраняем граф
        self.graph.save_now()
        
        # Проверяем, что файл существует
        assert os.path.exists(self.file_path)
        
        # Добавляем еще один узел, который не должен быть в файле
        self.graph.add_node("D", type="place", name="Diner")
        
        # Перезагружаем граф из файла
        self.graph.reload()
        
        # Проверяем, что узел C загружен, а узел D нет
        assert self.graph.has_node("C")
        assert not self.graph.has_node("D")
    
    def test_close(self):
        """Проверка закрытия графа с сохранением изменений."""
        # Добавляем новый узел
        self.graph.add_node("C", type="place", name="Cafe")
        
        # Закрываем граф
        self.graph.close()
        
        # Проверяем, что файл существует
        assert os.path.exists(self.file_path)
        
        # Создаем новый граф с тем же файлом
        new_graph = PersistentSemGraph(self.file_path)
        
        # Проверяем, что данные были сохранены
        assert new_graph.has_node("C")
        
        # Закрываем новый граф
        new_graph.close()