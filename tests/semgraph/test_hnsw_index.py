"""Тесты для индекса HNSW."""

import os
import tempfile
import numpy as np
import pytest

# Пропускаем тесты, если библиотека hnswlib не установлена
hnswlib_spec = pytest.importorskip("hnswlib")

from neurograph.semgraph.index.hnsw import HNSWIndex
from neurograph.core.errors import VectorError


class TestHNSWIndex:
    """Тесты для индекса HNSW."""
    
    def test_init(self):
        """Проверка инициализации индекса."""
        index = HNSWIndex(dim=10, max_elements=100)
        
        assert index.dim == 10
        assert index.max_elements == 100
    
    def test_add_item(self):
        """Проверка добавления элемента в индекс."""
        index = HNSWIndex(dim=3)
        
        # Добавляем элемент
        index.add_item("item1", np.array([1.0, 0.0, 0.0]))
        
        # Проверяем, что элемент добавлен
        assert "item1" in index.id_to_index
    
    def test_add_item_invalid_dimension(self):
        """Проверка обработки векторов неверной размерности."""
        index = HNSWIndex(dim=3)
        
        # Пытаемся добавить вектор неверной размерности
        with pytest.raises(VectorError):
            index.add_item("item1", np.array([1.0, 0.0]))
    
    def test_search(self):
        """Проверка поиска ближайших соседей."""
        index = HNSWIndex(dim=3)
        
        # Добавляем несколько элементов
        index.add_item("item1", np.array([1.0, 0.0, 0.0]))
        index.add_item("item2", np.array([0.0, 1.0, 0.0]))
        index.add_item("item3", np.array([0.0, 0.0, 1.0]))
        index.add_item("item4", np.array([0.9, 0.1, 0.0]))
        
        # Ищем ближайших соседей
        results = index.search(np.array([1.0, 0.0, 0.0]), k=2)
        
        # Проверяем результаты
        assert len(results) == 2
        assert results[0][0] == "item1"  # Первый результат должен быть item1
        assert results[1][0] == "item4"  # Второй результат должен быть item4
        assert results[0][1] > 0.9  # Высокое сходство
    
    def test_remove_item(self):
        """Проверка удаления элемента из индекса."""
        index = HNSWIndex(dim=3)
        
        # Добавляем элемент
        index.add_item("item1", np.array([1.0, 0.0, 0.0]))
        
        # Удаляем элемент
        result = index.remove_item("item1")
        
        # Проверяем результат
        assert result is True
        assert "item1" not in index.id_to_index
        
        # Пытаемся удалить несуществующий элемент
        result = index.remove_item("non_existent")
        assert result is False
    
    def test_save_load(self):
        """Проверка сохранения и загрузки индекса."""
        # Создаем индекс и добавляем элементы
        index = HNSWIndex(dim=3)
        index.add_item("item1", np.array([1.0, 0.0, 0.0]))
        index.add_item("item2", np.array([0.0, 1.0, 0.0]))
        
        # Создаем временную директорию для файлов
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "index_test")
            
            # Сохраняем индекс
            index.save(file_path)
            
            # Загружаем индекс
            loaded_index = HNSWIndex.load(file_path)
            
            # Проверяем параметры
            assert loaded_index.dim == index.dim
            assert loaded_index.max_elements == index.max_elements
            
            # Проверяем наличие элементов
            assert "item1" in loaded_index.id_to_index
            assert "item2" in loaded_index.id_to_index
            
            # Проверяем поиск
            results = loaded_index.search(np.array([1.0, 0.0, 0.0]), k=1)
            assert results[0][0] == "item1"