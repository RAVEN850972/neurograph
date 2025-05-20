"""Тесты для динамических векторных представлений."""

import os
import tempfile
import numpy as np
import pytest

from neurograph.contextvec.impl.dynamic import DynamicContextVectors
from neurograph.core.errors import InvalidVectorDimensionError


class TestDynamicContextVectors:
    """Тесты для класса DynamicContextVectors."""
    
    def test_init(self):
        """Проверка инициализации."""
        vectors = DynamicContextVectors(vector_size=100, use_indexing=False)
        
        assert vectors.vector_size == 100
        assert vectors.use_indexing is False
        assert len(vectors.vectors) == 0
    
    def test_create_vector(self):
        """Проверка создания векторного представления."""
        vectors = DynamicContextVectors(vector_size=3, use_indexing=False)
        
        # Создаем вектор
        result = vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
        
        # Проверяем результат
        assert result is True
        assert vectors.has_key("key1")
        
        # Проверяем, что вектор нормализован
        vector = vectors.get_vector("key1")
        assert np.allclose(vector, np.array([1.0, 0.0, 0.0]))
        
        # Попытка создать вектор неверной размерности
        with pytest.raises(InvalidVectorDimensionError):
            vectors.create_vector("key2", np.array([1.0, 0.0]))
    
    def test_get_vector(self):
        """Проверка получения векторного представления."""
        vectors = DynamicContextVectors(vector_size=3, use_indexing=False)
        
        # Создаем вектор
        vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
        
        # Получаем вектор
        vector = vectors.get_vector("key1")
        assert np.allclose(vector, np.array([1.0, 0.0, 0.0]))
        
        # Попытка получить несуществующий вектор
        assert vectors.get_vector("non_existent") is None
    
    def test_similarity(self):
        """Проверка вычисления близости между векторами."""
        vectors = DynamicContextVectors(vector_size=3, use_indexing=False)
        
        # Создаем векторы
        vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
        vectors.create_vector("key2", np.array([0.0, 1.0, 0.0]))
        vectors.create_vector("key3", np.array([1.0, 1.0, 0.0]))
        
        # Вычисляем близость
        assert np.isclose(vectors.similarity("key1", "key1"), 1.0)
        assert np.isclose(vectors.similarity("key1", "key2"), 0.0)
        
        # Близость между векторами под углом 45 градусов
        sim = vectors.similarity("key1", "key3")
        assert sim is not None
        assert np.isclose(sim, 1.0 / np.sqrt(2))
        
        # Близость с несуществующим ключом
        assert vectors.similarity("key1", "non_existent") is None
    
    def test_get_most_similar(self):
        """Проверка поиска наиболее похожих векторов."""
        vectors = DynamicContextVectors(vector_size=3, use_indexing=False)
        
        # Создаем векторы
        vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
        vectors.create_vector("key2", np.array([0.9, 0.1, 0.0]))
        vectors.create_vector("key3", np.array([0.7, 0.3, 0.0]))
        vectors.create_vector("key4", np.array([0.0, 1.0, 0.0]))
        
        # Поиск похожих векторов
        results = vectors.get_most_similar("key1", top_n=2)
        
        # Проверяем результаты
        assert len(results) == 2
        assert results[0][0] == "key2"  # Наиболее похожий
        assert results[1][0] == "key3"  # Менее похожий
        
        # Поиск для несуществующего ключа
        assert vectors.get_most_similar("non_existent") == []
    
    def test_remove_vector(self):
        """Проверка удаления векторного представления."""
        vectors = DynamicContextVectors(vector_size=3, use_indexing=False)
        
        # Создаем вектор
        vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
        assert vectors.has_key("key1")
        
        # Удаляем вектор
        result = vectors.remove_vector("key1")
        assert result is True
        assert not vectors.has_key("key1")
        
        # Попытка удалить несуществующий вектор
        result = vectors.remove_vector("non_existent")
        assert result is False
    
    def test_update_vector(self):
        """Проверка обновления векторного представления."""
        vectors = DynamicContextVectors(vector_size=3, use_indexing=False)
        
        # Создаем вектор
        vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
        
        # Обновляем вектор
        result = vectors.update_vector("key1", np.array([0.0, 1.0, 0.0]), learning_rate=0.5)
        assert result is True
        
        # Проверяем результат (вектор должен быть смешан и нормализован)
        vector = vectors.get_vector("key1")
        expected = np.array([0.5, 0.5, 0.0]) / np.sqrt(0.5**2 + 0.5**2)
        assert np.allclose(vector, expected)
        
        # Попытка обновить несуществующий вектор
        result = vectors.update_vector("non_existent", np.array([1.0, 0.0, 0.0]))
        assert result is False
        
        # Попытка обновить вектор неверной размерности
        with pytest.raises(InvalidVectorDimensionError):
            vectors.update_vector("key1", np.array([1.0, 0.0]))
    
    def test_average_vectors(self):
        """Проверка усреднения векторов."""
        vectors = DynamicContextVectors(vector_size=3, use_indexing=False)
        
        # Создаем векторы
        vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
        vectors.create_vector("key2", np.array([0.0, 1.0, 0.0]))
        
        # Вычисляем среднее
        avg = vectors.average_vectors(["key1", "key2"])
        
        # Проверяем результат
        expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        assert np.allclose(avg, expected)
        
        # Среднее с несуществующим ключом
        avg = vectors.average_vectors(["key1", "non_existent"])
        assert np.allclose(avg, np.array([1.0, 0.0, 0.0]))
        
        # Среднее только для несуществующих ключей
        assert vectors.average_vectors(["non_existent1", "non_existent2"]) is None
    
    def test_save_load(self):
        """Проверка сохранения и загрузки векторных представлений."""
        vectors = DynamicContextVectors(vector_size=3, use_indexing=False)
        
        # Создаем векторы
        vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
        vectors.create_vector("key2", np.array([0.0, 1.0, 0.0]))
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Сохраняем векторы
            result = vectors.save(temp_path)
            assert result is True
            
            # Загружаем векторы
            loaded_vectors = DynamicContextVectors.load(temp_path)
            
            # Проверяем параметры
            assert loaded_vectors.vector_size == vectors.vector_size
            
            # Проверяем векторы
            assert loaded_vectors.has_key("key1")
            assert loaded_vectors.has_key("key2")
            assert np.allclose(loaded_vectors.get_vector("key1"), np.array([1.0, 0.0, 0.0]))
            assert np.allclose(loaded_vectors.get_vector("key2"), np.array([0.0, 1.0, 0.0]))
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)