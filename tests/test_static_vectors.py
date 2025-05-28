"""Тесты для статических векторных представлений."""

import pytest
import numpy as np

from neurograph.contextvec.impl.static import StaticContextVectors


def test_create_get_vector():
    """Тест создания и получения векторного представления."""
    vectors = StaticContextVectors(vector_size=3)
    
    # Создание вектора
    result = vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
    assert result is True
    
    # Проверка, что вектор был создан
    assert vectors.has_key("key1")
    
    # Получение вектора
    vector = vectors.get_vector("key1")
    assert vector is not None
    assert vector.shape == (3,)
    # Вектор должен быть нормализован
    assert np.allclose(vector, np.array([1.0, 0.0, 0.0]))
    
    # Получение несуществующего вектора
    vector = vectors.get_vector("non_existent")
    assert vector is None


def test_similarity():
    """Тест вычисления близости между векторами."""
    vectors = StaticContextVectors(vector_size=3)
    
    # Создание векторов
    vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
    vectors.create_vector("key2", np.array([0.0, 1.0, 0.0]))
    vectors.create_vector("key3", np.array([1.0, 1.0, 0.0]))
    
    # Проверка близости между ортогональными векторами
    sim = vectors.similarity("key1", "key2")
    assert sim is not None
    assert np.isclose(sim, 0.0)
    
    # Проверка близости между векторами под углом 45 градусов
    sim = vectors.similarity("key1", "key3")
    assert sim is not None
    assert np.isclose(sim, 1.0 / np.sqrt(2))
    
    # Проверка близости с несуществующим ключом
    sim = vectors.similarity("key1", "non_existent")
    assert sim is None


def test_most_similar():
    """Тест поиска наиболее похожих векторов."""
    vectors = StaticContextVectors(vector_size=3)
    
    # Создание векторов
    vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
    vectors.create_vector("key2", np.array([0.9, 0.1, 0.0]))
    vectors.create_vector("key3", np.array([0.7, 0.3, 0.0]))
    vectors.create_vector("key4", np.array([0.0, 1.0, 0.0]))
    
    # Получение наиболее похожих на key1
    similar = vectors.get_most_similar("key1", top_n=2)
    assert len(similar) == 2
    
    # Проверка, что ключи возвращаются в порядке убывания сходства
    assert similar[0][0] == "key2"  # Наиболее похожий
    assert similar[1][0] == "key3"  # Менее похожий
    
    # Проверка с несуществующим ключом
    similar = vectors.get_most_similar("non_existent")
    assert similar == []


def test_remove_vector():
    """Тест удаления векторного представления."""
    vectors = StaticContextVectors(vector_size=3)
    
    # Создание вектора
    vectors.create_vector("key1", np.array([1.0, 0.0, 0.0]))
    assert vectors.has_key("key1")
    
    # Удаление вектора
    result = vectors.remove_vector("key1")
    assert result is True
    assert not vectors.has_key("key1")
    
    # Попытка удаления несуществующего вектора
    result = vectors.remove_vector("non_existent")
    assert result is False