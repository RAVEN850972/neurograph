"""Тесты для легковесных моделей векторных представлений."""

import os
import tempfile
import numpy as np
import pytest

from neurograph.contextvec.models.lightweight import HashingVectorizer, RandomProjection


class TestHashingVectorizer:
    """Тесты для класса HashingVectorizer."""
    
    def test_init(self):
        """Проверка инициализации векторизатора."""
        vectorizer = HashingVectorizer(vector_size=100, lowercase=True, ngram_range=(1, 2))
        
        assert vectorizer.vector_size == 100
        assert vectorizer.lowercase is True
        assert vectorizer.ngram_range == (1, 2)
    
    def test_preprocess_text(self):
        """Проверка предварительной обработки текста."""
        vectorizer = HashingVectorizer(lowercase=True)
        
        # Проверяем обработку текста
        processed = vectorizer._preprocess_text("Hello, World! This is a TEST.")
        assert processed == "hello world this is a test"
        
        # Проверяем с отключенным lowercase
        vectorizer = HashingVectorizer(lowercase=False)
        processed = vectorizer._preprocess_text("Hello, World!")
        assert processed == "Hello World"
    
    def test_extract_ngrams(self):
        """Проверка извлечения n-грамм."""
        vectorizer = HashingVectorizer(ngram_range=(1, 2))
        
        # Проверяем извлечение n-грамм
        ngrams = vectorizer._extract_ngrams("hello world")
        
        # Ожидаемые n-граммы: ["hello", "world", "hello world"]
        assert len(ngrams) == 3
        assert "hello" in ngrams
        assert "world" in ngrams
        assert "hello world" in ngrams
        
        # Проверяем с другим диапазоном n-грамм
        vectorizer = HashingVectorizer(ngram_range=(2, 3))
        ngrams = vectorizer._extract_ngrams("hello world test")
        
        # Ожидаемые n-граммы: ["hello world", "world test", "hello world test"]
        assert len(ngrams) == 3
        assert "hello world" in ngrams
        assert "world test" in ngrams
        assert "hello world test" in ngrams
    
    def test_transform(self):
        """Проверка преобразования текста в вектор."""
        vectorizer = HashingVectorizer(vector_size=100)
        
        # Преобразуем текст
        vector = vectorizer.transform("hello world")
        
        # Проверяем размерность вектора
        assert vector.shape == (100,)
        
        # Проверяем, что вектор нормализован
        assert np.isclose(np.linalg.norm(vector), 1.0) or np.isclose(np.linalg.norm(vector), 0.0)
        
        # Преобразуем другой текст
        vector2 = vectorizer.transform("hello world")
        
        # Проверяем, что одинаковый текст дает одинаковые векторы
        assert np.allclose(vector, vector2)
        
        # Преобразуем другой текст
        vector3 = vectorizer.transform("different text")
        
        # Проверяем, что разный текст дает разные векторы
        assert not np.allclose(vector, vector3)
    
    def test_transform_batch(self):
        """Проверка преобразования нескольких текстов."""
        vectorizer = HashingVectorizer(vector_size=100)
        
        # Преобразуем несколько текстов
        texts = ["hello world", "another text", "third example"]
        vectors = vectorizer.transform_batch(texts)
        
        # Проверяем размерность матрицы
        assert vectors.shape == (3, 100)
        
        # Проверяем, что каждая строка - нормализованный вектор
        for i in range(3):
            assert np.isclose(np.linalg.norm(vectors[i]), 1.0) or np.isclose(np.linalg.norm(vectors[i]), 0.0)


class TestRandomProjection:
    """Тесты для класса RandomProjection."""
    
    def test_init(self):
        """Проверка инициализации проекции."""
        projection = RandomProjection(input_dim=100, output_dim=50, seed=42)
        
        assert projection.input_dim == 100
        assert projection.output_dim == 50
        assert projection.projection_matrix.shape == (50, 100)
        
        # Проверяем, что строки матрицы проекции нормализованы
        for i in range(50):
            assert np.isclose(np.linalg.norm(projection.projection_matrix[i]), 1.0)
    
    def test_transform(self):
        """Проверка преобразования вектора."""
        projection = RandomProjection(input_dim=3, output_dim=2, seed=42)
        
        # Преобразуем вектор
        input_vector = np.array([1.0, 0.0, 0.0])
        output_vector = projection.transform(input_vector)
        
        # Проверяем размерность выходного вектора
        assert output_vector.shape == (2,)
        
        # Проверяем, что выходной вектор нормализован
        assert np.isclose(np.linalg.norm(output_vector), 1.0)
        
        # Попытка преобразовать вектор неверной размерности
        with pytest.raises(ValueError):
            projection.transform(np.array([1.0, 0.0]))
    
    def test_transform_batch(self):
        """Проверка преобразования нескольких векторов."""
        projection = RandomProjection(input_dim=3, output_dim=2, seed=42)
        
        # Преобразуем несколько векторов
        input_vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        output_vectors = projection.transform_batch(input_vectors)
        
        # Проверяем размерность выходной матрицы
        assert output_vectors.shape == (3, 2)
        
        # Проверяем, что каждая строка - нормализованный вектор
        for i in range(3):
            assert np.isclose(np.linalg.norm(output_vectors[i]), 1.0)
        
        # Попытка преобразовать векторы неверной размерности
        with pytest.raises(ValueError):
            projection.transform_batch(np.array([[1.0, 0.0], [0.0, 1.0]]))
    
    def test_save_load(self):
        """Проверка сохранения и загрузки проекции."""
        projection = RandomProjection(input_dim=100, output_dim=50, seed=42)
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Сохраняем проекцию
            projection.save(temp_path)
            
            # Загружаем проекцию
            loaded_projection = RandomProjection.load(temp_path)
            
            # Проверяем параметры
            assert loaded_projection.input_dim == projection.input_dim
            assert loaded_projection.output_dim == projection.output_dim
            
            # Проверяем матрицу проекции
            assert np.allclose(loaded_projection.projection_matrix, projection.projection_matrix)
            
            # Проверяем, что преобразования дают одинаковый результат
            input_vector = np.ones(100)
            assert np.allclose(
                loaded_projection.transform(input_vector),
                projection.transform(input_vector)
            )
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)