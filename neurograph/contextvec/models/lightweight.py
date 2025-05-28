"""Легковесные модели для создания векторных представлений."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import re
from collections import Counter

from neurograph.core.logging import get_logger


logger = get_logger("contextvec.models.lightweight")


class HashingVectorizer:
    """Векторизатор на основе хеширования, не требующий предварительного словаря."""
    
    def __init__(self, vector_size: int = 100, lowercase: bool = True, 
                ngram_range: Tuple[int, int] = (1, 2)):
        """Инициализирует векторизатор.
        
        Args:
            vector_size: Размерность векторов.
            lowercase: Преобразовывать ли текст в нижний регистр.
            ngram_range: Диапазон n-грамм (min, max).
        """
        self.vector_size = vector_size
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        logger.info(f"Инициализирован HashingVectorizer с размером вектора {vector_size}, "
                   f"ngram_range={ngram_range}")
    
    def _preprocess_text(self, text: str) -> str:
        """Предварительная обработка текста.
        
        Args:
            text: Исходный текст.
            
        Returns:
            Обработанный текст.
        """
        # Преобразуем в нижний регистр, если требуется
        if self.lowercase:
            text = text.lower()
        
        # Удаляем лишние пробелы и специальные символы
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_ngrams(self, text: str) -> List[str]:
        """Извлекает n-граммы из текста.
        
        Args:
            text: Обработанный текст.
            
        Returns:
            Список n-грамм.
        """
        tokens = text.split()
        ngrams = []
        
        # Извлекаем n-граммы в заданном диапазоне
        for n in range(self.ngram_range[0], min(self.ngram_range[1] + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i:i+n]))
        
        return ngrams
    
    def _hash_function(self, feature: str) -> int:
        """Хеш-функция для отображения текстовых признаков в индексы.
        
        Args:
            feature: Текстовый признак.
            
        Returns:
            Индекс в векторе.
        """
        # Используем встроенную хеш-функцию для преобразования строки в число
        hash_value = hash(feature)
        # Берем модуль по размеру вектора
        index = abs(hash_value) % self.vector_size
        
        return index
    
    def transform(self, text: str) -> np.ndarray:
        """Преобразует текст в векторное представление.
        
        Args:
            text: Исходный текст.
            
        Returns:
            Векторное представление.
        """
        # Инициализируем вектор нулями
        vector = np.zeros(self.vector_size, dtype=np.float32)
        
        # Предварительная обработка текста
        processed_text = self._preprocess_text(text)
        
        # Если текст пустой после обработки, возвращаем нулевой вектор
        if not processed_text:
            return vector
        
        # Извлекаем n-граммы
        ngrams = self._extract_ngrams(processed_text)
        
        # Подсчитываем частоту n-грамм
        ngram_counts = Counter(ngrams)
        
        # Заполняем вектор
        for ngram, count in ngram_counts.items():
            index = self._hash_function(ngram)
            # Используем tf (частоту термина)
            vector[index] += count
        
        # Нормализуем вектор, если он не нулевой
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def transform_batch(self, texts: List[str]) -> np.ndarray:
        """Преобразует список текстов в матрицу векторов.
        
        Args:
            texts: Список текстов.
            
        Returns:
            Матрица векторов (количество текстов x размерность вектора).
        """
        vectors = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        
        for i, text in enumerate(texts):
            vectors[i] = self.transform(text)
        
        return vectors


class RandomProjection:
    """Модель для снижения размерности векторов с помощью случайной проекции."""
    
    def __init__(self, input_dim: int, output_dim: int, seed: Optional[int] = None):
        """Инициализирует модель.
        
        Args:
            input_dim: Входная размерность.
            output_dim: Выходная размерность.
            seed: Начальное значение для генератора случайных чисел.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Инициализируем генератор случайных чисел
        rng = np.random.RandomState(seed)
        
        # Создаем матрицу проекции
        # Нормализуем строки, чтобы сохранить длину вектора
        self.projection_matrix = rng.normal(size=(output_dim, input_dim))
        for i in range(output_dim):
            self.projection_matrix[i] /= np.linalg.norm(self.projection_matrix[i])
        
        logger.info(f"Инициализирована RandomProjection {input_dim} -> {output_dim}")
    
    def transform(self, vector: np.ndarray) -> np.ndarray:
        """Преобразует вектор в пространство меньшей размерности.
        
        Args:
            vector: Входной вектор.
            
        Returns:
            Вектор в пространстве меньшей размерности.
        """
        if vector.shape != (self.input_dim,):
            raise ValueError(f"Входной вектор должен иметь размерность {self.input_dim}, получен {vector.shape}")
        
        # Применяем проекцию
        result = np.dot(self.projection_matrix, vector)
        
        # Нормализуем результат
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result
    
    def transform_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Преобразует матрицу векторов в пространство меньшей размерности.
        
        Args:
            vectors: Матрица векторов (количество векторов x input_dim).
            
        Returns:
            Матрица векторов в пространстве меньшей размерности.
        """
        if vectors.shape[1] != self.input_dim:
            raise ValueError(f"Входные векторы должны иметь размерность {self.input_dim}, получены {vectors.shape[1]}")
        
        # Применяем проекцию ко всем векторам
        result = np.dot(vectors, self.projection_matrix.T)
        
        # Нормализуем каждый вектор
        for i in range(result.shape[0]):
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] = result[i] / norm
        
        return result
    
    def save(self, file_path: str) -> None:
        """Сохраняет модель в файл.
        
        Args:
            file_path: Путь к файлу.
        """
        np.savez(file_path, 
                projection_matrix=self.projection_matrix, 
                input_dim=self.input_dim, 
                output_dim=self.output_dim)
        
        logger.info(f"Модель RandomProjection сохранена в файл: {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> "RandomProjection":
        """Загружает модель из файла.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Загруженная модель.
        """
        data = np.load(file_path)
        
        # Создаем экземпляр класса
        instance = cls(input_dim=int(data["input_dim"]), output_dim=int(data["output_dim"]))
        
        # Загружаем матрицу проекции
        instance.projection_matrix = data["projection_matrix"]
        
        logger.info(f"Модель RandomProjection загружена из файла: {file_path}")
        return instance