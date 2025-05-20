"""Адаптер для моделей Word2Vec."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import os
import importlib.util
import re

from neurograph.core.logging import get_logger


logger = get_logger("contextvec.adapters.word2vec")


class Word2VecAdapter:
    """Адаптер для моделей Word2Vec для создания эмбеддингов."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Инициализирует адаптер.
        
        Args:
            model_path: Путь к предварительно обученной модели Word2Vec.
            
        Raises:
            ImportError: Если библиотека gensim не установлена.
        """
        # Проверяем, установлена ли библиотека gensim
        if not importlib.util.find_spec("gensim"):
            error_msg = "Библиотека gensim не установлена. Установите ее с помощью pip install gensim"
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        # Импортируем внутри метода, чтобы не требовать наличия библиотеки для остальных частей системы
        from gensim.models import KeyedVectors, Word2Vec
        
        self.model = None
        self.word_vectors = None
        
        if model_path is not None:
            try:
                # Проверяем формат файла модели
                if model_path.endswith('.bin') or model_path.endswith('.vec'):
                    # Загружаем бинарный формат word2vec или текстовый формат
                    self.word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=model_path.endswith('.bin'))
                else:
                    # Загружаем модель gensim
                    self.model = Word2Vec.load(model_path)
                    self.word_vectors = self.model.wv
                
                self.vector_size = self.word_vectors.vector_size
                logger.info(f"Инициализирован Word2VecAdapter с моделью из {model_path}, "
                           f"размерность вектора: {self.vector_size}")
            except Exception as e:
                error_msg = f"Ошибка при загрузке модели {model_path}: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            error_msg = "Необходимо указать путь к модели Word2Vec"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Предварительная обработка текста.
        
        Args:
            text: Исходный текст.
            
        Returns:
            Список токенов.
        """
        # Преобразуем в нижний регистр
        text = text.lower()
        
        # Удаляем лишние пробелы и специальные символы
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Токенизируем
        tokens = text.strip().split()
        
        return tokens
    
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Создает векторное представление для текста.
        
        Args:
            text: Текст для кодирования.
            normalize: Нормализовать ли векторы.
            
        Returns:
            Векторное представление (среднее векторов слов).
        """
        if not text:
            # Возвращаем нулевой вектор для пустого текста
            return np.zeros(self.vector_size, dtype=np.float32)
        
        # Предварительная обработка текста
        tokens = self._preprocess_text(text)
        
        # Если текст не содержит токенов, возвращаем нулевой вектор
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)
        
        # Получаем векторы для каждого слова
        word_vectors = []
        for token in tokens:
            if token in self.word_vectors:
                word_vectors.append(self.word_vectors[token])
        
        # Если нет найденных векторов, возвращаем нулевой вектор
        if not word_vectors:
            return np.zeros(self.vector_size, dtype=np.float32)
        
        # Вычисляем среднее векторов слов
        text_vector = np.mean(word_vectors, axis=0)
        
        # Нормализуем вектор, если требуется
        if normalize:
            norm = np.linalg.norm(text_vector)
            if norm > 0:
                text_vector = text_vector / norm
        
        return text_vector
    
    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Создает векторные представления для списка текстов.
        
        Args:
            texts: Список текстов для кодирования.
            normalize: Нормализовать ли векторы.
            
        Returns:
            Матрица векторных представлений.
        """
        if not texts:
            return np.array([])
        
        vectors = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        
        for i, text in enumerate(texts):
            vectors[i] = self.encode(text, normalize=normalize)
        
        return vectors
    
    def get_vector_size(self) -> int:
        """Возвращает размерность векторов.
        
        Returns:
            Размерность векторов.
        """
        return self.vector_size
    
    def get_most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Находит наиболее похожие слова.
        
        Args:
            word: Исходное слово.
            top_n: Количество похожих слов для возврата.
            
        Returns:
            Список кортежей (слово, сходство).
        """
        if not self.word_vectors or word not in self.word_vectors:
            return []
        
        return self.word_vectors.most_similar(word, topn=top_n)