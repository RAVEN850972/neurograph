"""Адаптер для моделей Sentence Transformers."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import os
import importlib.util

from neurograph.core.logging import get_logger


logger = get_logger("contextvec.adapters.sentence")


class SentenceTransformerAdapter:
    """Адаптер для моделей Sentence Transformers для создания эмбеддингов."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Инициализирует адаптер.
        
        Args:
            model_name: Имя модели Sentence Transformers.
            
        Raises:
            ImportError: Если библиотека sentence-transformers не установлена.
        """
        # Проверяем, установлена ли библиотека sentence-transformers
        if not importlib.util.find_spec("sentence_transformers"):
            error_msg = ("Библиотека sentence-transformers не установлена. "
                       "Установите ее с помощью pip install sentence-transformers")
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        # Импортируем внутри метода, чтобы не требовать наличия библиотеки для остальных частей системы
        from sentence_transformers import SentenceTransformer
        
        try:
            self.model = SentenceTransformer(model_name)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Инициализирован SentenceTransformerAdapter с моделью {model_name}, "
                       f"размерность вектора: {self.vector_size}")
        except Exception as e:
            error_msg = f"Ошибка при загрузке модели {model_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Создает векторное представление для текста.
        
        Args:
            text: Текст для кодирования.
            normalize: Нормализовать ли векторы.
            
        Returns:
            Векторное представление.
        """
        if not text:
            # Возвращаем нулевой вектор для пустого текста
            return np.zeros(self.vector_size, dtype=np.float32)
        
        vector = self.model.encode(text, normalize_embeddings=normalize)
        return vector
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, 
                    normalize: bool = True) -> np.ndarray:
        """Создает векторные представления для списка текстов.
        
        Args:
            texts: Список текстов для кодирования.
            batch_size: Размер пакета для обработки.
            normalize: Нормализовать ли векторы.
            
        Returns:
            Матрица векторных представлений.
        """
        if not texts:
            return np.array([])
        
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize)
        return vectors
    
    def get_vector_size(self) -> int:
        """Возвращает размерность векторов.
        
        Returns:
            Размерность векторов.
        """
        return self.vector_size