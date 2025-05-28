"""Реализация статических векторных представлений."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cosine

from neurograph.contextvec.base import IContextVectors


class StaticContextVectors(IContextVectors):
    """Статические векторные представления для слов, фраз и понятий."""
    
    def __init__(self, vector_size: int = 100):
        """Инициализирует хранилище векторных представлений.
        
        Args:
            vector_size: Размерность векторов.
        """
        self.vector_size = vector_size
        self.vectors: Dict[str, np.ndarray] = {}
        
    def create_vector(self, key: str, vector: np.ndarray) -> bool:
        """Создает или обновляет векторное представление для ключа."""
        if vector.shape != (self.vector_size,):
            return False
            
        # Нормализуем вектор для косинусного сходства
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        self.vectors[key] = vector
        return True
        
    def get_vector(self, key: str) -> Optional[np.ndarray]:
        """Возвращает векторное представление для ключа."""
        return self.vectors.get(key)
        
    def similarity(self, key1: str, key2: str) -> Optional[float]:
        """Вычисляет косинусную близость между векторами для двух ключей."""
        if key1 not in self.vectors or key2 not in self.vectors:
            return None
            
        # Косинусное сходство (1 - косинусное расстояние)
        return 1.0 - cosine(self.vectors[key1], self.vectors[key2])
        
    def get_most_similar(self, key: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Возвращает список наиболее похожих ключей."""
        if key not in self.vectors:
            return []
            
        similarities = []
        for other_key in self.vectors:
            if other_key != key:
                sim = self.similarity(key, other_key)
                if sim is not None:
                    similarities.append((other_key, sim))
                    
        # Сортируем по убыванию сходства и берем top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
        
    def has_key(self, key: str) -> bool:
        """Проверяет наличие ключа в словаре векторов."""
        return key in self.vectors
        
    def get_all_keys(self) -> List[str]:
        """Возвращает список всех ключей."""
        return list(self.vectors.keys())
        
    def remove_vector(self, key: str) -> bool:
        """Удаляет векторное представление для ключа."""
        if key in self.vectors:
            del self.vectors[key]
            return True
        return False