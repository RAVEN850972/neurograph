"""Базовые классы и интерфейсы для работы с векторными представлениями."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np


class IContextVectors(ABC):
    """Интерфейс для работы с векторными представлениями.
    
    Этот интерфейс определяет методы для создания, получения и
    сравнения векторных представлений слов, фраз и понятий.
    """
    
    @abstractmethod
    def create_vector(self, key: str, vector: np.ndarray) -> bool:
        """Создает или обновляет векторное представление для ключа.
        
        Args:
            key: Ключ для доступа к вектору (слово, фраза, понятие).
            vector: Векторное представление.
            
        Returns:
            True, если вектор успешно создан или обновлен, иначе False.
        """
        pass
        
    @abstractmethod
    def get_vector(self, key: str) -> Optional[np.ndarray]:
        """Возвращает векторное представление для ключа.
        
        Args:
            key: Ключ для доступа к вектору.
            
        Returns:
            Векторное представление или None, если ключ не найден.
        """
        pass
        
    @abstractmethod
    def similarity(self, key1: str, key2: str) -> Optional[float]:
        """Вычисляет косинусную близость между векторами для двух ключей.
        
        Args:
            key1: Первый ключ.
            key2: Второй ключ.
            
        Returns:
            Значение косинусной близости в диапазоне [-1, 1] или None, 
            если хотя бы один из ключей не найден.
        """
        pass
        
    @abstractmethod
    def get_most_similar(self, key: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Возвращает список наиболее похожих ключей.
        
        Args:
            key: Исходный ключ.
            top_n: Количество похожих ключей для возврата.
            
        Returns:
            Список кортежей (ключ, значение близости).
        """
        pass
        
    @abstractmethod
    def has_key(self, key: str) -> bool:
        """Проверяет наличие ключа в словаре векторов.
        
        Args:
            key: Проверяемый ключ.
            
        Returns:
            True, если ключ существует, иначе False.
        """
        pass
        
    @abstractmethod
    def get_all_keys(self) -> List[str]:
        """Возвращает список всех ключей.
        
        Returns:
            Список всех ключей в словаре векторов.
        """
        pass
        
    @abstractmethod
    def remove_vector(self, key: str) -> bool:
        """Удаляет векторное представление для ключа.
        
        Args:
            key: Ключ для удаления.
            
        Returns:
            True, если вектор успешно удален, иначе False.
        """
        pass