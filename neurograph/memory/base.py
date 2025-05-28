"""Базовые классы и интерфейсы для работы с памятью."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import time


class MemoryItem:
    """Элемент памяти с контентом и векторным представлением."""
    
    def __init__(self, content: str, embedding: np.ndarray, 
                 content_type: str = "text", metadata: Optional[Dict[str, Any]] = None):
        """Инициализирует элемент памяти.
        
        Args:
            content: Содержимое элемента памяти.
            embedding: Векторное представление содержимого.
            content_type: Тип содержимого (например, "text", "image").
            metadata: Дополнительные метаданные элемента.
        """
        self.content = content
        self.embedding = embedding
        self.content_type = content_type
        self.metadata = metadata or {}
        
        # Используем timestamp вместо datetime для единообразия
        current_time = time.time()
        self.created_at = current_time
        self.last_accessed_at = current_time
        self.access_count = 0
        self.id: Optional[str] = None
    
    def access(self) -> None:
        """Отмечает доступ к элементу памяти."""
        self.last_accessed_at = time.time()
        self.access_count += 1
        
    def __str__(self) -> str:
        return f"MemoryItem(id={self.id}, content_type={self.content_type}, content='{self.content[:30]}...')"


class IMemory(ABC):
    """Интерфейс для системы памяти.
    
    Этот интерфейс определяет методы для работы с системой памяти,
    включая добавление, поиск и удаление элементов.
    """
    
    @abstractmethod
    def add(self, item: MemoryItem) -> str:
        """Добавляет элемент в память и возвращает его ID.
        
        Args:
            item: Элемент для добавления в память.
            
        Returns:
            Уникальный идентификатор добавленного элемента.
        """
        pass
        
    @abstractmethod
    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Возвращает элемент памяти по ID.
        
        Args:
            item_id: Идентификатор элемента.
            
        Returns:
            Элемент памяти или None, если элемент не найден.
        """
        pass
        
    @abstractmethod
    def search(self, query: Union[str, np.ndarray], limit: int = 10) -> List[Tuple[str, float]]:
        """Ищет элементы, похожие на запрос.
        
        Args:
            query: Текстовый запрос или векторное представление.
            limit: Максимальное количество результатов.
            
        Returns:
            Список кортежей (id, score), где score - мера сходства.
        """
        pass
        
    @abstractmethod
    def remove(self, item_id: str) -> bool:
        """Удаляет элемент из памяти.
        
        Args:
            item_id: Идентификатор элемента.
            
        Returns:
            True, если элемент был удален, иначе False.
        """
        pass
        
    @abstractmethod
    def clear(self) -> None:
        """Очищает память."""
        pass
        
    @abstractmethod
    def size(self) -> int:
        """Возвращает размер памяти (количество элементов).
        
        Returns:
            Количество элементов в памяти.
        """
        pass