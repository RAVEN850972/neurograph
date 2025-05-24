"""Реализация индекса HNSW для быстрого поиска в графе."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from hnswlib import Index

from neurograph.core.errors import VectorError
from neurograph.core.logging import get_logger


logger = get_logger("semgraph.index.hnsw")


class HNSWIndex:
    """Индекс на основе Hierarchical Navigable Small World для быстрого поиска ближайших соседей."""
    
    def __init__(self, dim: int, max_elements: int = 10000, ef_construction: int = 200, M: int = 16):
        """Инициализирует индекс HNSW.
        
        Args:
            dim: Размерность векторов.
            max_elements: Максимальное количество элементов в индексе.
            ef_construction: Параметр ef_construction для HNSW.
            M: Параметр M для HNSW.
        """
        self.dim = dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        
        # Создаем индекс для косинусного расстояния
        self.index = Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        
        # Маппинг индексов к идентификаторам и обратно
        self.index_to_id: Dict[int, str] = {}
        self.id_to_index: Dict[str, int] = {}
        
        # Счетчик для назначения индексов
        self.next_index = 0
        
        logger.debug(f"Создан индекс HNSW с размерностью {dim}, максимальным количеством элементов {max_elements}")
    
    def add_item(self, item_id: str, vector: np.ndarray) -> None:
        """Добавляет элемент в индекс.
        
        Args:
            item_id: Идентификатор элемента.
            vector: Векторное представление элемента.
            
        Raises:
            VectorError: Если вектор имеет неправильную размерность.
        """
        if vector.shape != (self.dim,):
            raise VectorError(f"Вектор должен иметь размерность {self.dim}, получен {vector.shape}")
        
        # Нормализуем вектор для косинусного расстояния
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Проверяем, есть ли элемент уже в индексе
        if item_id in self.id_to_index:
            index = self.id_to_index[item_id]
            # Обновляем существующий элемент
            self.index.mark_deleted(index)
            self.index.add_items(vector, index)
            logger.debug(f"Обновлен элемент в индексе: {item_id} -> {index}")
        else:
            # Добавляем новый элемент
            self.index.add_items(vector, self.next_index)
            self.id_to_index[item_id] = self.next_index
            self.index_to_id[self.next_index] = item_id
            self.next_index += 1
            logger.debug(f"Добавлен элемент в индекс: {item_id} -> {self.next_index - 1}")
    
    def remove_item(self, item_id: str) -> bool:
        """Удаляет элемент из индекса.
        
        Args:
            item_id: Идентификатор элемента.
            
        Returns:
            True, если элемент был удален, иначе False.
        """
        if item_id not in self.id_to_index:
            return False
        
        index = self.id_to_index[item_id]
        self.index.mark_deleted(index)
        
        del self.id_to_index[item_id]
        del self.index_to_id[index]
        
        logger.debug(f"Удален элемент из индекса: {item_id}")
        return True
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Ищет ближайшие элементы к запросу.
        
        Args:
            query_vector: Вектор запроса.
            k: Количество ближайших соседей для возврата.
            
        Returns:
            Список кортежей (id, расстояние).
            
        Raises:
            VectorError: Если вектор запроса имеет неправильную размерность.
        """
        if query_vector.shape != (self.dim,):
            raise VectorError(f"Вектор запроса должен иметь размерность {self.dim}, получен {query_vector.shape}")
        
        # Нормализуем вектор запроса
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        # Получаем ближайших соседей
        indices, distances = self.index.knn_query(query_vector, k=min(k, len(self.id_to_index)))
        
        # Преобразуем индексы в идентификаторы и конвертируем расстояния в сходство
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx in self.index_to_id:
                # Преобразуем косинусное расстояние в косинусное сходство (1 - расстояние)
                similarity = 1.0 - dist
                results.append((self.index_to_id[idx], similarity))
        
        return results
    
    def save(self, file_path: str) -> None:
        """Сохраняет индекс в файл.
        
        Args:
            file_path: Путь к файлу.
        """
        self.index.save_index(f"{file_path}.hnsw")
        
        # Сохраняем маппинги и параметры
        import pickle
        with open(f"{file_path}.meta", "wb") as f:
            pickle.dump({
                "dim": self.dim,
                "max_elements": self.max_elements,
                "ef_construction": self.ef_construction,
                "M": self.M,
                "index_to_id": self.index_to_id,
                "id_to_index": self.id_to_index,
                "next_index": self.next_index
            }, f)
        
        logger.info(f"Индекс сохранен в файл: {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> "HNSWIndex":
        """Загружает индекс из файла.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Загруженный индекс.
        """
        import pickle
        with open(f"{file_path}.meta", "rb") as f:
            meta = pickle.load(f)
        
        # Создаем экземпляр с теми же параметрами
        index = cls(
            dim=meta["dim"],
            max_elements=meta["max_elements"],
            ef_construction=meta["ef_construction"],
            M=meta["M"]
        )
        
        # Загружаем индекс
        index.index.load_index(f"{file_path}.hnsw")
        
        # Восстанавливаем маппинги и счетчик
        index.index_to_id = meta["index_to_id"]
        index.id_to_index = meta["id_to_index"]
        index.next_index = meta["next_index"]
        
        logger.info(f"Индекс загружен из файла: {file_path}")
        return index