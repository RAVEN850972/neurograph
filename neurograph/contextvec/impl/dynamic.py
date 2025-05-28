"""Реализация динамических векторных представлений."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cosine
import threading

from neurograph.contextvec.base import IContextVectors
from neurograph.core.errors import InvalidVectorDimensionError, VectorNotFoundError
from neurograph.core.logging import get_logger


logger = get_logger("contextvec.impl.dynamic")


class DynamicContextVectors(IContextVectors):
    """Динамические векторные представления, обновляемые в процессе работы."""
    
    def __init__(self, vector_size: int = 100, use_indexing: bool = True):
        """Инициализирует хранилище векторных представлений.
        
        Args:
            vector_size: Размерность векторов.
            use_indexing: Использовать ли индексацию для быстрого поиска.
        """
        self.vector_size = vector_size
        self.vectors: Dict[str, np.ndarray] = {}
        self.use_indexing = use_indexing
        self.index = None
        self._lock = threading.RLock()
        
        # Инициализация индекса, если требуется
        if use_indexing:
            try:
                from neurograph.semgraph.index.hnsw import HNSWIndex
                self.index = HNSWIndex(dim=vector_size)
                logger.info(f"Инициализирован индекс HNSW для быстрого поиска")
            except ImportError:
                logger.warning("Не удалось инициализировать индекс HNSW. Функциональность поиска будет ограничена.")
                self.use_indexing = False
    
    def create_vector(self, key: str, vector: np.ndarray) -> bool:
        """Создает или обновляет векторное представление для ключа."""
        if vector.shape != (self.vector_size,):
            logger.error(f"Попытка создать вектор с неправильной размерностью: {vector.shape} != ({self.vector_size},)")
            raise InvalidVectorDimensionError(f"Вектор должен иметь размерность {self.vector_size}, получен {vector.shape}")
        
        # Нормализуем вектор для косинусного сходства
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        with self._lock:
            self.vectors[key] = vector
            
            # Обновляем индекс, если он используется
            if self.use_indexing and self.index is not None:
                try:
                    self.index.add_item(key, vector)
                except Exception as e:
                    logger.error(f"Ошибка при добавлении вектора в индекс: {str(e)}")
        
        logger.debug(f"Создан/обновлен вектор для ключа: {key}")
        return True
    
    def get_vector(self, key: str) -> Optional[np.ndarray]:
        """Возвращает векторное представление для ключа."""
        with self._lock:
            if key not in self.vectors:
                return None
            return self.vectors[key].copy()
    
    def similarity(self, key1: str, key2: str) -> Optional[float]:
        """Вычисляет косинусную близость между векторами для двух ключей."""
        with self._lock:
            if key1 not in self.vectors or key2 not in self.vectors:
                return None
            
            # Косинусное сходство (1 - косинусное расстояние)
            return 1.0 - cosine(self.vectors[key1], self.vectors[key2])
    
    def get_most_similar(self, key: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Возвращает список наиболее похожих ключей."""
        with self._lock:
            if key not in self.vectors:
                logger.warning(f"Попытка найти похожие векторы для несуществующего ключа: {key}")
                return []
            
            query_vector = self.vectors[key]
            
            # Используем индекс для быстрого поиска, если доступен
            if self.use_indexing and self.index is not None:
                try:
                    return self.index.search(query_vector, k=top_n + 1)  # +1 для самого ключа
                except Exception as e:
                    logger.error(f"Ошибка при поиске с использованием индекса: {str(e)}")
                    # Fallback к линейному поиску
            
            # Линейный поиск (если индекс не используется или произошла ошибка)
            similarities = []
            for other_key, other_vector in self.vectors.items():
                if other_key != key:
                    sim = 1.0 - cosine(query_vector, other_vector)
                    similarities.append((other_key, sim))
            
            # Сортируем по убыванию сходства и берем top_n
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_n]
    
    def has_key(self, key: str) -> bool:
        """Проверяет наличие ключа в словаре векторов."""
        with self._lock:
            return key in self.vectors
    
    def get_all_keys(self) -> List[str]:
        """Возвращает список всех ключей."""
        with self._lock:
            return list(self.vectors.keys())
    
    def remove_vector(self, key: str) -> bool:
        """Удаляет векторное представление для ключа."""
        with self._lock:
            if key not in self.vectors:
                return False
            
            # Удаляем вектор из словаря
            del self.vectors[key]
            
            # Удаляем из индекса, если он используется
            if self.use_indexing and self.index is not None:
                try:
                    self.index.remove_item(key)
                except Exception as e:
                    logger.error(f"Ошибка при удалении вектора из индекса: {str(e)}")
            
            logger.debug(f"Удален вектор для ключа: {key}")
            return True
    
    def update_vector(self, key: str, new_vector: np.ndarray, learning_rate: float = 0.1) -> bool:
        """Обновляет вектор с учетом нового векторного представления.
        
        Args:
            key: Ключ для обновления.
            new_vector: Новое векторное представление.
            learning_rate: Скорость обучения (вес нового вектора).
            
        Returns:
            True, если вектор успешно обновлен, иначе False.
        """
        if new_vector.shape != (self.vector_size,):
            logger.error(f"Попытка обновить вектор с неправильной размерностью: {new_vector.shape} != ({self.vector_size},)")
            raise InvalidVectorDimensionError(f"Вектор должен иметь размерность {self.vector_size}, получен {new_vector.shape}")
        
        with self._lock:
            if key not in self.vectors:
                logger.warning(f"Попытка обновить несуществующий вектор: {key}")
                return False
            
            # Нормализуем новый вектор
            norm = np.linalg.norm(new_vector)
            if norm > 0:
                new_vector = new_vector / norm
            
            # Обновляем вектор с помощью интерполяции
            current_vector = self.vectors[key]
            updated_vector = (1 - learning_rate) * current_vector + learning_rate * new_vector
            
            # Нормализуем результат
            norm = np.linalg.norm(updated_vector)
            if norm > 0:
                updated_vector = updated_vector / norm
            
            self.vectors[key] = updated_vector
            
            # Обновляем индекс, если он используется
            if self.use_indexing and self.index is not None:
                try:
                    self.index.add_item(key, updated_vector)
                except Exception as e:
                    logger.error(f"Ошибка при обновлении вектора в индексе: {str(e)}")
            
            logger.debug(f"Обновлен вектор для ключа: {key} с learning_rate={learning_rate}")
            return True
    
    def average_vectors(self, keys: List[str]) -> Optional[np.ndarray]:
        """Вычисляет среднее значение векторов для заданных ключей.
        
        Args:
            keys: Список ключей для усреднения.
            
        Returns:
            Усредненный вектор или None, если ни один из ключей не найден.
        """
        with self._lock:
            valid_vectors = []
            
            for key in keys:
                if key in self.vectors:
                    valid_vectors.append(self.vectors[key])
                else:
                    logger.warning(f"Ключ не найден при усреднении векторов: {key}")
            
            if not valid_vectors:
                logger.error("Нет действительных векторов для усреднения")
                return None
            
            # Вычисляем среднее векторов
            avg_vector = np.mean(valid_vectors, axis=0)
            
            # Нормализуем результат
            norm = np.linalg.norm(avg_vector)
            if norm > 0:
                avg_vector = avg_vector / norm
            
            return avg_vector
    
    def save(self, file_path: str) -> bool:
        """Сохраняет векторные представления в файл.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            True, если сохранение успешно, иначе False.
        """
        try:
            with self._lock:
                # Сохраняем векторы
                data = {
                    "vector_size": self.vector_size,
                    "vectors": {key: vector.tolist() for key, vector in self.vectors.items()}
                }
                
                from neurograph.core.utils.serialization import JSONSerializer
                JSONSerializer.save_to_file(data, file_path)
                
                # Сохраняем индекс, если он используется
                if self.use_indexing and self.index is not None:
                    try:
                        self.index.save(f"{file_path}.index")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении индекса: {str(e)}")
                
                logger.info(f"Векторные представления сохранены в файл: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении векторных представлений: {str(e)}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> "DynamicContextVectors":
        """Загружает векторные представления из файла.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Экземпляр DynamicContextVectors с загруженными векторами.
        """
        from neurograph.core.utils.serialization import JSONSerializer
        data = JSONSerializer.load_from_file(file_path)
        
        vector_size = data["vector_size"]
        vectors_dict = data["vectors"]
        
        # Создаем экземпляр класса
        instance = cls(vector_size=vector_size)
        
        # Загружаем векторы
        for key, vector_list in vectors_dict.items():
            instance.create_vector(key, np.array(vector_list, dtype=np.float32))
        
        # Загружаем индекс, если он существует
        if instance.use_indexing and instance.index is not None:
            try:
                from neurograph.semgraph.index.hnsw import HNSWIndex
                instance.index = HNSWIndex.load(f"{file_path}.index")
            except Exception as e:
                logger.error(f"Ошибка при загрузке индекса: {str(e)}")
        
        logger.info(f"Векторные представления загружены из файла: {file_path}")
        return instance