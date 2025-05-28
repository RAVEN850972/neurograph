# neurograph/semgraph/neural/embedding.py

from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
import os
import json

from neurograph.semgraph.base import ISemGraph
from neurograph.core.logging import get_logger

logger = get_logger("semgraph.neural.embedding")

class GraphEmbedding:
    """Класс для создания и использования векторных представлений графа."""
    
    def __init__(self, graph: ISemGraph, embedding_dim: int = 100):
        """Инициализирует класс для векторных представлений графа.
        
        Args:
            graph: Граф для создания векторных представлений.
            embedding_dim: Размерность векторных представлений.
        """
        self.graph = graph
        self.embedding_dim = embedding_dim
        
        # Словарь для хранения векторных представлений узлов
        self.node_embeddings: Dict[str, np.ndarray] = {}
        
        # Словарь для хранения векторных представлений ребер
        self.edge_type_embeddings: Dict[str, np.ndarray] = {}
    
    def initialize_random_embeddings(self, seed: Optional[int] = None) -> None:
        """Инициализирует случайные векторные представления для узлов и типов ребер.
        
        Args:
            seed: Начальное значение для генератора случайных чисел.
        """
        # Устанавливаем seed, если указан
        rng = np.random.RandomState(seed)
        
        # Инициализируем векторные представления для всех узлов
        for node_id in self.graph.get_all_nodes():
            self.node_embeddings[node_id] = rng.normal(0, 1, self.embedding_dim)
            
            # Нормализуем вектор
            self.node_embeddings[node_id] /= np.linalg.norm(self.node_embeddings[node_id])
        
        # Получаем все типы ребер
        edge_types = set()
        for _, _, edge_type in self.graph.get_all_edges():
            edge_types.add(edge_type)
        
        # Инициализируем векторные представления для всех типов ребер
        for edge_type in edge_types:
            self.edge_type_embeddings[edge_type] = rng.normal(0, 1, self.embedding_dim)
            
            # Нормализуем вектор
            self.edge_type_embeddings[edge_type] /= np.linalg.norm(self.edge_type_embeddings[edge_type])
        
        logger.info(f"Инициализированы случайные векторные представления для {len(self.node_embeddings)} узлов и {len(self.edge_type_embeddings)} типов ребер")
    
    def train_embeddings(self, num_iterations: int = 1000, learning_rate: float = 0.01,
                       regularization: float = 0.01, batch_size: int = 32,
                       negative_samples: int = 5, seed: Optional[int] = None) -> None:
        """Обучает векторные представления на графе с использованием TransE.
        
        Args:
            num_iterations: Количество итераций обучения.
            learning_rate: Скорость обучения.
            regularization: Коэффициент регуляризации.
            batch_size: Размер пакета для обучения.
            negative_samples: Количество отрицательных примеров на один положительный.
            seed: Начальное значение для генератора случайных чисел.
        """
        # Устанавливаем seed, если указан
        rng = np.random.RandomState(seed)
        
        # Инициализируем случайные векторные представления, если это не было сделано ранее
        if not self.node_embeddings:
            self.initialize_random_embeddings(seed)
        
        # Получаем все триплеты (субъект, предикат, объект)
        triplets = []
        for source, target, edge_type in self.graph.get_all_edges():
            triplets.append((source, edge_type, target))
        
        # Количество триплетов
        num_triplets = len(triplets)
        
        if num_triplets == 0:
            logger.warning("Граф не содержит ребер для обучения векторных представлений")
            return
        
        # Обучение
        for iteration in range(num_iterations):
            # Перемешиваем триплеты
            rng.shuffle(triplets)
            
            total_loss = 0.0
            
            # Проходим по пакетам
            for batch_start in range(0, num_triplets, batch_size):
                batch_end = min(batch_start + batch_size, num_triplets)
                batch_triplets = triplets[batch_start:batch_end]
                
                batch_loss = 0.0
                
                # Обрабатываем каждый триплет в пакете
                for subject, predicate, obj in batch_triplets:
                    # Получаем векторные представления
                    subject_embedding = self.node_embeddings[subject]
                    predicate_embedding = self.edge_type_embeddings[predicate]
                    object_embedding = self.node_embeddings[obj]
                    
                    # Генерируем отрицательные примеры
                    for _ in range(negative_samples):
                        # С вероятностью 50% заменяем субъект или объект
                        if rng.random() < 0.5:
                            # Заменяем субъект
                            corrupt_subject = triplets[rng.randint(num_triplets)][0]
                            while corrupt_subject == subject:
                                corrupt_subject = triplets[rng.randint(num_triplets)][0]
                                
                            corrupt_subject_embedding = self.node_embeddings[corrupt_subject]
                            
                            # Обновляем векторные представления
                            pos_score = self._score_triplet(subject_embedding, predicate_embedding, object_embedding)
                            neg_score = self._score_triplet(corrupt_subject_embedding, predicate_embedding, object_embedding)
                            
                            loss = max(0, 1 + pos_score - neg_score)
                            batch_loss += loss
                            
                            if loss > 0:
                                # Градиентный спуск для положительного триплета
                                grad_subject = predicate_embedding + object_embedding
                                grad_predicate = subject_embedding - object_embedding
                                grad_object = subject_embedding - predicate_embedding
                                
                                self.node_embeddings[subject] -= learning_rate * (grad_subject + regularization * subject_embedding)
                                self.edge_type_embeddings[predicate] -= learning_rate * (grad_predicate + regularization * predicate_embedding)
                                self.node_embeddings[obj] -= learning_rate * (grad_object + regularization * object_embedding)
                                
                                # Градиентный спуск для отрицательного триплета
                                grad_corrupt_subject = -(predicate_embedding + object_embedding)
                                
                                self.node_embeddings[corrupt_subject] -= learning_rate * (grad_corrupt_subject + regularization * corrupt_subject_embedding)
                        else:
                            # Заменяем объект
                            corrupt_object = triplets[rng.randint(num_triplets)][2]
                            while corrupt_object == obj:
                                corrupt_object = triplets[rng.randint(num_triplets)][2]
                                
                            corrupt_object_embedding = self.node_embeddings[corrupt_object]
                            
                            # Обновляем векторные представления
                            pos_score = self._score_triplet(subject_embedding, predicate_embedding, object_embedding)
                            neg_score = self._score_triplet(subject_embedding, predicate_embedding, corrupt_object_embedding)
                            
                            loss = max(0, 1 + pos_score - neg_score)
                            batch_loss += loss
                            
                            if loss > 0:
                                # Градиентный спуск для положительного триплета
                                grad_subject = predicate_embedding + object_embedding
                                grad_predicate = subject_embedding - object_embedding
                                grad_object = subject_embedding - predicate_embedding
                                
                                self.node_embeddings[subject] -= learning_rate * (grad_subject + regularization * subject_embedding)
                                self.edge_type_embeddings[predicate] -= learning_rate * (grad_predicate + regularization * predicate_embedding)
                                self.node_embeddings[obj] -= learning_rate * (grad_object + regularization * object_embedding)
                                
                                # Градиентный спуск для отрицательного триплета
                                grad_corrupt_object = -(subject_embedding - predicate_embedding)
                                
                                self.node_embeddings[corrupt_object] -= learning_rate * (grad_corrupt_object + regularization * corrupt_object_embedding)
                
                # Нормализуем все векторные представления после обновления
                for node_id in self.node_embeddings:
                    norm = np.linalg.norm(self.node_embeddings[node_id])
                    if norm > 0:
                        self.node_embeddings[node_id] /= norm
                
                for edge_type in self.edge_type_embeddings:
                    norm = np.linalg.norm(self.edge_type_embeddings[edge_type])
                    if norm > 0:
                        self.edge_type_embeddings[edge_type] /= norm
                
                total_loss += batch_loss
            
            # Логируем процесс обучения
            if (iteration + 1) % 10 == 0 or iteration == 0:
                logger.info(f"Итерация {iteration + 1}/{num_iterations}, потери: {total_loss:.4f}")
        
        logger.info(f"Обучение векторных представлений завершено, итераций: {num_iterations}, финальные потери: {total_loss:.4f}")
    
    def _score_triplet(self, subject_embedding: np.ndarray, predicate_embedding: np.ndarray, 
                     object_embedding: np.ndarray) -> float:
        """Вычисляет оценку для триплета (расстояние h + r ~ t).
        
        Args:
            subject_embedding: Векторное представление субъекта.
            predicate_embedding: Векторное представление предиката.
            object_embedding: Векторное представление объекта.
            
        Returns:
            Оценка триплета (чем меньше, тем лучше).
        """
        # TransE: ||h + r - t||
        score = np.linalg.norm(subject_embedding + predicate_embedding - object_embedding)
        return score
    
    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Возвращает векторное представление узла.
        
        Args:
            node_id: Идентификатор узла.
            
        Returns:
            Векторное представление узла или None, если узел не найден.
        """
        return self.node_embeddings.get(node_id)
    
    def get_edge_type_embedding(self, edge_type: str) -> Optional[np.ndarray]:
        """Возвращает векторное представление типа ребра.
        
        Args:
            edge_type: Тип ребра.
            
        Returns:
            Векторное представление типа ребра или None, если тип не найден.
        """
        return self.edge_type_embeddings.get(edge_type)
    
    def save(self, file_path: str) -> None:
        """Сохраняет векторные представления в файл.
        
        Args:
            file_path: Путь к файлу.
        """
        data = {
            "embedding_dim": self.embedding_dim,
            "node_embeddings": {node_id: embedding.tolist() for node_id, embedding in self.node_embeddings.items()},
            "edge_type_embeddings": {edge_type: embedding.tolist() for edge_type, embedding in self.edge_type_embeddings.items()}
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Векторные представления сохранены в файл: {file_path}")
    
    @classmethod
    def load(cls, file_path: str, graph: ISemGraph) -> "GraphEmbedding":
        """Загружает векторные представления из файла.
        
        Args:
            file_path: Путь к файлу.
            graph: Граф для связывания с векторными представлениями.
            
        Returns:
            Загруженный экземпляр GraphEmbedding.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        embedding_dim = data["embedding_dim"]
        
        instance = cls(graph, embedding_dim)
        
        instance.node_embeddings = {node_id: np.array(embedding) for node_id, embedding in data["node_embeddings"].items()}
        instance.edge_type_embeddings = {edge_type: np.array(embedding) for edge_type, embedding in data["edge_type_embeddings"].items()}
        
        logger.info(f"Векторные представления загружены из файла: {file_path}")
        
        return instance