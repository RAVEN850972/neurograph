# neurograph/semgraph/query/search.py

from typing import Dict, List, Any, Tuple, Optional, Set, Union
import numpy as np
import re

from neurograph.semgraph.base import ISemGraph
from neurograph.core.logging import get_logger

logger = get_logger("semgraph.query.search")

class GraphSearcher:
    """Класс для выполнения различных типов поиска в графе."""
    
    def __init__(self, graph: ISemGraph):
        """Инициализирует поиск в графе.
        
        Args:
            graph: Граф для поиска.
        """
        self.graph = graph
    
    def search_by_attribute(self, attribute_name: str, attribute_value: Any, 
                          node_types: Optional[List[str]] = None) -> List[str]:
        """Ищет узлы с заданным значением атрибута.
        
        Args:
            attribute_name: Имя атрибута.
            attribute_value: Значение атрибута.
            node_types: Список типов узлов для фильтрации (если None, ищет везде).
            
        Returns:
            Список идентификаторов найденных узлов.
        """
        results = []
        
        for node_id in self.graph.get_all_nodes():
            node_attrs = self.graph.get_node(node_id) or {}
            
            # Если указаны типы узлов, проверяем тип
            if node_types is not None:
                node_type = node_attrs.get("type")
                if node_type not in node_types:
                    continue
            
            # Проверяем значение атрибута
            if attribute_name in node_attrs and node_attrs[attribute_name] == attribute_value:
                results.append(node_id)
        
        return results
    
    def search_by_regex(self, attribute_name: str, pattern: str,
                      node_types: Optional[List[str]] = None) -> List[str]:
        """Ищет узлы, значение атрибута которых соответствует регулярному выражению.
        
        Args:
            attribute_name: Имя атрибута.
            pattern: Регулярное выражение.
            node_types: Список типов узлов для фильтрации (если None, ищет везде).
            
        Returns:
            Список идентификаторов найденных узлов.
        """
        results = []
        regex = re.compile(pattern)
        
        for node_id in self.graph.get_all_nodes():
            node_attrs = self.graph.get_node(node_id) or {}
            
            # Если указаны типы узлов, проверяем тип
            if node_types is not None:
                node_type = node_attrs.get("type")
                if node_type not in node_types:
                    continue
            
            # Проверяем значение атрибута
            if attribute_name in node_attrs:
                attr_value = str(node_attrs[attribute_name])
                if regex.search(attr_value):
                    results.append(node_id)
        
        return results

    def search_by_distance(self, node_id: str, max_distance: int = 2,
                        edge_types: Optional[List[str]] = None) -> Dict[str, int]:
        """Ищет узлы, находящиеся на расстоянии не более max_distance от заданного узла.
        
        Args:
            node_id: Идентификатор исходного узла.
            max_distance: Максимальное расстояние.
            edge_types: Список типов ребер для рассмотрения (если None, рассматриваются все).
            
        Returns:
            Словарь {node_id: distance}.
        """
        if not self.graph.has_node(node_id):
            return {}
        
        # Используем алгоритм поиска в ширину
        distances = {node_id: 0}
        queue = [(node_id, 0)]
        visited = {node_id}
        
        while queue:
            current, distance = queue.pop(0)
            
            # Не идем дальше максимального расстояния
            if distance >= max_distance:
                continue
            
            # Получаем соседей текущего узла
            edges = []
            for source, target, edge_type in self.graph.get_all_edges():
                if source == current:
                    # Если указаны типы ребер, фильтруем по ним
                    if edge_types is not None and edge_type not in edge_types:
                        continue
                    
                    edges.append((target, edge_type))
            
            for neighbor, _ in edges:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distance + 1
                    queue.append((neighbor, distance + 1))
        
        return distances

    def search_nodes_with_similar_connections(self, node_id: str, min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """Ищет узлы, имеющие похожие связи с другими узлами.
        
        Args:
            node_id: Идентификатор исходного узла.
            min_similarity: Минимальное значение сходства для включения в результаты.
            
        Returns:
            Список кортежей (node_id, similarity_score).
        """
        if not self.graph.has_node(node_id):
            return []
        
        # Получаем соседей исходного узла
        source_neighbors = set()
        for source, target, _ in self.graph.get_all_edges():
            if source == node_id:
                source_neighbors.add(target)
        
        results = []
        
        for other_node in self.graph.get_all_nodes():
            if other_node == node_id:
                continue
            
            # Получаем соседей текущего узла
            other_neighbors = set()
            for source, target, _ in self.graph.get_all_edges():
                if source == other_node:
                    other_neighbors.add(target)
            
            # Вычисляем меру сходства Жаккара
            if not source_neighbors and not other_neighbors:
                similarity = 0.0
            else:
                intersection = len(source_neighbors.intersection(other_neighbors))
                union = len(source_neighbors.union(other_neighbors))
                similarity = intersection / union
            
            if similarity >= min_similarity:
                results.append((other_node, similarity))
        
        # Сортируем по убыванию значения сходства
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def search_subgraph_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, str]]:
        """Ищет подграфы, соответствующие заданному паттерну.
        
        Пример паттерна:
        {
            "nodes": [
                {"id": "A", "type": "Person"},
                {"id": "B", "type": "City"}
            ],
            "edges": [
                {"source": "A", "target": "B", "type": "lives_in"}
            ]
        }
        
        Args:
            pattern: Паттерн для поиска.
            
        Returns:
            Список соответствий, где каждое соответствие - словарь {pattern_id: node_id}.
        """
        pattern_nodes = pattern.get("nodes", [])
        pattern_edges = pattern.get("edges", [])
        
        if not pattern_nodes:
            return []
        
        # Начинаем с первого узла паттерна
        first_pattern_node = pattern_nodes[0]
        first_pattern_id = first_pattern_node["id"]
        first_pattern_type = first_pattern_node.get("type")
        
        # Находим все узлы графа, соответствующие первому узлу паттерна
        candidates = []
        for node_id in self.graph.get_all_nodes():
            node_attrs = self.graph.get_node(node_id) or {}
            
            # Если в паттерне указан тип, проверяем его
            if first_pattern_type is not None and node_attrs.get("type") != first_pattern_type:
                continue
            
            # Проверяем атрибуты узла
            match = True
            for attr_name, attr_value in first_pattern_node.items():
                if attr_name == "id":
                    continue
                
                if attr_name not in node_attrs or node_attrs[attr_name] != attr_value:
                    match = False
                    break
            
            if match:
                candidates.append({first_pattern_id: node_id})
        
        # Если нет кандидатов, возвращаем пустой список
        if not candidates:
            return []
        
        # Расширяем каждое соответствие
        results = []
        for candidate in candidates:
            complete_matches = self._expand_match(candidate, pattern_nodes, pattern_edges)
            results.extend(complete_matches)
        
        return results

    def _expand_match(self, partial_match: Dict[str, str], pattern_nodes: List[Dict[str, Any]], 
                    pattern_edges: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Расширяет частичное соответствие паттерну.
        
        Args:
            partial_match: Текущее частичное соответствие.
            pattern_nodes: Узлы паттерна.
            pattern_edges: Ребра паттерна.
            
        Returns:
            Список полных соответствий.
        """
        # Если все узлы паттерна уже сопоставлены, возвращаем текущее соответствие
        if len(partial_match) == len(pattern_nodes):
            return [partial_match]
        
        # Находим следующий узел паттерна, который нужно сопоставить
        next_pattern_node = None
        for node in pattern_nodes:
            if node["id"] not in partial_match:
                # Проверяем, есть ли связь с уже сопоставленными узлами
                connected = False
                for edge in pattern_edges:
                    if ((edge["source"] == node["id"] and edge["target"] in partial_match) or
                        (edge["target"] == node["id"] and edge["source"] in partial_match)):
                        connected = True
                        break
                
                if connected:
                    next_pattern_node = node
                    break
        
        # Если нет следующего узла, который можно сопоставить, возвращаем текущее соответствие
        if next_pattern_node is None:
            return [partial_match] if len(partial_match) == len(pattern_nodes) else []
        
        next_pattern_id = next_pattern_node["id"]
        next_pattern_type = next_pattern_node.get("type")
        
        # Находим связи следующего узла паттерна с уже сопоставленными узлами
        connections = []
        for edge in pattern_edges:
            if edge["source"] == next_pattern_id and edge["target"] in partial_match:
                connections.append({
                    "direction": "outgoing",
                    "pattern_id": edge["target"],
                    "real_id": partial_match[edge["target"]],
                    "edge_type": edge.get("type")
                })
            elif edge["target"] == next_pattern_id and edge["source"] in partial_match:
                connections.append({
                    "direction": "incoming",
                    "pattern_id": edge["source"],
                    "real_id": partial_match[edge["source"]],
                    "edge_type": edge.get("type")
                })
        
        # Находим кандидатов для следующего узла
        candidates = set()
        
        for connection in connections:
            real_id = connection["real_id"]
            edge_type = connection["edge_type"]
            
            if connection["direction"] == "outgoing":
                # Ищем исходящие ребра
                for source, target, type_val in self.graph.get_all_edges():
                    if source == real_id and (edge_type is None or type_val == edge_type):
                        candidates.add(target)
            else:
                # Ищем входящие ребра
                for source, target, type_val in self.graph.get_all_edges():
                    if target == real_id and (edge_type is None or type_val == edge_type):
                        candidates.add(source)
        
        # Фильтруем кандидатов по типу и атрибутам
        filtered_candidates = []
        for node_id in candidates:
            # Пропускаем узлы, которые уже сопоставлены
            if node_id in partial_match.values():
                continue
                
            node_attrs = self.graph.get_node(node_id) or {}
            
            # Если в паттерне указан тип, проверяем его
            if next_pattern_type is not None and node_attrs.get("type") != next_pattern_type:
                continue
            
            # Проверяем атрибуты узла
            match = True
            for attr_name, attr_value in next_pattern_node.items():
                if attr_name == "id":
                    continue
                
                if attr_name not in node_attrs or node_attrs[attr_name] != attr_value:
                    match = False
                    break
            
            if match:
                filtered_candidates.append(node_id)
        
        # Расширяем соответствие для каждого подходящего кандидата
        results = []
        for candidate in filtered_candidates:
            new_match = partial_match.copy()
            new_match[next_pattern_id] = candidate
            
            # Рекурсивно расширяем соответствие
            complete_matches = self._expand_match(new_match, pattern_nodes, pattern_edges)
            results.extend(complete_matches)
        
        return results


class SemanticSearcher:
    """Класс для выполнения семантического поиска в графе с использованием векторных представлений."""

    def __init__(self, graph: ISemGraph, vectors_provider: Any):
        """Инициализирует семантический поиск.
        
        Args:
            graph: Граф для поиска.
            vectors_provider: Провайдер векторных представлений для текста.
        """
        self.graph = graph
        self.vectors_provider = vectors_provider

    def search_by_text_similarity(self, query: str, attribute_name: str = "text", 
                                top_n: int = 10) -> List[Tuple[str, float]]:
        """Ищет узлы, текстовый атрибут которых семантически похож на запрос.
        
        Args:
            query: Текстовый запрос.
            attribute_name: Имя атрибута с текстом.
            top_n: Количество результатов для возврата.
            
        Returns:
            Список кортежей (node_id, similarity_score).
        """
        # Получаем векторное представление запроса
        query_vector = self.vectors_provider.encode(query)
        
        # Поиск узлов с текстовым атрибутом
        results = []
        
        for node_id in self.graph.get_all_nodes():
            node_attrs = self.graph.get_node(node_id) or {}
            
            # Проверяем наличие текстового атрибута
            if attribute_name not in node_attrs:
                continue
                
            text = str(node_attrs[attribute_name])
            
            # Получаем векторное представление текста
            text_vector = self.vectors_provider.encode(text)
            
            # Вычисляем косинусное сходство
            similarity = self._cosine_similarity(query_vector, text_vector)
            
            results.append((node_id, similarity))
        
        # Сортируем по убыванию значения сходства
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_n]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет косинусное сходство между двумя векторами.
        
        Args:
            vec1: Первый вектор.
            vec2: Второй вектор.
            
        Returns:
            Значение косинусного сходства.
        """
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)