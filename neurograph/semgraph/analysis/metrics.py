# neurograph/semgraph/analysis/metrics.py

from typing import Dict, List, Any, Tuple, Optional, Set
import math
import networkx as nx

from neurograph.semgraph.base import ISemGraph
from neurograph.core.logging import get_logger

logger = get_logger("semgraph.analysis.metrics")

class GraphMetrics:
    """Класс для расчета метрик графа."""
    
    def __init__(self, graph: ISemGraph):
        """Инициализирует класс метрик.
        
        Args:
            graph: Граф для анализа.
        """
        self.graph = graph
        
        # Конвертируем граф в формат NetworkX для использования алгоритмов NetworkX
        self.nx_graph = self._convert_to_networkx()
    
    def _convert_to_networkx(self) -> nx.MultiDiGraph:
        """Конвертирует граф в формат NetworkX.
        
        Returns:
            Граф NetworkX.
        """
        G = nx.MultiDiGraph()
        
        # Добавляем узлы
        for node_id in self.graph.get_all_nodes():
            node_attrs = self.graph.get_node(node_id) or {}
            G.add_node(node_id, **node_attrs)
        
        # Добавляем ребра
        for source, target, edge_type in self.graph.get_all_edges():
            edge_attrs = self.graph.get_edge(source, target, edge_type) or {}
            weight = edge_attrs.get("weight", 1.0)
            G.add_edge(source, target, key=edge_type, weight=weight, **edge_attrs)
        
        return G
    
    def get_node_count(self) -> int:
        """Возвращает количество узлов в графе.
        
        Returns:
            Количество узлов.
        """
        return len(self.graph.get_all_nodes())
    
    def get_edge_count(self) -> int:
        """Возвращает количество ребер в графе.
        
        Returns:
            Количество ребер.
        """
        return len(self.graph.get_all_edges())
    
    def get_node_degrees(self) -> Dict[str, Dict[str, int]]:
        """Возвращает степени узлов (входящие и исходящие).
        
        Returns:
            Словарь вида {node_id: {"in": in_degree, "out": out_degree, "total": total_degree}}.
        """
        result = {}
        
        for node_id in self.graph.get_all_nodes():
            in_degree = 0
            out_degree = 0
            
            # Считаем исходящие ребра
            out_degree = len(self.graph.get_neighbors(node_id))
            
            # Считаем входящие ребра
            for source, target, _ in self.graph.get_all_edges():
                if target == node_id:
                    in_degree += 1
            
            result[node_id] = {
                "in": in_degree,
                "out": out_degree,
                "total": in_degree + out_degree
            }
        
        return result
    
    def get_central_nodes(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Возвращает наиболее центральные узлы по метрике PageRank.
        
        Args:
            top_n: Количество узлов для возврата.
            
        Returns:
            Список кортежей (node_id, centrality_score).
        """
        try:
            # Используем алгоритм PageRank из NetworkX
            pagerank = nx.pagerank(self.nx_graph, weight="weight")
            
            # Сортируем по убыванию значения центральности
            sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_nodes[:top_n]
        except Exception as e:
            logger.error(f"Ошибка при расчете центральности узлов: {str(e)}")
            return []
    
    def get_shortest_paths(self, source: str, targets: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
        """Возвращает кратчайшие пути от исходного узла до целевых узлов.
        
        Args:
            source: Исходный узел.
            targets: Список целевых узлов.
            
        Returns:
            Словарь вида {target: path}, где path - список ребер в пути.
        """
        from neurograph.semgraph.query.path import PathFinder
        
        path_finder = PathFinder(self.graph)
        
        result = {}
        for target in targets:
            if target == source:
                result[target] = []
                continue
                
            path = path_finder.find_weighted_shortest_path(source, target)
            result[target] = path
        
        return result
    
    def get_connected_components(self) -> List[Set[str]]:
        """Возвращает связные компоненты графа.
        
        Returns:
            Список множеств узлов, где каждое множество представляет одну связную компоненту.
        """
        # Создаем неориентированный граф для нахождения связных компонент
        undirected_graph = self.nx_graph.to_undirected()
        
        # Находим связные компоненты
        components = list(nx.connected_components(undirected_graph))
        
        return components
    
    def get_communities(self, resolution: float = 1.0) -> List[Set[str]]:
        """Находит сообщества в графе, используя метод Лувена.
        
        Args:
            resolution: Параметр разрешения (влияет на размер сообществ).
            
        Returns:
            Список множеств узлов, где каждое множество представляет одно сообщество.
        """
        try:
            from community import best_partition
            
            # Создаем неориентированный граф для нахождения сообществ
            undirected_graph = self.nx_graph.to_undirected()
            
            # Находим сообщества с помощью метода Лувена
            partition = best_partition(undirected_graph, resolution=resolution)
            
            # Группируем узлы по сообществам
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = set()
                communities[community_id].add(node)
            
            return list(communities.values())
        except ImportError:
            logger.warning("Библиотека community не установлена. Используется альтернативный метод нахождения сообществ.")
            
            # Используем алгоритм модулярности вместо метода Лувена
            undirected_graph = self.nx_graph.to_undirected()
            communities = nx.algorithms.community.greedy_modularity_communities(undirected_graph)
            return [set(community) for community in communities]
        except Exception as e:
            logger.error(f"Ошибка при нахождении сообществ: {str(e)}")
            return []
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """Вычисляет все метрики графа.
        
        Returns:
            Словарь с метриками.
        """
        metrics = {
            "node_count": self.get_node_count(),
            "edge_count": self.get_edge_count(),
            "density": self.get_edge_count() / (self.get_node_count() * (self.get_node_count() - 1)) if self.get_node_count() > 1 else 0,
            "central_nodes": self.get_central_nodes(),
            "connected_components": len(self.get_connected_components()),
            "average_degree": sum(data["total"] for data in self.get_node_degrees().values()) / self.get_node_count() if self.get_node_count() > 0 else 0
        }
        
        return metrics