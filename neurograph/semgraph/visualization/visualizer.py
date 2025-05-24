"""Визуализация графа."""

from typing import Dict, List, Set, Tuple, Any, Optional, Union
import os
import tempfile
from pathlib import Path
import json

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from neurograph.semgraph.base import ISemGraph
from neurograph.core.logging import get_logger


logger = get_logger("semgraph.visualization.visualizer")


class GraphVisualizer:
    """Класс для визуализации графа."""
    
    def __init__(self, graph: ISemGraph):
        """Инициализирует визуализатор.
        
        Args:
            graph: Граф для визуализации.
        """
        self.graph = graph
        self.colors = list(mcolors.TABLEAU_COLORS)
        self.edge_type_colors = {}
    
    def visualize(self, output_path: Optional[Union[str, Path]] = None, 
                 show: bool = True, 
                 node_size: int = 500,
                 figsize: Tuple[int, int] = (12, 8)) -> Optional[plt.Figure]:
        """Визуализирует граф.
        
        Args:
            output_path: Путь для сохранения изображения (если None, не сохраняется).
            show: Показывать ли изображение.
            node_size: Размер узлов.
            figsize: Размер фигуры.
            
        Returns:
            Объект Figure, если show=False, иначе None.
        """
        # Создаем граф NetworkX
        nx_graph = nx.MultiDiGraph()
        
        # Добавляем узлы
        for node_id in self.graph.get_all_nodes():
            node_attrs = self.graph.get_node(node_id) or {}
            nx_graph.add_node(node_id, **node_attrs)
        
        # Добавляем ребра и определяем цвета для типов ребер
        for source, target, edge_type in self.graph.get_all_edges():
            # Определяем цвет для типа ребра, если еще не определен
            if edge_type not in self.edge_type_colors:
                self.edge_type_colors[edge_type] = self.colors[len(self.edge_type_colors) % len(self.colors)]
            
            edge_attrs = self.graph.get_edge(source, target, edge_type) or {}
            edge_color = self.edge_type_colors[edge_type]
            nx_graph.add_edge(source, target, key=edge_type, color=edge_color, **edge_attrs)
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=figsize)
        
        # Вычисляем позиции узлов
        pos = nx.spring_layout(nx_graph, seed=42)
        
        # Рисуем узлы
        nx.draw_networkx_nodes(nx_graph, pos, node_size=node_size, node_color="skyblue", alpha=0.8)
        
        # Рисуем метки узлов
        nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight="bold")
        
        # Рисуем ребра с разными цветами для разных типов
        for edge_type, color in self.edge_type_colors.items():
            # Фильтруем ребра по типу
            edges = [(s, t) for s, t, k in nx_graph.edges(keys=True) if k == edge_type]
            if edges:
                nx.draw_networkx_edges(nx_graph, pos, edgelist=edges, edge_color=color, 
                                       width=1.5, alpha=0.7, arrows=True, 
                                       connectionstyle="arc3,rad=0.1")
        
        # Добавляем легенду для типов ребер
        if self.edge_type_colors:
            legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=edge_type) 
                              for edge_type, color in self.edge_type_colors.items()]
            plt.legend(handles=legend_elements, loc="upper right")
        
        # Удаляем оси
        plt.axis("off")
        
        # Добавляем заголовок
        plt.title(f"Граф из {len(nx_graph.nodes)} узлов и {nx_graph.number_of_edges()} ребер")
        
        # Сохраняем изображение, если указан путь
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            logger.info(f"Изображение графа сохранено в: {output_path}")
        
        # Показываем изображение, если требуется
        if show:
            plt.show()
            return None
        
        return fig
    
    def visualize_subgraph(self, node_ids: List[str], 
                          output_path: Optional[Union[str, Path]] = None,
                          show: bool = True,
                          include_neighbors: bool = False,
                          node_size: int = 500,
                          figsize: Tuple[int, int] = (12, 8)) -> Optional[plt.Figure]:
        """Визуализирует подграф.
        
        Args:
            node_ids: Идентификаторы узлов для включения в подграф.
            output_path: Путь для сохранения изображения (если None, не сохраняется).
            show: Показывать ли изображение.
            include_neighbors: Включать ли соседей указанных узлов.
            node_size: Размер узлов.
            figsize: Размер фигуры.
            
        Returns:
            Объект Figure, если show=False, иначе None.
        """
        # Создаем множество узлов для включения
        nodes_to_include = set(node_id for node_id in node_ids if self.graph.has_node(node_id))
        
        # Добавляем соседей, если требуется
        if include_neighbors:
            neighbors = set()
            for node_id in nodes_to_include:
                neighbors.update(self.graph.get_neighbors(node_id))
            nodes_to_include.update(neighbors)
        
        # Создаем граф NetworkX
        nx_graph = nx.MultiDiGraph()
        
        # Добавляем узлы
        for node_id in nodes_to_include:
            node_attrs = self.graph.get_node(node_id) or {}
            nx_graph.add_node(node_id, **node_attrs)
        
        # Добавляем ребра между включенными узлами
        for source, target, edge_type in self.graph.get_all_edges():
            if source in nodes_to_include and target in nodes_to_include:
                # Определяем цвет для типа ребра, если еще не определен
                if edge_type not in self.edge_type_colors:
                    self.edge_type_colors[edge_type] = self.colors[len(self.edge_type_colors) % len(self.colors)]
                
                edge_attrs = self.graph.get_edge(source, target, edge_type) or {}
                edge_color = self.edge_type_colors[edge_type]
                nx_graph.add_edge(source, target, key=edge_type, color=edge_color, **edge_attrs)
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=figsize)
        
        # Если граф пустой, выводим сообщение и возвращаем фигуру
        if len(nx_graph.nodes) == 0:
            plt.text(0.5, 0.5, "Пустой подграф", ha="center", va="center")
            plt.axis("off")
            
            if output_path:
                plt.savefig(output_path, bbox_inches="tight", dpi=300)
            
            if show:
                plt.show()
                return None
            
            return fig
        
        # Вычисляем позиции узлов
        pos = nx.spring_layout(nx_graph, seed=42)
        
        # Определяем цвета узлов: выделяем исходные узлы
        node_colors = ["red" if node in node_ids else "skyblue" for node in nx_graph.nodes]
        
        # Рисуем узлы
        nx.draw_networkx_nodes(nx_graph, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
        
        # Рисуем метки узлов
        nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight="bold")
        
        # Рисуем ребра с разными цветами для разных типов
        for edge_type, color in self.edge_type_colors.items():
            # Фильтруем ребра по типу
            edges = [(s, t) for s, t, k in nx_graph.edges(keys=True) if k == edge_type]
            if edges:
                nx.draw_networkx_edges(nx_graph, pos, edgelist=edges, edge_color=color, 
                                       width=1.5, alpha=0.7, arrows=True, 
                                       connectionstyle="arc3,rad=0.1")
        
        # Добавляем легенду для типов ребер
        if self.edge_type_colors:
            legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=edge_type) 
                              for edge_type, color in self.edge_type_colors.items()]
            # Добавляем легенду для цветов узлов
            legend_elements.extend([
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Исходные узлы"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="skyblue", markersize=10, label="Соседи")
            ])
            plt.legend(handles=legend_elements, loc="upper right")
        
        # Удаляем оси
        plt.axis("off")
        
        # Добавляем заголовок
        plt.title(f"Подграф из {len(nx_graph.nodes)} узлов и {nx_graph.number_of_edges()} ребер")
        
        # Сохраняем изображение, если указан путь
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            logger.info(f"Изображение подграфа сохранено в: {output_path}")
        
        # Показываем изображение, если требуется
        if show:
            plt.show()
            return None
        
        return fig
    
    # Добавленные методы
    def save_as_graphml(self, output_path: str) -> None:
        """Сохраняет граф в формате GraphML.
        
        Args:
            output_path: Путь для сохранения файла.
        """
        # Создаем граф NetworkX
        nx_graph = nx.MultiDiGraph()
        
        # Добавляем узлы
        for node_id in self.graph.get_all_nodes():
            node_attrs = self.graph.get_node(node_id) or {}
            nx_graph.add_node(node_id, **node_attrs)
        
        # Добавляем ребра
        for source, target, edge_type in self.graph.get_all_edges():
            edge_attrs = self.graph.get_edge(source, target, edge_type) or {}
            nx_graph.add_edge(source, target, key=edge_type, **edge_attrs)
        
        # Сохраняем в формате GraphML
        nx.write_graphml(nx_graph, output_path)
        logger.info(f"Граф сохранен в формате GraphML: {output_path}")

    def save_as_gexf(self, output_path: str) -> None:
        """Сохраняет граф в формате GEXF (для Gephi).
        
        Args:
            output_path: Путь для сохранения файла.
        """
        # Создаем граф NetworkX
        nx_graph = nx.MultiDiGraph()
        
        # Добавляем узлы
        for node_id in self.graph.get_all_nodes():
            node_attrs = self.graph.get_node(node_id) or {}
            nx_graph.add_node(node_id, **node_attrs)
        
        # Добавляем ребра
        for source, target, edge_type in self.graph.get_all_edges():
            edge_attrs = self.graph.get_edge(source, target, edge_type) or {}
            nx_graph.add_edge(source, target, key=edge_type, **edge_attrs)
        
        # Сохраняем в формате GEXF
        nx.write_gexf(nx_graph, output_path)
        logger.info(f"Граф сохранен в формате GEXF: {output_path}")

    def export_to_cytoscape(self, output_path: str) -> None:
        """Экспортирует граф в формат Cytoscape.js.
        
        Args:
            output_path: Путь для сохранения файла.
        """
        elements = {
            "nodes": [],
            "edges": []
        }
        
        # Добавляем узлы
        for node_id in self.graph.get_all_nodes():
            node_attrs = self.graph.get_node(node_id) or {}
            node_obj = {
                "data": {
                    "id": node_id,
                    **node_attrs
                }
            }
            elements["nodes"].append(node_obj)
        
        # Добавляем ребра
        edge_count = 0
        for source, target, edge_type in self.graph.get_all_edges():
            edge_attrs = self.graph.get_edge(source, target, edge_type) or {}
            edge_id = f"{source}-{target}-{edge_type}"
            
            edge_obj = {
                "data": {
                    "id": edge_id,
                    "source": source,
                    "target": target,
                    "edge_type": edge_type,
                    **edge_attrs
                }
            }
            elements["edges"].append(edge_obj)
            edge_count += 1
        
        # Сохраняем в формате JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(elements, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Граф экспортирован в формат Cytoscape.js: {output_path} ({len(elements['nodes'])} узлов, {edge_count} ребер)")