# neurograph/propagation/visualizer.py
"""
Визуализация процесса распространения активации.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Any, Tuple
import networkx as nx
from datetime import datetime
import json

from neurograph.propagation.base import PropagationResult, IPropagationVisualizer
from neurograph.core.logging import get_logger


class PropagationVisualizer(IPropagationVisualizer):
    """Визуализатор распространения активации."""
    
    def __init__(self):
        self.logger = get_logger("propagation_visualizer")
        
        # Настройки цветовой схемы
        self.activation_colormap = LinearSegmentedColormap.from_list(
            "activation",
            ["#FFFFFF", "#FFE6E6", "#FFB3B3", "#FF8080", "#FF4D4D", "#FF0000", "#CC0000"],
            N=256
        )
        
        # Настройки размеров и стилей
        self.default_figsize = (12, 8)
        self.default_node_size = 300
        self.default_edge_width = 1.0
        self.default_font_size = 8
        
    def visualize_propagation(self, 
                            result: PropagationResult,
                            graph,
                            save_path: Optional[str] = None,
                            show_animation: bool = False,
                            **kwargs) -> None:
        """Основная визуализация результата распространения."""
        
        if show_animation and len(result.activation_history) > 1:
            self._create_animated_visualization(result, graph, save_path, **kwargs)
        else:
            self._create_static_visualization(result, graph, save_path, **kwargs)
    
    def _create_static_visualization(self, 
                                   result: PropagationResult,
                                   graph,
                                   save_path: Optional[str],
                                   **kwargs) -> None:
        """Создание статической визуализации."""
        
        figsize = kwargs.get('figsize', self.default_figsize)
        node_size_base = kwargs.get('node_size', self.default_node_size)
        show_labels = kwargs.get('show_labels', True)
        layout_type = kwargs.get('layout', 'spring')
        
        # Создание NetworkX графа
        nx_graph = self._convert_to_networkx(graph, result)
        
        # Создание layout
        pos = self._create_layout(nx_graph, layout_type)
        
        # Создание фигуры
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Основной граф с активациями
        self._draw_activation_graph(ax1, nx_graph, pos, result, node_size_base, show_labels)
        
        # График истории активации
        self._draw_activation_history(ax2, result)
        
        # Настройка заголовков и информации
        self._add_visualization_info(fig, result)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Визуализация сохранена: {save_path}")
        
        plt.show()
    
    def _create_animated_visualization(self, 
                                     result: PropagationResult,
                                     graph,
                                     save_path: Optional[str],
                                     **kwargs) -> None:
        """Создание анимированной визуализации."""
        
        figsize = kwargs.get('figsize', self.default_figsize)
        node_size_base = kwargs.get('node_size', self.default_node_size)
        show_labels = kwargs.get('show_labels', False)  # Отключаем для анимации
        layout_type = kwargs.get('layout', 'spring')
        interval = kwargs.get('interval', 500)  # мс между кадрами
        
        # Создание NetworkX графа
        nx_graph = self._convert_to_networkx(graph, result)
        pos = self._create_layout(nx_graph, layout_type)
        
        # Подготовка данных для анимации
        all_nodes = set()
        for step_activations in result.activation_history:
            all_nodes.update(step_activations.keys())
        
        # Создание фигуры
        fig, ax = plt.subplots(figsize=figsize)
        
        def animate(frame):
            ax.clear()
            
            if frame < len(result.activation_history):
                current_activations = result.activation_history[frame]
                
                # Создание временного результата для текущего кадра
                temp_result = PropagationResult(
                    success=True,
                    activated_nodes={
                        node_id: result.activated_nodes.get(node_id, None)
                        for node_id in current_activations.keys()
                        if result.activated_nodes.get(node_id, None) is not None
                    }
                )
                
                # Обновляем уровни активации для текущего кадра
                for node_id, activation_level in current_activations.items():
                    if node_id in temp_result.activated_nodes:
                        temp_result.activated_nodes[node_id].activation_level = activation_level
                
                self._draw_activation_graph(ax, nx_graph, pos, temp_result, 
                                          node_size_base, show_labels)
                
                ax.set_title(f"Распространение активации - Итерация {frame + 1}/{len(result.activation_history)}")
            
            return ax.collections + ax.texts
        
        # Создание анимации
        anim = animation.FuncAnimation(
            fig, animate, frames=len(result.activation_history),
            interval=interval, blit=False, repeat=True
        )
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=2)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=2)
            else:
                anim.save(f"{save_path}.gif", writer='pillow', fps=2)
            
            self.logger.info(f"Анимация сохранена: {save_path}")
        
        plt.show()
    
    def _convert_to_networkx(self, graph, result: PropagationResult) -> nx.DiGraph:
        """Преобразование графа знаний в NetworkX граф."""
        
        nx_graph = nx.DiGraph()
        
        # Добавляем все узлы из результата
        for node_id in result.activated_nodes.keys():
            node_data = graph.get_node(node_id) if graph.has_node(node_id) else {}
            nx_graph.add_node(node_id, **node_data)
        
        # Добавляем ребра между активными узлами
        for source_id in result.activated_nodes.keys():
            if graph.has_node(source_id):
                neighbors = graph.get_neighbors(source_id)
                for neighbor_id in neighbors:
                    if neighbor_id in result.activated_nodes:
                        edge_data = graph.get_edge(source_id, neighbor_id) or {}
                        nx_graph.add_edge(source_id, neighbor_id, **edge_data)
        
        return nx_graph
    
    def _create_layout(self, nx_graph: nx.DiGraph, layout_type: str) -> Dict[str, Tuple[float, float]]:
        """Создание layout для визуализации."""
        
        if layout_type == 'spring':
            return nx.spring_layout(nx_graph, k=1, iterations=50)
        elif layout_type == 'circular':
            return nx.circular_layout(nx_graph)
        elif layout_type == 'kamada_kawai':
            return nx.kamada_kawai_layout(nx_graph)
        elif layout_type == 'spectral':
            return nx.spectral_layout(nx_graph)
        elif layout_type == 'shell':
            return nx.shell_layout(nx_graph)
        else:
            # По умолчанию spring
            return nx.spring_layout(nx_graph, k=1, iterations=50)
    
    def _draw_activation_graph(self, 
                             ax, 
                             nx_graph: nx.DiGraph,
                             pos: Dict[str, Tuple[float, float]],
                             result: PropagationResult,
                             node_size_base: int,
                             show_labels: bool) -> None:
        """Отрисовка графа с активациями."""
        
        # Подготовка данных для визуализации
        node_colors = []
        node_sizes = []
        edge_colors = []
        edge_widths = []
        
        # Цвета и размеры узлов
        for node_id in nx_graph.nodes():
            if node_id in result.activated_nodes:
                activation = result.activated_nodes[node_id]
                activation_level = activation.activation_level
                
                # Цвет по уровню активации
                node_colors.append(activation_level)
                
                # Размер по уровню активации
                size_multiplier = max(0.5, activation_level * 2)
                node_sizes.append(node_size_base * size_multiplier)
            else:
                node_colors.append(0.0)
                node_sizes.append(node_size_base * 0.3)
        
        # Цвета и ширины ребер
        for source, target in nx_graph.edges():
            source_activation = result.activated_nodes.get(source)
            target_activation = result.activated_nodes.get(target)
            
            if source_activation and target_activation:
                # Цвет ребра как среднее активаций концов
                avg_activation = (source_activation.activation_level + target_activation.activation_level) / 2
                edge_colors.append(avg_activation)
                edge_widths.append(self.default_edge_width * (1 + avg_activation))
            else:
                edge_colors.append(0.1)
                edge_widths.append(self.default_edge_width * 0.5)
        
        # Отрисовка ребер
        if nx_graph.edges():
            nx.draw_networkx_edges(
                nx_graph, pos, ax=ax,
                edge_color=edge_colors,
                width=edge_widths,
                alpha=0.6,
                edge_cmap=self.activation_colormap,
                edge_vmin=0, edge_vmax=1
            )
        
        # Отрисовка узлов
        nodes = nx.draw_networkx_nodes(
            nx_graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=self.activation_colormap,
            vmin=0, vmax=1,
            alpha=0.8
        )
        
        # Подписи узлов
        if show_labels:
            # Показываем только наиболее активные узлы
            labels_to_show = {}
            for node_id in nx_graph.nodes():
                if node_id in result.activated_nodes:
                    activation_level = result.activated_nodes[node_id].activation_level
                    if activation_level > 0.3:  # Порог для показа подписи
                        labels_to_show[node_id] = f"{node_id}\n{activation_level:.2f}"
            
            if labels_to_show:
                nx.draw_networkx_labels(
                    nx_graph, pos, labels_to_show, ax=ax,
                    font_size=self.default_font_size,
                    font_weight='bold'
                )
        
        # Настройка осей
        ax.set_title("Распространение активации по графу")
        ax.axis('off')
        
        # Добавление colorbar
        if nodes is not None:
            cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Уровень активации')
    
    def _draw_activation_history(self, ax, result: PropagationResult) -> None:
        """Отрисовка истории активации."""
        
        if not result.activation_history:
            ax.text(0.5, 0.5, 'Нет данных об истории активации', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Подготовка данных
        iterations = list(range(1, len(result.activation_history) + 1))
        
        # Находим узлы с наибольшей максимальной активацией
        max_activations = {}
        for step_data in result.activation_history:
            for node_id, activation in step_data.items():
                if node_id not in max_activations:
                    max_activations[node_id] = activation
                else:
                    max_activations[node_id] = max(max_activations[node_id], activation)
        
        # Берем топ-5 узлов для отображения
        top_nodes = sorted(max_activations.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Строим графики для каждого узла
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_nodes)))
        
        for i, (node_id, _) in enumerate(top_nodes):
            node_history = []
            for step_data in result.activation_history:
                node_history.append(step_data.get(node_id, 0.0))
            
            ax.plot(iterations, node_history, 
                   label=node_id, color=colors[i], 
                   marker='o', markersize=4, linewidth=2)
        
        # Общая активация системы
        total_activations = []
        for step_data in result.activation_history:
            total_activations.append(sum(step_data.values()))
        
        ax2 = ax.twinx()
        ax2.plot(iterations, total_activations, 
                'k--', alpha=0.7, linewidth=2, label='Общая активация')
        ax2.set_ylabel('Общая активация', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        
        # Настройка основного графика
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Уровень активации')
        ax.set_title('История активации узлов')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _add_visualization_info(self, fig, result: PropagationResult) -> None:
        """Добавление информационного блока к визуализации."""
        
        info_text = [
            f"Статистика распространения:",
            f"• Активированных узлов: {len(result.activated_nodes)}",
            f"• Итераций: {result.iterations_used}",
            f"• Сходимость: {'Да' if result.convergence_achieved else 'Нет'}",
            f"• Время обработки: {result.processing_time:.3f}с",
            f"• Макс. активация: {result.max_activation_reached:.3f}",
            f"• Общая активация: {result.total_activation:.3f}"
        ]
        
        fig.text(0.02, 0.98, '\n'.join(info_text), 
                transform=fig.transFigure, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    def create_activation_heatmap(self,
                                activations: Dict[str, float],
                                graph,
                                save_path: Optional[str] = None,
                                **kwargs) -> None:
        """Создание тепловой карты активации."""
        
        figsize = kwargs.get('figsize', (10, 8))
        layout_type = kwargs.get('layout', 'spring')
        show_colorbar = kwargs.get('show_colorbar', True)
        
        # Создание NetworkX графа
        nx_graph = nx.DiGraph()
        
        # Добавляем узлы с активациями
        for node_id, activation_level in activations.items():
            if graph.has_node(node_id):
                node_data = graph.get_node(node_id) or {}
                nx_graph.add_node(node_id, activation=activation_level, **node_data)
        
        # Добавляем ребра между узлами
        for node_id in activations.keys():
            if graph.has_node(node_id):
                neighbors = graph.get_neighbors(node_id)
                for neighbor_id in neighbors:
                    if neighbor_id in activations:
                        edge_data = graph.get_edge(node_id, neighbor_id) or {}
                        nx_graph.add_edge(node_id, neighbor_id, **edge_data)
        
        if not nx_graph.nodes():
            self.logger.warning("Нет узлов для создания тепловой карты")
            return
        
        # Создание layout
        pos = self._create_layout(nx_graph, layout_type)
        
        # Создание фигуры
        fig, ax = plt.subplots(figsize=figsize)
        
        # Подготовка данных для визуализации
        node_colors = [activations.get(node_id, 0.0) for node_id in nx_graph.nodes()]
        node_sizes = [300 + 700 * activations.get(node_id, 0.0) for node_id in nx_graph.nodes()]
        
        # Отрисовка ребер
        nx.draw_networkx_edges(
            nx_graph, pos, ax=ax,
            edge_color='gray',
            width=1.0,
            alpha=0.5
        )
        
        # Отрисовка узлов
        nodes = nx.draw_networkx_nodes(
            nx_graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=self.activation_colormap,
            vmin=0, vmax=1,
            alpha=0.8
        )
        
        # Подписи для высокоактивных узлов
        high_activation_labels = {
            node_id: f"{node_id}\n{activations[node_id]:.2f}"
            for node_id in nx_graph.nodes()
            if activations.get(node_id, 0.0) > 0.5
        }
        
        if high_activation_labels:
            nx.draw_networkx_labels(
                nx_graph, pos, high_activation_labels, ax=ax,
                font_size=8, font_weight='bold'
            )
        
        # Настройка визуализации
        ax.set_title("Тепловая карта активации узлов")
        ax.axis('off')
        
        # Добавление colorbar
        if show_colorbar and nodes is not None:
            cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Уровень активации')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Тепловая карта сохранена: {save_path}")
        
        plt.show()
    
    def create_propagation_flow_diagram(self,
                                      result: PropagationResult,
                                      graph,
                                      save_path: Optional[str] = None,
                                      max_depth: int = 3) -> None:
        """Создание диаграммы потоков распространения."""
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Группировка узлов по глубине распространения
        depth_groups = {}
        for node_id, activation in result.activated_nodes.items():
            depth = activation.propagation_depth
            if depth <= max_depth:
                if depth not in depth_groups:
                    depth_groups[depth] = []
                depth_groups[depth].append((node_id, activation))
        
        # Создание позиций для узлов по уровням
        pos = {}
        colors = []
        sizes = []
        labels = {}
        
        for depth, nodes_at_depth in depth_groups.items():
            y_positions = np.linspace(-1, 1, len(nodes_at_depth))
            x_position = depth * 2
            
            for i, (node_id, activation) in enumerate(nodes_at_depth):
                pos[node_id] = (x_position, y_positions[i])
                colors.append(activation.activation_level)
                sizes.append(300 + 500 * activation.activation_level)
                
                if activation.activation_level > 0.3:
                    labels[node_id] = f"{node_id}\n{activation.activation_level:.2f}"
        
        # Создание NetworkX графа для отрисовки связей
        nx_graph = nx.DiGraph()
        for node_id in pos.keys():
            nx_graph.add_node(node_id)
        
        # Добавление ребер между узлами разных уровней
        for source_id, source_activation in result.activated_nodes.items():
            if source_id in pos:
                for target_id in source_activation.source_nodes:
                    if target_id in pos and source_id != target_id:
                        nx_graph.add_edge(target_id, source_id)
        
        # Отрисовка ребер с направлением
        nx.draw_networkx_edges(
            nx_graph, pos, ax=ax,
            edge_color='gray',
            width=2.0,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->'
        )
        
        # Отрисовка узлов
        nodes = nx.draw_networkx_nodes(
            nx_graph, pos, ax=ax,
            node_color=colors,
            node_size=sizes,
            cmap=self.activation_colormap,
            vmin=0, vmax=1,
            alpha=0.8
        )
        
        # Подписи узлов
        if labels:
            nx.draw_networkx_labels(
                nx_graph, pos, labels, ax=ax,
                font_size=8, font_weight='bold'
            )
        
        # Подписи уровней глубины
        for depth in depth_groups.keys():
            ax.text(depth * 2, -1.5, f"Глубина {depth}", 
                   ha='center', va='top', fontsize=12, fontweight='bold')
        
        # Настройка визуализации
        ax.set_title("Диаграмма потоков распространения активации")
        ax.set_xlabel("Глубина распространения")
        ax.axis('equal')
        
        # Добавление colorbar
        if nodes is not None:
            cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Уровень активации')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Диаграмма потоков сохранена: {save_path}")
        
        plt.show()
    
    def create_convergence_plot(self,
                              result: PropagationResult,
                              save_path: Optional[str] = None) -> None:
        """Создание графика сходимости."""
        
        if len(result.activation_history) < 2:
            self.logger.warning("Недостаточно данных для графика сходимости")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        iterations = list(range(1, len(result.activation_history) + 1))
        
        # График изменения общей активации
        total_activations = [sum(step.values()) for step in result.activation_history]
        ax1.plot(iterations, total_activations, 'b-', linewidth=2, marker='o')
        ax1.set_ylabel('Общая активация')
        ax1.set_title('Сходимость: Общая активация системы')
        ax1.grid(True, alpha=0.3)
        
        # График изменений между итерациями
        changes = []
        for i in range(1, len(result.activation_history)):
            prev_step = result.activation_history[i-1]
            curr_step = result.activation_history[i]
            
            # Вычисляем среднее изменение
            all_nodes = set(prev_step.keys()) | set(curr_step.keys())
            total_change = sum(
                abs(curr_step.get(node, 0) - prev_step.get(node, 0))
                for node in all_nodes
            )
            avg_change = total_change / len(all_nodes) if all_nodes else 0
            changes.append(avg_change)
        
        if changes:
            ax2.plot(iterations[1:], changes, 'r-', linewidth=2, marker='s')
            ax2.set_ylabel('Среднее изменение')
            ax2.set_xlabel('Итерация')
            ax2.set_title('Сходимость: Изменения между итерациями')
            ax2.grid(True, alpha=0.3)
            
            # Добавляем линию порога сходимости, если результат содержит эту информацию
            if hasattr(result, 'convergence_threshold') and result.convergence_threshold:
                ax2.axhline(y=result.convergence_threshold, color='g', 
                           linestyle='--', label='Порог сходимости')
                ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"График сходимости сохранен: {save_path}")
        
        plt.show()
    
    def export_visualization_data(self,
                                result: PropagationResult,
                                graph,
                                export_path: str) -> None:
        """Экспорт данных визуализации в JSON формат."""
        
        # Подготовка данных для экспорта
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_nodes": len(result.activated_nodes),
                "iterations": result.iterations_used,
                "convergence_achieved": result.convergence_achieved,
                "processing_time": result.processing_time
            },
            "nodes": {},
            "edges": [],
            "activation_history": result.activation_history,
            "initial_nodes": list(result.initial_nodes)
        }
        
        # Экспорт информации об узлах
        for node_id, activation in result.activated_nodes.items():
            node_data = graph.get_node(node_id) if graph.has_node(node_id) else {}
            
            export_data["nodes"][node_id] = {
                "activation_level": activation.activation_level,
                "previous_activation": activation.previous_activation,
                "propagation_depth": activation.propagation_depth,
                "source_nodes": list(activation.source_nodes),
                "metadata": activation.metadata,
                "node_attributes": node_data
            }
        
        # Экспорт информации о рёбрах
        for node_id in result.activated_nodes.keys():
            if graph.has_node(node_id):
                neighbors = graph.get_neighbors(node_id)
                for neighbor_id in neighbors:
                    if neighbor_id in result.activated_nodes:
                        edge_data = graph.get_edge(node_id, neighbor_id) or {}
                        export_data["edges"].append({
                            "source": node_id,
                            "target": neighbor_id,
                            "weight": edge_data.get("weight", 1.0),
                            "edge_attributes": edge_data
                        })
        
        # Сохранение в файл
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Данные визуализации экспортированы: {export_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка экспорта данных визуализации: {e}")
    
    def create_multi_step_comparison(self,
                                   results: List[PropagationResult],
                                   labels: List[str],
                                   save_path: Optional[str] = None) -> None:
        """Создание сравнительной визуализации нескольких результатов."""
        
        if len(results) != len(labels):
            raise ValueError("Количество результатов должно совпадать с количеством меток")
        
        fig, axes = plt.subplots(2, len(results), figsize=(5 * len(results), 10))
        
        if len(results) == 1:
            axes = axes.reshape(2, 1)
        
        for i, (result, label) in enumerate(zip(results, labels)):
            # График общей активации
            if result.activation_history:
                iterations = range(1, len(result.activation_history) + 1)
                total_activations = [sum(step.values()) for step in result.activation_history]
                
                axes[0, i].plot(iterations, total_activations, 'b-', linewidth=2, marker='o')
                axes[0, i].set_title(f'{label}\nОбщая активация')
                axes[0, i].set_ylabel('Активация')
                axes[0, i].grid(True, alpha=0.3)
            
            # Столбчатая диаграмма топ узлов
            if result.activated_nodes:
                top_nodes = sorted(
                    result.activated_nodes.items(),
                    key=lambda x: x[1].activation_level,
                    reverse=True
                )[:10]
                
                node_names = [node_id[:8] + '...' if len(node_id) > 8 else node_id 
                             for node_id, _ in top_nodes]
                activations = [activation.activation_level for _, activation in top_nodes]
                
                bars = axes[1, i].bar(range(len(node_names)), activations, 
                                     color=plt.cm.viridis(np.array(activations)))
                axes[1, i].set_title(f'{label}\nТоп-{len(top_nodes)} узлов')
                axes[1, i].set_ylabel('Уровень активации')
                axes[1, i].set_xticks(range(len(node_names)))
                axes[1, i].set_xticklabels(node_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Сравнительная визуализация сохранена: {save_path}")
        
        plt.show()
    
    def create_activation_distribution_plot(self,
                                          result: PropagationResult,
                                          save_path: Optional[str] = None) -> None:
        """Создание графика распределения активации."""
        
        if not result.activated_nodes:
            self.logger.warning("Нет данных об активации для построения распределения")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Гистограмма уровней активации
        activation_levels = [act.activation_level for act in result.activated_nodes.values()]
        
        ax1.hist(activation_levels, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Уровень активации')
        ax1.set_ylabel('Количество узлов')
        ax1.set_title('Распределение уровней активации')
        ax1.grid(True, alpha=0.3)
        
        # Добавляем статистики
        mean_activation = np.mean(activation_levels)
        median_activation = np.median(activation_levels)
        ax1.axvline(mean_activation, color='red', linestyle='--', 
                   label=f'Среднее: {mean_activation:.3f}')
        ax1.axvline(median_activation, color='green', linestyle='--', 
                   label=f'Медиана: {median_activation:.3f}')
        ax1.legend()
        
        # Распределение по глубине распространения
        depths = [act.propagation_depth for act in result.activated_nodes.values()]
        depth_counts = {}
        for depth in depths:
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        sorted_depths = sorted(depth_counts.items())
        depth_labels, depth_values = zip(*sorted_depths) if sorted_depths else ([], [])
        
        ax2.bar(depth_labels, depth_values, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Глубина распространения')
        ax2.set_ylabel('Количество узлов')
        ax2.set_title('Распределение по глубине распространения')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"График распределения сохранен: {save_path}")
        
        plt.show()