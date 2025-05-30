# neurograph/propagation/engine.py
"""
Основной движок распространения активации.
"""

import time
import math
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, deque
from datetime import datetime

from neurograph.core.logging import get_logger
from neurograph.core.events import publish
from neurograph.propagation.base import (
    IPropagationEngine, PropagationConfig, PropagationResult, NodeActivation,
    PropagationMode, GraphNotSetError, InvalidConfigurationError,
    ConvergenceError, NodeNotFoundError
)
from neurograph.propagation.functions import (
    ActivationFunctionFactory, DecayFunctionFactory, LateralInhibitionProcessor,
    normalize_activations, compute_convergence_metric, compute_activation_entropy
)


class SpreadingActivationEngine(IPropagationEngine):
    """Основной движок распространения активации по графу знаний."""
    
    def __init__(self, graph=None):
        self.graph = graph
        self.logger = get_logger("spreading_activation")
        
        # Текущее состояние активаций
        self.current_activations: Dict[str, NodeActivation] = {}
        
        # Компоненты движка
        self.lateral_inhibition = LateralInhibitionProcessor()
        
        # Статистика
        self.statistics = {
            "total_propagations": 0,
            "successful_propagations": 0,
            "total_processing_time": 0.0,
            "average_iterations": 0.0,
            "convergence_failures": 0,
            "nodes_processed": 0,
            "max_activations_reached": 0.0
        }
        
        # Кеш для оптимизации
        self._neighbor_cache: Dict[str, List[str]] = {}
        self._distance_cache: Dict[Tuple[str, str], int] = {}
        
    def set_graph(self, graph) -> None:
        """Установка графа знаний."""
        self.graph = graph
        self.reset_activations()
        self._clear_caches()
        self.logger.info("Граф установлен для распространения активации")
    
    def reset_activations(self) -> None:
        """Сброс всех активаций."""
        self.current_activations.clear()
        self.logger.debug("Активации сброшены")
    
    def get_node_activation(self, node_id: str) -> Optional[NodeActivation]:
        """Получение текущей активации узла."""
        return self.current_activations.get(node_id)
    
    def update_node_activation(self, node_id: str, activation_level: float) -> bool:
        """Обновление активации узла."""
        if not self.graph or not self.graph.has_node(node_id):
            return False
        
        if node_id not in self.current_activations:
            self.current_activations[node_id] = NodeActivation(node_id=node_id)
        
        self.current_activations[node_id].update_activation(activation_level)
        return True
    
    def propagate(self, 
                 initial_nodes: Dict[str, float], 
                 config: PropagationConfig) -> PropagationResult:
        """Основной метод распространения активации."""
        
        start_time = time.time()
        
        try:
            # Валидация
            self._validate_input(initial_nodes, config)
            
            # Инициализация
            result = self._initialize_propagation(initial_nodes, config)
            
            # Выполнение распространения
            if config.propagation_mode == PropagationMode.SPREADING:
                result = self._spreading_propagation(result, config)
            elif config.propagation_mode == PropagationMode.FOCUSING:
                result = self._focusing_propagation(result, config)
            elif config.propagation_mode == PropagationMode.BIDIRECTIONAL:
                result = self._bidirectional_propagation(result, config)
            elif config.propagation_mode == PropagationMode.CONSTRAINED:
                result = self._constrained_propagation(result, config)
            else:
                raise InvalidConfigurationError(f"Неизвестный режим распространения: {config.propagation_mode}")
            
            # Финализация
            result.processing_time = time.time() - start_time
            result.success = True
            
            # Обновление статистики
            self._update_statistics(result)
            
            # Публикация события
            publish("propagation.completed", {
                "initial_nodes_count": len(initial_nodes),
                "activated_nodes_count": len(result.activated_nodes),
                "iterations_used": result.iterations_used,
                "convergence_achieved": result.convergence_achieved,
                "processing_time": result.processing_time,
                "max_activation": result.max_activation_reached
            })
            
            self.logger.info(
                f"Распространение завершено: {len(result.activated_nodes)} узлов активировано "
                f"за {result.iterations_used} итераций ({result.processing_time:.3f}с)"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.statistics["convergence_failures"] += 1
            
            error_result = PropagationResult(
                success=False,
                activated_nodes={},
                processing_time=processing_time,
                error_message=str(e)
            )
            
            publish("propagation.error", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "initial_nodes_count": len(initial_nodes),
                "processing_time": processing_time
            })
            
            self.logger.error(f"Ошибка распространения: {e}")
            return error_result
    
    def _validate_input(self, initial_nodes: Dict[str, float], config: PropagationConfig) -> None:
        """Валидация входных данных."""
        
        if not self.graph:
            raise GraphNotSetError()
        
        if not initial_nodes:
            raise InvalidConfigurationError("Не указаны начальные узлы для активации")
        
        # Проверка существования узлов в графе
        for node_id in initial_nodes.keys():
            if not self.graph.has_node(node_id):
                raise NodeNotFoundError(node_id)
        
        # Валидация конфигурации
        is_valid, errors = config.validate()
        if not is_valid:
            raise InvalidConfigurationError("Некорректная конфигурация", errors)
        
        # Проверка уровней активации
        for node_id, activation in initial_nodes.items():
            if not 0.0 <= activation <= 1.0:
                raise InvalidConfigurationError(
                    f"Некорректный уровень активации для узла {node_id}: {activation}"
                )
    
    def _initialize_propagation(self, 
                               initial_nodes: Dict[str, float], 
                               config: PropagationConfig) -> PropagationResult:
        """Инициализация процесса распространения."""
        
        # Создаем результат
        result = PropagationResult(
            success=False,
            activated_nodes={},
            initial_nodes=set(initial_nodes.keys())
        )
        
        # Сброс текущих активаций
        self.reset_activations()
        
        # Инициализация начальных узлов
        for node_id, activation_level in initial_nodes.items():
            node_activation = NodeActivation(
                node_id=node_id,
                activation_level=activation_level,
                propagation_depth=0,
                metadata={"initial_node": True}
            )
            
            self.current_activations[node_id] = node_activation
            result.activated_nodes[node_id] = node_activation
        
        # Запись начального состояния в историю
        initial_state = {node_id: act.activation_level 
                        for node_id, act in result.activated_nodes.items()}
        result.activation_history.append(initial_state)
        
        return result
    
    def _spreading_propagation(self, 
                              result: PropagationResult, 
                              config: PropagationConfig) -> PropagationResult:
        """Расходящееся распространение активации."""
        
        activation_func = ActivationFunctionFactory.create(config.activation_function)
        decay_func = DecayFunctionFactory.create(config.decay_function)
        
        for iteration in range(config.max_iterations):
            iteration_start = time.time()
            
            # Создаем копию текущих активаций для обновления
            new_activations = {}
            for node_id, activation in result.activated_nodes.items():
                new_activations[node_id] = NodeActivation(
                    node_id=activation.node_id,
                    activation_level=activation.activation_level,
                    previous_activation=activation.previous_activation,
                    activation_time=activation.activation_time,
                    source_nodes=activation.source_nodes.copy(),
                    propagation_depth=activation.propagation_depth,
                    metadata=activation.metadata.copy()
                )
            
            # Распространение от каждого активного узла
            nodes_to_process = [
                (node_id, activation) 
                for node_id, activation in result.activated_nodes.items()
                if activation.activation_level >= config.activation_threshold
            ]
            
            for source_node_id, source_activation in nodes_to_process:
                self._propagate_from_node(
                    source_node_id, source_activation, new_activations,
                    activation_func, config
                )
            
            # Применение затухания
            self._apply_decay(new_activations, decay_func, config)
            
            # Применение латерального торможения
            if config.lateral_inhibition:
                new_activations = self.lateral_inhibition.apply_inhibition(
                    new_activations, self.graph, config
                )
            
            # Фильтрация узлов по порогу активации
            filtered_activations = {
                node_id: activation
                for node_id, activation in new_activations.items()
                if activation.activation_level >= config.activation_threshold
            }
            
            # Ограничение количества активных узлов
            if len(filtered_activations) > config.max_active_nodes:
                sorted_activations = sorted(
                    filtered_activations.items(),
                    key=lambda x: x[1].activation_level,
                    reverse=True
                )
                filtered_activations = dict(sorted_activations[:config.max_active_nodes])
            
            # Обновление результата
            result.activated_nodes = filtered_activations
            result.iterations_used = iteration + 1
            
            # Запись истории
            current_state = {node_id: act.activation_level 
                           for node_id, act in filtered_activations.items()}
            result.activation_history.append(current_state)
            
            # Вычисление метрик
            max_activation = max(
                (act.activation_level for act in filtered_activations.values()),
                default=0.0
            )
            result.max_activation_reached = max(result.max_activation_reached, max_activation)
            result.total_activation = sum(
                act.activation_level for act in filtered_activations.values()
            )
            
            # Проверка сходимости
            if len(result.activation_history) >= 2:
                convergence = compute_convergence_metric(result.activation_history, window_size=3)
                if convergence <= config.convergence_threshold:
                    result.convergence_achieved = True
                    self.logger.debug(f"Сходимость достигнута на итерации {iteration + 1}")
                    break
            
            # Проверка на отсутствие активных узлов
            if not filtered_activations:
                self.logger.debug(f"Нет активных узлов на итерации {iteration + 1}")
                break
            
            # Публикация промежуточного события
            if iteration % 10 == 0:  # Каждые 10 итераций
                publish("propagation.iteration", {
                    "iteration": iteration + 1,
                    "active_nodes_count": len(filtered_activations),
                    "max_activation": max_activation,
                    "convergence_metric": convergence if 'convergence' in locals() else None
                })
        
        return result
    
    def _propagate_from_node(self, 
                           source_node_id: str,
                           source_activation: NodeActivation,
                           target_activations: Dict[str, NodeActivation],
                           activation_func,
                           config: PropagationConfig) -> None:
        """Распространение активации от одного узла к соседям."""
        
        # Получаем соседей узла
        neighbors = self._get_neighbors_cached(source_node_id, config)
        
        for neighbor_id, edge_weight in neighbors:
            # Проверка глубины распространения
            if source_activation.propagation_depth >= config.max_propagation_depth:
                continue
            
            # Вычисление входного сигнала для соседа
            input_signal = source_activation.activation_level * edge_weight
            
            # Применение функции активации
            if config.activation_function == config.activation_function.SOFTMAX:
                # Для softmax нужны все соседи
                all_neighbors_signals = [
                    target_activations.get(n_id, NodeActivation(n_id)).activation_level
                    for n_id, _ in neighbors
                ]
                activation_params = dict(config.activation_params)
                activation_params["all_values"] = all_neighbors_signals
            else:
                activation_params = config.activation_params
            
            new_activation_level = activation_func.compute(input_signal, **activation_params)
            
            # Создание или обновление активации соседа
            if neighbor_id not in target_activations:
                target_activations[neighbor_id] = NodeActivation(
                    node_id=neighbor_id,
                    propagation_depth=source_activation.propagation_depth + 1
                )
            
            neighbor_activation = target_activations[neighbor_id]
            
            # Комбинирование активаций (аддитивная модель)
            combined_activation = min(1.0, 
                neighbor_activation.activation_level + new_activation_level * 0.5
            )
            
            neighbor_activation.update_activation(combined_activation, source_node_id)
            neighbor_activation.propagation_depth = min(
                neighbor_activation.propagation_depth,
                source_activation.propagation_depth + 1
            )
    
    def _get_neighbors_cached(self, node_id: str, config: PropagationConfig) -> List[Tuple[str, float]]:
        """Получение соседей узла с кешированием."""
        
        cache_key = f"{node_id}_{hash(tuple(config.edge_type_filters))}"
        
        if cache_key not in self._neighbor_cache:
            neighbors = []
            
            # Получаем всех соседей
            all_neighbors = self.graph.get_neighbors(node_id)
            
            for neighbor_id in all_neighbors:
                # Применяем фильтры типов узлов
                if config.node_type_filters:
                    neighbor_node = self.graph.get_node(neighbor_id)
                    if neighbor_node:
                        node_type = neighbor_node.get('type', '')
                        if node_type not in config.node_type_filters:
                            continue
                
                # Получаем информацию о ребре
                edge_data = self.graph.get_edge(node_id, neighbor_id)
                
                # Применяем фильтры типов ребер
                if config.edge_type_filters:
                    # Проверяем все возможные типы ребер между узлами
                    valid_edge_found = False
                    for edge_type in config.edge_type_filters:
                        specific_edge = self.graph.get_edge(node_id, neighbor_id, edge_type)
                        if specific_edge:
                            edge_data = specific_edge
                            valid_edge_found = True
                            break
                    
                    if not valid_edge_found:
                        continue
                
                # Определяем вес ребра
                if edge_data:
                    edge_weight = edge_data.get('weight', config.default_weight)
                else:
                    edge_weight = config.default_weight
                
                neighbors.append((neighbor_id, edge_weight))
            
            # Нормализация весов, если включена
            if config.weight_normalization and neighbors:
                total_weight = sum(weight for _, weight in neighbors)
                if total_weight > 0:
                    neighbors = [
                        (neighbor_id, weight / total_weight)
                        for neighbor_id, weight in neighbors
                    ]
            
            self._neighbor_cache[cache_key] = neighbors
        
        return self._neighbor_cache[cache_key]
    
    def _apply_decay(self, 
                    activations: Dict[str, NodeActivation],
                    decay_func,
                    config: PropagationConfig) -> None:
        """Применение затухания к активациям."""
        
        for activation in activations.values():
            current_level = activation.activation_level
            
            # Применяем затухание
            new_level = decay_func.compute(
                current_level, 
                config.time_step_duration,
                **config.decay_params
            )
            
            activation.update_activation(new_level)
    
    def _focusing_propagation(self, 
                            result: PropagationResult, 
                            config: PropagationConfig) -> PropagationResult:
        """Сходящееся распространение активации (активация движется к центральным узлам)."""
        
        # Аналогично spreading, но инвертируем направление распространения
        # и используем входящие связи вместо исходящих
        
        activation_func = ActivationFunctionFactory.create(config.activation_function)
        decay_func = DecayFunctionFactory.create(config.decay_function)
        
        for iteration in range(config.max_iterations):
            # Копируем текущие активации
            new_activations = {
                node_id: NodeActivation(
                    node_id=activation.node_id,
                    activation_level=activation.activation_level,
                    previous_activation=activation.previous_activation,
                    activation_time=activation.activation_time,
                    source_nodes=activation.source_nodes.copy(),
                    propagation_depth=activation.propagation_depth,
                    metadata=activation.metadata.copy()
                )
                for node_id, activation in result.activated_nodes.items()
            }
            
            # Распространение к узлам-родителям (обратное направление)
            for target_node_id, target_activation in result.activated_nodes.items():
                if target_activation.activation_level < config.activation_threshold:
                    continue
                
                # Находим узлы, которые указывают на текущий
                incoming_nodes = self._get_incoming_neighbors(target_node_id, config)
                
                for source_node_id, edge_weight in incoming_nodes:
                    input_signal = target_activation.activation_level * edge_weight
                    new_activation_level = activation_func.compute(input_signal, **config.activation_params)
                    
                    if source_node_id not in new_activations:
                        new_activations[source_node_id] = NodeActivation(
                            node_id=source_node_id,
                            propagation_depth=target_activation.propagation_depth + 1
                        )
                    
                    source_activation = new_activations[source_node_id]
                    combined_activation = min(1.0, 
                        source_activation.activation_level + new_activation_level * 0.5
                    )
                    source_activation.update_activation(combined_activation, target_node_id)
            
            # Применение затухания и фильтрация
            self._apply_decay(new_activations, decay_func, config)
            
            # Фильтрация и ограничение
            filtered_activations = {
                node_id: activation
                for node_id, activation in new_activations.items()
                if activation.activation_level >= config.activation_threshold
            }
            
            if len(filtered_activations) > config.max_active_nodes:
                sorted_activations = sorted(
                    filtered_activations.items(),
                    key=lambda x: x[1].activation_level,
                    reverse=True
                )
                filtered_activations = dict(sorted_activations[:config.max_active_nodes])
            
            # Обновление результата
            result.activated_nodes = filtered_activations
            result.iterations_used = iteration + 1
            
            # Запись истории и проверка сходимости
            current_state = {node_id: act.activation_level 
                           for node_id, act in filtered_activations.items()}
            result.activation_history.append(current_state)
            
            if len(result.activation_history) >= 2:
                convergence = compute_convergence_metric(result.activation_history, window_size=3)
                if convergence <= config.convergence_threshold:
                    result.convergence_achieved = True
                    break
            
            if not filtered_activations:
                break
        
        return result
    
    def _bidirectional_propagation(self, 
                                 result: PropagationResult, 
                                 config: PropagationConfig) -> PropagationResult:
        """Двунаправленное распространение активации."""
        
        # Комбинируем расходящееся и сходящееся распространение
        
        # Сначала выполняем расходящееся распространение
        spreading_result = self._spreading_propagation(result, config)
        
        # Затем применяем сходящееся к полученным активациям
        focusing_result = self._focusing_propagation(spreading_result, config)
        
        # Комбинируем результаты
        combined_activations = {}
        
        # Объединяем активации из обоих режимов
        all_nodes = set(spreading_result.activated_nodes.keys()) | set(focusing_result.activated_nodes.keys())
        
        for node_id in all_nodes:
            spreading_activation = spreading_result.activated_nodes.get(node_id)
            focusing_activation = focusing_result.activated_nodes.get(node_id)
            
            if spreading_activation and focusing_activation:
                # Усредняем активации
                avg_level = (spreading_activation.activation_level + focusing_activation.activation_level) / 2.0
                combined_activation = NodeActivation(
                    node_id=node_id,
                    activation_level=avg_level,
                    propagation_depth=min(spreading_activation.propagation_depth, focusing_activation.propagation_depth),
                    metadata={"bidirectional": True}
                )
                combined_activation.source_nodes.update(spreading_activation.source_nodes)
                combined_activation.source_nodes.update(focusing_activation.source_nodes)
            elif spreading_activation:
                combined_activation = spreading_activation
            else:
                combined_activation = focusing_activation
            
            combined_activations[node_id] = combined_activation
        
        # Обновляем результат
        result.activated_nodes = combined_activations
        result.iterations_used = max(spreading_result.iterations_used, focusing_result.iterations_used)
        result.convergence_achieved = spreading_result.convergence_achieved and focusing_result.convergence_achieved
        
        return result
    
    def _constrained_propagation(self, 
                               result: PropagationResult, 
                               config: PropagationConfig) -> PropagationResult:
        """Ограниченное распространение с дополнительными условиями."""
        
        activation_func = ActivationFunctionFactory.create(config.activation_function)
        decay_func = DecayFunctionFactory.create(config.decay_function)
        
        # Дополнительные ограничения для constrained режима
        max_depth_per_branch = config.max_propagation_depth // 2
        activation_boost_threshold = 0.7  # Порог для усиления активации
        
        for iteration in range(config.max_iterations):
            new_activations = {
                node_id: NodeActivation(
                    node_id=activation.node_id,
                    activation_level=activation.activation_level,
                    previous_activation=activation.previous_activation,
                    activation_time=activation.activation_time,
                    source_nodes=activation.source_nodes.copy(),
                    propagation_depth=activation.propagation_depth,
                    metadata=activation.metadata.copy()
                )
                for node_id, activation in result.activated_nodes.items()
            }
            
            # Применяем ограниченное распространение
            for source_node_id, source_activation in result.activated_nodes.items():
                if source_activation.activation_level < config.activation_threshold:
                    continue
                
                # Ограничение по глубине
                if source_activation.propagation_depth >= max_depth_per_branch:
                    continue
                
                # Получаем соседей с дополнительной фильтрацией
                neighbors = self._get_constrained_neighbors(source_node_id, source_activation, config)
                
                for neighbor_id, edge_weight, constraint_factor in neighbors:
                    # Применяем ограничивающий фактор
                    adjusted_signal = source_activation.activation_level * edge_weight * constraint_factor
                    
                    # Усиление для высокоактивированных узлов
                    if source_activation.activation_level >= activation_boost_threshold:
                        adjusted_signal *= 1.5
                    
                    new_activation_level = activation_func.compute(adjusted_signal, **config.activation_params)
                    
                    if neighbor_id not in new_activations:
                        new_activations[neighbor_id] = NodeActivation(
                            node_id=neighbor_id,
                            propagation_depth=source_activation.propagation_depth + 1,
                            metadata={"constrained": True}
                        )
                    
                    neighbor_activation = new_activations[neighbor_id]
                    combined_activation = min(1.0, 
                        neighbor_activation.activation_level + new_activation_level * 0.3  # Меньший вклад
                    )
                    neighbor_activation.update_activation(combined_activation, source_node_id)
            
            # Применение затухания и специальных ограничений
            self._apply_decay(new_activations, decay_func, config)
            self._apply_constrained_filtering(new_activations, config)
            
            # Фильтрация по порогу с повышенными требованиями
            higher_threshold = config.activation_threshold * 1.5
            filtered_activations = {
                node_id: activation
                for node_id, activation in new_activations.items()
                if activation.activation_level >= higher_threshold
            }
            
            # Более строгое ограничение количества узлов
            max_constrained_nodes = min(config.max_active_nodes // 2, 50)
            if len(filtered_activations) > max_constrained_nodes:
                sorted_activations = sorted(
                    filtered_activations.items(),
                    key=lambda x: x[1].activation_level,
                    reverse=True
                )
                filtered_activations = dict(sorted_activations[:max_constrained_nodes])
            
            result.activated_nodes = filtered_activations
            result.iterations_used = iteration + 1
            
            # Запись истории и проверка сходимости
            current_state = {node_id: act.activation_level 
                           for node_id, act in filtered_activations.items()}
            result.activation_history.append(current_state)
            
            if len(result.activation_history) >= 2:
                convergence = compute_convergence_metric(result.activation_history, window_size=2)
                if convergence <= config.convergence_threshold * 0.5:  # Более строгая сходимость
                    result.convergence_achieved = True
                    break
            
            if not filtered_activations:
                break
        
        return result
    
    def _get_incoming_neighbors(self, node_id: str, config: PropagationConfig) -> List[Tuple[str, float]]:
        """Получение узлов, которые указывают на данный узел."""
        
        incoming_neighbors = []
        
        # Перебираем все узлы графа для поиска входящих связей
        all_nodes = self.graph.get_all_nodes()
        
        for potential_source in all_nodes:
            if potential_source == node_id:
                continue
            
            # Проверяем, есть ли ребро от potential_source к node_id
            if self.graph.has_edge(potential_source, node_id):
                edge_data = self.graph.get_edge(potential_source, node_id)
                edge_weight = edge_data.get('weight', config.default_weight) if edge_data else config.default_weight
                incoming_neighbors.append((potential_source, edge_weight))
        
        return incoming_neighbors
    
    def _get_constrained_neighbors(self, 
                                 node_id: str, 
                                 source_activation: NodeActivation, 
                                 config: PropagationConfig) -> List[Tuple[str, float, float]]:
        """Получение соседей с дополнительными ограничениями."""
        
        base_neighbors = self._get_neighbors_cached(node_id, config)
        constrained_neighbors = []
        
        for neighbor_id, edge_weight in base_neighbors:
            # Вычисляем ограничивающий фактор на основе различных критериев
            constraint_factor = 1.0
            
            # Фактор на основе типа узла
            neighbor_node = self.graph.get_node(neighbor_id)
            if neighbor_node:
                node_type = neighbor_node.get('type', '')
                if node_type in ['peripheral', 'noise']:
                    constraint_factor *= 0.5
                elif node_type in ['central', 'important']:
                    constraint_factor *= 1.5
            
            # Фактор на основе глубины распространения
            depth_factor = 1.0 / (source_activation.propagation_depth + 1)
            constraint_factor *= depth_factor
            
            # Фактор на основе предыдущей активации
            if neighbor_id in self.current_activations:
                prev_activation = self.current_activations[neighbor_id].activation_level
                if prev_activation > 0.8:  # Уже сильно активирован
                    constraint_factor *= 0.3
                elif prev_activation > 0.5:
                    constraint_factor *= 0.7
            
            constrained_neighbors.append((neighbor_id, edge_weight, constraint_factor))
        
        return constrained_neighbors
    
    def _apply_constrained_filtering(self, 
                                   activations: Dict[str, NodeActivation],
                                   config: PropagationConfig) -> None:
        """Применение дополнительной фильтрации для constrained режима."""
        
        # Подавление слабых активаций
        weak_threshold = config.activation_threshold * 0.5
        nodes_to_suppress = []
        
        for node_id, activation in activations.items():
            # Подавляем узлы с очень низкой активацией
            if activation.activation_level < weak_threshold:
                nodes_to_suppress.append(node_id)
                continue
            
            # Подавляем узлы, которые активированы только из одного источника
            if len(activation.source_nodes) == 1 and activation.activation_level < config.activation_threshold * 0.8:
                activation.activation_level *= 0.5
            
            # Усиливаем узлы с множественными источниками активации
            if len(activation.source_nodes) >= 3:
                boost_factor = min(1.5, 1.0 + len(activation.source_nodes) * 0.1)
                activation.activation_level = min(1.0, activation.activation_level * boost_factor)
        
        # Удаляем подавленные узлы
        for node_id in nodes_to_suppress:
            if node_id in activations:
                del activations[node_id]
    
    def _update_statistics(self, result: PropagationResult) -> None:
        """Обновление статистики работы движка."""
        
        self.statistics["total_propagations"] += 1
        
        if result.success:
            self.statistics["successful_propagations"] += 1
        
        self.statistics["total_processing_time"] += result.processing_time
        
        # Обновление средних значений
        total_successful = self.statistics["successful_propagations"]
        if total_successful > 0:
            self.statistics["average_iterations"] = (
                (self.statistics["average_iterations"] * (total_successful - 1) + result.iterations_used) / total_successful
            )
        
        self.statistics["nodes_processed"] += len(result.activated_nodes)
        self.statistics["max_activations_reached"] = max(
            self.statistics["max_activations_reached"],
            result.max_activation_reached
        )
    
    def _clear_caches(self) -> None:
        """Очистка кешей."""
        self._neighbor_cache.clear()
        self._distance_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики работы движка."""
        
        stats = self.statistics.copy()
        
        # Дополнительные вычисляемые метрики
        if stats["total_propagations"] > 0:
            stats["success_rate"] = stats["successful_propagations"] / stats["total_propagations"]
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_propagations"]
        else:
            stats["success_rate"] = 0.0
            stats["average_processing_time"] = 0.0
        
        stats["current_active_nodes"] = len(self.current_activations)
        stats["cache_size"] = len(self._neighbor_cache)
        
        return stats
    
    def get_activation_summary(self) -> Dict[str, Any]:
        """Получение сводки по текущим активациям."""
        
        if not self.current_activations:
            return {
                "total_nodes": 0,
                "active_nodes": 0,
                "max_activation": 0.0,
                "min_activation": 0.0,
                "avg_activation": 0.0,
                "activation_entropy": 0.0
            }
        
        activations_values = [act.activation_level for act in self.current_activations.values()]
        activation_dict = {node_id: act.activation_level for node_id, act in self.current_activations.items()}
        
        return {
            "total_nodes": len(self.current_activations),
            "active_nodes": sum(1 for val in activations_values if val >= 0.1),
            "max_activation": max(activations_values),
            "min_activation": min(activations_values),
            "avg_activation": sum(activations_values) / len(activations_values),
            "activation_entropy": compute_activation_entropy(activation_dict),
            "most_active_nodes": sorted(
                [(node_id, act.activation_level) for node_id, act in self.current_activations.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def export_activation_state(self) -> Dict[str, Any]:
        """Экспорт текущего состояния активаций."""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "activations": {
                node_id: {
                    "activation_level": act.activation_level,
                    "previous_activation": act.previous_activation,
                    "activation_time": act.activation_time.isoformat(),
                    "source_nodes": list(act.source_nodes),
                    "propagation_depth": act.propagation_depth,
                    "metadata": act.metadata
                }
                for node_id, act in self.current_activations.items()
            },
            "statistics": self.get_statistics(),
            "summary": self.get_activation_summary()
        }
    
    def import_activation_state(self, state_data: Dict[str, Any]) -> bool:
        """Импорт состояния активаций."""
        
        try:
            self.reset_activations()
            
            activations_data = state_data.get("activations", {})
            for node_id, act_data in activations_data.items():
                activation = NodeActivation(
                    node_id=node_id,
                    activation_level=act_data["activation_level"],
                    previous_activation=act_data["previous_activation"],
                    propagation_depth=act_data["propagation_depth"],
                    metadata=act_data.get("metadata", {})
                )
                
                activation.source_nodes = set(act_data.get("source_nodes", []))
                
                # Парсим время активации
                if "activation_time" in act_data:
                    try:
                        activation.activation_time = datetime.fromisoformat(act_data["activation_time"])
                    except:
                        activation.activation_time = datetime.now()
                
                self.current_activations[node_id] = activation
            
            self.logger.info(f"Импортировано состояние с {len(self.current_activations)} активациями")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка импорта состояния активаций: {e}")
            self.reset_activations()
            return False