# neurograph/propagation/functions.py
"""
Реализация функций активации, затухания и латерального торможения.
"""

import numpy as np
import math
from typing import Dict, Any, Set, List
from neurograph.propagation.base import (
    IActivationFunction, IDecayFunction, ILateralInhibition,
    ActivationFunction, DecayFunction, NodeActivation, PropagationConfig
)


class ActivationFunctions:
    """Коллекция функций активации."""
    
    class LinearActivation(IActivationFunction):
        """Линейная функция активации."""
        
        def compute(self, input_value: float, **params) -> float:
            slope = params.get("slope", 1.0)
            intercept = params.get("intercept", 0.0)
            result = slope * input_value + intercept
            return max(0.0, min(1.0, result))
        
        def derivative(self, input_value: float, **params) -> float:
            return params.get("slope", 1.0)
    
    class SigmoidActivation(IActivationFunction):
        """Сигмоидальная функция активации."""
        
        def compute(self, input_value: float, **params) -> float:
            steepness = params.get("steepness", 1.0)
            try:
                return 1.0 / (1.0 + math.exp(-steepness * input_value))
            except OverflowError:
                return 0.0 if input_value < 0 else 1.0
        
        def derivative(self, input_value: float, **params) -> float:
            steepness = params.get("steepness", 1.0)
            sigmoid_val = self.compute(input_value, **params)
            return steepness * sigmoid_val * (1.0 - sigmoid_val)
    
    class TanhActivation(IActivationFunction):
        """Гиперболический тангенс."""
        
        def compute(self, input_value: float, **params) -> float:
            steepness = params.get("steepness", 1.0)
            result = math.tanh(steepness * input_value)
            # Нормализуем в диапазон [0, 1]
            return (result + 1.0) / 2.0
        
        def derivative(self, input_value: float, **params) -> float:
            steepness = params.get("steepness", 1.0)
            tanh_val = math.tanh(steepness * input_value)
            return steepness * (1.0 - tanh_val * tanh_val) / 2.0
    
    class ReLUActivation(IActivationFunction):
        """Rectified Linear Unit."""
        
        def compute(self, input_value: float, **params) -> float:
            threshold = params.get("threshold", 0.0)
            max_val = params.get("max_value", 1.0)
            if input_value <= threshold:
                return 0.0
            return min(input_value - threshold, max_val)
        
        def derivative(self, input_value: float, **params) -> float:
            threshold = params.get("threshold", 0.0)
            return 1.0 if input_value > threshold else 0.0
    
    class ThresholdActivation(IActivationFunction):
        """Пороговая функция активации."""
        
        def compute(self, input_value: float, **params) -> float:
            threshold = params.get("threshold", 0.5)
            output_high = params.get("output_high", 1.0)
            output_low = params.get("output_low", 0.0)
            return output_high if input_value >= threshold else output_low
        
        def derivative(self, input_value: float, **params) -> float:
            # Производная пороговой функции = дельта-функция (аппроксимируем нулем)
            return 0.0
    
    class GaussianActivation(IActivationFunction):
        """Гауссовская функция активации."""
        
        def compute(self, input_value: float, **params) -> float:
            center = params.get("center", 0.5)
            width = params.get("width", 0.2)
            amplitude = params.get("amplitude", 1.0)
            
            exponent = -((input_value - center) ** 2) / (2 * width ** 2)
            try:
                return amplitude * math.exp(exponent)
            except OverflowError:
                return 0.0
        
        def derivative(self, input_value: float, **params) -> float:
            center = params.get("center", 0.5)
            width = params.get("width", 0.2)
            gaussian_val = self.compute(input_value, **params)
            return -gaussian_val * (input_value - center) / (width ** 2)
    
    class SoftmaxActivation(IActivationFunction):
        """Softmax функция активации (требует контекст всех узлов)."""
        
        def compute(self, input_value: float, **params) -> float:
            # Для softmax нужны все значения, используем упрощенную версию
            temperature = params.get("temperature", 1.0)
            all_values = params.get("all_values", [input_value])
            
            try:
                exp_values = [math.exp(val / temperature) for val in all_values]
                exp_sum = sum(exp_values)
                
                if exp_sum == 0:
                    return 1.0 / len(all_values)
                
                exp_input = math.exp(input_value / temperature)
                return exp_input / exp_sum
            except OverflowError:
                return 0.0
        
        def derivative(self, input_value: float, **params) -> float:
            # Упрощенная производная softmax
            softmax_val = self.compute(input_value, **params)
            return softmax_val * (1.0 - softmax_val)


class DecayFunctions:
    """Коллекция функций затухания."""
    
    class ExponentialDecay(IDecayFunction):
        """Экспоненциальное затухание."""
        
        def compute(self, current_activation: float, time_step: float, **params) -> float:
            decay_rate = params.get("rate", 0.1)
            half_life = params.get("half_life", None)
            
            if half_life is not None:
                # Вычисляем decay_rate из периода полураспада
                decay_rate = math.log(2) / half_life
            
            decay_factor = math.exp(-decay_rate * time_step)
            return current_activation * decay_factor
    
    class LinearDecay(IDecayFunction):
        """Линейное затухание."""
        
        def compute(self, current_activation: float, time_step: float, **params) -> float:
            decay_rate = params.get("rate", 0.1)
            max_decay = params.get("max_decay", current_activation)
            
            decay_amount = decay_rate * time_step
            decay_amount = min(decay_amount, max_decay)
            
            return max(0.0, current_activation - decay_amount)
    
    class LogarithmicDecay(IDecayFunction):
        """Логарифмическое затухание."""
        
        def compute(self, current_activation: float, time_step: float, **params) -> float:
            decay_rate = params.get("rate", 0.1)
            base = params.get("base", math.e)
            
            if current_activation <= 0:
                return 0.0
            
            decay_factor = 1.0 - (decay_rate * math.log(time_step + 1, base))
            decay_factor = max(0.0, decay_factor)
            
            return current_activation * decay_factor
    
    class PowerDecay(IDecayFunction):
        """Степенное затухание."""
        
        def compute(self, current_activation: float, time_step: float, **params) -> float:
            decay_rate = params.get("rate", 0.1)
            power = params.get("power", 2.0)
            
            decay_factor = 1.0 / (1.0 + decay_rate * (time_step ** power))
            return current_activation * decay_factor
    
    class StepDecay(IDecayFunction):
        """Ступенчатое затухание."""
        
        def compute(self, current_activation: float, time_step: float, **params) -> float:
            decay_rate = params.get("rate", 0.1)
            step_size = params.get("step_size", 1.0)
            
            steps = int(time_step / step_size)
            decay_factor = (1.0 - decay_rate) ** steps
            
            return current_activation * decay_factor
    
    class NoDecay(IDecayFunction):
        """Без затухания."""
        
        def compute(self, current_activation: float, time_step: float, **params) -> float:
            return current_activation


class LateralInhibitionProcessor(ILateralInhibition):
    """Процессор латерального торможения."""
    
    def apply_inhibition(self, 
                        activations: Dict[str, NodeActivation],
                        graph,
                        config: PropagationConfig) -> Dict[str, NodeActivation]:
        """Применение латерального торможения между соседними узлами."""
        
        if not config.lateral_inhibition:
            return activations
        
        inhibited_activations = {node_id: activation.__class__(
            node_id=activation.node_id,
            activation_level=activation.activation_level,
            previous_activation=activation.previous_activation,
            activation_time=activation.activation_time,
            source_nodes=activation.source_nodes.copy(),
            propagation_depth=activation.propagation_depth,
            metadata=activation.metadata.copy()
        ) for node_id, activation in activations.items()}
        
        inhibition_strength = config.inhibition_strength
        inhibition_radius = config.inhibition_radius
        
        # Для каждого активного узла применяем торможение к соседям
        for node_id, activation in activations.items():
            if activation.activation_level < config.activation_threshold:
                continue
            
            # Находим узлы в радиусе торможения
            neighbors_to_inhibit = self._find_neighbors_in_radius(
                node_id, graph, inhibition_radius
            )
            
            # Применяем торможение
            for neighbor_id in neighbors_to_inhibit:
                if neighbor_id in inhibited_activations and neighbor_id != node_id:
                    neighbor_activation = inhibited_activations[neighbor_id]
                    
                    # Вычисляем силу торможения на основе расстояния и активации
                    distance = self._get_distance(node_id, neighbor_id, graph)
                    distance_factor = 1.0 / (distance + 1.0)  # Ближе = сильнее торможение
                    
                    inhibition = (activation.activation_level * 
                                inhibition_strength * 
                                distance_factor)
                    
                    # Применяем торможение
                    new_activation = max(0.0, 
                                       neighbor_activation.activation_level - inhibition)
                    
                    neighbor_activation.update_activation(new_activation, f"inhibited_by_{node_id}")
                    
                    # Записываем информацию о торможении
                    if "inhibition_sources" not in neighbor_activation.metadata:
                        neighbor_activation.metadata["inhibition_sources"] = []
                    neighbor_activation.metadata["inhibition_sources"].append({
                        "source": node_id,
                        "inhibition_amount": inhibition,
                        "distance": distance
                    })
        
        return inhibited_activations
    
    def _find_neighbors_in_radius(self, node_id: str, graph, radius: int) -> Set[str]:
        """Поиск соседей в заданном радиусе."""
        visited = set()
        current_level = {node_id}
        all_neighbors = set()
        
        for depth in range(radius):
            next_level = set()
            
            for current_node in current_level:
                if current_node in visited:
                    continue
                visited.add(current_node)
                
                # Получаем прямых соседей
                neighbors = graph.get_neighbors(current_node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        all_neighbors.add(neighbor)
            
            current_level = next_level
            
            if not current_level:  # Нет больше соседей
                break
        
        return all_neighbors
    
    def _get_distance(self, node1: str, node2: str, graph) -> int:
        """Вычисление кратчайшего расстояния между узлами."""
        if node1 == node2:
            return 0
        
        # BFS для поиска кратчайшего пути
        visited = {node1}
        queue = [(node1, 0)]
        
        while queue:
            current_node, distance = queue.pop(0)
            
            neighbors = graph.get_neighbors(current_node)
            for neighbor in neighbors:
                if neighbor == node2:
                    return distance + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        return float('inf')  # Нет пути


class ActivationFunctionFactory:
    """Фабрика функций активации."""
    
    _functions = {
        ActivationFunction.LINEAR: ActivationFunctions.LinearActivation,
        ActivationFunction.SIGMOID: ActivationFunctions.SigmoidActivation,
        ActivationFunction.TANH: ActivationFunctions.TanhActivation,
        ActivationFunction.RELU: ActivationFunctions.ReLUActivation,
        ActivationFunction.THRESHOLD: ActivationFunctions.ThresholdActivation,
        ActivationFunction.GAUSSIAN: ActivationFunctions.GaussianActivation,
        ActivationFunction.SOFTMAX: ActivationFunctions.SoftmaxActivation,
    }
    
    @classmethod
    def create(cls, function_type: ActivationFunction) -> IActivationFunction:
        """Создание функции активации по типу."""
        if function_type not in cls._functions:
            raise ValueError(f"Неизвестный тип функции активации: {function_type}")
        
        return cls._functions[function_type]()
    
    @classmethod
    def get_available_functions(cls) -> List[ActivationFunction]:
        """Получение списка доступных функций."""
        return list(cls._functions.keys())


class DecayFunctionFactory:
    """Фабрика функций затухания."""
    
    _functions = {
        DecayFunction.EXPONENTIAL: DecayFunctions.ExponentialDecay,
        DecayFunction.LINEAR: DecayFunctions.LinearDecay,
        DecayFunction.LOGARITHMIC: DecayFunctions.LogarithmicDecay,
        DecayFunction.POWER: DecayFunctions.PowerDecay,
        DecayFunction.STEP: DecayFunctions.StepDecay,
        DecayFunction.NONE: DecayFunctions.NoDecay,
    }
    
    @classmethod
    def create(cls, function_type: DecayFunction) -> IDecayFunction:
        """Создание функции затухания по типу."""
        if function_type not in cls._functions:
            raise ValueError(f"Неизвестный тип функции затухания: {function_type}")
        
        return cls._functions[function_type]()
    
    @classmethod
    def get_available_functions(cls) -> List[DecayFunction]:
        """Получение списка доступных функций."""
        return list(cls._functions.keys())


# Вспомогательные функции для работы с активацией

def normalize_activations(activations: Dict[str, float], 
                         method: str = "softmax",
                         temperature: float = 1.0) -> Dict[str, float]:
    """Нормализация активаций различными методами."""
    
    if not activations:
        return {}
    
    values = list(activations.values())
    
    if method == "softmax":
        # Softmax нормализация
        exp_values = [math.exp(val / temperature) for val in values]
        exp_sum = sum(exp_values)
        
        if exp_sum == 0:
            # Равномерное распределение при переполнении
            uniform_value = 1.0 / len(activations)
            return {node_id: uniform_value for node_id in activations.keys()}
        
        normalized = {}
        for i, node_id in enumerate(activations.keys()):
            normalized[node_id] = exp_values[i] / exp_sum
        
        return normalized
    
    elif method == "min_max":
        # Min-Max нормализация
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            # Все значения одинаковые
            return {node_id: 0.5 for node_id in activations.keys()}
        
        return {
            node_id: (val - min_val) / (max_val - min_val)
            for node_id, val in activations.items()
        }
    
    elif method == "z_score":
        # Z-score нормализация
        mean_val = sum(values) / len(values)
        variance = sum((val - mean_val) ** 2 for val in values) / len(values)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return {node_id: 0.5 for node_id in activations.keys()}
        
        z_scores = {
            node_id: (val - mean_val) / std_dev
            for node_id, val in activations.items()
        }
        
        # Преобразуем Z-scores в диапазон [0, 1] через sigmoid
        sigmoid_func = ActivationFunctions.SigmoidActivation()
        return {
            node_id: sigmoid_func.compute(z_score)
            for node_id, z_score in z_scores.items()
        }
    
    elif method == "l2":
        # L2 нормализация
        l2_norm = math.sqrt(sum(val ** 2 for val in values))
        
        if l2_norm == 0:
            return {node_id: 0.0 for node_id in activations.keys()}
        
        return {
            node_id: val / l2_norm
            for node_id, val in activations.items()
        }
    
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")


def compute_activation_entropy(activations: Dict[str, float]) -> float:
    """Вычисление энтропии распределения активации."""
    
    if not activations:
        return 0.0
    
    # Нормализуем активации как вероятности
    total_activation = sum(activations.values())
    
    if total_activation == 0:
        return 0.0
    
    probabilities = [val / total_activation for val in activations.values()]
    
    # Вычисляем энтропию
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


def compute_activation_sparsity(activations: Dict[str, float], 
                               threshold: float = 0.1) -> float:
    """Вычисление разреженности активации."""
    
    if not activations:
        return 1.0
    
    active_count = sum(1 for val in activations.values() if val >= threshold)
    total_count = len(activations)
    
    return 1.0 - (active_count / total_count)


def find_activation_peaks(activations: Dict[str, float], 
                         min_prominence: float = 0.1) -> List[str]:
    """Поиск пиков активации."""
    
    # Сортируем узлы по уровню активации
    sorted_nodes = sorted(activations.items(), key=lambda x: x[1], reverse=True)
    
    peaks = []
    for node_id, activation_level in sorted_nodes:
        if activation_level < min_prominence:
            break
        
        # Проверяем, является ли это локальным максимумом
        # (упрощенная версия без учета топологии графа)
        is_peak = True
        
        # Если активация значительно выше среднего, считаем пиком
        avg_activation = sum(activations.values()) / len(activations)
        if activation_level > avg_activation + min_prominence:
            peaks.append(node_id)
    
    return peaks


def compute_convergence_metric(history: List[Dict[str, float]], 
                              window_size: int = 5) -> float:
    """Вычисление метрики сходимости на основе истории активаций."""
    
    if len(history) < window_size + 1:
        return float('inf')  # Недостаточно данных
    
    # Берем последние window_size итераций
    recent_history = history[-window_size-1:]
    
    # Вычисляем изменения между соседними итерациями
    changes = []
    
    for i in range(1, len(recent_history)):
        prev_activations = recent_history[i-1]
        curr_activations = recent_history[i]
        
        # Находим общие узлы
        common_nodes = set(prev_activations.keys()) & set(curr_activations.keys())
        
        if not common_nodes:
            continue
        
        # Вычисляем среднее изменение
        total_change = sum(
            abs(curr_activations.get(node, 0) - prev_activations.get(node, 0))
            for node in common_nodes
        )
        
        avg_change = total_change / len(common_nodes)
        changes.append(avg_change)
    
    if not changes:
        return float('inf')
    
    # Возвращаем среднее изменение за период
    return sum(changes) / len(changes)