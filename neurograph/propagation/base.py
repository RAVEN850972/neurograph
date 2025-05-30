# neurograph/propagation/base.py
"""
Базовые интерфейсы и классы данных для модуля распространения активации.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime
import uuid
import numpy as np


class ActivationFunction(Enum):
    """Типы функций активации."""
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    SOFTMAX = "softmax"
    THRESHOLD = "threshold"
    GAUSSIAN = "gaussian"


class DecayFunction(Enum):
    """Типы функций затухания."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    POWER = "power"
    STEP = "step"
    NONE = "none"


class PropagationMode(Enum):
    """Режимы распространения активации."""
    SPREADING = "spreading"          # Расходящееся распространение
    FOCUSING = "focusing"            # Сходящееся распространение
    BIDIRECTIONAL = "bidirectional" # Двунаправленное
    CONSTRAINED = "constrained"     # Ограниченное по условиям


@dataclass
class NodeActivation:
    """Состояние активации узла."""
    node_id: str                            # ID узла
    activation_level: float = 0.0           # Уровень активации (0.0-1.0)
    previous_activation: float = 0.0        # Предыдущий уровень
    activation_time: datetime = field(default_factory=datetime.now)
    source_nodes: Set[str] = field(default_factory=set)  # Источники активации
    propagation_depth: int = 0              # Глубина от источника
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_activation(self, new_level: float, source_node: str = None):
        """Обновление уровня активации."""
        self.previous_activation = self.activation_level
        self.activation_level = max(0.0, min(1.0, new_level))
        self.activation_time = datetime.now()
        if source_node:
            self.source_nodes.add(source_node)
    
    def decay(self, decay_rate: float = 0.1):
        """Применение затухания к активации."""
        self.previous_activation = self.activation_level
        self.activation_level = max(0.0, self.activation_level - decay_rate)
        if self.activation_level != self.previous_activation:
            self.activation_time = datetime.now()
    
    def is_active(self, threshold: float = 0.1) -> bool:
        """Проверка активности узла."""
        return self.activation_level >= threshold


@dataclass
class PropagationConfig:
    """Конфигурация распространения активации."""
    # Основные параметры
    max_iterations: int = 100               # Максимальное количество итераций
    convergence_threshold: float = 0.001    # Порог сходимости
    activation_threshold: float = 0.1       # Порог активации
    max_active_nodes: int = 1000           # Максимальное количество активных узлов
    
    # Функции активации и затухания
    activation_function: ActivationFunction = ActivationFunction.SIGMOID
    decay_function: DecayFunction = DecayFunction.EXPONENTIAL
    
    # Параметры функций
    activation_params: Dict[str, float] = field(default_factory=lambda: {"steepness": 1.0})
    decay_params: Dict[str, float] = field(default_factory=lambda: {"rate": 0.1})
    
    # Режим распространения
    propagation_mode: PropagationMode = PropagationMode.SPREADING
    max_propagation_depth: int = 10         # Максимальная глубина распространения
    
    # Латеральное торможение
    lateral_inhibition: bool = True
    inhibition_strength: float = 0.2
    inhibition_radius: int = 2              # Радиус торможения
    
    # Временные параметры
    time_steps: int = 10                    # Количество временных шагов
    time_step_duration: float = 0.1         # Длительность шага (секунды)
    
    # Веса связей
    default_weight: float = 1.0
    weight_normalization: bool = True
    bidirectional_weights: bool = False
    
    # Ограничения
    edge_type_filters: List[str] = field(default_factory=list)  # Фильтр типов ребер
    node_type_filters: List[str] = field(default_factory=list)  # Фильтр типов узлов
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Валидация конфигурации."""
        errors = []
        
        if self.max_iterations <= 0:
            errors.append("max_iterations должно быть больше 0")
        
        if not 0 <= self.convergence_threshold <= 1:
            errors.append("convergence_threshold должно быть между 0 и 1")
            
        if not 0 <= self.activation_threshold <= 1:
            errors.append("activation_threshold должно быть между 0 и 1")
            
        if self.max_active_nodes <= 0:
            errors.append("max_active_nodes должно быть больше 0")
            
        if self.max_propagation_depth <= 0:
            errors.append("max_propagation_depth должно быть больше 0")
            
        if not 0 <= self.inhibition_strength <= 1:
            errors.append("inhibition_strength должно быть между 0 и 1")
            
        return len(errors) == 0, errors


@dataclass
class PropagationResult:
    """Результат распространения активации."""
    success: bool                                          # Успешность распространения
    activated_nodes: Dict[str, NodeActivation]            # Активированные узлы
    activation_history: List[Dict[str, float]] = field(default_factory=list)
    convergence_achieved: bool = False                     # Достигнута ли сходимость
    iterations_used: int = 0                              # Использованные итерации
    processing_time: float = 0.0                          # Время обработки
    initial_nodes: Set[str] = field(default_factory=set)  # Начальные узлы
    max_activation_reached: float = 0.0                   # Максимальная активация
    total_activation: float = 0.0                         # Общая активация
    propagation_paths: List[List[str]] = field(default_factory=list)  # Пути распространения
    inhibited_nodes: Set[str] = field(default_factory=set)  # Заторможенные узлы
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def get_most_activated_nodes(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Получение наиболее активированных узлов."""
        sorted_nodes = sorted(
            [(node_id, activation.activation_level) 
             for node_id, activation in self.activated_nodes.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_nodes[:top_n]
    
    def get_activation_levels(self) -> Dict[str, float]:
        """Получение уровней активации всех узлов."""
        return {
            node_id: activation.activation_level 
            for node_id, activation in self.activated_nodes.items()
        }
    
    def filter_active_nodes(self, threshold: float = 0.1) -> Dict[str, NodeActivation]:
        """Фильтрация активных узлов по порогу."""
        return {
            node_id: activation 
            for node_id, activation in self.activated_nodes.items()
            if activation.activation_level >= threshold
        }


class IPropagationEngine(ABC):
    """Основной интерфейс движка распространения активации."""
    
    @abstractmethod
    def propagate(self, 
                 initial_nodes: Dict[str, float], 
                 config: PropagationConfig) -> PropagationResult:
        """
        Выполняет распространение активации от начальных узлов.
        
        Args:
            initial_nodes: Словарь {node_id: initial_activation_level}
            config: Конфигурация распространения
            
        Returns:
            PropagationResult: Результат распространения
        """
        pass
    
    @abstractmethod
    def set_graph(self, graph) -> None:
        """Установка графа знаний для распространения."""
        pass
    
    @abstractmethod
    def reset_activations(self) -> None:
        """Сброс всех активаций к нулевому состоянию."""
        pass
    
    @abstractmethod
    def get_node_activation(self, node_id: str) -> Optional[NodeActivation]:
        """Получение текущей активации узла."""
        pass
    
    @abstractmethod
    def update_node_activation(self, node_id: str, activation_level: float) -> bool:
        """Обновление активации узла."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики работы движка."""
        pass


class IActivationFunction(ABC):
    """Интерфейс функции активации."""
    
    @abstractmethod
    def compute(self, input_value: float, **params) -> float:
        """Вычисление функции активации."""
        pass
    
    @abstractmethod
    def derivative(self, input_value: float, **params) -> float:
        """Вычисление производной функции активации."""
        pass


class IDecayFunction(ABC):
    """Интерфейс функции затухания."""
    
    @abstractmethod
    def compute(self, current_activation: float, time_step: float, **params) -> float:
        """Вычисление затухания активации."""
        pass


class ILateralInhibition(ABC):
    """Интерфейс латерального торможения."""
    
    @abstractmethod
    def apply_inhibition(self, 
                        activations: Dict[str, NodeActivation],
                        graph,
                        config: PropagationConfig) -> Dict[str, NodeActivation]:
        """Применение латерального торможения."""
        pass


class IPropagationVisualizer(ABC):
    """Интерфейс визуализации распространения активации."""
    
    @abstractmethod
    def visualize_propagation(self, 
                            result: PropagationResult,
                            graph,
                            save_path: Optional[str] = None,
                            show_animation: bool = False) -> None:
        """Визуализация результата распространения."""
        pass
    
    @abstractmethod
    def create_activation_heatmap(self,
                                activations: Dict[str, float],
                                graph,
                                save_path: Optional[str] = None) -> None:
        """Создание тепловой карты активации."""
        pass


# Исключения для модуля
class PropagationError(Exception):
    """Базовое исключение модуля распространения."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PROPAGATION_ERROR"
        self.details = details or {}


class GraphNotSetError(PropagationError):
    """Исключение при отсутствии установленного графа."""
    
    def __init__(self, message: str = "Граф не установлен для распространения"):
        super().__init__(message, "GRAPH_NOT_SET")


class InvalidConfigurationError(PropagationError):
    """Исключение при неверной конфигурации."""
    
    def __init__(self, message: str, config_errors: List[str] = None):
        super().__init__(message, "INVALID_CONFIGURATION")
        self.config_errors = config_errors or []


class ConvergenceError(PropagationError):
    """Исключение при проблемах сходимости."""
    
    def __init__(self, message: str, iterations_used: int = 0):
        super().__init__(message, "CONVERGENCE_ERROR")
        self.iterations_used = iterations_used


class NodeNotFoundError(PropagationError):
    """Исключение при отсутствии узла в графе."""
    
    def __init__(self, node_id: str):
        super().__init__(f"Узел {node_id} не найден в графе", "NODE_NOT_FOUND")
        self.node_id = node_id