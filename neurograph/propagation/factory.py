# neurograph/propagation/factory.py
"""
Фабрика для создания компонентов распространения активации и утилиты.
"""

from typing import Dict, Any, Optional, List, Type
from neurograph.core import Configuration
from neurograph.core.logging import get_logger
from neurograph.propagation.base import (
    IPropagationEngine, PropagationConfig, ActivationFunction, DecayFunction,
    PropagationMode, InvalidConfigurationError
)
from neurograph.propagation.engine import SpreadingActivationEngine
from neurograph.propagation.visualizer import PropagationVisualizer


class PropagationFactory:
    """Фабрика для создания компонентов распространения активации."""
    
    _engines: Dict[str, Type[IPropagationEngine]] = {
        "spreading_activation": SpreadingActivationEngine,
    }
    
    _logger = get_logger("propagation_factory")
    
    @classmethod
    def create_engine(cls, 
                     engine_type: str = "spreading_activation",
                     graph=None,
                     **kwargs) -> IPropagationEngine:
        """
        Создание движка распространения активации.
        
        Args:
            engine_type: Тип движка ("spreading_activation")
            graph: Граф знаний для распространения
            **kwargs: Дополнительные параметры для движка
            
        Returns:
            IPropagationEngine: Экземпляр движка распространения
        """
        
        if engine_type not in cls._engines:
            available = ", ".join(cls._engines.keys())
            raise ValueError(f"Неизвестный тип движка: {engine_type}. Доступные: {available}")
        
        engine_class = cls._engines[engine_type]
        engine = engine_class(graph=graph, **kwargs)
        
        cls._logger.info(f"Создан движок распространения: {engine_type}")
        return engine
    
    @classmethod
    def create_config(cls,
                     activation_function: ActivationFunction = ActivationFunction.SIGMOID,
                     decay_function: DecayFunction = DecayFunction.EXPONENTIAL,
                     propagation_mode: PropagationMode = PropagationMode.SPREADING,
                     **kwargs) -> PropagationConfig:
        """
        Создание конфигурации распространения.
        
        Args:
            activation_function: Функция активации
            decay_function: Функция затухания
            propagation_mode: Режим распространения
            **kwargs: Дополнительные параметры конфигурации
            
        Returns:
            PropagationConfig: Конфигурация распространения
        """
        
        config = PropagationConfig(
            activation_function=activation_function,
            decay_function=decay_function,
            propagation_mode=propagation_mode,
            **kwargs
        )
        
        # Валидация конфигурации
        is_valid, errors = config.validate()
        if not is_valid:
            raise InvalidConfigurationError("Некорректная конфигурация", errors)
        
        cls._logger.debug(f"Создана конфигурация распространения: {propagation_mode}")
        return config
    
    @classmethod
    def create_visualizer(cls) -> PropagationVisualizer:
        """Создание визуализатора распространения."""
        return PropagationVisualizer()
    
    @classmethod
    def register_engine(cls, name: str, engine_class: Type[IPropagationEngine]) -> None:
        """Регистрация нового типа движка."""
        cls._engines[name] = engine_class
        cls._logger.info(f"Зарегистрирован движок: {name}")
    
    @classmethod
    def get_available_engines(cls) -> List[str]:
        """Получение списка доступных движков."""
        return list(cls._engines.keys())
    
    @classmethod
    def create_from_config(cls, config: Configuration) -> IPropagationEngine:
        """Создание движка из конфигурации."""
        
        engine_type = config.get("engine.type", "spreading_activation")
        engine_params = config.get("engine.params", {})
        
        return cls.create_engine(engine_type, **engine_params)


# Готовые функции для быстрого создания компонентов

def create_default_engine(graph=None) -> IPropagationEngine:
    """Создание движка с настройками по умолчанию."""
    return PropagationFactory.create_engine("spreading_activation", graph=graph)


def create_high_performance_engine(graph=None) -> IPropagationEngine:
    """Создание высокопроизводительного движка."""
    return PropagationFactory.create_engine(
        "spreading_activation",
        graph=graph
    )


def create_research_engine(graph=None) -> IPropagationEngine:
    """Создание движка для исследований с расширенными возможностями."""
    return PropagationFactory.create_engine(
        "spreading_activation",
        graph=graph
    )


def create_default_config() -> PropagationConfig:
    """Создание конфигурации по умолчанию."""
    return PropagationFactory.create_config()


def create_fast_config() -> PropagationConfig:
    """Создание конфигурации для быстрого распространения."""
    return PropagationFactory.create_config(
        max_iterations=50,
        convergence_threshold=0.01,
        activation_threshold=0.2,
        max_active_nodes=100,
        time_steps=5
    )


def create_precise_config() -> PropagationConfig:
    """Создание конфигурации для точного распространения."""
    return PropagationFactory.create_config(
        max_iterations=200,
        convergence_threshold=0.0001,
        activation_threshold=0.05,
        max_active_nodes=1000,
        time_steps=20,
        lateral_inhibition=True,
        weight_normalization=True
    )


def create_experimental_config() -> PropagationConfig:
    """Создание экспериментальной конфигурации."""
    return PropagationFactory.create_config(
        activation_function=ActivationFunction.TANH,
        decay_function=DecayFunction.POWER,
        propagation_mode=PropagationMode.BIDIRECTIONAL,
        max_iterations=500,
        convergence_threshold=0.00001,
        lateral_inhibition=True,
        inhibition_strength=0.3,
        time_steps=50
    )


# Утилиты для работы с конфигурациями

class ConfigurationPresets:
    """Готовые наборы конфигураций для различных сценариев."""
    
    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        """Конфигурация для разработки."""
        return {
            "max_iterations": 20,
            "convergence_threshold": 0.01,
            "activation_threshold": 0.3,
            "max_active_nodes": 50,
            "activation_function": ActivationFunction.SIGMOID,
            "decay_function": DecayFunction.LINEAR,
            "propagation_mode": PropagationMode.SPREADING,
            "lateral_inhibition": False,
            "time_steps": 3
        }
    
    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """Конфигурация для продакшена."""
        return {
            "max_iterations": 100,
            "convergence_threshold": 0.001,
            "activation_threshold": 0.1,
            "max_active_nodes": 500,
            "activation_function": ActivationFunction.SIGMOID,
            "decay_function": DecayFunction.EXPONENTIAL,
            "propagation_mode": PropagationMode.SPREADING,
            "lateral_inhibition": True,
            "inhibition_strength": 0.2,
            "time_steps": 10,
            "weight_normalization": True
        }
    
    @staticmethod
    def get_memory_efficient_config() -> Dict[str, Any]:
        """Конфигурация для ограниченной памяти."""
        return {
            "max_iterations": 30,
            "convergence_threshold": 0.005,
            "activation_threshold": 0.4,
            "max_active_nodes": 25,
            "activation_function": ActivationFunction.THRESHOLD,
            "decay_function": DecayFunction.STEP,
            "propagation_mode": PropagationMode.CONSTRAINED,
            "lateral_inhibition": False,
            "time_steps": 2
        }
    
    @staticmethod
    def get_research_config() -> Dict[str, Any]:
        """Конфигурация для исследований."""
        return {
            "max_iterations": 1000,
            "convergence_threshold": 0.00001,
            "activation_threshold": 0.01,
            "max_active_nodes": 2000,
            "activation_function": ActivationFunction.GAUSSIAN,
            "decay_function": DecayFunction.POWER,
            "propagation_mode": PropagationMode.BIDIRECTIONAL,
            "lateral_inhibition": True,
            "inhibition_strength": 0.15,
            "inhibition_radius": 3,
            "time_steps": 100,
            "weight_normalization": True,
            "bidirectional_weights": True
        }


class PropagationConfigBuilder:
    """Строитель конфигураций распространения."""
    
    def __init__(self):
        self._config_data = {}
        self.reset()
    
    def reset(self):
        """Сброс к значениям по умолчанию."""
        self._config_data = {
            "max_iterations": 100,
            "convergence_threshold": 0.001,
            "activation_threshold": 0.1,
            "max_active_nodes": 1000,
            "activation_function": ActivationFunction.SIGMOID,
            "decay_function": DecayFunction.EXPONENTIAL,
            "propagation_mode": PropagationMode.SPREADING,
            "lateral_inhibition": True,
            "inhibition_strength": 0.2,
            "inhibition_radius": 2,
            "time_steps": 10,
            "time_step_duration": 0.1,
            "default_weight": 1.0,
            "weight_normalization": True,
            "bidirectional_weights": False,
            "max_propagation_depth": 10,
            "edge_type_filters": [],
            "node_type_filters": [],
            "activation_params": {"steepness": 1.0},
            "decay_params": {"rate": 0.1}
        }
        return self
    
    def set_performance_mode(self, mode: str):
        """Установка режима производительности."""
        if mode == "fast":
            self._config_data.update({
                "max_iterations": 50,
                "convergence_threshold": 0.01,
                "activation_threshold": 0.2,
                "max_active_nodes": 100,
                "time_steps": 5,
                "lateral_inhibition": False
            })
        elif mode == "balanced":
            self._config_data.update({
                "max_iterations": 100,
                "convergence_threshold": 0.001,
                "activation_threshold": 0.1,
                "max_active_nodes": 500,
                "time_steps": 10,
                "lateral_inhibition": True
            })
        elif mode == "precise":
            self._config_data.update({
                "max_iterations": 200,
                "convergence_threshold": 0.0001,
                "activation_threshold": 0.05,
                "max_active_nodes": 1000,
                "time_steps": 20,
                "lateral_inhibition": True
            })
        else:
            raise ValueError(f"Неизвестный режим производительности: {mode}")
        
        return self
    
    def set_activation_function(self, func: ActivationFunction, **params):
        """Установка функции активации."""
        self._config_data["activation_function"] = func
        if params:
            self._config_data["activation_params"] = params
        return self
    
    def set_decay_function(self, func: DecayFunction, **params):
        """Установка функции затухания."""
        self._config_data["decay_function"] = func
        if params:
            self._config_data["decay_params"] = params
        return self
    
    def set_propagation_mode(self, mode: PropagationMode):
        """Установка режима распространения."""
        self._config_data["propagation_mode"] = mode
        return self
    
    def set_iterations(self, max_iterations: int, convergence_threshold: float = None):
        """Установка параметров итераций."""
        self._config_data["max_iterations"] = max_iterations
        if convergence_threshold is not None:
            self._config_data["convergence_threshold"] = convergence_threshold
        return self
    
    def set_activation_limits(self, threshold: float, max_nodes: int = None):
        """Установка пороговых значений активации."""
        self._config_data["activation_threshold"] = threshold
        if max_nodes is not None:
            self._config_data["max_active_nodes"] = max_nodes
        return self
    
    def set_lateral_inhibition(self, enabled: bool, strength: float = 0.2, radius: int = 2):
        """Настройка латерального торможения."""
        self._config_data["lateral_inhibition"] = enabled
        if enabled:
            self._config_data["inhibition_strength"] = strength
            self._config_data["inhibition_radius"] = radius
        return self
    
    def set_temporal_params(self, time_steps: int, step_duration: float = 0.1):
        """Установка временных параметров."""
        self._config_data["time_steps"] = time_steps
        self._config_data["time_step_duration"] = step_duration
        return self
    
    def set_depth_limit(self, max_depth: int):
        """Установка максимальной глубины распространения."""
        self._config_data["max_propagation_depth"] = max_depth
        return self
    
    def set_filters(self, edge_types: List[str] = None, node_types: List[str] = None):
        """Установка фильтров типов узлов и ребер."""
        if edge_types is not None:
            self._config_data["edge_type_filters"] = edge_types
        if node_types is not None:
            self._config_data["node_type_filters"] = node_types
        return self
    
    def set_weight_params(self, default_weight: float = 1.0, 
                         normalize: bool = True, bidirectional: bool = False):
        """Установка параметров весов."""
        self._config_data["default_weight"] = default_weight
        self._config_data["weight_normalization"] = normalize
        self._config_data["bidirectional_weights"] = bidirectional
        return self
    
    def build(self) -> PropagationConfig:
        """Построение конфигурации."""
        return PropagationConfig(**self._config_data)


# Утилиты для анализа и диагностики

class PropagationDiagnostics:
    """Диагностические утилиты для распространения активации."""
    
    @staticmethod
    def analyze_config(config: PropagationConfig) -> Dict[str, Any]:
        """Анализ конфигурации на потенциальные проблемы."""
        
        analysis = {
            "performance_rating": "unknown",
            "memory_usage_estimate": "unknown",
            "convergence_likelihood": "unknown",
            "warnings": [],
            "recommendations": []
        }
        
        # Анализ производительности
        performance_score = 100
        
        if config.max_iterations > 500:
            performance_score -= 30
            analysis["warnings"].append("Большое количество итераций может замедлить выполнение")
        
        if config.max_active_nodes > 2000:
            performance_score -= 20
            analysis["warnings"].append("Большое количество активных узлов увеличит потребление памяти")
        
        if config.lateral_inhibition and config.inhibition_radius > 3:
            performance_score -= 15
            analysis["warnings"].append("Большой радиус торможения замедлит обработку")
        
        if config.time_steps > 50:
            performance_score -= 10
            analysis["warnings"].append("Много временных шагов увеличат время обработки")
        
        # Рейтинг производительности
        if performance_score >= 80:
            analysis["performance_rating"] = "excellent"
        elif performance_score >= 60:
            analysis["performance_rating"] = "good"
        elif performance_score >= 40:
            analysis["performance_rating"] = "fair"
        else:
            analysis["performance_rating"] = "poor"
        
        # Оценка использования памяти
        memory_score = config.max_active_nodes * config.time_steps
        if memory_score < 1000:
            analysis["memory_usage_estimate"] = "low"
        elif memory_score < 10000:
            analysis["memory_usage_estimate"] = "medium"
        else:
            analysis["memory_usage_estimate"] = "high"
        
        # Оценка вероятности сходимости
        if config.convergence_threshold > 0.01:
            analysis["convergence_likelihood"] = "very_high"
        elif config.convergence_threshold > 0.001:
            analysis["convergence_likelihood"] = "high"
        elif config.convergence_threshold > 0.0001:
            analysis["convergence_likelihood"] = "medium"
        else:
            analysis["convergence_likelihood"] = "low"
            analysis["warnings"].append("Очень строгий порог сходимости может препятствовать завершению")
        
        # Рекомендации
        if config.activation_threshold < 0.05:
            analysis["recommendations"].append("Рассмотрите повышение порога активации для фильтрации шума")
        
        if not config.weight_normalization and config.propagation_mode != PropagationMode.CONSTRAINED:
            analysis["recommendations"].append("Включите нормализацию весов для стабильной активации")
        
        if config.lateral_inhibition and config.inhibition_strength > 0.5:
            analysis["recommendations"].append("Высокая сила торможения может подавить активацию")
        
        return analysis
    
    @staticmethod
    def estimate_execution_time(config: PropagationConfig, 
                               graph_size: int) -> Dict[str, float]:
        """Оценка времени выполнения на основе конфигурации и размера графа."""
        
        # Базовые коэффициенты (приблизительные)
        base_time_per_node = 0.001  # секунды
        iteration_overhead = 0.01   # секунды на итерацию
        inhibition_overhead = 0.5   # множитель для латерального торможения
        
        # Оценка времени на узел
        time_per_node = base_time_per_node
        
        if config.lateral_inhibition:
            time_per_node *= inhibition_overhead
        
        if config.propagation_mode == PropagationMode.BIDIRECTIONAL:
            time_per_node *= 1.5
        elif config.propagation_mode == PropagationMode.CONSTRAINED:
            time_per_node *= 0.8
        
        # Оценка общего времени
        active_nodes = min(config.max_active_nodes, graph_size)
        
        estimated_time = (
            active_nodes * time_per_node * config.max_iterations +
            config.max_iterations * iteration_overhead
        )
        
        return {
            "estimated_seconds": estimated_time,
            "estimated_minutes": estimated_time / 60,
            "confidence": "low",  # Оценка приблизительная
            "factors": {
                "graph_size": graph_size,
                "active_nodes": active_nodes,
                "iterations": config.max_iterations,
                "lateral_inhibition": config.lateral_inhibition,
                "propagation_mode": config.propagation_mode.value
            }
        }
    
    @staticmethod
    def validate_config_compatibility(config: PropagationConfig, 
                                    graph_info: Dict[str, Any]) -> List[str]:
        """Проверка совместимости конфигурации с графом."""
        
        issues = []
        
        graph_size = graph_info.get("node_count", 0)
        edge_count = graph_info.get("edge_count", 0)
        available_edge_types = graph_info.get("edge_types", [])
        available_node_types = graph_info.get("node_types", [])
        
        # Проверка размера графа
        if graph_size == 0:
            issues.append("Граф не содержит узлов")
            return issues
        
        if edge_count == 0:
            issues.append("Граф не содержит ребер - распространение невозможно")
        
        # Проверка фильтров
        if config.edge_type_filters:
            missing_edge_types = set(config.edge_type_filters) - set(available_edge_types)
            if missing_edge_types:
                issues.append(f"Фильтры ребер содержат отсутствующие типы: {missing_edge_types}")
        
        if config.node_type_filters:
            missing_node_types = set(config.node_type_filters) - set(available_node_types)
            if missing_node_types:
                issues.append(f"Фильтры узлов содержат отсутствующие типы: {missing_node_types}")
        
        # Проверка производительности
        if config.max_active_nodes > graph_size:
            issues.append(f"Максимальное количество активных узлов ({config.max_active_nodes}) "
                         f"превышает размер графа ({graph_size})")
        
        # Проверка связности для латерального торможения
        if config.lateral_inhibition:
            avg_degree = edge_count / graph_size if graph_size > 0 else 0
            if avg_degree < 2:
                issues.append("Низкая связность графа может снизить эффективность латерального торможения")
        
        return issues


# Утилитные функции для быстрого создания и настройки

def quick_propagate(graph, 
                   initial_nodes: Dict[str, float],
                   mode: str = "balanced") -> Any:
    """Быстрое распространение активации с предустановленными настройками."""
    
    engine = create_default_engine(graph)
    
    if mode == "fast":
        config = create_fast_config()
    elif mode == "precise":
        config = create_precise_config()
    elif mode == "experimental":
        config = create_experimental_config()
    else:
        config = create_default_config()
    
    return engine.propagate(initial_nodes, config)


def create_custom_config_from_dict(config_dict: Dict[str, Any]) -> PropagationConfig:
    """Создание конфигурации из словаря."""
    
    # Преобразование строковых значений в enum'ы
    if "activation_function" in config_dict:
        if isinstance(config_dict["activation_function"], str):
            config_dict["activation_function"] = ActivationFunction(config_dict["activation_function"])
    
    if "decay_function" in config_dict:
        if isinstance(config_dict["decay_function"], str):
            config_dict["decay_function"] = DecayFunction(config_dict["decay_function"])
    
    if "propagation_mode" in config_dict:
        if isinstance(config_dict["propagation_mode"], str):
            config_dict["propagation_mode"] = PropagationMode(config_dict["propagation_mode"])
    
    return PropagationConfig(**config_dict)


def benchmark_config(config: PropagationConfig, 
                    graph, 
                    test_nodes: Dict[str, float],
                    runs: int = 3) -> Dict[str, Any]:
    """Бенчмарк конфигурации на тестовых данных."""
    
    engine = create_default_engine(graph)
    results = []
    
    for run in range(runs):
        result = engine.propagate(test_nodes, config)
        results.append({
            "success": result.success,
            "processing_time": result.processing_time,
            "iterations_used": result.iterations_used,
            "convergence_achieved": result.convergence_achieved,
            "activated_nodes_count": len(result.activated_nodes),
            "max_activation": result.max_activation_reached
        })
    
    # Агрегированная статистика
    if results:
        successful_runs = [r for r in results if r["success"]]
        
        benchmark_stats = {
            "total_runs": runs,
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / runs,
            "avg_processing_time": sum(r["processing_time"] for r in successful_runs) / len(successful_runs) if successful_runs else 0,
            "avg_iterations": sum(r["iterations_used"] for r in successful_runs) / len(successful_runs) if successful_runs else 0,
            "convergence_rate": sum(1 for r in successful_runs if r["convergence_achieved"]) / len(successful_runs) if successful_runs else 0,
            "avg_activated_nodes": sum(r["activated_nodes_count"] for r in successful_runs) / len(successful_runs) if successful_runs else 0,
            "max_activation_avg": sum(r["max_activation"] for r in successful_runs) / len(successful_runs) if successful_runs else 0,
            "individual_results": results
        }
    else:
        benchmark_stats = {
            "total_runs": runs,
            "successful_runs": 0,
            "success_rate": 0.0,
            "error": "Все запуски завершились неудачей"
        }
    
    return benchmark_stats