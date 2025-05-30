# tests/test_propagation/test_base.py
"""
Тесты для базовых классов и интерфейсов модуля распространения активации.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from neurograph.propagation.base import (
    NodeActivation, PropagationConfig, PropagationResult,
    ActivationFunction, DecayFunction, PropagationMode,
    PropagationError, GraphNotSetError, InvalidConfigurationError,
    ConvergenceError, NodeNotFoundError
)


class TestNodeActivation:
    """Тесты для класса NodeActivation."""
    
    def test_node_activation_creation(self):
        """Тест создания активации узла."""
        activation = NodeActivation("test_node")
        
        assert activation.node_id == "test_node"
        assert activation.activation_level == 0.0
        assert activation.previous_activation == 0.0
        assert activation.propagation_depth == 0
        assert isinstance(activation.source_nodes, set)
        assert len(activation.source_nodes) == 0
        assert isinstance(activation.metadata, dict)
        assert isinstance(activation.activation_time, datetime)
    
    def test_node_activation_with_parameters(self):
        """Тест создания активации с параметрами."""
        now = datetime.now()
        metadata = {"test": "value"}
        source_nodes = {"source1", "source2"}
        
        activation = NodeActivation(
            node_id="test_node",
            activation_level=0.5,
            previous_activation=0.3,
            activation_time=now,
            source_nodes=source_nodes,
            propagation_depth=2,
            metadata=metadata
        )
        
        assert activation.node_id == "test_node"
        assert activation.activation_level == 0.5
        assert activation.previous_activation == 0.3
        assert activation.activation_time == now
        assert activation.source_nodes == source_nodes
        assert activation.propagation_depth == 2
        assert activation.metadata == metadata
    
    def test_update_activation(self):
        """Тест обновления активации."""
        activation = NodeActivation("test_node", activation_level=0.3)
        old_time = activation.activation_time
        
        # Ждем немного для изменения времени
        import time
        time.sleep(0.01)
        
        activation.update_activation(0.7, "source_node")
        
        assert activation.previous_activation == 0.3
        assert activation.activation_level == 0.7
        assert "source_node" in activation.source_nodes
        assert activation.activation_time > old_time
    
    def test_update_activation_bounds(self):
        """Тест границ при обновлении активации."""
        activation = NodeActivation("test_node")
        
        # Тест верхней границы
        activation.update_activation(1.5)
        assert activation.activation_level == 1.0
        
        # Тест нижней границы
        activation.update_activation(-0.5)
        assert activation.activation_level == 0.0
    
    def test_decay(self):
        """Тест затухания активации."""
        activation = NodeActivation("test_node", activation_level=0.8)
        
        activation.decay(0.2)
        
        assert activation.previous_activation == 0.8
        assert activation.activation_level == 0.6
    
    def test_decay_bounds(self):
        """Тест границ при затухании."""
        activation = NodeActivation("test_node", activation_level=0.1)
        
        activation.decay(0.2)
        
        assert activation.activation_level == 0.0
    
    def test_is_active(self):
        """Тест проверки активности узла."""
        activation = NodeActivation("test_node")
        
        # Неактивный узел
        assert not activation.is_active(0.1)
        
        # Активный узел
        activation.activation_level = 0.5
        assert activation.is_active(0.1)
        assert activation.is_active(0.5)
        assert not activation.is_active(0.6)


class TestPropagationConfig:
    """Тесты для класса PropagationConfig."""
    
    def test_default_config_creation(self):
        """Тест создания конфигурации по умолчанию."""
        config = PropagationConfig()
        
        assert config.max_iterations == 100
        assert config.convergence_threshold == 0.001
        assert config.activation_threshold == 0.1
        assert config.max_active_nodes == 1000
        assert config.activation_function == ActivationFunction.SIGMOID
        assert config.decay_function == DecayFunction.EXPONENTIAL
        assert config.propagation_mode == PropagationMode.SPREADING
        assert config.lateral_inhibition is True
        assert config.inhibition_strength == 0.2
        assert config.inhibition_radius == 2
        assert config.time_steps == 10
        assert config.time_step_duration == 0.1
        assert config.max_propagation_depth == 10
    
    def test_custom_config_creation(self):
        """Тест создания пользовательской конфигурации."""
        config = PropagationConfig(
            max_iterations=50,
            convergence_threshold=0.01,
            activation_function=ActivationFunction.RELU,
            decay_function=DecayFunction.LINEAR,
            propagation_mode=PropagationMode.FOCUSING,
            lateral_inhibition=False
        )
        
        assert config.max_iterations == 50
        assert config.convergence_threshold == 0.01
        assert config.activation_function == ActivationFunction.RELU
        assert config.decay_function == DecayFunction.LINEAR
        assert config.propagation_mode == PropagationMode.FOCUSING
        assert config.lateral_inhibition is False
    
    def test_config_validation_valid(self):
        """Тест валидации корректной конфигурации."""
        config = PropagationConfig()
        is_valid, errors = config.validate()
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_config_validation_invalid_iterations(self):
        """Тест валидации с некорректным количеством итераций."""
        config = PropagationConfig(max_iterations=0)
        is_valid, errors = config.validate()
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("max_iterations" in error for error in errors)
    
    def test_config_validation_invalid_threshold(self):
        """Тест валидации с некорректным порогом."""
        config = PropagationConfig(convergence_threshold=1.5)
        is_valid, errors = config.validate()
        
        assert is_valid is False
        assert any("convergence_threshold" in error for error in errors)
    
    def test_config_validation_multiple_errors(self):
        """Тест валидации с множественными ошибками."""
        config = PropagationConfig(
            max_iterations=-1,
            convergence_threshold=2.0,
            activation_threshold=-0.5,
            max_active_nodes=0
        )
        is_valid, errors = config.validate()
        
        assert is_valid is False
        assert len(errors) >= 4


class TestPropagationResult:
    """Тесты для класса PropagationResult."""
    
    def test_result_creation(self):
        """Тест создания результата."""
        result = PropagationResult(success=True, activated_nodes={})
        
        assert result.success is True
        assert isinstance(result.activated_nodes, dict)
        assert isinstance(result.activation_history, list)
        assert result.convergence_achieved is False
        assert result.iterations_used == 0
        assert result.processing_time == 0.0
        assert isinstance(result.initial_nodes, set)
        assert result.max_activation_reached == 0.0
        assert result.total_activation == 0.0
        assert isinstance(result.propagation_paths, list)
        assert isinstance(result.inhibited_nodes, set)
        assert isinstance(result.metadata, dict)
        assert result.error_message is None
    
    def test_get_most_activated_nodes(self, sample_activations):
        """Тест получения наиболее активных узлов."""
        result = PropagationResult(success=True, activated_nodes=sample_activations)
        
        top_nodes = result.get_most_activated_nodes(2)
        
        assert len(top_nodes) == 2
        assert top_nodes[0][0] == "node_1"  # Наибольшая активация
        assert top_nodes[0][1] == 0.8
        assert top_nodes[1][0] == "node_2"
        assert top_nodes[1][1] == 0.6
    
    def test_get_activation_levels(self, sample_activations):
        """Тест получения уровней активации."""
        result = PropagationResult(success=True, activated_nodes=sample_activations)
        
        levels = result.get_activation_levels()
        
        assert len(levels) == 4
        assert levels["node_1"] == 0.8
        assert levels["node_2"] == 0.6
        assert levels["node_3"] == 0.4
        assert levels["node_4"] == 0.2
    
    def test_filter_active_nodes(self, sample_activations):
        """Тест фильтрации активных узлов."""
        result = PropagationResult(success=True, activated_nodes=sample_activations)
        
        # Фильтр с порогом 0.5
        active = result.filter_active_nodes(0.5)
        assert len(active) == 2
        assert "node_1" in active
        assert "node_2" in active
        
        # Фильтр с порогом 0.3
        active = result.filter_active_nodes(0.3)
        assert len(active) == 3
        
        # Фильтр с высоким порогом
        active = result.filter_active_nodes(0.9)
        assert len(active) == 0


class TestEnumerations:
    """Тесты для перечислений."""
    
    def test_activation_function_values(self):
        """Тест значений функций активации."""
        assert ActivationFunction.LINEAR.value == "linear"
        assert ActivationFunction.SIGMOID.value == "sigmoid"
        assert ActivationFunction.TANH.value == "tanh"
        assert ActivationFunction.RELU.value == "relu"
        assert ActivationFunction.SOFTMAX.value == "softmax"
        assert ActivationFunction.THRESHOLD.value == "threshold"
        assert ActivationFunction.GAUSSIAN.value == "gaussian"
    
    def test_decay_function_values(self):
        """Тест значений функций затухания."""
        assert DecayFunction.EXPONENTIAL.value == "exponential"
        assert DecayFunction.LINEAR.value == "linear"
        assert DecayFunction.LOGARITHMIC.value == "logarithmic"
        assert DecayFunction.POWER.value == "power"
        assert DecayFunction.STEP.value == "step"
        assert DecayFunction.NONE.value == "none"
    
    def test_propagation_mode_values(self):
        """Тест значений режимов распространения."""
        assert PropagationMode.SPREADING.value == "spreading"
        assert PropagationMode.FOCUSING.value == "focusing"
        assert PropagationMode.BIDIRECTIONAL.value == "bidirectional"
        assert PropagationMode.CONSTRAINED.value == "constrained"


class TestExceptions:
    """Тесты для исключений модуля."""
    
    def test_propagation_error(self):
        """Тест базового исключения PropagationError."""
        error = PropagationError("Test error")
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == "PROPAGATION_ERROR"
        assert error.details == {}
        
        # С дополнительными параметрами
        details = {"param": "value"}
        error = PropagationError("Test error", "CUSTOM_CODE", details)
        
        assert error.error_code == "CUSTOM_CODE"
        assert error.details == details
    
    def test_graph_not_set_error(self):
        """Тест исключения GraphNotSetError."""
        error = GraphNotSetError()
        
        assert "Граф не установлен" in str(error)
        assert error.error_code == "GRAPH_NOT_SET"
        
        # С пользовательским сообщением
        error = GraphNotSetError("Custom message")
        assert str(error) == "Custom message"
    
    def test_invalid_configuration_error(self):
        """Тест исключения InvalidConfigurationError."""
        config_errors = ["error1", "error2"]
        error = InvalidConfigurationError("Config invalid", config_errors)
        
        assert str(error) == "Config invalid"
        assert error.error_code == "INVALID_CONFIGURATION"
        assert error.config_errors == config_errors
    
    def test_convergence_error(self):
        """Тест исключения ConvergenceError."""
        error = ConvergenceError("No convergence", 50)
        
        assert str(error) == "No convergence"
        assert error.error_code == "CONVERGENCE_ERROR"
        assert error.iterations_used == 50
    
    def test_node_not_found_error(self):
        """Тест исключения NodeNotFoundError."""
        error = NodeNotFoundError("missing_node")
        
        assert "missing_node не найден" in str(error)
        assert error.error_code == "NODE_NOT_FOUND"
        assert error.node_id == "missing_node"


class TestParametrizedConfigurations:
    """Параметризованные тесты конфигураций."""
    
    @pytest.mark.parametrize("activation_func", [
        ActivationFunction.SIGMOID,
        ActivationFunction.TANH,
        ActivationFunction.RELU,
        ActivationFunction.LINEAR
    ])
    def test_activation_function_configs(self, activation_func):
        """Тест конфигураций с различными функциями активации."""
        config = PropagationConfig(activation_function=activation_func)
        
        assert config.activation_function == activation_func
        is_valid, errors = config.validate()
        assert is_valid is True
    
    @pytest.mark.parametrize("decay_func", [
        DecayFunction.EXPONENTIAL,
        DecayFunction.LINEAR,
        DecayFunction.POWER,
        DecayFunction.NONE
    ])
    def test_decay_function_configs(self, decay_func):
        """Тест конфигураций с различными функциями затухания."""
        config = PropagationConfig(decay_function=decay_func)
        
        assert config.decay_function == decay_func
        is_valid, errors = config.validate()
        assert is_valid is True
    
    @pytest.mark.parametrize("prop_mode", [
        PropagationMode.SPREADING,
        PropagationMode.FOCUSING,
        PropagationMode.BIDIRECTIONAL,
        PropagationMode.CONSTRAINED
    ])
    def test_propagation_mode_configs(self, prop_mode):
        """Тест конфигураций с различными режимами распространения."""
        config = PropagationConfig(propagation_mode=prop_mode)
        
        assert config.propagation_mode == prop_mode
        is_valid, errors = config.validate()
        assert is_valid is True
    
    @pytest.mark.parametrize("iterations,threshold,valid", [
        (1, 0.001, True),
        (100, 0.001, True),
        (1000, 0.0001, True),
        (0, 0.001, False),      # Некорректные итерации
        (-1, 0.001, False),     # Отрицательные итерации
        (100, 1.5, False),      # Некорректный порог
        (100, -0.1, False),     # Отрицательный порог
    ])
    def test_validation_combinations(self, iterations, threshold, valid):
        """Тест различных комбинаций параметров валидации."""
        config = PropagationConfig(
            max_iterations=iterations,
            convergence_threshold=threshold
        )
        
        is_valid, errors = config.validate()
        assert is_valid == valid
        
        if not valid:
            assert len(errors) > 0


class TestConfigurationEdgeCases:
    """Тесты граничных случаев конфигурации."""
    
    def test_minimal_valid_config(self):
        """Тест минимальной валидной конфигурации."""
        config = PropagationConfig(
            max_iterations=1,
            convergence_threshold=1.0,
            activation_threshold=1.0,
            max_active_nodes=1,
            max_propagation_depth=1
        )
        
        is_valid, errors = config.validate()
        assert is_valid is True
    
    def test_maximal_valid_config(self):
        """Тест максимальной валидной конфигурации."""
        config = PropagationConfig(
            max_iterations=10000,
            convergence_threshold=0.0,
            activation_threshold=0.0,
            max_active_nodes=100000,
            max_propagation_depth=100,
            inhibition_strength=1.0
        )
        
        is_valid, errors = config.validate()
        assert is_valid is True
    
    def test_edge_case_thresholds(self):
        """Тест граничных значений порогов."""
        # Граничные валидные значения
        config = PropagationConfig(
            convergence_threshold=0.0,
            activation_threshold=0.0,
            inhibition_strength=0.0
        )
        is_valid, _ = config.validate()
        assert is_valid is True
        
        config = PropagationConfig(
            convergence_threshold=1.0,
            activation_threshold=1.0,
            inhibition_strength=1.0
        )
        is_valid, _ = config.validate()
        assert is_valid is True
    
    def test_empty_filters(self):
        """Тест пустых фильтров."""
        config = PropagationConfig(
            edge_type_filters=[],
            node_type_filters=[]
        )
        
        is_valid, errors = config.validate()
        assert is_valid is True
        assert config.edge_type_filters == []
        assert config.node_type_filters == []
    
    def test_complex_filters(self):
        """Тест сложных фильтров."""
        config = PropagationConfig(
            edge_type_filters=["type1", "type2", "type3"],
            node_type_filters=["node_type1", "node_type2"]
        )
        
        is_valid, errors = config.validate()
        assert is_valid is True
        assert len(config.edge_type_filters) == 3
        assert len(config.node_type_filters) == 2


class TestNodeActivationEdgeCases:
    """Тесты граничных случаев активации узлов."""
    
    def test_activation_with_empty_sources(self):
        """Тест активации без источников."""
        activation = NodeActivation("node")
        
        activation.update_activation(0.5)
        assert len(activation.source_nodes) == 0
        
        activation.update_activation(0.7, None)
        assert len(activation.source_nodes) == 0
        
        activation.update_activation(0.9, "")
        assert "" in activation.source_nodes
    
    def test_activation_with_many_sources(self):
        """Тест активации с множественными источниками."""
        activation = NodeActivation("node")
        
        sources = [f"source_{i}" for i in range(100)]
        for source in sources:
            activation.update_activation(0.5, source)
        
        assert len(activation.source_nodes) == 100
        assert all(f"source_{i}" in activation.source_nodes for i in range(100))
    
    def test_repeated_updates(self):
        """Тест повторных обновлений."""
        activation = NodeActivation("node")
        
        # Множественные обновления от одного источника
        for i in range(10):
            activation.update_activation(0.1 * i, "same_source")
        
        assert len(activation.source_nodes) == 1
        assert "same_source" in activation.source_nodes
        assert activation.activation_level == 0.9
    
    def test_extreme_decay_values(self):
        """Тест экстремальных значений затухания."""
        activation = NodeActivation("node", activation_level=1.0)
        
        # Очень большое затухание
        activation.decay(2.0)
        assert activation.activation_level == 0.0
        
        # Затухание от нуля
        activation.activation_level = 0.0
        activation.decay(0.5)
        assert activation.activation_level == 0.0
    
    def test_precision_boundaries(self):
        """Тест границ точности."""
        activation = NodeActivation("node")
        
        # Очень маленькие значения
        activation.update_activation(1e-10)
        assert activation.activation_level == 1e-10
        
        # Значения близкие к границам
        activation.update_activation(0.9999999)
        assert activation.activation_level == 0.9999999
        
        activation.update_activation(1.0000001)
        assert activation.activation_level == 1.0


class TestResultAnalysis:
    """Тесты для анализа результатов."""
    
    def test_empty_result_analysis(self):
        """Тест анализа пустого результата."""
        result = PropagationResult(success=True, activated_nodes={})
        
        top_nodes = result.get_most_activated_nodes(5)
        assert len(top_nodes) == 0
        
        levels = result.get_activation_levels()
        assert len(levels) == 0
        
        active = result.filter_active_nodes(0.1)
        assert len(active) == 0
    
    def test_single_node_result(self):
        """Тест результата с одним узлом."""
        activation = NodeActivation("single", activation_level=0.75)
        result = PropagationResult(success=True, activated_nodes={"single": activation})
        
        top_nodes = result.get_most_activated_nodes(5)
        assert len(top_nodes) == 1
        assert top_nodes[0] == ("single", 0.75)
        
        levels = result.get_activation_levels()
        assert levels == {"single": 0.75}
        
        active = result.filter_active_nodes(0.5)
        assert len(active) == 1
        assert "single" in active
    
    def test_large_result_analysis(self):
        """Тест анализа большого результата."""
        # Создаем большой набор активаций
        activations = {}
        for i in range(1000):
            activation_level = i / 1000.0  # От 0.0 до 0.999
            activations[f"node_{i}"] = NodeActivation(
                f"node_{i}", 
                activation_level=activation_level
            )
        
        result = PropagationResult(success=True, activated_nodes=activations)
        
        # Топ узлы должны быть с наибольшими номерами
        top_nodes = result.get_most_activated_nodes(10)
        assert len(top_nodes) == 10
        assert top_nodes[0][0] == "node_999"
        assert top_nodes[0][1] == 0.999
        
        # Фильтрация должна работать корректно
        active = result.filter_active_nodes(0.95)
        assert len(active) == 50  # Узлы 950-999
        
        active = result.filter_active_nodes(0.0)
        assert len(active) == 999  # Все кроме node_0 (0.0)