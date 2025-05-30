# tests/test_propagation/test_functions.py
"""
Тесты для функций активации, затухания и латерального торможения.
"""

import pytest
import numpy as np
import math
from unittest.mock import Mock, patch

from neurograph.propagation.functions import (
    ActivationFunctions, DecayFunctions, LateralInhibitionProcessor,
    ActivationFunctionFactory, DecayFunctionFactory,
    normalize_activations, compute_activation_entropy,
    compute_activation_sparsity, find_activation_peaks,
    compute_convergence_metric
)
from neurograph.propagation.base import (
    ActivationFunction, DecayFunction, NodeActivation, PropagationConfig
)


class TestActivationFunctions:
    """Тесты для функций активации."""
    
    def test_linear_activation(self):
        """Тест линейной функции активации."""
        linear = ActivationFunctions.LinearActivation()
        
        # Базовое поведение
        assert linear.compute(0.5) == 0.5
        assert linear.compute(0.0) == 0.0
        assert linear.compute(1.0) == 1.0
        
        # С параметрами
        assert linear.compute(0.5, slope=2.0) == 1.0
        assert linear.compute(0.5, slope=0.5) == 0.25
        assert linear.compute(0.5, intercept=0.2) == 0.7
        
        # Границы
        assert linear.compute(2.0) == 1.0  # Ограничение сверху
        assert linear.compute(-1.0) == 0.0  # Ограничение снизу
        
        # Производная
        assert linear.derivative(0.5) == 1.0
        assert linear.derivative(0.5, slope=2.0) == 2.0
    
    def test_sigmoid_activation(self):
        """Тест сигмоидальной функции активации."""
        sigmoid = ActivationFunctions.SigmoidActivation()
        
        # Основные точки
        assert abs(sigmoid.compute(0.0) - 0.5) < 0.001
        assert sigmoid.compute(-100) < 0.001  # Близко к 0
        assert sigmoid.compute(100) > 0.999   # Близко к 1
        
        # Монотонность
        assert sigmoid.compute(-1) < sigmoid.compute(0) < sigmoid.compute(1)
        
        # С параметром steepness
        steep_result = sigmoid.compute(1.0, steepness=2.0)
        normal_result = sigmoid.compute(1.0, steepness=1.0)
        assert steep_result > normal_result
        
        # Производная
        derivative = sigmoid.derivative(0.0, steepness=1.0)
        assert abs(derivative - 0.25) < 0.001  # d/dx sigmoid(0) = 0.25
    
    def test_tanh_activation(self):
        """Тест гиперболического тангенса."""
        tanh_func = ActivationFunctions.TanhActivation()
        
        # Основные точки
        assert abs(tanh_func.compute(0.0) - 0.5) < 0.001  # Нормализованный tanh(0) = 0.5
        assert tanh_func.compute(-100) < 0.001
        assert tanh_func.compute(100) > 0.999
        
        # Монотонность
        assert tanh_func.compute(-1) < tanh_func.compute(0) < tanh_func.compute(1)
        
        # Производная
        derivative = tanh_func.derivative(0.0, steepness=1.0)
        assert derivative > 0
    
    def test_relu_activation(self):
        """Тест ReLU функции активации."""
        relu = ActivationFunctions.ReLUActivation()
        
        # Основное поведение
        assert relu.compute(0.5) == 0.5
        assert relu.compute(-0.5) == 0.0
        assert relu.compute(0.0) == 0.0
        assert relu.compute(1.5) == 1.0  # Ограничение max_value
        
        # С параметрами
        assert relu.compute(0.5, threshold=0.3) == 0.2  # 0.5 - 0.3
        assert relu.compute(0.2, threshold=0.3) == 0.0  # Ниже порога
        assert relu.compute(2.0, max_value=1.5) == 1.5
        
        # Производная
        assert relu.derivative(0.5) == 1.0
        assert relu.derivative(-0.5) == 0.0
        assert relu.derivative(0.0, threshold=0.1) == 0.0
    
    def test_threshold_activation(self):
        """Тест пороговой функции активации."""
        threshold = ActivationFunctions.ThresholdActivation()
        
        # Основное поведение
        assert threshold.compute(0.6) == 1.0  # Выше порога 0.5
        assert threshold.compute(0.4) == 0.0  # Ниже порога 0.5
        assert threshold.compute(0.5) == 1.0  # На пороге
        
        # С параметрами
        assert threshold.compute(0.3, threshold=0.2) == 1.0
        assert threshold.compute(0.1, threshold=0.2) == 0.0
        assert threshold.compute(0.6, output_high=0.8, output_low=0.2) == 0.8
        
        # Производная (должна быть 0 везде)
        assert threshold.derivative(0.6) == 0.0
        assert threshold.derivative(0.4) == 0.0
    
    def test_gaussian_activation(self):
        """Тест гауссовской функции активации."""
        gaussian = ActivationFunctions.GaussianActivation()
        
        # В центре должен быть максимум
        center_value = gaussian.compute(0.5, center=0.5, width=0.2, amplitude=1.0)
        off_center_value = gaussian.compute(0.8, center=0.5, width=0.2, amplitude=1.0)
        assert center_value > off_center_value
        
        # Амплитуда
        assert abs(gaussian.compute(0.5, center=0.5, width=0.2, amplitude=0.8) - 0.8) < 0.001
        
        # Ширина влияет на форму
        narrow_value = gaussian.compute(0.6, center=0.5, width=0.1, amplitude=1.0)
        wide_value = gaussian.compute(0.6, center=0.5, width=0.3, amplitude=1.0)
        assert narrow_value < wide_value
        
        # Производная
        derivative = gaussian.derivative(0.5, center=0.5, width=0.2)
        assert abs(derivative) < 0.001  # В центре производная близка к 0
    
    def test_softmax_activation(self):
        """Тест Softmax функции активации."""
        softmax = ActivationFunctions.SoftmaxActivation()
        
        # С набором значений
        all_values = [1.0, 2.0, 3.0]
        
        result1 = softmax.compute(1.0, all_values=all_values)
        result2 = softmax.compute(2.0, all_values=all_values)
        result3 = softmax.compute(3.0, all_values=all_values)
        
        # Результаты должны быть в порядке возрастания
        assert result1 < result2 < result3
        
        # Сумма должна быть близка к 1 (проверим приблизительно)
        total = result1 + result2 + result3
        assert abs(total - 1.0) < 0.1
        
        # С температурой
        high_temp_result = softmax.compute(1.0, all_values=all_values, temperature=2.0)
        normal_temp_result = softmax.compute(1.0, all_values=all_values, temperature=1.0)
        # При высокой температуре распределение более равномерное
        
        # Производная
        derivative = softmax.derivative(1.0, all_values=all_values)
        assert derivative >= 0


class TestDecayFunctions:
    """Тесты для функций затухания."""
    
    def test_exponential_decay(self):
        """Тест экспоненциального затухания."""
        exp_decay = DecayFunctions.ExponentialDecay()
        
        # Основное поведение
        result = exp_decay.compute(1.0, 1.0, rate=0.1)
        assert 0.9 < result < 1.0  # Должно уменьшиться
        
        # Большее время = больше затухания
        result1 = exp_decay.compute(1.0, 1.0, rate=0.1)
        result2 = exp_decay.compute(1.0, 2.0, rate=0.1)
        assert result2 < result1
        
        # Большая скорость = больше затухания
        result1 = exp_decay.compute(1.0, 1.0, rate=0.1)
        result2 = exp_decay.compute(1.0, 1.0, rate=0.2)
        assert result2 < result1
        
        # С периодом полураспада
        half_life_result = exp_decay.compute(1.0, 1.0, half_life=1.0)
        assert abs(half_life_result - 0.5) < 0.1
    
    def test_linear_decay(self):
        """Тест линейного затухания."""
        linear_decay = DecayFunctions.LinearDecay()
        
        # Основное поведение
        result = linear_decay.compute(1.0, 1.0, rate=0.1)
        assert abs(result - 0.9) < 0.001
        
        # Не может быть отрицательным
        result = linear_decay.compute(0.5, 10.0, rate=0.1)
        assert result == 0.0
        
        # С максимальным затуханием
        result = linear_decay.compute(1.0, 1.0, rate=0.2, max_decay=0.1)
        assert abs(result - 0.9) < 0.001
    
    def test_logarithmic_decay(self):
        """Тест логарифмического затухания."""
        log_decay = DecayFunctions.LogarithmicDecay()
        
        # Основное поведение
        result = log_decay.compute(1.0, 1.0, rate=0.1)
        assert 0.9 < result < 1.0
        
        # С нулевой активацией
        result = log_decay.compute(0.0, 1.0, rate=0.1)
        assert result == 0.0
        
        # С разными основаниями
        result_e = log_decay.compute(1.0, 1.0, rate=0.1, base=math.e)
        result_10 = log_decay.compute(1.0, 1.0, rate=0.1, base=10)
        # Результаты должны отличаться
    
    def test_power_decay(self):
        """Тест степенного затухания."""
        power_decay = DecayFunctions.PowerDecay()
        
        # Основное поведение
        result = power_decay.compute(1.0, 1.0, rate=0.1, power=2.0)
        assert 0.9 < result < 1.0
        
        # Разные степени
        result1 = power_decay.compute(1.0, 2.0, rate=0.1, power=1.0)
        result2 = power_decay.compute(1.0, 2.0, rate=0.1, power=2.0)
        assert result2 < result1  # Большая степень = больше затухания
    
    def test_step_decay(self):
        """Тест ступенчатого затухания."""
        step_decay = DecayFunctions.StepDecay()
        
        # Основное поведение
        result = step_decay.compute(1.0, 1.0, rate=0.1, step_size=1.0)
        assert abs(result - 0.9) < 0.001
        
        # Разные размеры шагов
        result1 = step_decay.compute(1.0, 2.5, rate=0.1, step_size=1.0)  # 2 шага
        result2 = step_decay.compute(1.0, 2.5, rate=0.1, step_size=0.5)  # 5 шагов
        assert result2 < result1
    
    def test_no_decay(self):
        """Тест отсутствия затухания."""
        no_decay = DecayFunctions.NoDecay()
        
        # Значение не должно изменяться
        assert no_decay.compute(1.0, 100.0) == 1.0
        assert no_decay.compute(0.5, 1000.0) == 0.5
        assert no_decay.compute(0.0, 1.0) == 0.0


class TestLateralInhibition:
    """Тесты для латерального торможения."""
    
    def test_lateral_inhibition_basic(self, simple_graph):
        """Тест базового латерального торможения."""
        inhibition = LateralInhibitionProcessor()
        
        # Создаем активации
        activations = {
            "A": NodeActivation("A", activation_level=0.9),
            "B": NodeActivation("B", activation_level=0.6),
            "C": NodeActivation("C", activation_level=0.3)
        }
        
        config = PropagationConfig(
            lateral_inhibition=True,
            inhibition_strength=0.2,
            inhibition_radius=1
        )
        
        result = inhibition.apply_inhibition(activations, simple_graph, config)
        
        # A должен затормозить соседей
        assert result["A"].activation_level == 0.9  # Не изменился
        assert result["B"].activation_level < 0.6   # Заторможен
        
        # Проверяем метаданные торможения
        assert "inhibition_sources" in result["B"].metadata
    
    def test_lateral_inhibition_disabled(self, simple_graph):
        """Тест отключенного латерального торможения."""
        inhibition = LateralInhibitionProcessor()
        
        activations = {
            "A": NodeActivation("A", activation_level=0.9),
            "B": NodeActivation("B", activation_level=0.6)
        }
        
        config = PropagationConfig(lateral_inhibition=False)
        
        result = inhibition.apply_inhibition(activations, simple_graph, config)
        
        # Активации не должны измениться
        assert result["A"].activation_level == 0.9
        assert result["B"].activation_level == 0.6
    
    def test_inhibition_radius(self, complex_graph):
        """Тест радиуса торможения."""
        inhibition = LateralInhibitionProcessor()
        
        activations = {
            "node_0": NodeActivation("node_0", activation_level=1.0),
            "node_1": NodeActivation("node_1", activation_level=0.5),
            "node_2": NodeActivation("node_2", activation_level=0.5),
            "node_3": NodeActivation("node_3", activation_level=0.5)
        }
        
        config = PropagationConfig(
            lateral_inhibition=True,
            inhibition_strength=0.3,
            inhibition_radius=2
        )
        
        result = inhibition.apply_inhibition(activations, complex_graph, config)
        
        # node_0 должен затормозить узлы в радиусе 2
        assert result["node_0"].activation_level == 1.0  # Источник не изменился
        assert result["node_1"].activation_level < 0.5   # Радиус 1
        assert result["node_2"].activation_level < 0.5   # Радиус 2
    
    def test_inhibition_strength(self, simple_graph):
        """Тест силы торможения."""
        inhibition = LateralInhibitionProcessor()
        
        activations = {
            "A": NodeActivation("A", activation_level=0.8),
            "B": NodeActivation("B", activation_level=0.5)
        }
        
        # Слабое торможение
        config_weak = PropagationConfig(
            lateral_inhibition=True,
            inhibition_strength=0.1,
            inhibition_radius=1
        )
        
        result_weak = inhibition.apply_inhibition(activations.copy(), simple_graph, config_weak)
        
        # Сильное торможение
        config_strong = PropagationConfig(
            lateral_inhibition=True,
            inhibition_strength=0.5,
            inhibition_radius=1
        )
        
        activations_copy = {
            "A": NodeActivation("A", activation_level=0.8),
            "B": NodeActivation("B", activation_level=0.5)
        }
        result_strong = inhibition.apply_inhibition(activations_copy, simple_graph, config_strong)
        
        # Сильное торможение должно больше уменьшить активацию
        assert result_strong["B"].activation_level < result_weak["B"].activation_level


class TestActivationFunctionFactory:
    """Тесты для фабрики функций активации."""
    
    def test_create_sigmoid(self):
        """Тест создания сигмоидальной функции."""
        func = ActivationFunctionFactory.create(ActivationFunction.SIGMOID)
        assert isinstance(func, ActivationFunctions.SigmoidActivation)
        
        # Проверяем, что функция работает
        result = func.compute(0.0)
        assert abs(result - 0.5) < 0.001
    
    def test_create_relu(self):
        """Тест создания ReLU функции."""
        func = ActivationFunctionFactory.create(ActivationFunction.RELU)
        assert isinstance(func, ActivationFunctions.ReLUActivation)
        
        assert func.compute(0.5) == 0.5
        assert func.compute(-0.5) == 0.0
    
    def test_create_all_functions(self):
        """Тест создания всех доступных функций."""
        available = ActivationFunctionFactory.get_available_functions()
        
        for func_type in available:
            func = ActivationFunctionFactory.create(func_type)
            assert func is not None
            
            # Проверяем базовую функциональность
            result = func.compute(0.5)
            assert isinstance(result, (int, float))
            assert 0.0 <= result <= 1.0
    
    def test_create_invalid_function(self):
        """Тест создания несуществующей функции."""
        with pytest.raises(ValueError, match="Неизвестный тип функции активации"):
            ActivationFunctionFactory.create("invalid_function")


class TestDecayFunctionFactory:
    """Тесты для фабрики функций затухания."""
    
    def test_create_exponential(self):
        """Тест создания экспоненциального затухания."""
        func = DecayFunctionFactory.create(DecayFunction.EXPONENTIAL)
        assert isinstance(func, DecayFunctions.ExponentialDecay)
        
        result = func.compute(1.0, 1.0, rate=0.1)
        assert 0.8 < result < 1.0
    
    def test_create_linear(self):
        """Тест создания линейного затухания."""
        func = DecayFunctionFactory.create(DecayFunction.LINEAR)
        assert isinstance(func, DecayFunctions.LinearDecay)
        
        result = func.compute(1.0, 1.0, rate=0.1)
        assert abs(result - 0.9) < 0.001
    
    def test_create_all_functions(self):
        """Тест создания всех доступных функций затухания."""
        available = DecayFunctionFactory.get_available_functions()
        
        for func_type in available:
            func = DecayFunctionFactory.create(func_type)
            assert func is not None
            
            # Проверяем базовую функциональность
            result = func.compute(1.0, 1.0)
            assert isinstance(result, (int, float))
            assert result >= 0.0  # Затухание не может давать отрицательные значения
    
    def test_create_invalid_function(self):
        """Тест создания несуществующей функции затухания."""
        with pytest.raises(ValueError, match="Неизвестный тип функции затухания"):
            DecayFunctionFactory.create("invalid_decay")


class TestUtilityFunctions:
    """Тесты для утилитных функций."""
    
    def test_normalize_activations_softmax(self):
        """Тест нормализации активаций методом softmax."""
        activations = {"A": 1.0, "B": 2.0, "C": 3.0}
        
        normalized = normalize_activations(activations, method="softmax")
        
        # Проверяем, что сумма равна 1
        total = sum(normalized.values())
        assert abs(total - 1.0) < 0.001
        
        # Проверяем порядок
        assert normalized["C"] > normalized["B"] > normalized["A"]
        
        # Все значения положительные
        assert all(val > 0 for val in normalized.values())
    
    def test_normalize_activations_min_max(self):
        """Тест нормализации min-max."""
        activations = {"A": 1.0, "B": 3.0, "C": 5.0}
        
        normalized = normalize_activations(activations, method="min_max")
        
        # Проверяем границы
        assert normalized["A"] == 0.0  # Минимум
        assert normalized["C"] == 1.0  # Максимум
        assert normalized["B"] == 0.5  # Середина
    
    def test_normalize_activations_z_score(self):
        """Тест нормализации z-score."""
        activations = {"A": 1.0, "B": 2.0, "C": 3.0}
        
        normalized = normalize_activations(activations, method="z_score")
        
        # Все значения должны быть между 0 и 1 (после sigmoid)
        assert all(0.0 <= val <= 1.0 for val in normalized.values())
    
    def test_normalize_activations_l2(self):
        """Тест L2 нормализации."""
        activations = {"A": 3.0, "B": 4.0}  # 3-4-5 треугольник
        
        normalized = normalize_activations(activations, method="l2")
        
        # Проверяем L2 норму
        l2_norm = sum(val**2 for val in normalized.values())**0.5
        assert abs(l2_norm - 1.0) < 0.001
    
    def test_normalize_activations_edge_cases(self):
        """Тест граничных случаев нормализации."""
        # Пустой словарь
        assert normalize_activations({}) == {}
        
        # Одинаковые значения
        activations = {"A": 5.0, "B": 5.0, "C": 5.0}
        normalized = normalize_activations(activations, method="min_max")
        assert all(val == 0.5 for val in normalized.values())
        
        # Нулевые значения
        activations = {"A": 0.0, "B": 0.0}
        normalized = normalize_activations(activations, method="l2")
        assert all(val == 0.0 for val in normalized.values())
    
    def test_normalize_activations_invalid_method(self):
        """Тест неверного метода нормализации."""
        activations = {"A": 1.0, "B": 2.0}
        
        with pytest.raises(ValueError, match="Неизвестный метод нормализации"):
            normalize_activations(activations, method="invalid_method")
    
    def test_compute_activation_entropy(self):
        """Тест вычисления энтропии активации."""
        # Равномерное распределение имеет максимальную энтропию
        uniform = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        entropy_uniform = compute_activation_entropy(uniform)
        
        # Неравномерное распределение имеет меньшую энтропию
        skewed = {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}
        entropy_skewed = compute_activation_entropy(skewed)
        
        assert entropy_uniform > entropy_skewed
        
        # Полностью детерминированное распределение имеет нулевую энтропию
        deterministic = {"A": 1.0, "B": 0.0, "C": 0.0}
        entropy_det = compute_activation_entropy(deterministic)
        assert entropy_det == 0.0
        
        # Пустое распределение
        assert compute_activation_entropy({}) == 0.0
        
        # Нулевое распределение
        zeros = {"A": 0.0, "B": 0.0}
        assert compute_activation_entropy(zeros) == 0.0
    
    def test_compute_activation_sparsity(self):
        """Тест вычисления разреженности активации."""
        # Плотная активация
        dense = {"A": 0.8, "B": 0.7, "C": 0.6, "D": 0.5}
        sparsity_dense = compute_activation_sparsity(dense, threshold=0.1)
        assert sparsity_dense == 0.0  # Все узлы активны
        
        # Разреженная активация
        sparse = {"A": 0.8, "B": 0.05, "C": 0.03, "D": 0.02}
        sparsity_sparse = compute_activation_sparsity(sparse, threshold=0.1)
        assert sparsity_sparse == 0.75  # 3 из 4 неактивны
        
        # Пустая активация
        assert compute_activation_sparsity({}) == 1.0
        
        # С разными порогами
        activations = {"A": 0.5, "B": 0.3, "C": 0.1}
        sparsity_low = compute_activation_sparsity(activations, threshold=0.05)
        sparsity_high = compute_activation_sparsity(activations, threshold=0.4)
        assert sparsity_high > sparsity_low
    
    def test_find_activation_peaks(self):
        """Тест поиска пиков активации."""
        # Активации с явными пиками
        activations = {
            "peak1": 0.9,
            "peak2": 0.8,
            "normal1": 0.3,
            "normal2": 0.2,
            "low": 0.05
        }
        
        peaks = find_activation_peaks(activations, min_prominence=0.1)
        
        # Должны найти пики
        assert "peak1" in peaks
        assert "peak2" in peaks
        assert "normal1" in peaks  # Выше среднего + prominence
        
        # Низкие значения не должны быть пиками
        assert "low" not in peaks
        
        # С высоким порогом
        high_peaks = find_activation_peaks(activations, min_prominence=0.7)
        assert len(high_peaks) <= 2  # Только самые высокие
        
        # Пустые активации
        assert find_activation_peaks({}) == []
    
    def test_compute_convergence_metric(self):
        """Тест вычисления метрики сходимости."""
        # История с убывающими изменениями (сходимость)
        history_converging = [
            {"A": 0.0, "B": 0.0},
            {"A": 0.8, "B": 0.5},
            {"A": 0.82, "B": 0.51},
            {"A": 0.821, "B": 0.511},
            {"A": 0.8211, "B": 0.5111}
        ]
        
        convergence = compute_convergence_metric(history_converging, window_size=3)
        assert convergence < 0.01  # Низкие изменения = сходимость
        
        # История с большими изменениями (нет сходимости)
        history_diverging = [
            {"A": 0.0, "B": 0.0},
            {"A": 0.5, "B": 0.3},
            {"A": 0.8, "B": 0.1},
            {"A": 0.2, "B": 0.9},
            {"A": 0.9, "B": 0.2}
        ]
        
        divergence = compute_convergence_metric(history_diverging, window_size=3)
        assert divergence > 0.1  # Большие изменения = расходимость
        
        # Недостаточно данных
        short_history = [{"A": 0.5}]
        assert compute_convergence_metric(short_history, window_size=3) == float('inf')
        
        # Пустая история
        assert compute_convergence_metric([], window_size=3) == float('inf')


class TestFunctionParametrization:
    """Параметризованные тесты функций."""
    
    @pytest.mark.parametrize("activation_func", [
        ActivationFunction.SIGMOID,
        ActivationFunction.TANH,
        ActivationFunction.RELU,
        ActivationFunction.LINEAR
    ])
    def test_activation_function_bounds(self, activation_func):
        """Тест границ для различных функций активации."""
        func = ActivationFunctionFactory.create(activation_func)
        
        # Все функции должны возвращать значения в разумных пределах
        for input_val in [-10, -1, 0, 0.5, 1, 10]:
            result = func.compute(input_val)
            assert isinstance(result, (int, float))
            assert not math.isnan(result)
            assert not math.isinf(result)
            
            # Большинство функций ограничены [0, 1]
            if activation_func != ActivationFunction.LINEAR:
                assert 0.0 <= result <= 1.0
    
    @pytest.mark.parametrize("decay_func", [
        DecayFunction.EXPONENTIAL,
        DecayFunction.LINEAR,
        DecayFunction.POWER,
        DecayFunction.NONE
    ])
    def test_decay_function_properties(self, decay_func):
        """Тест свойств различных функций затухания."""
        func = DecayFunctionFactory.create(decay_func)
        
        # Затухание не должно увеличивать активацию
        initial = 1.0
        result = func.compute(initial, 1.0)
        
        if decay_func != DecayFunction.NONE:
            assert result <= initial
        else:
            assert result == initial
        
        # Результат не должен быть отрицательным
        assert result >= 0.0
        
        # Не должно быть NaN или inf
        assert not math.isnan(result)
        assert not math.isinf(result)
    
    @pytest.mark.parametrize("method", ["softmax", "min_max", "z_score", "l2"])
    def test_normalization_methods(self, method):
        """Тест различных методов нормализации."""
        activations = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 0.5}
        
        normalized = normalize_activations(activations, method=method)
        
        # Основные проверки для всех методов
        assert len(normalized) == len(activations)
        assert all(key in normalized for key in activations.keys())
        assert all(isinstance(val, (int, float)) for val in normalized.values())
        assert all(not math.isnan(val) for val in normalized.values())
        assert all(not math.isinf(val) for val in normalized.values())
        
        # Специфичные проверки для методов
        if method == "softmax":
            assert abs(sum(normalized.values()) - 1.0) < 0.001
            assert all(val > 0 for val in normalized.values())
        elif method == "min_max":
            values = list(normalized.values())
            assert min(values) == 0.0
            assert max(values) == 1.0
        elif method == "l2":
            l2_norm = sum(val**2 for val in normalized.values())**0.5
            assert abs(l2_norm - 1.0) < 0.001


class TestEdgeCasesAndErrorHandling:
    """Тесты граничных случаев и обработки ошибок."""
    
    def test_activation_function_extreme_inputs(self):
        """Тест функций активации с экстремальными входами."""
        sigmoid = ActivationFunctions.SigmoidActivation()
        
        # Очень большие положительные значения
        result = sigmoid.compute(1000)
        assert result > 0.999
        assert not math.isinf(result)
        
        # Очень большие отрицательные значения
        result = sigmoid.compute(-1000)
        assert result < 0.001
        assert not math.isinf(result)
        
        # NaN входы не должны ломать функцию
        result = sigmoid.compute(float('nan'))
        # Результат может быть NaN, но не должно быть исключения
    
    def test_decay_function_edge_cases(self):
        """Тест функций затухания в граничных случаях."""
        exp_decay = DecayFunctions.ExponentialDecay()
        
        # Нулевая активация
        result = exp_decay.compute(0.0, 1.0, rate=0.1)
        assert result == 0.0
        
        # Нулевое время
        result = exp_decay.compute(1.0, 0.0, rate=0.1)
        assert result == 1.0
        
        # Очень большая скорость затухания
        result = exp_decay.compute(1.0, 1.0, rate=100.0)
        assert result >= 0.0  # Не должно быть отрицательным
        
        # Отрицательная скорость (некорректное использование)
        result = exp_decay.compute(1.0, 1.0, rate=-0.1)
        # Результат может быть больше 1, но не должно быть исключения
    
    def test_utility_functions_empty_inputs(self):
        """Тест утилитных функций с пустыми входами."""
        # Пустые активации
        assert normalize_activations({}) == {}
        assert compute_activation_entropy({}) == 0.0
        assert compute_activation_sparsity({}) == 1.0
        assert find_activation_peaks({}) == []
        
        # Пустая история
        assert compute_convergence_metric([]) == float('inf')
        assert compute_convergence_metric([{}]) == float('inf')
    
    def test_lateral_inhibition_edge_cases(self, simple_graph):
        """Тест латерального торможения в граничных случаях."""
        inhibition = LateralInhibitionProcessor()
        
        # Пустые активации
        result = inhibition.apply_inhibition({}, simple_graph, PropagationConfig())
        assert result == {}
        
        # Одна активация
        single_activation = {"A": NodeActivation("A", activation_level=0.5)}
        config = PropagationConfig(lateral_inhibition=True)
        result = inhibition.apply_inhibition(single_activation, simple_graph, config)
        assert result["A"].activation_level == 0.5  # Не должна измениться
        
        # Очень высокая сила торможения
        activations = {
            "A": NodeActivation("A", activation_level=1.0),
            "B": NodeActivation("B", activation_level=0.5)
        }
        config = PropagationConfig(
            lateral_inhibition=True,
            inhibition_strength=2.0  # Больше 1.0
        )
        result = inhibition.apply_inhibition(activations, simple_graph, config)
        assert result["B"].activation_level >= 0.0  # Не должна быть отрицательной