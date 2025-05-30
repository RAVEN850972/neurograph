# tests/test_propagation/conftest.py
"""
Конфигурация и фикстуры для тестирования модуля распространения активации.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from neurograph.propagation.base import (
    PropagationConfig, NodeActivation, ActivationFunction, 
    DecayFunction, PropagationMode
)
from neurograph.propagation.engine import SpreadingActivationEngine
from neurograph.propagation.visualizer import PropagationVisualizer
from neurograph.propagation.factory import PropagationFactory


# Mock классы для тестирования

class MockGraph:
    """Mock граф для тестирования."""
    
    def __init__(self, nodes: List[str] = None, edges: List[tuple] = None):
        self.nodes = {}
        self.edges = {}
        
        # Добавляем узлы
        if nodes:
            for node in nodes:
                self.add_node(node)
        
        # Добавляем ребра
        if edges:
            for edge in edges:
                if len(edge) == 2:
                    source, target = edge
                    self.add_edge(source, target)
                elif len(edge) == 3:
                    source, target, edge_type = edge
                    self.add_edge(source, target, edge_type)
                elif len(edge) == 4:
                    source, target, edge_type, weight = edge
                    self.add_edge(source, target, edge_type, weight=weight)
    
    def add_node(self, node_id: str, **attributes):
        """Добавление узла."""
        self.nodes[node_id] = attributes
    
    def add_edge(self, source: str, target: str, edge_type: str = "default", weight: float = 1.0, **attributes):
        """Добавление ребра."""
        edge_key = (source, target, edge_type)
        self.edges[edge_key] = {"weight": weight, "type": edge_type, **attributes}
    
    def has_node(self, node_id: str) -> bool:
        """Проверка существования узла."""
        return node_id in self.nodes
    
    def has_edge(self, source: str, target: str, edge_type: str = None) -> bool:
        """Проверка существования ребра."""
        if edge_type:
            return (source, target, edge_type) in self.edges
        else:
            return any((source, target, et) in self.edges for et in self._get_edge_types())
    
    def get_node(self, node_id: str) -> Dict[str, Any]:
        """Получение данных узла."""
        return self.nodes.get(node_id, {})
    
    def get_edge(self, source: str, target: str, edge_type: str = None) -> Dict[str, Any]:
        """Получение данных ребра."""
        if edge_type:
            return self.edges.get((source, target, edge_type))
        else:
            # Возвращаем первое найденное ребро
            for et in self._get_edge_types():
                edge_data = self.edges.get((source, target, et))
                if edge_data:
                    return edge_data
            return None
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Получение соседей узла."""
        neighbors = []
        for (source, target, _) in self.edges.keys():
            if source == node_id and target not in neighbors:
                neighbors.append(target)
        return neighbors
    
    def get_all_nodes(self) -> List[str]:
        """Получение всех узлов."""
        return list(self.nodes.keys())
    
    def get_all_edges(self) -> List[tuple]:
        """Получение всех ребер."""
        return [(source, target, edge_type) for source, target, edge_type in self.edges.keys()]
    
    def _get_edge_types(self) -> List[str]:
        """Получение всех типов ребер."""
        return list(set(edge_type for _, _, edge_type in self.edges.keys()))


class MockEncoder:
    """Mock энкодер для тестирования интеграции."""
    
    def __init__(self, vector_size: int = 384):
        self.vector_size = vector_size
    
    def encode(self, text: str) -> np.ndarray:
        """Создание фиктивного вектора."""
        # Используем хеш для воспроизводимости
        hash_value = hash(text) % 1000000
        np.random.seed(hash_value)
        return np.random.random(self.vector_size)


# Фикстуры для тестирования

@pytest.fixture
def simple_graph():
    """Простой граф для базовых тестов."""
    nodes = ["A", "B", "C", "D", "E"]
    edges = [
        ("A", "B", "default", 0.8),
        ("B", "C", "default", 0.7),
        ("C", "D", "default", 0.6),
        ("A", "E", "default", 0.5),
        ("E", "D", "default", 0.4)
    ]
    return MockGraph(nodes, edges)


@pytest.fixture
def complex_graph():
    """Сложный граф для продвинутых тестов."""
    nodes = [f"node_{i}" for i in range(10)]
    edges = []
    
    # Создаем связанный граф
    for i in range(9):
        edges.append((f"node_{i}", f"node_{i+1}", "sequential", 0.8))
    
    # Добавляем дополнительные связи
    edges.extend([
        ("node_0", "node_5", "shortcut", 0.6),
        ("node_2", "node_7", "cross", 0.5),
        ("node_3", "node_8", "cross", 0.4),
        ("node_1", "node_9", "longcut", 0.3)
    ])
    
    return MockGraph(nodes, edges)


@pytest.fixture
def hierarchical_graph():
    """Иерархический граф для тестирования глубины."""
    nodes = []
    edges = []
    
    # Создаем 3 уровня по 3 узла
    for level in range(3):
        for i in range(3):
            node_id = f"L{level}_N{i}"
            nodes.append(node_id)
            
            # Связи с предыдущим уровнем
            if level > 0:
                parent_id = f"L{level-1}_N{i}"
                edges.append((parent_id, node_id, "hierarchy", 0.9))
                
                # Дополнительные связи
                if i < 2:
                    sibling_parent = f"L{level-1}_N{i+1}"
                    edges.append((sibling_parent, node_id, "cross_hierarchy", 0.3))
    
    return MockGraph(nodes, edges)


@pytest.fixture
def default_config():
    """Конфигурация по умолчанию."""
    return PropagationConfig()


@pytest.fixture
def fast_config():
    """Быстрая конфигурация для тестов."""
    return PropagationConfig(
        max_iterations=10,
        convergence_threshold=0.1,
        activation_threshold=0.2,
        max_active_nodes=20,
        time_steps=3
    )


@pytest.fixture
def precise_config():
    """Точная конфигурация для тестов."""
    return PropagationConfig(
        max_iterations=200,
        convergence_threshold=0.0001,
        activation_threshold=0.05,
        max_active_nodes=100,
        lateral_inhibition=True,
        time_steps=20
    )


@pytest.fixture
def engine_with_simple_graph(simple_graph):
    """Движок с простым графом."""
    engine = SpreadingActivationEngine(simple_graph)
    return engine


@pytest.fixture
def engine_with_complex_graph(complex_graph):
    """Движок со сложным графом."""
    engine = SpreadingActivationEngine(complex_graph)
    return engine


@pytest.fixture
def visualizer():
    """Визуализатор для тестов."""
    return PropagationVisualizer()


@pytest.fixture
def mock_encoder():
    """Mock энкодер."""
    return MockEncoder()


@pytest.fixture
def sample_activations():
    """Примеры активаций для тестов."""
    return {
        "node_1": NodeActivation("node_1", activation_level=0.8, propagation_depth=0),
        "node_2": NodeActivation("node_2", activation_level=0.6, propagation_depth=1),
        "node_3": NodeActivation("node_3", activation_level=0.4, propagation_depth=1),
        "node_4": NodeActivation("node_4", activation_level=0.2, propagation_depth=2)
    }


@pytest.fixture
def initial_nodes_simple():
    """Простые начальные узлы."""
    return {"A": 1.0}


@pytest.fixture
def initial_nodes_multiple():
    """Множественные начальные узлы."""
    return {"A": 1.0, "C": 0.5}


@pytest.fixture
def initial_nodes_complex():
    """Сложные начальные узлы."""
    return {
        "node_0": 1.0,
        "node_5": 0.7,
        "node_9": 0.3
    }


# Фикстуры для различных режимов распространения

@pytest.fixture
def spreading_config():
    """Конфигурация для расходящегося распространения."""
    return PropagationConfig(
        propagation_mode=PropagationMode.SPREADING,
        max_iterations=50,
        activation_threshold=0.1
    )


@pytest.fixture
def focusing_config():
    """Конфигурация для сходящегося распространения."""
    return PropagationConfig(
        propagation_mode=PropagationMode.FOCUSING,
        max_iterations=50,
        activation_threshold=0.1
    )


@pytest.fixture
def bidirectional_config():
    """Конфигурация для двунаправленного распространения."""
    return PropagationConfig(
        propagation_mode=PropagationMode.BIDIRECTIONAL,
        max_iterations=50,
        activation_threshold=0.1
    )


@pytest.fixture
def constrained_config():
    """Конфигурация для ограниченного распространения."""
    return PropagationConfig(
        propagation_mode=PropagationMode.CONSTRAINED,
        max_iterations=30,
        activation_threshold=0.2
    )


# Фикстуры для различных функций активации

@pytest.fixture
def sigmoid_config():
    """Конфигурация с сигмоидальной активацией."""
    return PropagationConfig(
        activation_function=ActivationFunction.SIGMOID,
        activation_params={"steepness": 1.0}
    )


@pytest.fixture
def relu_config():
    """Конфигурация с ReLU активацией."""
    return PropagationConfig(
        activation_function=ActivationFunction.RELU,
        activation_params={"threshold": 0.1, "max_value": 1.0}
    )


@pytest.fixture
def threshold_config():
    """Конфигурация с пороговой активацией."""
    return PropagationConfig(
        activation_function=ActivationFunction.THRESHOLD,
        activation_params={"threshold": 0.5, "output_high": 1.0, "output_low": 0.0}
    )


# Фикстуры для функций затухания

@pytest.fixture
def exponential_decay_config():
    """Конфигурация с экспоненциальным затуханием."""
    return PropagationConfig(
        decay_function=DecayFunction.EXPONENTIAL,
        decay_params={"rate": 0.1}
    )


@pytest.fixture
def linear_decay_config():
    """Конфигурация с линейным затуханием."""
    return PropagationConfig(
        decay_function=DecayFunction.LINEAR,
        decay_params={"rate": 0.05}
    )


@pytest.fixture
def no_decay_config():
    """Конфигурация без затухания."""
    return PropagationConfig(
        decay_function=DecayFunction.NONE
    )


# Параметризованные фикстуры

@pytest.fixture(params=[
    ActivationFunction.SIGMOID,
    ActivationFunction.TANH,
    ActivationFunction.RELU,
    ActivationFunction.THRESHOLD
])
def activation_function(request):
    """Параметризованная фикстура функций активации."""
    return request.param


@pytest.fixture(params=[
    DecayFunction.EXPONENTIAL,
    DecayFunction.LINEAR,
    DecayFunction.POWER,
    DecayFunction.NONE
])
def decay_function(request):
    """Параметризованная фикстура функций затухания."""
    return request.param


@pytest.fixture(params=[
    PropagationMode.SPREADING,
    PropagationMode.FOCUSING,
    PropagationMode.BIDIRECTIONAL,
    PropagationMode.CONSTRAINED
])
def propagation_mode(request):
    """Параметризованная фикстура режимов распространения."""
    return request.param


# Утилитные фикстуры

@pytest.fixture
def temp_file_path(tmp_path):
    """Временный путь к файлу."""
    return tmp_path / "test_output.png"


@pytest.fixture
def temp_json_path(tmp_path):
    """Временный путь к JSON файлу."""
    return tmp_path / "test_data.json"


# Фикстуры для интеграционного тестирования

@pytest.fixture
def mock_memory():
    """Mock объект памяти."""
    memory = Mock()
    memory.add = Mock(return_value="item_123")
    memory.get = Mock(return_value=None)
    memory.size = Mock(return_value=0)
    memory.get_recent_items = Mock(return_value=[])
    return memory


@pytest.fixture
def mock_processor():
    """Mock объект процессора."""
    processor = Mock()
    processor.add_rule = Mock(return_value="rule_123")
    processor.derive = Mock()
    return processor


# Фикстуры для производительности

@pytest.fixture
def performance_graph():
    """Граф для тестирования производительности."""
    nodes = [f"perf_node_{i}" for i in range(100)]
    edges = []
    
    # Создаем плотно связанный граф
    for i in range(99):
        edges.append((f"perf_node_{i}", f"perf_node_{i+1}", "chain", 0.7))
        
        # Добавляем случайные связи
        if i % 5 == 0 and i < 95:
            edges.append((f"perf_node_{i}", f"perf_node_{i+5}", "skip", 0.5))
    
    return MockGraph(nodes, edges)


@pytest.fixture
def performance_config():
    """Конфигурация для тестов производительности."""
    return PropagationConfig(
        max_iterations=100,
        convergence_threshold=0.001,
        activation_threshold=0.1,
        max_active_nodes=50,
        lateral_inhibition=True,
        time_steps=10
    )


# Вспомогательные функции для тестов

def assert_valid_propagation_result(result, expected_success=True):
    """Проверка валидности результата распространения."""
    assert result is not None
    assert hasattr(result, 'success')
    assert hasattr(result, 'activated_nodes')
    assert hasattr(result, 'processing_time')
    assert hasattr(result, 'iterations_used')
    
    if expected_success:
        assert result.success is True
        assert isinstance(result.activated_nodes, dict)
        assert result.processing_time >= 0
        assert result.iterations_used >= 0
    
    return True


def assert_valid_node_activation(activation):
    """Проверка валидности активации узла."""
    assert isinstance(activation, NodeActivation)
    assert 0.0 <= activation.activation_level <= 1.0
    assert activation.propagation_depth >= 0
    assert isinstance(activation.source_nodes, set)
    return True


def create_test_config(**overrides):
    """Создание тестовой конфигурации с переопределениями."""
    default_params = {
        "max_iterations": 20,
        "convergence_threshold": 0.01,
        "activation_threshold": 0.2,
        "max_active_nodes": 20,
        "time_steps": 3
    }
    default_params.update(overrides)
    return PropagationConfig(**default_params)


# Маркеры для pytest

def pytest_configure(config):
    """Конфигурация pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "visualization: marks tests that require matplotlib"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance tests"
    )


# Настройки для пропуска тестов

def pytest_collection_modifyitems(config, items):
    """Модификация коллекции тестов."""
    
    # Проверяем доступность matplotlib для тестов визуализации
    try:
        import matplotlib
        matplotlib_available = True
    except ImportError:
        matplotlib_available = False
    
    # Добавляем маркеры пропуска
    skip_visualization = pytest.mark.skip(reason="matplotlib not available")
    
    for item in items:
        if "visualization" in item.keywords and not matplotlib_available:
            item.add_marker(skip_visualization)