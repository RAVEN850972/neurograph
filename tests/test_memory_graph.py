"""Тесты для реализации графа в памяти."""

import pytest

from neurograph.semgraph.impl.memory_graph import MemoryEfficientSemGraph


def test_add_node():
    """Тест добавления узла в граф."""
    graph = MemoryEfficientSemGraph()
    
    # Добавление узла с атрибутами
    graph.add_node("A", type="node", value=1)
    
    # Проверка, что узел был добавлен
    assert graph.has_node("A")
    
    # Проверка атрибутов узла
    node = graph.get_node("A")
    assert node is not None
    assert node["type"] == "node"
    assert node["value"] == 1


def test_add_edge():
    """Тест добавления ребра в граф."""
    graph = MemoryEfficientSemGraph()
    
    # Добавление ребра (узлы должны быть созданы автоматически)
    graph.add_edge("A", "B", "relation", weight=2.0, property="value")
    
    # Проверка, что узлы были созданы
    assert graph.has_node("A")
    assert graph.has_node("B")
    
    # Проверка, что ребро было создано
    assert graph.has_edge("A", "B", "relation")
    
    # Проверка атрибутов ребра
    edge = graph.get_edge("A", "B", "relation")
    assert edge is not None
    assert edge["weight"] == 2.0
    assert edge["property"] == "value"


def test_get_neighbors():
    """Тест получения соседей узла."""
    graph = MemoryEfficientSemGraph()
    
    # Создание графа
    graph.add_edge("A", "B", "relation1")
    graph.add_edge("A", "C", "relation1")
    graph.add_edge("A", "D", "relation2")
    
    # Получение всех соседей
    neighbors = graph.get_neighbors("A")
    assert set(neighbors) == {"B", "C", "D"}
    
    # Получение соседей с определенным типом ребра
    neighbors_rel1 = graph.get_neighbors("A", "relation1")
    assert set(neighbors_rel1) == {"B", "C"}
    
    neighbors_rel2 = graph.get_neighbors("A", "relation2")
    assert set(neighbors_rel2) == {"D"}


def test_update_edge_weight():
    """Тест обновления веса ребра."""
    graph = MemoryEfficientSemGraph()
    
    # Добавление ребра
    graph.add_edge("A", "B", "relation", weight=1.0)
    
    # Проверка начального веса
    weight = graph.get_edge_weight("A", "B", "relation")
    assert weight == 1.0
    
    # Обновление веса
    result = graph.update_edge_weight("A", "B", 2.0, "relation")
    assert result is True
    
    # Проверка обновленного веса
    weight = graph.get_edge_weight("A", "B", "relation")
    assert weight == 2.0
    
    # Попытка обновления несуществующего ребра
    result = graph.update_edge_weight("X", "Y", 1.0)
    assert result is False


def test_get_all_nodes_and_edges():
    """Тест получения всех узлов и ребер графа."""
    graph = MemoryEfficientSemGraph()
    
    # Создание графа
    graph.add_node("A", type="node")
    graph.add_node("B", type="node")
    graph.add_node("C", type="node")
    
    graph.add_edge("A", "B", "relation1")
    graph.add_edge("B", "C", "relation2")
    
    # Получение всех узлов
    nodes = graph.get_all_nodes()
    assert set(nodes) == {"A", "B", "C"}
    
    # Получение всех ребер
    edges = graph.get_all_edges()
    assert set((s, t, k) for s, t, k in edges) == {("A", "B", "relation1"), ("B", "C", "relation2")}