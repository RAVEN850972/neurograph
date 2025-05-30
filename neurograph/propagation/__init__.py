# neurograph/propagation/__init__.py
"""
Модуль распространения активации по графу знаний системы NeuroGraph.

Этот модуль предоставляет функциональность для распространения активации
по семантическому графу, имитируя процессы активации в нейронных сетях
и когнитивных системах.

Основные возможности:
- Различные алгоритмы распространения активации
- Настраиваемые функции активации и затухания
- Латеральное торможение между узлами
- Визуализация процесса распространения
- Анализ результатов и конвергенции

Примеры использования:
    >>> from neurograph.propagation import create_default_engine, create_default_config
    >>> 
    >>> # Создание движка и конфигурации
    >>> engine = create_default_engine(graph)
    >>> config = create_default_config()
    >>> 
    >>> # Запуск распространения
    >>> initial_nodes = {"concept_1": 1.0, "concept_2": 0.8}
    >>> result = engine.propagate(initial_nodes, config)
    >>> 
    >>> # Анализ результатов
    >>> print(f"Активировано узлов: {len(result.activated_nodes)}")
    >>> print(f"Сходимость: {result.convergence_achieved}")
"""

__version__ = "0.1.0"

# Базовые интерфейсы и классы данных
from .base import (
    # Основные интерфейсы
    IPropagationEngine,
    IActivationFunction,
    IDecayFunction,
    ILateralInhibition,
    IPropagationVisualizer,
    
    # Классы данных
    NodeActivation,
    PropagationConfig,
    PropagationResult,
    
    # Перечисления
    ActivationFunction,
    DecayFunction,
    PropagationMode,
    
    # Исключения
    PropagationError,
    GraphNotSetError,
    InvalidConfigurationError,
    ConvergenceError,
    NodeNotFoundError
)

# Основной движок распространения
from .engine import SpreadingActivationEngine

# Функции активации и затухания
from .functions import (
    ActivationFunctions,
    DecayFunctions,
    LateralInhibitionProcessor,
    ActivationFunctionFactory,
    DecayFunctionFactory,
    normalize_activations,
    compute_activation_entropy,
    compute_activation_sparsity,
    find_activation_peaks,
    compute_convergence_metric
)

# Визуализация
from .visualizer import PropagationVisualizer

# Фабрика и утилиты
from .factory import (
    PropagationFactory,
    ConfigurationPresets,
    PropagationConfigBuilder,
    PropagationDiagnostics,
    
    # Готовые функции создания
    create_default_engine,
    create_high_performance_engine,
    create_research_engine,
    create_default_config,
    create_fast_config,
    create_precise_config,
    create_experimental_config,
    
    # Утилитные функции
    quick_propagate,
    create_custom_config_from_dict,
    benchmark_config
)

# Публичный API для быстрого доступа
__all__ = [
    # Основные интерфейсы
    "IPropagationEngine",
    "IActivationFunction", 
    "IDecayFunction",
    "ILateralInhibition",
    "IPropagationVisualizer",
    
    # Классы данных
    "NodeActivation",
    "PropagationConfig", 
    "PropagationResult",
    
    # Перечисления
    "ActivationFunction",
    "DecayFunction",
    "PropagationMode",
    
    # Основные реализации
    "SpreadingActivationEngine",
    "PropagationVisualizer",
    
    # Фабрики
    "PropagationFactory",
    "ActivationFunctionFactory",
    "DecayFunctionFactory",
    
    # Утилиты
    "ConfigurationPresets",
    "PropagationConfigBuilder",
    "PropagationDiagnostics",
    
    # Быстрые создания
    "create_default_engine",
    "create_high_performance_engine", 
    "create_research_engine",
    "create_default_config",
    "create_fast_config",
    "create_precise_config",
    "create_experimental_config",
    
    # Утилитные функции
    "quick_propagate",
    "create_custom_config_from_dict",
    "benchmark_config",
    "normalize_activations",
    "compute_activation_entropy",
    "compute_activation_sparsity",
    "find_activation_peaks",
    "compute_convergence_metric",
    
    # Исключения
    "PropagationError",
    "GraphNotSetError", 
    "InvalidConfigurationError",
    "ConvergenceError",
    "NodeNotFoundError"
]


def get_module_info() -> dict:
    """Получение информации о модуле."""
    return {
        "name": "neurograph.propagation",
        "version": __version__,
        "description": "Модуль распространения активации по графу знаний",
        "author": __author__,
        "components": {
            "engines": PropagationFactory.get_available_engines(),
            "activation_functions": [func.value for func in ActivationFunction],
            "decay_functions": [func.value for func in DecayFunction],
            "propagation_modes": [mode.value for mode in PropagationMode]
        },
        "features": [
            "Множественные алгоритмы распространения активации",
            "Настраиваемые функции активации и затухания", 
            "Латеральное торможение между узлами",
            "Различные режимы распространения",
            "Визуализация процесса и результатов",
            "Анализ сходимости и производительности",
            "Готовые конфигурации для разных сценариев"
        ]
    }


def create_system_with_propagation(graph, config_preset: str = "balanced"):
    """
    Создание полной системы распространения активации.
    
    Args:
        graph: Граф знаний для распространения
        config_preset: Предустановка конфигурации ("fast", "balanced", "precise", "research")
        
    Returns:
        Tuple[IPropagationEngine, PropagationConfig, PropagationVisualizer]: 
        Кортеж с движком, конфигурацией и визуализатором
    """
    
    # Создание движка
    engine = create_default_engine(graph)
    
    # Создание конфигурации
    if config_preset == "fast":
        config = create_fast_config()
    elif config_preset == "precise":
        config = create_precise_config()
    elif config_preset == "research":
        config = create_experimental_config()
    else:  # balanced
        config = create_default_config()
    
    # Создание визуализатора
    visualizer = PropagationFactory.create_visualizer()
    
    return engine, config, visualizer


def propagate_and_visualize(graph, 
                          initial_nodes: dict, 
                          config_preset: str = "balanced",
                          save_path: str = None,
                          show_animation: bool = False):
    """
    Выполнение распространения активации с автоматической визуализацией.
    
    Args:
        graph: Граф знаний
        initial_nodes: Словарь начальных узлов {node_id: activation_level}
        config_preset: Предустановка конфигурации
        save_path: Путь для сохранения визуализации
        show_animation: Показать анимацию процесса
        
    Returns:
        PropagationResult: Результат распространения активации
    """
    
    engine, config, visualizer = create_system_with_propagation(graph, config_preset)
    
    # Выполнение распространения
    result = engine.propagate(initial_nodes, config)
    
    # Визуализация результатов
    if result.success:
        visualizer.visualize_propagation(
            result, graph, 
            save_path=save_path,
            show_animation=show_animation
        )
    else:
        print(f"Ошибка распространения: {result.error_message}")
    
    return result


# Интеграция с другими модулями NeuroGraph

def integrate_with_memory(propagation_result, memory_module, encoder=None):
    """
    Интеграция результатов распространения с модулем памяти.
    
    Args:
        propagation_result: Результат распространения активации
        memory_module: Экземпляр модуля Memory
        encoder: Энкодер для создания векторных представлений
    """
    
    if not propagation_result.success:
        return
    
    # Добавление активированных концептов в память
    for node_id, activation in propagation_result.activated_nodes.items():
        if activation.activation_level > 0.5:  # Высокоактивированные узлы
            
            # Создание содержимого для памяти
            content = f"Активированный концепт: {node_id}"
            
            # Создание векторного представления (если есть энкодер)
            if encoder:
                embedding = encoder.encode(content)
            else:
                import numpy as np
                embedding = np.random.random(384)  # Заглушка
            
            # Создание элемента памяти
            from neurograph.memory.base import MemoryItem
            memory_item = MemoryItem(
                content=content,
                embedding=embedding,
                content_type="activated_concept",
                metadata={
                    "activation_level": activation.activation_level,
                    "propagation_depth": activation.propagation_depth,
                    "source_nodes": list(activation.source_nodes),
                    "activation_source": "propagation"
                }
            )
            
            memory_module.add(memory_item)


def integrate_with_semgraph(propagation_result, graph):
    """
    Интеграция результатов распространения с семантическим графом.
    
    Args:
        propagation_result: Результат распространения активации
        graph: Экземпляр семантического графа
    """
    
    if not propagation_result.success:
        return
    
    # Добавление метаданных активации к узлам
    for node_id, activation in propagation_result.activated_nodes.items():
        if graph.has_node(node_id):
            # Обновление узла с информацией об активации
            current_data = graph.get_node(node_id) or {}
            current_data.update({
                "last_activation_level": activation.activation_level,
                "last_activation_time": activation.activation_time.isoformat(),
                "activation_count": current_data.get("activation_count", 0) + 1,
                "max_activation_reached": max(
                    current_data.get("max_activation_reached", 0),
                    activation.activation_level
                )
            })
            
            # Сохранение обновленной информации
            graph.add_node(node_id, **current_data)
    
    # Создание временных связей между сильно активированными узлами
    high_activation_nodes = [
        node_id for node_id, activation in propagation_result.activated_nodes.items()
        if activation.activation_level > 0.7
    ]
    
    for i, node1 in enumerate(high_activation_nodes):
        for node2 in high_activation_nodes[i+1:]:
            # Создание временной связи совместной активации
            edge_data = graph.get_edge(node1, node2) or {}
            edge_data["co_activation_count"] = edge_data.get("co_activation_count", 0) + 1
            edge_data["last_co_activation"] = propagation_result.activated_nodes[node1].activation_time.isoformat()
            
            # Если связи не было, создаем временную
            if not graph.has_edge(node1, node2):
                graph.add_edge(node1, node2, "co_activated", 
                             weight=0.1, temporary=True, **edge_data)
            else:
                # Обновляем существующую связь
                current_edge = graph.get_edge(node1, node2)
                current_edge.update(edge_data)
                graph.add_edge(node1, node2, current_edge.get("type", "related"), **current_edge)


def integrate_with_processor(propagation_result, processor_module):
    """
    Интеграция результатов распространения с модулем процессора.
    
    Args:
        propagation_result: Результат распространения активации
        processor_module: Экземпляр модуля Processor
    """
    
    if not propagation_result.success:
        return
    
    from neurograph.processor.base import ProcessingContext
    
    # Создание контекста на основе активированных узлов
    context = ProcessingContext()
    
    for node_id, activation in propagation_result.activated_nodes.items():
        # Добавление фактов об активации
        fact_key = f"activated_{node_id}"
        context.add_fact(fact_key, True, activation.activation_level)
        
        # Добавление фактов о связях между активированными узлами
        for source_node in activation.source_nodes:
            if source_node in propagation_result.activated_nodes:
                relation_fact = f"{source_node}_activated_with_{node_id}"
                context.add_fact(relation_fact, True, 
                                min(activation.activation_level, 
                                    propagation_result.activated_nodes[source_node].activation_level))
    
    # Добавление контекстной информации
    context.query_params.update({
        "propagation_session": True,
        "max_activation": propagation_result.max_activation_reached,
        "total_activation": propagation_result.total_activation,
        "convergence_achieved": propagation_result.convergence_achieved
    })
    
    return context


# Готовые сценарии использования

def scenario_concept_exploration(graph, start_concept: str, exploration_depth: int = 3):
    """
    Сценарий исследования концептов вокруг заданного понятия.
    
    Args:
        graph: Граф знаний
        start_concept: Начальный концепт для исследования
        exploration_depth: Глубина исследования
        
    Returns:
        Dict: Результаты исследования с рекомендациями
    """
    
    if not graph.has_node(start_concept):
        return {"error": f"Концепт {start_concept} не найден в графе"}
    
    # Конфигурация для исследования
    config = PropagationConfigBuilder().reset()\
        .set_propagation_mode(PropagationMode.SPREADING)\
        .set_depth_limit(exploration_depth)\
        .set_activation_limits(threshold=0.2, max_nodes=50)\
        .set_lateral_inhibition(False)\
        .build()
    
    # Создание движка и запуск
    engine = create_default_engine(graph)
    result = engine.propagate({start_concept: 1.0}, config)
    
    if not result.success:
        return {"error": f"Ошибка распространения: {result.error_message}"}
    
    # Анализ результатов
    related_concepts = []
    for node_id, activation in result.activated_nodes.items():
        if node_id != start_concept and activation.activation_level > 0.3:
            node_data = graph.get_node(node_id) or {}
            related_concepts.append({
                "concept": node_id,
                "relevance": activation.activation_level,
                "distance": activation.propagation_depth,
                "type": node_data.get("type", "unknown"),
                "description": node_data.get("description", "")
            })
    
    # Сортировка по релевантности
    related_concepts.sort(key=lambda x: x["relevance"], reverse=True)
    
    return {
        "start_concept": start_concept,
        "related_concepts": related_concepts[:20],  # Топ-20
        "total_explored": len(result.activated_nodes),
        "exploration_successful": result.success,
        "convergence_achieved": result.convergence_achieved,
        "processing_time": result.processing_time
    }


def scenario_knowledge_activation(graph, query_concepts: list, focus_mode: bool = True):
    """
    Сценарий активации знаний по набору концептов.
    
    Args:
        graph: Граф знаний
        query_concepts: Список концептов для активации
        focus_mode: Использовать фокусирующий режим (сходящееся распространение)
        
    Returns:
        Dict: Активированные знания и их приоритеты
    """
    
    # Проверка наличия концептов
    available_concepts = {concept: 1.0 / len(query_concepts) 
                         for concept in query_concepts 
                         if graph.has_node(concept)}
    
    if not available_concepts:
        return {"error": "Ни один из запрашиваемых концептов не найден в графе"}
    
    # Выбор режима распространения
    mode = PropagationMode.FOCUSING if focus_mode else PropagationMode.BIDIRECTIONAL
    
    # Конфигурация для активации знаний
    config = PropagationConfigBuilder().reset()\
        .set_propagation_mode(mode)\
        .set_activation_limits(threshold=0.15, max_nodes=100)\
        .set_lateral_inhibition(True, strength=0.3)\
        .set_iterations(150, convergence_threshold=0.001)\
        .build()
    
    # Выполнение активации
    engine = create_default_engine(graph)
    result = engine.propagate(available_concepts, config)
    
    if not result.success:
        return {"error": f"Ошибка активации: {result.error_message}"}
    
    # Категоризация активированных знаний
    activated_knowledge = {
        "primary": [],      # Основные активированные концепты
        "secondary": [],    # Вторичные концепты
        "connections": [],  # Обнаруженные связи
        "insights": []      # Потенциальные инсайты
    }
    
    for node_id, activation in result.activated_nodes.items():
        node_data = graph.get_node(node_id) or {}
        
        concept_info = {
            "concept": node_id,
            "activation_level": activation.activation_level,
            "depth": activation.propagation_depth,
            "type": node_data.get("type", "unknown"),
            "sources": list(activation.source_nodes)
        }
        
        if activation.activation_level > 0.6:
            activated_knowledge["primary"].append(concept_info)
        elif activation.activation_level > 0.3:
            activated_knowledge["secondary"].append(concept_info)
        
        # Поиск интересных связей
        if len(activation.source_nodes) >= 2:
            activated_knowledge["connections"].append({
                "target": node_id,
                "sources": list(activation.source_nodes),
                "strength": activation.activation_level
            })
        
        # Поиск потенциальных инсайтов (неожиданных активаций)
        if (activation.propagation_depth >= 3 and 
            activation.activation_level > 0.4 and
            node_id not in query_concepts):
            activated_knowledge["insights"].append(concept_info)
    
    # Сортировка результатов
    for category in ["primary", "secondary", "insights"]:
        activated_knowledge[category].sort(
            key=lambda x: x["activation_level"], reverse=True
        )
    
    return {
        "query_concepts": query_concepts,
        "available_concepts": list(available_concepts.keys()),
        "activated_knowledge": activated_knowledge,
        "statistics": {
            "total_activated": len(result.activated_nodes),
            "convergence_achieved": result.convergence_achieved,
            "processing_time": result.processing_time,
            "max_activation": result.max_activation_reached
        }
    }


def scenario_semantic_similarity(graph, concept1: str, concept2: str, max_depth: int = 5):
    """
    Сценарий оценки семантической близости между двумя концептами.
    
    Args:
        graph: Граф знаний
        concept1: Первый концепт
        concept2: Второй концепт
        max_depth: Максимальная глубина поиска связей
        
    Returns:
        Dict: Оценка семантической близости и путей связи
    """
    
    if not graph.has_node(concept1) or not graph.has_node(concept2):
        return {"error": "Один или оба концепта не найдены в графе"}
    
    # Конфигурация для двунаправленного распространения
    config = PropagationConfigBuilder().reset()\
        .set_propagation_mode(PropagationMode.BIDIRECTIONAL)\
        .set_depth_limit(max_depth)\
        .set_activation_limits(threshold=0.1, max_nodes=200)\
        .set_lateral_inhibition(False)\
        .build()
    
    engine = create_default_engine(graph)
    
    # Распространение от первого концепта
    result1 = engine.propagate({concept1: 1.0}, config)
    
    # Сброс и распространение от второго концепта
    engine.reset_activations()
    result2 = engine.propagate({concept2: 1.0}, config)
    
    if not (result1.success and result2.success):
        return {"error": "Ошибка при распространении активации"}
    
    # Анализ пересечений активаций
    intersection_nodes = set(result1.activated_nodes.keys()) & set(result2.activated_nodes.keys())
    intersection_nodes.discard(concept1)
    intersection_nodes.discard(concept2)
    
    # Вычисление метрик близости
    similarity_metrics = {}
    
    if intersection_nodes:
        # Семантическое пересечение
        intersection_activations = []
        common_concepts = []
        
        for node_id in intersection_nodes:
            act1 = result1.activated_nodes[node_id].activation_level
            act2 = result2.activated_nodes[node_id].activation_level
            combined_activation = (act1 + act2) / 2
            
            intersection_activations.append(combined_activation)
            common_concepts.append({
                "concept": node_id,
                "activation_from_concept1": act1,
                "activation_from_concept2": act2,
                "combined_activation": combined_activation,
                "depth1": result1.activated_nodes[node_id].propagation_depth,
                "depth2": result2.activated_nodes[node_id].propagation_depth
            })
        
        # Метрики близости
        similarity_metrics = {
            "jaccard_similarity": len(intersection_nodes) / 
                                len(set(result1.activated_nodes.keys()) | set(result2.activated_nodes.keys())),
            "activation_overlap": sum(intersection_activations) / len(intersection_activations),
            "concept_overlap_count": len(intersection_nodes),
            "avg_activation_strength": sum(intersection_activations) / len(intersection_activations),
            "min_connection_depth": min(
                min(c["depth1"], c["depth2"]) for c in common_concepts
            ) if common_concepts else float('inf')
        }
        
        # Сортировка общих концептов по релевантности
        common_concepts.sort(key=lambda x: x["combined_activation"], reverse=True)
    else:
        similarity_metrics = {
            "jaccard_similarity": 0.0,
            "activation_overlap": 0.0,
            "concept_overlap_count": 0,
            "avg_activation_strength": 0.0,
            "min_connection_depth": float('inf')
        }
        common_concepts = []
    
    # Оценка общей семантической близости
    overall_similarity = (
        similarity_metrics["jaccard_similarity"] * 0.3 +
        similarity_metrics["activation_overlap"] * 0.4 +
        (1.0 / (similarity_metrics["min_connection_depth"] + 1)) * 0.3
    )
    
    return {
        "concept1": concept1,
        "concept2": concept2,
        "overall_similarity": min(1.0, overall_similarity),
        "similarity_metrics": similarity_metrics,
        "common_concepts": common_concepts[:10],  # Топ-10
        "analysis": {
            "similarity_level": (
                "very_high" if overall_similarity > 0.8 else
                "high" if overall_similarity > 0.6 else
                "medium" if overall_similarity > 0.4 else
                "low" if overall_similarity > 0.2 else
                "very_low"
            ),
            "connection_strength": (
                "direct" if similarity_metrics["min_connection_depth"] <= 1 else
                "close" if similarity_metrics["min_connection_depth"] <= 2 else
                "distant" if similarity_metrics["min_connection_depth"] <= 4 else
                "very_distant"
            )
        }
    }


# Вспомогательные функции для отладки и анализа

def debug_propagation(graph, initial_nodes: dict, config: PropagationConfig = None):
    """
    Отладочная функция для детального анализа процесса распространения.
    
    Args:
        graph: Граф знаний
        initial_nodes: Начальные узлы
        config: Конфигурация (по умолчанию создается отладочная)
        
    Returns:
        Dict: Детальная информация о процессе распространения
    """
    
    if config is None:
        config = PropagationConfigBuilder().reset()\
            .set_performance_mode("balanced")\
            .set_iterations(50, convergence_threshold=0.01)\
            .build()
    
    # Анализ конфигурации
    diagnostics = PropagationDiagnostics()
    config_analysis = diagnostics.analyze_config(config)
    
    # Информация о графе
    graph_info = {
        "node_count": len(graph.get_all_nodes()),
        "edge_count": len(graph.get_all_edges()),
        "edge_types": list(set(
            edge_data.get("type", "default") 
            for _, _, edge_data in [(s, t, graph.get_edge(s, t)) for s, t, _ in graph.get_all_edges()]
            if edge_data
        )),
        "node_types": list(set(
            node_data.get("type", "unknown")
            for node_data in [graph.get_node(node_id) for node_id in graph.get_all_nodes()]
            if node_data
        ))
    }
    
    # Проверка совместимости
    compatibility_issues = diagnostics.validate_config_compatibility(config, graph_info)
    
    # Оценка времени выполнения
    time_estimate = diagnostics.estimate_execution_time(config, graph_info["node_count"])
    
    # Выполнение распространения
    engine = create_default_engine(graph)
    result = engine.propagate(initial_nodes, config)
    
    # Анализ результатов
    debug_info = {
        "input_analysis": {
            "initial_nodes": initial_nodes,
            "initial_nodes_count": len(initial_nodes),
            "graph_info": graph_info
        },
        "config_analysis": config_analysis,
        "compatibility_check": {
            "issues": compatibility_issues,
            "has_issues": len(compatibility_issues) > 0
        },
        "performance_estimates": time_estimate,
        "execution_results": {
            "success": result.success,
            "error_message": result.error_message,
            "processing_time": result.processing_time,
            "iterations_used": result.iterations_used,
            "convergence_achieved": result.convergence_achieved,
            "activated_nodes_count": len(result.activated_nodes),
            "max_activation_reached": result.max_activation_reached,
            "total_activation": result.total_activation
        },
        "activation_analysis": {
            "activation_distribution": {
                node_id: activation.activation_level
                for node_id, activation in list(result.activated_nodes.items())[:10]
            } if result.success else {},
            "depth_distribution": {},
            "source_analysis": {}
        }
    }
    
    # Дополнительный анализ при успешном выполнении
    if result.success and result.activated_nodes:
        # Распределение по глубине
        depth_counts = {}
        for activation in result.activated_nodes.values():
            depth = activation.propagation_depth
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        debug_info["activation_analysis"]["depth_distribution"] = depth_counts
        
        # Анализ источников активации
        source_stats = {}
        for activation in result.activated_nodes.values():
            source_count = len(activation.source_nodes)
            source_stats[source_count] = source_stats.get(source_count, 0) + 1
        debug_info["activation_analysis"]["source_analysis"] = source_stats
    
    return debug_info


# Экспорт всех компонентов для внешнего использования
def get_all_components():
    """Получение всех доступных компонентов модуля."""
    return {
        "engines": PropagationFactory.get_available_engines(),
        "activation_functions": [func.value for func in ActivationFunction],
        "decay_functions": [func.value for func in DecayFunction], 
        "propagation_modes": [mode.value for mode in PropagationMode],
        "config_presets": [
            "development", "production", "memory_efficient", "research"
        ],
        "scenarios": [
            "concept_exploration", "knowledge_activation", "semantic_similarity"
        ]
    }