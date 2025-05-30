# neurograph/propagation/examples/basic_usage.py
"""
Примеры использования модуля распространения активации NeuroGraph.

Этот файл демонстрирует основные сценарии использования модуля propagation,
включая создание движков, конфигурацию параметров, выполнение распространения
и анализ результатов.
"""

import numpy as np
from typing import Dict, List
import time

# Импорты модуля распространения
from neurograph.propagation import (
    create_default_engine, create_default_config, create_fast_config,
    PropagationConfigBuilder, PropagationFactory, 
    ActivationFunction, DecayFunction, PropagationMode,
    quick_propagate, scenario_concept_exploration,
    scenario_knowledge_activation, scenario_semantic_similarity,
    propagate_and_visualize, debug_propagation
)

# Импорты других модулей для демонстрации интеграции
from neurograph.semgraph import SemGraphFactory


def example_basic_propagation():
    """Базовый пример распространения активации."""
    
    print("=== Базовый пример распространения активации ===")
    
    # Создание тестового графа
    graph = SemGraphFactory.create("memory_efficient")
    
    # Добавление узлов
    concepts = [
        ("python", {"type": "programming_language", "popularity": "high"}),
        ("programming", {"type": "activity", "domain": "computer_science"}),
        ("computer_science", {"type": "field", "level": "academic"}),
        ("artificial_intelligence", {"type": "field", "level": "research"}),
        ("machine_learning", {"type": "subdomain", "parent": "artificial_intelligence"}),
        ("neural_networks", {"type": "technique", "parent": "machine_learning"}),
        ("deep_learning", {"type": "technique", "parent": "neural_networks"}),
        ("data_science", {"type": "field", "level": "applied"}),
        ("statistics", {"type": "field", "level": "mathematical"}),
        ("mathematics", {"type": "field", "level": "fundamental"})
    ]
    
    for concept, attributes in concepts:
        graph.add_node(concept, **attributes)
    
    # Добавление связей
    relations = [
        ("python", "programming", "used_for", 0.9),
        ("programming", "computer_science", "part_of", 0.8),
        ("artificial_intelligence", "computer_science", "part_of", 0.9),
        ("machine_learning", "artificial_intelligence", "part_of", 0.9),
        ("neural_networks", "machine_learning", "technique_of", 0.8),
        ("deep_learning", "neural_networks", "extension_of", 0.9),
        ("machine_learning", "data_science", "used_in", 0.7),
        ("data_science", "statistics", "based_on", 0.8),
        ("statistics", "mathematics", "part_of", 0.9),
        ("python", "data_science", "used_for", 0.8),
        ("python", "machine_learning", "used_for", 0.9)
    ]
    
    for source, target, relation_type, weight in relations:
        graph.add_edge(source, target, relation_type, weight=weight)
    
    print(f"Создан граф с {len(graph.get_all_nodes())} узлами и {len(graph.get_all_edges())} связями")
    
    # Создание движка и конфигурации
    engine = create_default_engine(graph)
    config = create_default_config()
    
    # Начальные узлы для активации
    initial_nodes = {
        "python": 1.0,
        "machine_learning": 0.8
    }
    
    print(f"Начальные узлы: {initial_nodes}")
    
    # Выполнение распространения
    start_time = time.time()
    result = engine.propagate(initial_nodes, config)
    end_time = time.time()
    
    # Анализ результатов
    if result.success:
        print(f"✓ Распространение успешно завершено за {end_time - start_time:.3f}с")
        print(f"  Итераций: {result.iterations_used}")
        print(f"  Сходимость: {'Да' if result.convergence_achieved else 'Нет'}")
        print(f"  Активированных узлов: {len(result.activated_nodes)}")
        print(f"  Максимальная активация: {result.max_activation_reached:.3f}")
        
        print("\nТоп-5 активированных узлов:")
        top_nodes = result.get_most_activated_nodes(5)
        for node_id, activation_level in top_nodes:
            depth = result.activated_nodes[node_id].propagation_depth
            print(f"  {node_id}: {activation_level:.3f} (глубина: {depth})")
    else:
        print(f"✗ Ошибка распространения: {result.error_message}")
    
    return graph, result


def example_different_configurations():
    """Пример использования различных конфигураций."""
    
    print("\n=== Сравнение различных конфигураций ===")
    
    # Используем граф из предыдущего примера
    graph, _ = example_basic_propagation()
    
    initial_nodes = {"python": 1.0}
    
    # Различные конфигурации
    configs = {
        "Быстрая": create_fast_config(),
        "По умолчанию": create_default_config(),
        "Пользовательская": PropagationConfigBuilder().reset()
            .set_performance_mode("precise")
            .set_activation_function(ActivationFunction.TANH)
            .set_decay_function(DecayFunction.POWER)
            .set_lateral_inhibition(True, strength=0.3)
            .build()
    }
    
    engine = create_default_engine(graph)
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nТестирование конфигурации: {config_name}")
        
        engine.reset_activations()
        start_time = time.time()
        result = engine.propagate(initial_nodes, config)
        end_time = time.time()
        
        results[config_name] = {
            "result": result,
            "time": end_time - start_time
        }
        
        if result.success:
            print(f"  ✓ Время: {end_time - start_time:.3f}с")
            print(f"  ✓ Итераций: {result.iterations_used}")
            print(f"  ✓ Активированных узлов: {len(result.activated_nodes)}")
            print(f"  ✓ Сходимость: {'Да' if result.convergence_achieved else 'Нет'}")
        else:
            print(f"  ✗ Ошибка: {result.error_message}")
    
    # Сравнительная таблица
    print("\n--- Сравнительная таблица ---")
    print(f"{'Конфигурация':<15} {'Время (с)':<10} {'Итераций':<10} {'Узлов':<8} {'Сходимость'}")
    print("-" * 65)
    
    for config_name, data in results.items():
        result = data["result"]
        if result.success:
            convergence = "Да" if result.convergence_achieved else "Нет"
            print(f"{config_name:<15} {data['time']:<10.3f} {result.iterations_used:<10} "
                  f"{len(result.activated_nodes):<8} {convergence}")
        else:
            print(f"{config_name:<15} {'Ошибка':<10} {'-':<10} {'-':<8} {'-'}")


def example_propagation_modes():
    """Пример различных режимов распространения."""
    
    print("\n=== Различные режимы распространения ===")
    
    graph, _ = example_basic_propagation()
    initial_nodes = {"artificial_intelligence": 1.0}
    
    # Тестирование различных режимов
    modes = [
        PropagationMode.SPREADING,
        PropagationMode.FOCUSING, 
        PropagationMode.BIDIRECTIONAL,
        PropagationMode.CONSTRAINED
    ]
    
    engine = create_default_engine(graph)
    
    for mode in modes:
        print(f"\nРежим: {mode.value}")
        
        config = PropagationConfigBuilder().reset()\
            .set_propagation_mode(mode)\
            .set_activation_limits(threshold=0.15, max_nodes=50)\
            .build()
        
        engine.reset_activations()
        result = engine.propagate(initial_nodes, config)
        
        if result.success:
            print(f"  Активированных узлов: {len(result.activated_nodes)}")
            print(f"  Итераций: {result.iterations_used}")
            print(f"  Максимальная активация: {result.max_activation_reached:.3f}")
            
            # Показываем топ-3 узла
            top_nodes = result.get_most_activated_nodes(3)
            print("  Топ-3 узла:")
            for node_id, activation_level in top_nodes:
                if node_id != "artificial_intelligence":  # Исключаем начальный узел
                    print(f"    {node_id}: {activation_level:.3f}")
        else:
            print(f"  Ошибка: {result.error_message}")


def example_scenario_usage():
    """Пример использования готовых сценариев."""
    
    print("\n=== Использование готовых сценариев ===")
    
    graph, _ = example_basic_propagation()
    
    # Сценарий 1: Исследование концепта
    print("\n1. Исследование концепта 'machine_learning':")
    exploration_result = scenario_concept_exploration(
        graph, "machine_learning", exploration_depth=3
    )
    
    if "error" not in exploration_result:
        print(f"   Найдено связанных концептов: {len(exploration_result['related_concepts'])}")
        print("   Топ-3 связанных концепта:")
        for concept in exploration_result['related_concepts'][:3]:
            print(f"     {concept['concept']}: релевантность {concept['relevance']:.3f}, "
                  f"расстояние {concept['distance']}")
    else:
        print(f"   Ошибка: {exploration_result['error']}")
    
    # Сценарий 2: Активация знаний
    print("\n2. Активация знаний по запросу ['python', 'data_science']:")
    activation_result = scenario_knowledge_activation(
        graph, ["python", "data_science"], focus_mode=True
    )
    
    if "error" not in activation_result:
        knowledge = activation_result['activated_knowledge']
        print(f"   Основных концептов: {len(knowledge['primary'])}")
        print(f"   Вторичных концептов: {len(knowledge['secondary'])}")
        print(f"   Обнаруженных связей: {len(knowledge['connections'])}")
        print(f"   Потенциальных инсайтов: {len(knowledge['insights'])}")
        
        if knowledge['primary']:
            print("   Топ-3 основных концепта:")
            for concept in knowledge['primary'][:3]:
                print(f"     {concept['concept']}: активация {concept['activation_level']:.3f}")
    else:
        print(f"   Ошибка: {activation_result['error']}")
    
    # Сценарий 3: Семантическая близость
    print("\n3. Семантическая близость между 'python' и 'statistics':")
    similarity_result = scenario_semantic_similarity(
        graph, "python", "statistics", max_depth=4
    )
    
    if "error" not in similarity_result:
        print(f"   Общая близость: {similarity_result['overall_similarity']:.3f}")
        print(f"   Уровень близости: {similarity_result['analysis']['similarity_level']}")
        print(f"   Сила связи: {similarity_result['analysis']['connection_strength']}")
        print(f"   Общих концептов: {similarity_result['similarity_metrics']['concept_overlap_count']}")
        
        if similarity_result['common_concepts']:
            print("   Топ-3 общих концепта:")
            for concept in similarity_result['common_concepts'][:3]:
                print(f"     {concept['concept']}: комбинированная активация {concept['combined_activation']:.3f}")
    else:
        print(f"   Ошибка: {similarity_result['error']}")


def example_visualization():
    """Пример визуализации результатов распространения."""
    
    print("\n=== Визуализация результатов ===")
    
    graph, _ = example_basic_propagation()
    
    # Выполняем распространение с визуализацией
    initial_nodes = {"computer_science": 1.0}
    
    print("Выполнение распространения с автоматической визуализацией...")
    
    try:
        # Попытка создать визуализацию (может не работать в некоторых средах)
        result = propagate_and_visualize(
            graph, 
            initial_nodes, 
            config_preset="balanced",
            save_path="propagation_result.png",
            show_animation=False
        )
        
        if result.success:
            print("✓ Визуализация создана и сохранена в 'propagation_result.png'")
        else:
            print(f"✗ Ошибка при создании визуализации: {result.error_message}")
    
    except Exception as e:
        print(f"Визуализация недоступна (возможно, отсутствует matplotlib): {e}")
        
        # Альтернативный текстовый вывод результатов
        engine = create_default_engine(graph)
        config = create_default_config()
        result = engine.propagate(initial_nodes, config)
        
        if result.success:
            print("\nТекстовое представление результатов:")
            print("Активированные узлы по уровням:")
            
            # Группировка по глубине
            by_depth = {}
            for node_id, activation in result.activated_nodes.items():
                depth = activation.propagation_depth
                if depth not in by_depth:
                    by_depth[depth] = []
                by_depth[depth].append((node_id, activation.activation_level))
            
            for depth in sorted(by_depth.keys()):
                print(f"  Глубина {depth}:")
                nodes_at_depth = sorted(by_depth[depth], key=lambda x: x[1], reverse=True)
                for node_id, activation_level in nodes_at_depth:
                    print(f"    {node_id}: {activation_level:.3f}")


def example_integration_with_other_modules():
    """Пример интеграции с другими модулями NeuroGraph."""
    
    print("\n=== Интеграция с другими модулями ===")
    
    graph, _ = example_basic_propagation()
    
    # Выполняем распространение
    engine = create_default_engine(graph)
    config = create_default_config()
    initial_nodes = {"python": 1.0, "data_science": 0.7}
    
    result = engine.propagate(initial_nodes, config)
    
    if not result.success:
        print(f"Ошибка распространения: {result.error_message}")
        return
    
    print("Результат распространения получен, демонстрируем интеграцию:")
    
    # Интеграция с SemGraph (добавление метаданных активации)
    print("\n1. Интеграция с SemGraph:")
    
    from neurograph.propagation import integrate_with_semgraph
    integrate_with_semgraph(result, graph)
    
    # Проверяем добавленные метаданные
    updated_nodes = 0
    for node_id in result.activated_nodes.keys():
        node_data = graph.get_node(node_id)
        if node_data and "last_activation_level" in node_data:
            updated_nodes += 1
    
    print(f"   Обновлено узлов с метаданными активации: {updated_nodes}")
    
    # Интеграция с Memory (если доступен модуль)
    print("\n2. Интеграция с Memory:")
    
    try:
        from neurograph.memory import create_default_biomorphic_memory
        from neurograph.propagation import integrate_with_memory
        
        memory = create_default_biomorphic_memory()
        
        # Создаем простой энкодер-заглушку
        class MockEncoder:
            def encode(self, text):
                return np.random.random(384)
        
        encoder = MockEncoder()
        integrate_with_memory(result, memory, encoder)
        
        print(f"   Добавлено элементов в память: {memory.size()}")
        
        # Показываем несколько элементов из памяти
        recent_items = memory.get_recent_items(hours=1.0)
        print(f"   Недавних элементов: {len(recent_items)}")
        
        for item in recent_items[:3]:
            print(f"     {item.content} (тип: {item.content_type})")
    
    except ImportError:
        print("   Модуль Memory недоступен для демонстрации")
    
    # Интеграция с Processor (если доступен модуль)
    print("\n3. Интеграция с Processor:")
    
    try:
        from neurograph.propagation import integrate_with_processor
        
        # Создаем mock процессор
        class MockProcessor:
            def __init__(self):
                self.contexts = []
            
            def process_context(self, context):
                self.contexts.append(context)
                return context
        
        processor = MockProcessor()
        context = integrate_with_processor(result, processor)
        
        print(f"   Создан контекст с {len(context.facts)} фактами")
        print("   Примеры фактов активации:")
        
        fact_examples = list(context.facts.items())[:3]
        for fact_key, fact_data in fact_examples:
            print(f"     {fact_key}: {fact_data}")
    
    except ImportError:
        print("   Модуль Processor недоступен для демонстрации")


def example_performance_analysis():
    """Пример анализа производительности и диагностики."""
    
    print("\n=== Анализ производительности и диагностика ===")
    
    graph, _ = example_basic_propagation()
    
    # Создаем различные конфигурации для тестирования
    test_configs = {
        "fast": PropagationConfigBuilder().reset().set_performance_mode("fast").build(),
        "balanced": PropagationConfigBuilder().reset().set_performance_mode("balanced").build(),
        "precise": PropagationConfigBuilder().reset().set_performance_mode("precise").build()
    }
    
    initial_nodes = {"computer_science": 1.0}
    
    print("Бенчмарк различных конфигураций:")
    
    from neurograph.propagation import benchmark_config
    
    for config_name, config in test_configs.items():
        print(f"\nТестирование конфигурации: {config_name}")
        
        benchmark_result = benchmark_config(config, graph, initial_nodes, runs=3)
        
        if benchmark_result["success_rate"] > 0:
            print(f"  Успешных запусков: {benchmark_result['successful_runs']}/{benchmark_result['total_runs']}")
            print(f"  Среднее время: {benchmark_result['avg_processing_time']:.3f}с")
            print(f"  Средние итерации: {benchmark_result['avg_iterations']:.1f}")
            print(f"  Частота сходимости: {benchmark_result['convergence_rate']:.1%}")
            print(f"  Среднее активированных узлов: {benchmark_result['avg_activated_nodes']:.1f}")
        else:
            print(f"  Все запуски завершились неудачей")
    
    # Диагностический анализ
    print("\nДиагностический анализ конфигурации:")
    
    debug_result = debug_propagation(graph, initial_nodes)
    
    print(f"  Граф: {debug_result['input_analysis']['graph_info']['node_count']} узлов, "
          f"{debug_result['input_analysis']['graph_info']['edge_count']} связей")
    
    config_analysis = debug_result['config_analysis']
    print(f"  Оценка производительности: {config_analysis['performance_rating']}")
    print(f"  Использование памяти: {config_analysis['memory_usage_estimate']}")
    print(f"  Вероятность сходимости: {config_analysis['convergence_likelihood']}")
    
    if config_analysis['warnings']:
        print("  Предупреждения:")
        for warning in config_analysis['warnings']:
            print(f"    - {warning}")
    
    if config_analysis['recommendations']:
        print("  Рекомендации:")
        for recommendation in config_analysis['recommendations']:
            print(f"    - {recommendation}")


def example_custom_activation_functions():
    """Пример создания и использования пользовательских функций активации."""
    
    print("\n=== Пользовательские функции активации ===")
    
    graph, _ = example_basic_propagation()
    
    # Тестирование различных функций активации
    activation_functions = [
        ActivationFunction.SIGMOID,
        ActivationFunction.TANH,
        ActivationFunction.RELU,
        ActivationFunction.THRESHOLD,
        ActivationFunction.GAUSSIAN
    ]
    
    initial_nodes = {"machine_learning": 1.0}
    
    print("Сравнение функций активации:")
    
    for func in activation_functions:
        print(f"\nФункция активации: {func.value}")
        
        config = PropagationConfigBuilder().reset()\
            .set_activation_function(func)\
            .set_performance_mode("balanced")\
            .build()
        
        if func == ActivationFunction.THRESHOLD:
            # Специальные параметры для пороговой функции
            config.activation_params = {"threshold": 0.5, "output_high": 1.0, "output_low": 0.0}
        elif func == ActivationFunction.GAUSSIAN:
            # Специальные параметры для гауссовской функции
            config.activation_params = {"center": 0.5, "width": 0.3, "amplitude": 1.0}
        elif func == ActivationFunction.RELU:
            # Специальные параметры для ReLU
            config.activation_params = {"threshold": 0.1, "max_value": 1.0}
        
        engine = create_default_engine(graph)
        result = engine.propagate(initial_nodes, config)
        
        if result.success:
            active_nodes = len([a for a in result.activated_nodes.values() if a.activation_level > 0.1])
            max_activation = result.max_activation_reached
            
            print(f"  Активных узлов: {active_nodes}")
            print(f"  Максимальная активация: {max_activation:.3f}")
            print(f"  Итераций: {result.iterations_used}")
            print(f"  Сходимость: {'Да' if result.convergence_achieved else 'Нет'}")
        else:
            print(f"  Ошибка: {result.error_message}")


def main():
    """Главная функция для запуска всех примеров."""
    
    print("🧠 Примеры использования модуля Propagation NeuroGraph")
    print("=" * 60)
    
    try:
        # Базовые примеры
        example_basic_propagation()
        example_different_configurations()
        example_propagation_modes()
        
        # Готовые сценарии
        example_scenario_usage()
        
        # Визуализация (может не работать в некоторых средах)
        example_visualization()
        
        # Интеграция с другими модулями
        example_integration_with_other_modules()
        
        # Анализ производительности
        example_performance_analysis()
        
        # Пользовательские функции
        example_custom_activation_functions()
        
        print("\n" + "=" * 60)
        print("✅ Все примеры выполнены успешно!")
        print("\nДополнительные возможности:")
        print("- Создание анимированных визуализаций")
        print("- Интеграция с внешними векторными моделями")
        print("- Пользовательские функции затухания")
        print("- Экспорт результатов в различные форматы")
        print("- Подключение к реальным базам знаний")
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении примеров: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# Дополнительные утилитные функции для экспериментов

def create_larger_test_graph(size: int = 50):
    """Создание большего тестового графа для экспериментов."""
    
    graph = SemGraphFactory.create("memory_efficient")
    
    # Создание узлов
    domains = ["science", "technology", "art", "philosophy", "mathematics"]
    
    node_id = 0
    for domain in domains:
        # Корневой узел домена
        root_id = f"{domain}_root"
        graph.add_node(root_id, type="domain", level=0, domain=domain)
        
        # Подузлы домена
        for level in range(1, 4):  # 3 уровня глубины
            level_size = max(1, size // (len(domains) * level))
            
            for i in range(level_size):
                node_name = f"{domain}_l{level}_n{i}"
                graph.add_node(node_name, type="concept", level=level, domain=domain)
                
                # Связи с предыдущим уровнем
                if level == 1:
                    graph.add_edge(root_id, node_name, "contains", weight=0.8)
                else:
                    # Связь с случайным узлом предыдущего уровня
                    prev_level_nodes = [
                        n for n in graph.get_all_nodes() 
                        if n.startswith(f"{domain}_l{level-1}")
                    ]
                    if prev_level_nodes:
                        parent = np.random.choice(prev_level_nodes)
                        graph.add_edge(parent, node_name, "contains", weight=0.7)
                
                # Случайные связи между узлами одного уровня
                if np.random.random() < 0.3:
                    same_level_nodes = [
                        n for n in graph.get_all_nodes() 
                        if n.startswith(f"{domain}_l{level}") and n != node_name
                    ]
                    if same_level_nodes:
                        sibling = np.random.choice(same_level_nodes)
                        graph.add_edge(node_name, sibling, "related_to", weight=0.5)
        
        # Междоменные связи
        if node_id > 0:
            prev_domain = domains[node_id - 1]
            weight = 0.3 + np.random.random() * 0.4
            graph.add_edge(f"{prev_domain}_root", f"{domain}_root", "influences", weight=weight)
        
        node_id += 1
    
    print(f"Создан большой граф с {len(graph.get_all_nodes())} узлами и {len(graph.get_all_edges())} связями")
    return graph


def experiment_with_large_graph():
    """Эксперимент с большим графом."""
    
    print("\n=== Эксперимент с большим графом ===")
    
    # Создание большого графа
    large_graph = create_larger_test_graph(100)
    
    # Тестирование производительности
    initial_nodes = {"science_root": 1.0, "technology_root": 0.8}
    
    configs_to_test = {
        "fast": create_fast_config(),
        "default": create_default_config(),
        "precise": create_precise_config()
    }
    
    engine = create_default_engine(large_graph)
    
    for config_name, config in configs_to_test.items():
        print(f"\nТестирование конфигурации '{config_name}' на большом графе:")
        
        start_time = time.time()
        result = engine.propagate(initial_nodes, config)
        end_time = time.time()
        
        if result.success:
            print(f"  Время выполнения: {end_time - start_time:.3f}с")
            print(f"  Итераций: {result.iterations_used}")
            print(f"  Активированных узлов: {len(result.activated_nodes)}")
            print(f"  Процент от общего: {len(result.activated_nodes) / len(large_graph.get_all_nodes()) * 100:.1f}%")
            print(f"  Сходимость: {'Да' if result.convergence_achieved else 'Нет'}")
        else:
            print(f"  Ошибка: {result.error_message}")
        
        engine.reset_activations()


def run_extended_examples():
    """Запуск расширенных примеров."""
    
    print("\n🔬 Расширенные примеры и эксперименты")
    print("=" * 50)
    
    try:
        experiment_with_large_graph()
        
        print("\n✅ Расширенные примеры выполнены!")
        
    except Exception as e:
        print(f"\n❌ Ошибка в расширенных примерах: {e}")


# Точка входа для расширенных примеров
if __name__ == "__main__":
    main()
    run_extended_examples()