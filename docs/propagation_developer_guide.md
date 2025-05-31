# Модуль Propagation - Документация для разработчиков

## Обзор

Модуль `neurograph.propagation` реализует механизмы распространения активации по семантическому графу, имитируя процессы активации в нейронных сетях и когнитивных системах. Это один из самых сложных и мощных модулей системы NeuroGraph.

### Ключевые особенности

- **Множественные алгоритмы распространения** (spreading, focusing, bidirectional, constrained)
- **Настраиваемые функции активации и затухания** с богатым набором параметров
- **Латеральное торможение** между соседними узлами для конкуренции
- **Визуализация процесса** распространения в реальном времени
- **Анализ сходимости и производительности** с детальной диагностикой
- **Готовые сценарии** для типичных задач когнитивного моделирования

## Быстрый старт

```python
from neurograph.propagation import (
    create_default_engine, create_default_config,
    quick_propagate, scenario_concept_exploration
)
from neurograph.semgraph import SemGraphFactory

# Создание графа знаний
graph = SemGraphFactory.create("memory_efficient")
graph.add_node("python", type="language")
graph.add_node("programming", type="activity")
graph.add_edge("python", "programming", "used_for", weight=0.9)

# Простое распространение активации
result = quick_propagate(graph, {"python": 1.0}, mode="balanced")

# Готовый сценарий исследования концепта
exploration = scenario_concept_exploration(graph, "python", exploration_depth=3)
```

## Архитектура модуля

### Структура директории

```
propagation/
├── __init__.py          # Основной API и готовые функции
├── base.py              # Базовые интерфейсы и структуры данных
├── engine.py            # Основной движок распространения
├── functions.py         # Функции активации, затухания и торможения
├── factory.py           # Фабрики и утилиты создания компонентов
├── visualizer.py        # Визуализация процесса распространения
└── examples/            # Примеры использования
    └── basic_usage.py      # Базовые примеры
```

### Диаграмма компонентов

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  SemGraph       │───▶│ PropagationEngine│───▶│ PropagationResult│
│  (граф знаний)  │    │  (движок)        │    │  (результаты)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ PropagationConfig│
                       │ (конфигурация)   │
                       └──────────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
            ┌─────────────┐ ┌─────────┐ ┌─────────────┐
            │Activation   │ │ Decay   │ │ Lateral     │
            │Functions    │ │Functions│ │ Inhibition  │
            └─────────────┘ └─────────┘ └─────────────┘
```

## Основные компоненты

### 1. Базовые структуры данных (base.py)

#### NodeActivation

Представляет состояние активации узла:

```python
from neurograph.propagation.base import NodeActivation
from datetime import datetime

# Создание активации узла
activation = NodeActivation(
    node_id="python",
    activation_level=0.8,
    propagation_depth=1,
    metadata={"source": "user_input"}
)

# Обновление активации
activation.update_activation(0.9, source_node="programming")

# Проверка активности
if activation.is_active(threshold=0.5):
    print(f"Узел {activation.node_id} активен: {activation.activation_level}")
```

#### PropagationConfig

Конфигурация параметров распространения:

```python
from neurograph.propagation.base import (
    PropagationConfig, ActivationFunction, DecayFunction, PropagationMode
)

# Базовая конфигурация
config = PropagationConfig(
    max_iterations=100,
    activation_threshold=0.1,
    activation_function=ActivationFunction.SIGMOID,
    propagation_mode=PropagationMode.SPREADING
)

# Валидация конфигурации
is_valid, errors = config.validate()
if not is_valid:
    print("Ошибки конфигурации:", errors)
```

#### PropagationResult

Результат распространения активации:

```python
# Анализ результатов
if result.success:
    print(f"Активировано узлов: {len(result.activated_nodes)}")
    print(f"Итераций: {result.iterations_used}")
    print(f"Сходимость: {result.convergence_achieved}")
    
    # Наиболее активированные узлы
    top_nodes = result.get_most_activated_nodes(5)
    for node_id, activation_level in top_nodes:
        print(f"{node_id}: {activation_level:.3f}")
    
    # Фильтрация активных узлов
    active_nodes = result.filter_active_nodes(threshold=0.3)
```

### 2. Движок распространения (engine.py)

#### SpreadingActivationEngine

Основной движок для распространения активации:

```python
from neurograph.propagation.engine import SpreadingActivationEngine
from neurograph.propagation import create_default_config

# Создание движка
engine = SpreadingActivationEngine(graph)

# Выполнение распространения
initial_nodes = {"python": 1.0, "ai": 0.8}
config = create_default_config()

result = engine.propagate(initial_nodes, config)

# Получение текущего состояния
current_activations = {
    node_id: engine.get_node_activation(node_id)
    for node_id in ["python", "programming", "ai"]
    if engine.get_node_activation(node_id)
}

# Статистика работы движка
stats = engine.get_statistics()
print(f"Общее время работы: {stats['total_processing_time']:.3f}с")
print(f"Успешных распространений: {stats['successful_propagations']}")
```

### 3. Функции активации и затухания (functions.py)

#### Функции активации

```python
from neurograph.propagation.functions import (
    ActivationFunctions, ActivationFunctionFactory
)
from neurograph.propagation.base import ActivationFunction

# Создание функций активации
sigmoid = ActivationFunctionFactory.create(ActivationFunction.SIGMOID)
tanh = ActivationFunctionFactory.create(ActivationFunction.TANH)
relu = ActivationFunctionFactory.create(ActivationFunction.RELU)

# Вычисление активации
input_signal = 0.7
sigmoid_output = sigmoid.compute(input_signal, steepness=2.0)
tanh_output = tanh.compute(input_signal, steepness=1.5)
relu_output = relu.compute(input_signal, threshold=0.2)

# Специальные функции
gaussian = ActivationFunctionFactory.create(ActivationFunction.GAUSSIAN)
gaussian_output = gaussian.compute(input_signal, center=0.5, width=0.3)

threshold = ActivationFunctionFactory.create(ActivationFunction.THRESHOLD)
threshold_output = threshold.compute(input_signal, threshold=0.6)
```

#### Функции затухания

```python
from neurograph.propagation.functions import DecayFunctionFactory
from neurograph.propagation.base import DecayFunction

# Различные типы затухания
exponential = DecayFunctionFactory.create(DecayFunction.EXPONENTIAL)
linear = DecayFunctionFactory.create(DecayFunction.LINEAR)
power = DecayFunctionFactory.create(DecayFunction.POWER)

# Применение затухания
current_activation = 0.8
time_step = 0.1

exp_decay = exponential.compute(current_activation, time_step, rate=0.1)
lin_decay = linear.compute(current_activation, time_step, rate=0.05)
pow_decay = power.compute(current_activation, time_step, rate=0.1, power=2.0)

# Затухание с периодом полураспада
half_life_decay = exponential.compute(
    current_activation, time_step, half_life=5.0
)
```

#### Латеральное торможение

```python
from neurograph.propagation.functions import LateralInhibitionProcessor

# Создание процессора торможения
inhibition = LateralInhibitionProcessor()

# Применение торможения (обычно вызывается автоматически)
# inhibited_activations = inhibition.apply_inhibition(
#     activations, graph, config
# )
```

#### Утилитные функции

```python
from neurograph.propagation.functions import (
    normalize_activations, compute_activation_entropy,
    find_activation_peaks, compute_convergence_metric
)

activations = {"python": 0.8, "java": 0.6, "c++": 0.4}

# Нормализация активаций
normalized = normalize_activations(activations, method="softmax")
print("Нормализованные активации:", normalized)

# Энтропия распределения
entropy = compute_activation_entropy(activations)
print(f"Энтропия активации: {entropy:.3f}")

# Поиск пиков активации
peaks = find_activation_peaks(activations, min_prominence=0.3)
print("Пики активации:", peaks)
```

### 4. Фабрики и строители (factory.py)

#### PropagationFactory

```python
from neurograph.propagation.factory import PropagationFactory

# Создание движка
engine = PropagationFactory.create_engine("spreading_activation", graph=graph)

# Создание конфигурации
config = PropagationFactory.create_config(
    activation_function=ActivationFunction.TANH,
    decay_function=DecayFunction.EXPONENTIAL,
    max_iterations=150
)

# Создание визуализатора
visualizer = PropagationFactory.create_visualizer()
```

#### PropagationConfigBuilder

Строитель для пошагового создания конфигураций:

```python
from neurograph.propagation.factory import PropagationConfigBuilder
from neurograph.propagation.base import PropagationMode

# Пошаговое построение конфигурации
config = PropagationConfigBuilder().reset()\
    .set_performance_mode("balanced")\
    .set_activation_function(ActivationFunction.SIGMOID, steepness=1.5)\
    .set_decay_function(DecayFunction.EXPONENTIAL, rate=0.1)\
    .set_propagation_mode(PropagationMode.BIDIRECTIONAL)\
    .set_activation_limits(threshold=0.15, max_nodes=500)\
    .set_lateral_inhibition(True, strength=0.3, radius=2)\
    .set_iterations(200, convergence_threshold=0.001)\
    .set_depth_limit(5)\
    .set_filters(edge_types=["related_to", "part_of"])\
    .build()
```

#### Готовые конфигурации

```python
from neurograph.propagation import (
    create_fast_config, create_precise_config, 
    create_experimental_config
)

# Быстрая конфигурация (для демо и тестов)
fast_config = create_fast_config()

# Точная конфигурация (для исследований)
precise_config = create_precise_config()

# Экспериментальная конфигурация (новые алгоритмы)
experimental_config = create_experimental_config()
```

#### Предустановки конфигураций

```python
from neurograph.propagation.factory import ConfigurationPresets

# Различные предустановки
dev_config_dict = ConfigurationPresets.get_development_config()
prod_config_dict = ConfigurationPresets.get_production_config()
memory_config_dict = ConfigurationPresets.get_memory_efficient_config()
research_config_dict = ConfigurationPresets.get_research_config()

# Создание конфигурации из словаря
from neurograph.propagation import create_custom_config_from_dict

config = create_custom_config_from_dict(prod_config_dict)
```

### 5. Визуализация (visualizer.py)

#### PropagationVisualizer

```python
from neurograph.propagation.visualizer import PropagationVisualizer

visualizer = PropagationVisualizer()

# Статическая визуализация результата
visualizer.visualize_propagation(
    result, graph,
    save_path="propagation.png",
    show_animation=False,
    figsize=(14, 10),
    node_size=500,
    show_labels=True,
    layout="spring"
)

# Анимированная визуализация
visualizer.visualize_propagation(
    result, graph,
    save_path="propagation_animation.gif",
    show_animation=True,
    interval=500  # миллисекунды между кадрами
)

# Тепловая карта активации
current_activations = result.get_activation_levels()
visualizer.create_activation_heatmap(
    current_activations, graph,
    save_path="heatmap.png"
)

# Диаграмма потоков распространения
visualizer.create_propagation_flow_diagram(
    result, graph,
    save_path="flow_diagram.png",
    max_depth=3
)

# График сходимости
visualizer.create_convergence_plot(
    result,
    save_path="convergence.png"
)
```

#### Экспорт данных визуализации

```python
# Экспорт в JSON для внешних инструментов
visualizer.export_visualization_data(
    result, graph, "visualization_data.json"
)

# Сравнительная визуализация
results_list = [result1, result2, result3]
labels = ["Конфигурация 1", "Конфигурация 2", "Конфигурация 3"]

visualizer.create_multi_step_comparison(
    results_list, labels,
    save_path="comparison.png"
)
```

## Режимы распространения

### 1. Spreading (Расходящееся)

Классическое распространение активации от источников к соседям:

```python
config = PropagationConfigBuilder().reset()\
    .set_propagation_mode(PropagationMode.SPREADING)\
    .set_depth_limit(4)\
    .build()

# Хорошо для: исследования концептов, поиска связанных идей
result = engine.propagate({"artificial_intelligence": 1.0}, config)
```

### 2. Focusing (Сходящееся)

Активация движется к центральным узлам:

```python
config = PropagationConfigBuilder().reset()\
    .set_propagation_mode(PropagationMode.FOCUSING)\
    .set_lateral_inhibition(True, strength=0.4)\
    .build()

# Хорошо для: поиска общих тем, выделения центральных концептов
result = engine.propagate({"python": 0.8, "java": 0.8, "c++": 0.6}, config)
```

### 3. Bidirectional (Двунаправленное)

Комбинация расходящегося и сходящегося:

```python
config = PropagationConfigBuilder().reset()\
    .set_propagation_mode(PropagationMode.BIDIRECTIONAL)\
    .set_iterations(150)\
    .build()

# Хорошо для: комплексного анализа, поиска скрытых связей
result = engine.propagate({"machine_learning": 1.0}, config)
```

### 4. Constrained (Ограниченное)

Распространение с дополнительными ограничениями:

```python
config = PropagationConfigBuilder().reset()\
    .set_propagation_mode(PropagationMode.CONSTRAINED)\
    .set_filters(
        edge_types=["part_of", "used_for"],
        node_types=["technology", "concept"]
    )\
    .build()

# Хорошо для: целенаправленного поиска, фильтрации результатов
result = engine.propagate({"programming": 1.0}, config)
```

## Готовые сценарии использования

### 1. Исследование концепта

```python
from neurograph.propagation import scenario_concept_exploration

# Исследование концепта "machine_learning"
exploration = scenario_concept_exploration(
    graph, "machine_learning", exploration_depth=3
)

if "error" not in exploration:
    print(f"Найдено концептов: {len(exploration['related_concepts'])}")
    
    # Топ-5 связанных концептов
    for concept in exploration['related_concepts'][:5]:
        print(f"  {concept['concept']}: {concept['relevance']:.3f}")
        print(f"    Расстояние: {concept['distance']}")
        print(f"    Тип: {concept['type']}")
```

### 2. Активация знаний

```python
from neurograph.propagation import scenario_knowledge_activation

# Активация знаний по запросу
activation = scenario_knowledge_activation(
    graph, 
    query_concepts=["python", "machine_learning", "data_science"],
    focus_mode=True
)

if "error" not in activation:
    knowledge = activation['activated_knowledge']
    
    print("Основные концепты:")
    for concept in knowledge['primary'][:5]:
        print(f"  {concept['concept']}: {concept['activation_level']:.3f}")
    
    print("Обнаруженные связи:")
    for connection in knowledge['connections'][:3]:
        print(f"  {connection['target']} <- {connection['sources']}")
    
    print("Потенциальные инсайты:")
    for insight in knowledge['insights'][:3]:
        print(f"  {insight['concept']}: {insight['activation_level']:.3f}")
```

### 3. Семантическая близость

```python
from neurograph.propagation import scenario_semantic_similarity

# Анализ близости между концептами
similarity = scenario_semantic_similarity(
    graph, "python", "javascript", max_depth=4
)

if "error" not in similarity:
    print(f"Общая близость: {similarity['overall_similarity']:.3f}")
    print(f"Уровень: {similarity['analysis']['similarity_level']}")
    print(f"Сила связи: {similarity['analysis']['connection_strength']}")
    
    print("Общие концепты:")
    for concept in similarity['common_concepts'][:5]:
        print(f"  {concept['concept']}: {concept['combined_activation']:.3f}")
```

## Диагностика и анализ производительности

### 1. Диагностика конфигурации

```python
from neurograph.propagation.factory import PropagationDiagnostics

diagnostics = PropagationDiagnostics()

# Анализ конфигурации
analysis = diagnostics.analyze_config(config)

print(f"Производительность: {analysis['performance_rating']}")
print(f"Использование памяти: {analysis['memory_usage_estimate']}")
print(f"Сходимость: {analysis['convergence_likelihood']}")

if analysis['warnings']:
    print("Предупреждения:")
    for warning in analysis['warnings']:
        print(f"  - {warning}")

if analysis['recommendations']:
    print("Рекомендации:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")
```

### 2. Оценка времени выполнения

```python
# Оценка времени выполнения
graph_info = {
    "node_count": len(graph.get_all_nodes()),
    "edge_count": len(graph.get_all_edges())
}

time_estimate = diagnostics.estimate_execution_time(config, graph_info["node_count"])

print(f"Ожидаемое время: {time_estimate['estimated_seconds']:.2f}с")
print(f"Факторы влияния:")
for factor, value in time_estimate['factors'].items():
    print(f"  {factor}: {value}")
```

### 3. Бенчмарк конфигураций

```python
from neurograph.propagation import benchmark_config

# Тестирование конфигурации
benchmark = benchmark_config(
    config, graph, 
    test_nodes={"python": 1.0}, 
    runs=5
)

print(f"Успешных запусков: {benchmark['successful_runs']}/{benchmark['total_runs']}")
print(f"Среднее время: {benchmark['avg_processing_time']:.3f}с")
print(f"Средние итерации: {benchmark['avg_iterations']:.1f}")
print(f"Частота сходимости: {benchmark['convergence_rate']:.1%}")
```

### 4. Полная диагностика

```python
from neurograph.propagation import debug_propagation

# Комплексная диагностика
debug_info = debug_propagation(graph, {"python": 1.0}, config)

print("=== Диагностика распространения ===")
print(f"Граф: {debug_info['input_analysis']['graph_info']}")
print(f"Конфигурация: {debug_info['config_analysis']['performance_rating']}")
print(f"Совместимость: {len(debug_info['compatibility_check']['issues'])} проблем")
print(f"Результат: {'✓' if debug_info['execution_results']['success'] else '✗'}")
```

## Интеграция с другими модулями

### 1. Интеграция с SemGraph

```python
from neurograph.propagation import integrate_with_semgraph

# Обогащение графа данными об активации
integrate_with_semgraph(result, graph)

# Проверка добавленных метаданных
for node_id in result.activated_nodes.keys():
    node_data = graph.get_node(node_id)
    if "last_activation_level" in node_data:
        print(f"{node_id}: последняя активация {node_data['last_activation_level']:.3f}")
```

### 2. Интеграция с Memory

```python
from neurograph.propagation import integrate_with_memory

# Создание энкодера для векторизации
class SimpleEncoder:
    def encode(self, text):
        return np.random.random(384)  # Заглушка

encoder = SimpleEncoder()

# Добавление активированных концептов в память
integrate_with_memory(result, memory_module, encoder)
```

### 3. Интеграция с Processor

```python
from neurograph.propagation import integrate_with_processor

# Создание контекста для логического вывода
context = integrate_with_processor(result, processor_module)

print(f"Создано фактов: {len(context.facts)}")
print(f"Параметры запроса: {context.query_params}")
```

## Продвинутые паттерны использования

### 1. Пользовательские функции активации

```python
from neurograph.propagation.base import IActivationFunction
from neurograph.propagation.functions import ActivationFunctionFactory

class CustomActivationFunction(IActivationFunction):
    """Пользовательская функция активации"""
    
    def compute(self, input_value: float, **params) -> float:
        # Комбинированная sigmoid + threshold функция
        threshold = params.get("threshold", 0.5)
        steepness = params.get("steepness", 1.0)
        
        if input_value < threshold:
            return 0.0
        
        # Применяем sigmoid к значениям выше порога
        shifted_input = (input_value - threshold) * steepness
        return 1.0 / (1.0 + math.exp(-shifted_input))
    
    def derivative(self, input_value: float, **params) -> float:
        # Производная для обучения
        threshold = params.get("threshold", 0.5)
        if input_value < threshold:
            return 0.0
        
        sigmoid_val = self.compute(input_value, **params)
        steepness = params.get("steepness", 1.0)
        return steepness * sigmoid_val * (1.0 - sigmoid_val)

# Регистрация и использование
# (В реальности нужно расширить enum ActivationFunction)
```

### 2. Адаптивная конфигурация

```python
class AdaptiveConfig:
    """Адаптивная конфигурация на основе характеристик графа"""
    
    @staticmethod
    def create_for_graph(graph) -> PropagationConfig:
        node_count = len(graph.get_all_nodes())
        edge_count = len(graph.get_all_edges())
        density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
        
        builder = PropagationConfigBuilder().reset()
        
        if node_count < 100:
            # Маленький граф - можем позволить себе точность
            builder.set_performance_mode("precise")
        elif node_count < 1000:
            # Средний граф - баланс
            builder.set_performance_mode("balanced")
        else:
            # Большой граф - скорость
            builder.set_performance_mode("fast")
        
        # Адаптация к плотности
        if density > 0.1:  # Плотный граф
            builder.set_lateral_inhibition(True, strength=0.4)
            builder.set_activation_limits(threshold=0.2, max_nodes=node_count // 4)
        else:  # Разреженный граф
            builder.set_lateral_inhibition(False)
            builder.set_activation_limits(threshold=0.1, max_nodes=node_count // 2)
        
        return builder.build()

# Использование
adaptive_config = AdaptiveConfig.create_for_graph(graph)
```

### 3. Многопоточное распространение

```python
import concurrent.futures
from typing import List, Dict

class ParallelPropagationEngine:
    """Параллельное выполнение распространения для разных начальных условий"""
    
    def __init__(self, graph, max_workers: int = 4):
        self.graph = graph
        self.max_workers = max_workers
    
    def parallel_propagate(self, 
                          initial_conditions: List[Dict[str, float]],
                          config: PropagationConfig) -> List[PropagationResult]:
        """Параллельное выполнение распространения"""
        
        def single_propagate(initial_nodes):
            engine = SpreadingActivationEngine(self.graph)
            return engine.propagate(initial_nodes, config)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(single_propagate, initial_nodes)
                for initial_nodes in initial_conditions
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Ошибка в параллельном распространении: {e}")
            
            return results

# Использование
parallel_engine = ParallelPropagationEngine(graph)

initial_conditions = [
    {"python": 1.0},
    {"java": 1.0}, 
    {"javascript": 1.0},
    {"c++": 1.0}
]

results = parallel_engine.parallel_propagate(initial_conditions, config)
```

### 4. Кэширование результатов

```python
import hashlib
import pickle
from functools import lru_cache

class CachedPropagationEngine:
    """Движок с кэшированием результатов"""
    
    def __init__(self, graph, cache_size: int = 1000):
        self.engine = SpreadingActivationEngine(graph)
        self.cache = {}
        self.cache_size = cache_size
    
    def _config_hash(self, config: PropagationConfig) -> str:
        """Создание хэша конфигурации для кэширования"""
        config_str = str(sorted(vars(config).items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _initial_nodes_hash(self, initial_nodes: Dict[str, float]) -> str:
        """Создание хэша начальных условий"""
        nodes_str = str(sorted(initial_nodes.items()))
        return hashlib.md5(nodes_str.encode()).hexdigest()
    
    def propagate_cached(self, 
                        initial_nodes: Dict[str, float],
                        config: PropagationConfig) -> PropagationResult:
        """Распространение с кэшированием"""
        
        # Создание ключа кэша
        cache_key = (
            self._initial_nodes_hash(initial_nodes),
            self._config_hash(config)
        )
        
        # Проверка кэша
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Выполнение распространения
        result = self.engine.propagate(initial_nodes, config)
        
        # Сохранение в кэш
        if len(self.cache) >= self.cache_size:
            # Удаляем старейший элемент (простая FIFO стратегия)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        
        return result

# Использование
cached_engine = CachedPropagationEngine(graph, cache_size=500)
result = cached_engine.propagate_cached({"python": 1.0}, config)
```

### 5. Мониторинг в реальном времени

```python
from neurograph.core.events import subscribe, publish
import time

class PropagationMonitor:
    """Мониторинг процесса распространения в реальном времени"""
    
    def __init__(self):
        self.stats = {
            "total_propagations": 0,
            "successful_propagations": 0,
            "failed_propagations": 0,
            "avg_processing_time": 0.0,
            "current_propagations": 0
        }
        
        # Подписка на события
        subscribe("propagation.started", self._on_propagation_started)
        subscribe("propagation.completed", self._on_propagation_completed)
        subscribe("propagation.error", self._on_propagation_error)
        subscribe("propagation.iteration", self._on_iteration_update)
    
    def _on_propagation_started(self, event_data):
        self.stats["current_propagations"] += 1
        print(f"▶ Начато распространение: {event_data.get('initial_nodes_count', 0)} начальных узлов")
    
    def _on_propagation_completed(self, event_data):
        self.stats["total_propagations"] += 1
        self.stats["successful_propagations"] += 1
        self.stats["current_propagations"] -= 1
        
        # Обновление средних значений
        processing_time = event_data.get("processing_time", 0)
        total = self.stats["successful_propagations"]
        current_avg = self.stats["avg_processing_time"]
        self.stats["avg_processing_time"] = (current_avg * (total - 1) + processing_time) / total
        
        print(f"✓ Завершено распространение: {event_data.get('activated_nodes_count', 0)} узлов активировано")
        print(f"  Время: {processing_time:.3f}с, Итераций: {event_data.get('iterations_used', 0)}")
    
    def _on_propagation_error(self, event_data):
        self.stats["total_propagations"] += 1
        self.stats["failed_propagations"] += 1
        self.stats["current_propagations"] -= 1
        
        print(f"✗ Ошибка распространения: {event_data.get('error_message', 'Неизвестная ошибка')}")
    
    def _on_iteration_update(self, event_data):
        iteration = event_data.get("iteration", 0)
        active_nodes = event_data.get("active_nodes_count", 0)
        
        if iteration % 10 == 0:  # Каждые 10 итераций
            print(f"  Итерация {iteration}: {active_nodes} активных узлов")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение текущей статистики"""
        success_rate = (
            self.stats["successful_propagations"] / self.stats["total_propagations"]
            if self.stats["total_propagations"] > 0 else 0
        )
        
        return {
            **self.stats,
            "success_rate": success_rate
        }

# Использование
monitor = PropagationMonitor()

# Выполнение с мониторингом
engine = create_default_engine(graph)
config = create_default_config()

# События будут автоматически отправляться в monitor
result = engine.propagate({"python": 1.0}, config)

# Просмотр статистики
stats = monitor.get_stats()
print(f"Статистика: {stats}")
```

## Оптимизация производительности

### 1. Профилирование

```python
import cProfile
import pstats
from neurograph.propagation import quick_propagate

def profile_propagation():
    """Профилирование распространения активации"""
    
    # Создание профайлера
    profiler = cProfile.Profile()
    
    # Запуск профилирования
    profiler.enable()
    
    # Выполнение распространения
    result = quick_propagate(graph, {"python": 1.0}, mode="balanced")
    
    profiler.disable()
    
    # Анализ результатов
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Топ-10 функций по времени
    
    return result

# Запуск профилирования
profile_propagation()
```

### 2. Оптимизация для больших графов

```python
def optimize_for_large_graph(graph_size: int) -> PropagationConfig:
    """Создание оптимизированной конфигурации для больших графов"""
    
    builder = PropagationConfigBuilder().reset()
    
    if graph_size > 10000:
        # Очень большой граф
        config = builder\
            .set_performance_mode("fast")\
            .set_activation_limits(threshold=0.3, max_nodes=min(1000, graph_size // 10))\
            .set_iterations(50, convergence_threshold=0.01)\
            .set_lateral_inhibition(False)\
            .set_depth_limit(3)\
            .build()
    
    elif graph_size > 1000:
        # Большой граф
        config = builder\
            .set_performance_mode("balanced")\
            .set_activation_limits(threshold=0.2, max_nodes=min(2000, graph_size // 5))\
            .set_iterations(100, convergence_threshold=0.005)\
            .set_lateral_inhibition(True, strength=0.2)\
            .set_depth_limit(4)\
            .build()
    
    else:
        # Обычный граф
        config = builder\
            .set_performance_mode("precise")\
            .build()
    
    return config

# Использование
optimized_config = optimize_for_large_graph(len(graph.get_all_nodes()))
```

### 3. Пакетная обработка

```python
class BatchPropagationProcessor:
    """Пакетная обработка множественных запросов распространения"""
    
    def __init__(self, graph, batch_size: int = 10):
        self.graph = graph
        self.batch_size = batch_size
        self.engine = SpreadingActivationEngine(graph)
    
    def process_batch(self, 
                     requests: List[Dict[str, Any]],
                     shared_config: PropagationConfig = None) -> List[PropagationResult]:
        """Обработка пакета запросов"""
        
        if shared_config is None:
            shared_config = create_default_config()
        
        results = []
        
        # Обработка по батчам
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            
            batch_results = []
            for request in batch:
                initial_nodes = request.get("initial_nodes", {})
                config = request.get("config", shared_config)
                
                # Сброс состояния между запросами
                self.engine.reset_activations()
                
                result = self.engine.propagate(initial_nodes, config)
                batch_results.append({
                    "request_id": request.get("id", f"batch_{i}"),
                    "result": result,
                    "success": result.success
                })
            
            results.extend(batch_results)
            
            # Опциональная пауза между батчами
            time.sleep(0.01)
        
        return results

# Использование
processor = BatchPropagationProcessor(graph, batch_size=5)

requests = [
    {"id": "req_1", "initial_nodes": {"python": 1.0}},
    {"id": "req_2", "initial_nodes": {"java": 1.0}},
    {"id": "req_3", "initial_nodes": {"ai": 0.8, "ml": 0.6}},
]

batch_results = processor.process_batch(requests)
```

## Тестирование

### 1. Модульные тесты

```python
import pytest
import numpy as np
from neurograph.propagation import (
    create_default_engine, create_default_config,
    SpreadingActivationEngine, PropagationConfigBuilder
)
from neurograph.propagation.base import NodeActivation, PropagationMode
from neurograph.semgraph import SemGraphFactory

class TestPropagationEngine:
    
    @pytest.fixture
    def simple_graph(self):
        """Простой граф для тестирования"""
        graph = SemGraphFactory.create("memory_efficient")
        
        # Линейная цепочка: A -> B -> C -> D
        nodes = ["A", "B", "C", "D"]
        for node in nodes:
            graph.add_node(node, type="test")
        
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1], "next", weight=1.0)
        
        return graph
    
    @pytest.fixture
    def engine(self, simple_graph):
        return SpreadingActivationEngine(simple_graph)
    
    def test_basic_propagation(self, engine):
        """Тест базового распространения"""
        config = create_default_config()
        initial_nodes = {"A": 1.0}
        
        result = engine.propagate(initial_nodes, config)
        
        assert result.success
        assert len(result.activated_nodes) > 1
        assert "A" in result.activated_nodes
        assert result.activated_nodes["A"].activation_level == 1.0
    
    def test_propagation_depth(self, engine):
        """Тест глубины распространения"""
        config = PropagationConfigBuilder().reset()\
            .set_depth_limit(2)\
            .set_activation_limits(threshold=0.01)\
            .build()
        
        result = engine.propagate({"A": 1.0}, config)
        
        assert result.success
        
        # Проверяем глубины
        max_depth = max(
            act.propagation_depth 
            for act in result.activated_nodes.values()
        )
        assert max_depth <= 2
    
    def test_activation_threshold(self, engine):
        """Тест порога активации"""
        config = PropagationConfigBuilder().reset()\
            .set_activation_limits(threshold=0.5)\
            .build()
        
        result = engine.propagate({"A": 0.3}, config)  # Низкая начальная активация
        
        # При высоком пороге должно быть мало активных узлов
        assert len(result.activated_nodes) <= 2
    
    def test_convergence(self, engine):
        """Тест сходимости"""
        config = PropagationConfigBuilder().reset()\
            .set_iterations(10, convergence_threshold=0.1)\
            .build()
        
        result = engine.propagate({"A": 1.0}, config)
        
        assert result.success
        # При таком пороге сходимость должна достигаться быстро
        assert result.iterations_used <= 10
    
    def test_different_modes(self, engine):
        """Тест различных режимов распространения"""
        initial_nodes = {"A": 1.0, "D": 0.8}
        
        modes = [
            PropagationMode.SPREADING,
            PropagationMode.FOCUSING,
            PropagationMode.BIDIRECTIONAL
        ]
        
        results = {}
        for mode in modes:
            config = PropagationConfigBuilder().reset()\
                .set_propagation_mode(mode)\
                .build()
            
            engine.reset_activations()
            result = engine.propagate(initial_nodes, config)
            results[mode] = result
            
            assert result.success
        
        # Разные режимы должны давать разные результаты
        spreading_nodes = len(results[PropagationMode.SPREADING].activated_nodes)
        bidirectional_nodes = len(results[PropagationMode.BIDIRECTIONAL].activated_nodes)
        
        # Двунаправленный режим обычно активирует больше узлов
        assert bidirectional_nodes >= spreading_nodes

class TestActivationFunctions:
    
    def test_sigmoid_function(self):
        """Тест сигмоидной функции"""
        from neurograph.propagation.functions import ActivationFunctionFactory
        from neurograph.propagation.base import ActivationFunction
        
        sigmoid = ActivationFunctionFactory.create(ActivationFunction.SIGMOID)
        
        # Тест базовых свойств
        assert sigmoid.compute(0.0) == 0.5
        assert sigmoid.compute(-10.0) < 0.1
        assert sigmoid.compute(10.0) > 0.9
        
        # Тест монотонности
        assert sigmoid.compute(0.5) > sigmoid.compute(0.0)
        assert sigmoid.compute(1.0) > sigmoid.compute(0.5)
    
    def test_threshold_function(self):
        """Тест пороговой функции"""
        from neurograph.propagation.functions import ActivationFunctionFactory
        from neurograph.propagation.base import ActivationFunction
        
        threshold = ActivationFunctionFactory.create(ActivationFunction.THRESHOLD)
        
        # Пороговая функция с порогом 0.5
        result_low = threshold.compute(0.3, threshold=0.5)
        result_high = threshold.compute(0.7, threshold=0.5)
        
        assert result_low == 0.0
        assert result_high == 1.0

class TestVisualization:
    
    @pytest.fixture
    def sample_result(self, simple_graph):
        """Пример результата для тестирования визуализации"""
        engine = SpreadingActivationEngine(simple_graph)
        config = create_default_config()
        return engine.propagate({"A": 1.0}, config)
    
    def test_static_visualization(self, sample_result, simple_graph, tmp_path):
        """Тест статической визуализации"""
        from neurograph.propagation.visualizer import PropagationVisualizer
        
        visualizer = PropagationVisualizer()
        output_path = tmp_path / "test_viz.png"
        
        # Не должно вызывать исключений
        try:
            visualizer.visualize_propagation(
                sample_result, simple_graph,
                save_path=str(output_path),
                show=False
            )
            
            # Файл должен быть создан
            assert output_path.exists()
            
        except ImportError:
            # matplotlib может быть недоступен в тестовой среде
            pytest.skip("matplotlib недоступен")
    
    def test_data_export(self, sample_result, simple_graph, tmp_path):
        """Тест экспорта данных визуализации"""
        from neurograph.propagation.visualizer import PropagationVisualizer
        
        visualizer = PropagationVisualizer()
        export_path = tmp_path / "test_export.json"
        
        visualizer.export_visualization_data(
            sample_result, simple_graph, str(export_path)
        )
        
        assert export_path.exists()
        
        # Проверяем содержимое
        import json
        with open(export_path) as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "nodes" in data
        assert "activation_history" in data

class TestIntegration:
    
    def test_scenario_concept_exploration(self, simple_graph):
        """Тест сценария исследования концепта"""
        from neurograph.propagation import scenario_concept_exploration
        
        result = scenario_concept_exploration(simple_graph, "A", exploration_depth=2)
        
        assert "error" not in result
        assert "related_concepts" in result
        assert result["start_concept"] == "A"
        assert result["exploration_successful"] is True
    
    def test_quick_propagate(self, simple_graph):
        """Тест быстрого распространения"""
        from neurograph.propagation import quick_propagate
        
        result = quick_propagate(simple_graph, {"A": 1.0}, mode="fast")
        
        assert result.success
        assert len(result.activated_nodes) > 0

# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__])
```

### 2. Интеграционные тесты

```python
import pytest
from neurograph.propagation import propagate_and_visualize, create_system_with_propagation

class TestPropagationIntegration:
    
    @pytest.fixture
    def complex_graph(self):
        """Сложный граф для интеграционных тестов"""
        from neurograph.semgraph import SemGraphFactory
        
        graph = SemGraphFactory.create("memory_efficient")
        
        # Создание иерархической структуры
        domains = ["programming", "ai", "web"]
        technologies = {
            "programming": ["python", "java", "c++"],
            "ai": ["machine_learning", "deep_learning", "nlp"],
            "web": ["html", "css", "javascript"]
        }
        
        # Добавление узлов
        for domain in domains:
            graph.add_node(domain, type="domain", level=0)
            
            for tech in technologies[domain]:
                graph.add_node(tech, type="technology", domain=domain, level=1)
                graph.add_edge(domain, tech, "contains", weight=0.8)
        
        # Междоменные связи
        graph.add_edge("python", "machine_learning", "used_for", weight=0.9)
        graph.add_edge("javascript", "web", "core_of", weight=0.9)
        graph.add_edge("python", "web", "used_for", weight=0.6)
        
        return graph
    
    def test_full_system_integration(self, complex_graph):
        """Тест полной интеграции системы"""
        
        # Создание полной системы
        engine, config, visualizer = create_system_with_propagation(
            complex_graph, config_preset="balanced"
        )
        
        # Выполнение распространения
        result = engine.propagate({"python": 1.0}, config)
        
        assert result.success
        assert len(result.activated_nodes) > 3
        
        # Проверка активации связанных концептов
        activated_ids = set(result.activated_nodes.keys())
        
        # Python должен активировать programming и machine_learning
        assert "programming" in activated_ids
        # machine_learning может быть активировано через python -> used_for
        
    def test_cross_module_integration(self, complex_graph):
        """Тест интеграции между модулями"""
        
        # Моки для других модулей
        class MockMemory:
            def __init__(self):
                self.items = []
            
            def add(self, item):
                self.items.append(item)
            
            def size(self):
                return len(self.items)
        
        class MockProcessor:
            def __init__(self):
                self.contexts = []
        
        from neurograph.propagation import (
            create_default_engine, create_default_config,
            integrate_with_semgraph, integrate_with_memory, integrate_with_processor
        )
        
        # Выполнение распространения
        engine = create_default_engine(complex_graph)
        config = create_default_config()
        result = engine.propagate({"python": 1.0, "ai": 0.8}, config)
        
        assert result.success
        
        # Интеграция с SemGraph
        integrate_with_semgraph(result, complex_graph)
        
        # Проверка обновления метаданных
        python_data = complex_graph.get_node("python")
        assert "last_activation_level" in python_data
        assert python_data["activation_count"] >= 1
        
        # Интеграция с Memory
        mock_memory = MockMemory()
        
        class MockEncoder:
            def encode(self, text):
                return [0.1] * 384
        
        integrate_with_memory(result, mock_memory, MockEncoder())
        assert mock_memory.size() > 0
        
        # Интеграция с Processor
        mock_processor = MockProcessor()
        context = integrate_with_processor(result, mock_processor)
        assert len(context.facts) > 0
```

### 3. Тесты производительности

```python
import time
import pytest
from neurograph.propagation import benchmark_config

class TestPropagationPerformance:
    
    def test_large_graph_performance(self):
        """Тест производительности на большом графе"""
        from neurograph.semgraph import SemGraphFactory
        
        # Создание большого графа
        graph = SemGraphFactory.create("memory_efficient")
        
        # Добавление 1000 узлов
        for i in range(1000):
            graph.add_node(f"node_{i}", type="test")
        
        # Добавление случайных связей
        import random
        random.seed(42)
        
        for i in range(2000):  # 2000 связей
            source = f"node_{random.randint(0, 999)}"
            target = f"node_{random.randint(0, 999)}"
            if source != target:
                graph.add_edge(source, target, "related", weight=random.random())
        
        # Тест производительности
        from neurograph.propagation import create_fast_config, create_default_engine
        
        config = create_fast_config()
        engine = create_default_engine(graph)
        
        start_time = time.time()
        result = engine.propagate({"node_0": 1.0}, config)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert result.success
        assert execution_time < 5.0  # Должно выполняться быстрее 5 секунд
        
        print(f"Время выполнения на 1000 узлах: {execution_time:.3f}с")
        print(f"Активировано узлов: {len(result.activated_nodes)}")
    
    @pytest.mark.parametrize("config_type", ["fast", "balanced", "precise"])
    def test_config_performance_comparison(self, config_type):
        """Сравнение производительности различных конфигураций"""
        from neurograph.semgraph import SemGraphFactory
        from neurograph.propagation import (
            create_fast_config, create_default_config, create_precise_config
        )
        
        # Средний граф
        graph = SemGraphFactory.create("memory_efficient")
        
        for i in range(100):
            graph.add_node(f"concept_{i}", type="concept")
        
        for i in range(200):
            import random
            random.seed(42 + i)
            source = f"concept_{random.randint(0, 99)}"
            target = f"concept_{random.randint(0, 99)}"
            if source != target:
                graph.add_edge(source, target, "related")
        
        # Выбор конфигурации
        configs = {
            "fast": create_fast_config(),
            "balanced": create_default_config(),
            "precise": create_precise_config()
        }
        
        config = configs[config_type]
        
        # Бенчмарк
        benchmark_result = benchmark_config(
            config, graph, {"concept_0": 1.0}, runs=3
        )
        
        assert benchmark_result["success_rate"] > 0.8
        assert benchmark_result["avg_processing_time"] < 2.0
        
        print(f"Конфигурация {config_type}:")
        print(f"  Среднее время: {benchmark_result['avg_processing_time']:.3f}с")
        print(f"  Средние итерации: {benchmark_result['avg_iterations']:.1f}")
```

## Примеры реальных применений

### 1. Система рекомендаций

```python
class ConceptRecommendationSystem:
    """Система рекомендаций концептов на основе распространения активации"""
    
    def __init__(self, knowledge_graph):
        self.graph = knowledge_graph
        self.engine = create_default_engine(knowledge_graph)
        
        # Специальная конфигурация для рекомендаций
        self.recommendation_config = PropagationConfigBuilder().reset()\
            .set_propagation_mode(PropagationMode.SPREADING)\
            .set_activation_limits(threshold=0.2, max_nodes=50)\
            .set_lateral_inhibition(True, strength=0.3)\
            .set_depth_limit(3)\
            .build()
    
    def recommend_concepts(self, 
                          user_interests: Dict[str, float],
                          exclude_known: List[str] = None,
                          top_n: int = 10) -> List[Dict[str, Any]]:
        """Рекомендация концептов на основе интересов пользователя"""
        
        # Выполнение распространения активации
        result = self.engine.propagate(user_interests, self.recommendation_config)
        
        if not result.success:
            return []
        
        exclude_known = exclude_known or []
        exclude_set = set(user_interests.keys()) | set(exclude_known)
        
        # Фильтрация и ранжирование рекомендаций
        recommendations = []
        for node_id, activation in result.activated_nodes.items():
            if node_id not in exclude_set:
                node_data = self.graph.get_node(node_id) or {}
                
                recommendations.append({
                    "concept_id": node_id,
                    "relevance_score": activation.activation_level,
                    "concept_type": node_data.get("type", "unknown"),
                    "description": node_data.get("description", ""),
                    "propagation_path_length": activation.propagation_depth,
                    "activation_sources": list(activation.source_nodes)
                })
        
        # Сортировка по релевантности
        recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return recommendations[:top_n]
    
    def explain_recommendation(self, concept_id: str, user_interests: Dict[str, float]) -> Dict[str, Any]:
        """Объяснение рекомендации концепта"""
        
        from neurograph.propagation import scenario_semantic_similarity
        
        explanations = []
        
        # Анализ связи с каждым интересом пользователя
        for interest, weight in user_interests.items():
            similarity = scenario_semantic_similarity(
                self.graph, interest, concept_id, max_depth=4
            )
            
            if "error" not in similarity and similarity["overall_similarity"] > 0.1:
                explanations.append({
                    "user_interest": interest,
                    "interest_weight": weight,
                    "similarity_score": similarity["overall_similarity"],
                    "connection_strength": similarity["analysis"]["connection_strength"],
                    "common_concepts": similarity["common_concepts"][:3]
                })
        
        return {
            "concept_id": concept_id,
            "explanations": explanations,
            "overall_relevance": sum(exp["similarity_score"] * exp["interest_weight"] 
                                   for exp in explanations)
        }

# Использование системы рекомендаций
recommender = ConceptRecommendationSystem(knowledge_graph)

user_profile = {
    "python": 0.9,
    "machine_learning": 0.8,
    "data_visualization": 0.6
}

recommendations = recommender.recommend_concepts(
    user_profile, 
    exclude_known=["pandas", "numpy"],  # Уже известные концепты
    top_n=5
)

for rec in recommendations:
    explanation = recommender.explain_recommendation(rec["concept_id"], user_profile)
    print(f"Рекомендация: {rec['concept_id']} (релевантность: {rec['relevance_score']:.3f})")
    print(f"Объяснение: общая релевантность {explanation['overall_relevance']:.3f}")
```

### 2. Автоматическое тегирование контента

```python
class ContentTagger:
    """Автоматическое тегирование контента на основе семантических связей"""
    
    def __init__(self, concept_graph):
        self.graph = concept_graph
        self.engine = create_default_engine(concept_graph)
        
        # Конфигурация для тегирования
        self.tagging_config = PropagationConfigBuilder().reset()\
            .set_propagation_mode(PropagationMode.FOCUSING)\
            .set_activation_limits(threshold=0.25, max_nodes=30)\
            .set_lateral_inhibition(True, strength=0.4)\
            .build()
    
    def extract_tags(self, 
                    content_concepts: List[str],
                    confidence_threshold: float = 0.3,
                    max_tags: int = 10) -> List[Dict[str, Any]]:
        """Извлечение тегов из концептов контента"""
        
        # Начальная активация концептов из контента
        initial_activation = {concept: 1.0 for concept in content_concepts 
                            if self.graph.has_node(concept)}
        
        if not initial_activation:
            return []
        
        # Распространение активации
        result = self.engine.propagate(initial_activation, self.tagging_config)
        
        if not result.success:
            return []
        
        # Извлечение потенциальных тегов
        tags = []
        for node_id, activation in result.activated_nodes.items():
            if activation.activation_level >= confidence_threshold:
                node_data = self.graph.get_node(node_id) or {}
                
                # Фильтрация по типу узла (теги обычно более общие концепты)
                node_type = node_data.get("type", "")
                
                if node_type in ["category", "domain", "topic"] or activation.propagation_depth <= 2:
                    tags.append({
                        "tag": node_id,
                        "confidence": activation.activation_level,
                        "type": node_type,
                        "sources": list(activation.source_nodes),
                        "description": node_data.get("description", "")
                    })
        
        # Сортировка и ограничение количества тегов
        tags.sort(key=lambda x: x["confidence"], reverse=True)
        return tags[:max_tags]
    
    def suggest_related_content(self, 
                              current_tags: List[str],
                              content_database: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Предложение связанного контента на основе тегов"""
        
        # Активация от текущих тегов
        initial_nodes = {tag: 1.0 for tag in current_tags if self.graph.has_node(tag)}
        
        if not initial_nodes:
            return []
        
        result = self.engine.propagate(initial_nodes, self.tagging_config)
        
        if not result.success:
            return []
        
        # Оценка релевантности контента из базы
        content_scores = []
        
        for content in content_database:
            content_tags = content.get("tags", [])
            
            # Вычисление общей релевантности
            total_relevance = 0.0
            matching_tags = 0
            
            for tag in content_tags:
                if tag in result.activated_nodes:
                    total_relevance += result.activated_nodes[tag].activation_level
                    matching_tags += 1
            
            if matching_tags > 0:
                avg_relevance = total_relevance / matching_tags
                content_scores.append({
                    **content,
                    "relevance_score": avg_relevance,
                    "matching_tags": matching_tags,
                    "total_tags": len(content_tags)
                })
        
        # Сортировка по релевантности
        content_scores.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return content_scores

# Использование системы тегирования
tagger = ContentTagger(concept_graph)

# Извлечение тегов из контента
article_concepts = ["neural_networks", "deep_learning", "computer_vision"]
tags = tagger.extract_tags(article_concepts, confidence_threshold=0.3)

print("Извлеченные теги:")
for tag in tags:
    print(f"  {tag['tag']}: {tag['confidence']:.3f} ({tag['type']})")

# База контента для рекомендаций
content_db = [
    {"id": "art1", "title": "Введение в CNN", "tags": ["neural_networks", "computer_vision"]},
    {"id": "art2", "title": "Обработка изображений", "tags": ["computer_vision", "image_processing"]},
    {"id": "art3", "title": "Рекуррентные сети", "tags": ["neural_networks", "nlp"]}
]

# Поиск связанного контента
tag_names = [tag["tag"] for tag in tags[:3]]
related_content = tagger.suggest_related_content(tag_names, content_db)

print("\nСвязанный контент:")
for content in related_content[:3]:
    print(f"  {content['title']}: релевантность {content['relevance_score']:.3f}")
```

### 3. Система когнитивного моделирования

```python
class CognitiveAssociationModel:
    """Модель когнитивных ассоциаций на основе распространения активации"""
    
    def __init__(self, semantic_network):
        self.network = semantic_network
        self.engine = create_default_engine(semantic_network)
        
        # Различные конфигурации для разных типов мышления
        self.configs = {
            "free_association": PropagationConfigBuilder().reset()
                .set_propagation_mode(PropagationMode.SPREADING)
                .set_activation_limits(threshold=0.1, max_nodes=100)
                .set_lateral_inhibition(False)
                .set_depth_limit(4)
                .build(),
            
            "focused_thinking": PropagationConfigBuilder().reset()
                .set_propagation_mode(PropagationMode.FOCUSING)
                .set_activation_limits(threshold=0.3, max_nodes=30)
                .set_lateral_inhibition(True, strength=0.5)
                .set_depth_limit(3)
                .build(),
            
            "creative_thinking": PropagationConfigBuilder().reset()
                .set_propagation_mode(PropagationMode.BIDIRECTIONAL)
                .set_activation_limits(threshold=0.15, max_nodes=80)
                .set_lateral_inhibition(True, strength=0.2)
                .set_depth_limit(5)
                .build()
        }
    
    def simulate_word_association(self, 
                                 stimulus_words: List[str],
                                 thinking_mode: str = "free_association",
                                 time_limit: float = 2.0) -> Dict[str, Any]:
        """Симуляция словесных ассоциаций"""
        
        # Подготовка начального состояния
        initial_activation = {}
        for word in stimulus_words:
            if self.network.has_node(word):
                initial_activation[word] = 1.0 / len(stimulus_words)
        
        if not initial_activation:
            return {"error": "Слова-стимулы не найдены в семантической сети"}
        
        config = self.configs.get(thinking_mode, self.configs["free_association"])
        
        # Ограничение времени через количество итераций
        max_iterations = int(time_limit * 50)  # 50 итераций на секунду
        config.max_iterations = min(config.max_iterations, max_iterations)
        
        # Выполнение симуляции
        start_time = time.time()
        result = self.engine.propagate(initial_activation, config)
        end_time = time.time()
        
        if not result.success:
            return {"error": f"Ошибка симуляции: {result.error_message}"}
        
        # Анализ результатов
        associations = []
        
        for node_id, activation in result.activated_nodes.items():
            if node_id not in stimulus_words and activation.activation_level > 0.2:
                node_data = self.network.get_node(node_id) or {}
                
                associations.append({
                    "word": node_id,
                    "activation_strength": activation.activation_level,
                    "association_path_length": activation.propagation_depth,
                    "word_type": node_data.get("type", "unknown"),
                    "frequency": node_data.get("frequency", 0),
                    "sources": list(activation.source_nodes)
                })
        
        # Сортировка по силе активации
        associations.sort(key=lambda x: x["activation_strength"], reverse=True)
        
        return {
            "stimulus_words": stimulus_words,
            "thinking_mode": thinking_mode,
            "processing_time": end_time - start_time,
            "total_associations": len(associations),
            "strong_associations": [a for a in associations if a["activation_strength"] > 0.5],
            "all_associations": associations,
            "network_statistics": {
                "total_activated_nodes": len(result.activated_nodes),
                "convergence_achieved": result.convergence_achieved,
                "iterations_used": result.iterations_used
            }
        }
    
    def analyze_conceptual_blending(self, 
                                  concept1: str, 
                                  concept2: str) -> Dict[str, Any]:
        """Анализ концептуального смешения двух понятий"""
        
        if not (self.network.has_node(concept1) and self.network.has_node(concept2)):
            return {"error": "Один или оба концепта не найдены в сети"}
        
        # Активация каждого концепта отдельно
        results = {}
        
        for concept in [concept1, concept2]:
            result = self.engine.propagate(
                {concept: 1.0}, 
                self.configs["creative_thinking"]
            )
            results[concept] = result
            self.engine.reset_activations()
        
        # Совместная активация
        joint_result = self.engine.propagate(
            {concept1: 0.7, concept2: 0.7},
            self.configs["creative_thinking"]
        )
        
        # Анализ пересечений и новых активаций
        individual_nodes = set()
        for result in results.values():
            individual_nodes.update(result.activated_nodes.keys())
        
        joint_nodes = set(joint_result.activated_nodes.keys())
        
        # Новые активации при совместном рассмотрении
        emergent_nodes = joint_nodes - individual_nodes
        
        # Усиленные активации
        enhanced_nodes = []
        for node_id in joint_nodes & individual_nodes:
            joint_activation = joint_result.activated_nodes[node_id].activation_level
            
            max_individual = max(
                results[concept1].activated_nodes.get(node_id, NodeActivation(node_id)).activation_level,
                results[concept2].activated_nodes.get(node_id, NodeActivation(node_id)).activation_level
            )
            
            if joint_activation > max_individual * 1.2:  # 20% усиление
                enhanced_nodes.append({
                    "node": node_id,
                    "joint_activation": joint_activation,
                    "max_individual": max_individual,
                    "enhancement_factor": joint_activation / max_individual
                })
        
        return {
            "concept1": concept1,
            "concept2": concept2,
            "individual_activations": {
                concept1: len(results[concept1].activated_nodes),
                concept2: len(results[concept2].activated_nodes)
            },
            "joint_activations": len(joint_nodes),
            "emergent_concepts": [
                {
                    "concept": node_id,
                    "activation": joint_result.activated_nodes[node_id].activation_level,
                    "description": self.network.get_node(node_id, {}).get("description", "")
                }
                for node_id in emergent_nodes
            ],
            "enhanced_concepts": enhanced_nodes,
            "blending_strength": len(emergent_nodes) / len(joint_nodes) if joint_nodes else 0
        }

# Использование когнитивной модели
cognitive_model = CognitiveAssociationModel(semantic_network)

# Симуляция свободных ассоциаций
association_result = cognitive_model.simulate_word_association(
    stimulus_words=["ocean", "music"],
    thinking_mode="free_association",
    time_limit=1.5
)

print("Словесные ассоциации:")
for assoc in association_result["strong_associations"][:5]:
    print(f"  {assoc['word']}: {assoc['activation_strength']:.3f}")

# Анализ концептуального смешения
blending_result = cognitive_model.analyze_conceptual_blending("music", "color")

print(f"\nКонцептуальное смешение:")
print(f"Сила смешения: {blending_result['blending_strength']:.3f}")
print("Эмерджентные концепты:")
for concept in blending_result["emergent_concepts"][:3]:
    print(f"  {concept['concept']}: {concept['activation']:.3f}")
```

## Лучшие практики и рекомендации

### 1. Выбор параметров

```python
# ✅ Хорошие практики настройки

def choose_optimal_config(graph_size: int, task_type: str) -> PropagationConfig:
    """Выбор оптимальной конфигурации на основе размера графа и типа задачи"""
    
    builder = PropagationConfigBuilder().reset()
    
    # Базовая настройка по размеру графа
    if graph_size < 100:
        builder.set_performance_mode("precise")
    elif graph_size < 1000:
        builder.set_performance_mode("balanced")
    else:
        builder.set_performance_mode("fast")
    
    # Настройка по типу задачи
    if task_type == "exploration":
        # Для исследования концептов
        builder.set_propagation_mode(PropagationMode.SPREADING)\
               .set_depth_limit(4)\
               .set_lateral_inhibition(False)
    
    elif task_type == "classification":
        # Для классификации и категоризации
        builder.set_propagation_mode(PropagationMode.FOCUSING)\
               .set_lateral_inhibition(True, strength=0.4)\
               .set_activation_limits(threshold=0.3)
    
    elif task_type == "similarity":
        # Для анализа сходства
        builder.set_propagation_mode(PropagationMode.BIDIRECTIONAL)\
               .set_depth_limit(5)\
               .set_activation_limits(threshold=0.15)
    
    elif task_type == "recommendation":
        # Для систем рекомендаций
        builder.set_propagation_mode(PropagationMode.SPREADING)\
               .set_lateral_inhibition(True, strength=0.3)\
               .set_activation_limits(max_nodes=50)
    
    return builder.build()

# Пример использования
config = choose_optimal_config(len(graph.get_all_nodes()), "exploration")
```

### 2. Обработка ошибок

```python
# ✅ Правильная обработка ошибок

def robust_propagation(graph, initial_nodes: Dict[str, float], 
                      config: PropagationConfig = None,
                      max_retries: int = 3) -> PropagationResult:
    """Устойчивое выполнение распространения с обработкой ошибок"""
    
    if config is None:
        config = create_default_config()
    
    engine = create_default_engine(graph)
    
    for attempt in range(max_retries):
        try:
            # Валидация входных данных
            valid_nodes = {
                node_id: level for node_id, level in initial_nodes.items()
                if graph.has_node(node_id) and 0 <= level <= 1
            }
            
            if not valid_nodes:
                return PropagationResult(
                    success=False,
                    activated_nodes={},
                    error_message="Нет валидных начальных узлов"
                )
            
            # Выполнение распространения
            result = engine.propagate(valid_nodes, config)
            
            if result.success:
                return result
            
            # При неудаче снижаем требования
            if attempt < max_retries - 1:
                config.convergence_threshold *= 2  # Ослабляем порог сходимости
                config.max_iterations = min(config.max_iterations * 2, 500)
                
                print(f"Попытка {attempt + 1} неудачна, корректируем параметры...")
        
        except Exception as e:
            if attempt == max_retries - 1:
                return PropagationResult(
                    success=False,
                    activated_nodes={},
                    error_message=f"Критическая ошибка: {str(e)}"
                )
            
            print(f"Ошибка на попытке {attempt + 1}: {e}")
    
    return PropagationResult(
        success=False,
        activated_nodes={},
        error_message="Превышено количество попыток"
    )

# Использование
result = robust_propagation(graph, {"python": 1.0})
if result.success:
    print("Распространение успешно")
else:
    print(f"Ошибка: {result.error_message}")
```

### 3. Мониторинг производительности

```python
# ✅ Мониторинг производительности

class PropagationProfiler:
    """Профайлер для анализа производительности распространения"""
    
    def __init__(self):
        self.metrics = []
    
    def profile_propagation(self, 
                          graph, 
                          initial_nodes: Dict[str, float],
                          config: PropagationConfig) -> Dict[str, Any]:
        """Профилирование выполнения распространения"""
        
        import psutil
        import tracemalloc
        
        # Начало трассировки памяти
        tracemalloc.start()
        
        # Измерение CPU и памяти до выполнения
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Выполнение распространения
        engine = create_default_engine(graph)
        start_time = time.time()
        
        result = engine.propagate(initial_nodes, config)
        
        end_time = time.time()
        
        # Измерение ресурсов после выполнения
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Пиковое использование памяти
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Сбор метрик
        profile_data = {
            "execution_time": end_time - start_time,
            "cpu_usage_change": cpu_after - cpu_before,
            "memory_usage_mb": memory_after - memory_before,
            "peak_memory_mb": peak / 1024 / 1024,
            "graph_size": len(graph.get_all_nodes()),
            "graph_edges": len(graph.get_all_edges()),
            "initial_nodes_count": len(initial_nodes),
            "result_success": result.success,
            "activated_nodes_count": len(result.activated_nodes) if result.success else 0,
            "iterations_used": result.iterations_used if result.success else 0,
            "convergence_achieved": result.convergence_achieved if result.success else False,
            "config_summary": {
                "mode": config.propagation_mode.value,
                "max_iterations": config.max_iterations,
                "threshold": config.activation_threshold,
                "lateral_inhibition": config.lateral_inhibition
            }
        }
        
        self.metrics.append(profile_data)
        
        return profile_data
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Анализ трендов производительности"""
        
        if not self.metrics:
            return {"error": "Нет данных для анализа"}
        
        # Вычисление средних значений
        avg_execution_time = sum(m["execution_time"] for m in self.metrics) / len(self.metrics)
        avg_memory_usage = sum(m["memory_usage_mb"] for m in self.metrics) / len(self.metrics)
        
        # Поиск корреляций
        graph_sizes = [m["graph_size"] for m in self.metrics]
        execution_times = [m["execution_time"] for m in self.metrics]
        
        # Простая корреляция (для полноценного анализа нужен scipy)
        size_time_correlation = "high" if max(execution_times) / min(execution_times) > 3 else "low"
        
        return {
            "total_executions": len(self.metrics),
            "avg_execution_time": avg_execution_time,
            "avg_memory_usage_mb": avg_memory_usage,
            "success_rate": sum(1 for m in self.metrics if m["result_success"]) / len(self.metrics),
            "size_time_correlation": size_time_correlation,
            "performance_recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        
        recommendations = []
        
        # Анализ времени выполнения
        slow_executions = [m for m in self.metrics if m["execution_time"] > 2.0]
        if slow_executions:
            recommendations.append("Рассмотрите использование более быстрых конфигураций для больших графов")
        
        # Анализ использования памяти
        memory_intensive = [m for m in self.metrics if m["memory_usage_mb"] > 100]
        if memory_intensive:
            recommendations.append("Ограничьте количество активных узлов для снижения потребления памяти")
        
        # Анализ сходимости
        no_convergence = [m for m in self.metrics if not m["convergence_achieved"]]
        if len(no_convergence) > len(self.metrics) * 0.3:
            recommendations.append("Увеличьте порог сходимости или количество итераций")
        
        return recommendations

# Использование профайлера
profiler = PropagationProfiler()

# Профилирование нескольких выполнений
test_configs = [create_fast_config(), create_default_config(), create_precise_config()]
test_scenarios = [{"python": 1.0}, {"ai": 0.8, "ml": 0.6}, {"programming": 1.0}]

for config in test_configs:
    for scenario in test_scenarios:
        profile_data = profiler.profile_propagation(graph, scenario, config)
        print(f"Конфигурация {config.propagation_mode.value}: {profile_data['execution_time']:.3f}s")

# Анализ трендов
trends = profiler.analyze_performance_trends()
print(f"\nСтатистика производительности:")
print(f"Среднее время выполнения: {trends['avg_execution_time']:.3f}s")
print(f"Среднее потребление памяти: {trends['avg_memory_usage_mb']:.1f}MB")
print(f"Частота успеха: {trends['success_rate']:.1%}")

if trends['performance_recommendations']:
    print("\nРекомендации:")
    for rec in trends['performance_recommendations']:
        print(f"  - {rec}")
```

## Заключение

Модуль Propagation представляет собой мощную и гибкую систему для распространения активации по семантическим графам. Он обеспечивает:

### Основные преимущества:
- **Универсальность**: подходит для широкого спектра задач от исследования концептов до рекомендательных систем
- **Гибкость**: множество настраиваемых параметров и режимов работы
- **Производительность**: оптимизированные алгоритмы и возможности профилирования
- **Визуализация**: богатые возможности визуализации процесса и результатов
- **Интеграция**: легкая интеграция с другими модулями NeuroGraph

### Ключевые возможности:
- Различные алгоритмы распространения (spreading, focusing, bidirectional, constrained)
- Настраиваемые функции активации и затухания
- Латеральное торможение для моделирования конкуренции
- Готовые сценарии для типичных задач
- Подробная диагностика и анализ производительности

### Области применения:
- Системы рекомендаций
- Автоматическое тегирование контента
- Когнитивное моделирование
- Семантический поиск и анализ
- Исследование концептуальных связей

Модуль предоставляет как простые в использовании готовые функции для быстрого старта, так и продвинутые возможности для глубокой настройки под специфические задачи. Архитектура модуля позволяет легко расширять функциональность и интегрироваться с другими компонентами системы NeuroGraph.