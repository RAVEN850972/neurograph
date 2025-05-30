# Модуль Propagation - Документация для разработчиков

## Обзор модуля

Модуль `propagation` является ключевым компонентом системы NeuroGraph, отвечающим за распространение активации по семантическому графу знаний. Он имитирует процессы активации в нейронных сетях и когнитивных системах, позволяя "пробуждать" связанные концепты и находить семантические связи.

### Основное назначение
- Распространение активации от заданных узлов по графу знаний
- Моделирование ассоциативного мышления и семантического поиска
- Обнаружение скрытых связей между концептами
- Приоритизация релевантных знаний для задач рассуждения

## Архитектура модуля

```
neurograph/propagation/
├── __init__.py          # Публичный API модуля
├── base.py              # Базовые интерфейсы и классы данных
├── engine.py            # Основной движок распространения
├── functions.py         # Функции активации, затухания, торможения
├── visualizer.py        # Визуализация процесса распространения
└── factory.py           # Фабрики и утилиты для создания компонентов
```

## Основные интерфейсы для интеграции

### 1. IPropagationEngine - Главный интерфейс движка

```python
from neurograph.propagation import IPropagationEngine, PropagationConfig, PropagationResult

class IPropagationEngine(ABC):
    @abstractmethod
    def propagate(self, initial_nodes: Dict[str, float], 
                 config: PropagationConfig) -> PropagationResult:
        """Основной метод распространения активации."""
        pass
    
    @abstractmethod
    def set_graph(self, graph) -> None:
        """Установка графа знаний."""
        pass
    
    @abstractmethod
    def reset_activations(self) -> None:
        """Сброс состояния активации."""
        pass
```

**Использование в других модулях:**
```python
# В модуле Memory для поиска связанных воспоминаний
propagation_engine.set_graph(memory_graph)
result = propagation_engine.propagate({"current_concept": 1.0}, config)
related_memories = result.get_most_activated_nodes(top_n=5)

# В модуле Processor для активации связанных правил
propagation_engine.set_graph(knowledge_graph)
result = propagation_engine.propagate(query_concepts, config)
relevant_rules = [node_id for node_id in result.activated_nodes 
                  if result.activated_nodes[node_id].activation_level > 0.5]
```

### 2. Классы данных для обмена

#### PropagationConfig - Конфигурация распространения
```python
@dataclass
class PropagationConfig:
    # Основные параметры
    max_iterations: int = 100               # Максимальное количество итераций
    convergence_threshold: float = 0.001    # Порог сходимости
    activation_threshold: float = 0.1       # Минимальный порог активации
    max_active_nodes: int = 1000           # Максимальное количество активных узлов
    
    # Функции активации и затухания
    activation_function: ActivationFunction = ActivationFunction.SIGMOID
    decay_function: DecayFunction = DecayFunction.EXPONENTIAL
    
    # Режим распространения
    propagation_mode: PropagationMode = PropagationMode.SPREADING
    max_propagation_depth: int = 10
    
    # Латеральное торможение
    lateral_inhibition: bool = True
    inhibition_strength: float = 0.2
```

#### PropagationResult - Результат распространения
```python
@dataclass
class PropagationResult:
    success: bool                                    # Успешность выполнения
    activated_nodes: Dict[str, NodeActivation]      # Активированные узлы
    convergence_achieved: bool = False               # Достигнута ли сходимость
    iterations_used: int = 0                        # Использованные итерации
    processing_time: float = 0.0                    # Время обработки
    max_activation_reached: float = 0.0             # Максимальная активация
    
    def get_most_activated_nodes(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Получение наиболее активированных узлов."""
        pass
    
    def filter_active_nodes(self, threshold: float = 0.1) -> Dict[str, NodeActivation]:
        """Фильтрация активных узлов по порогу."""
        pass
```

#### NodeActivation - Состояние активации узла
```python
@dataclass
class NodeActivation:
    node_id: str                           # ID узла
    activation_level: float = 0.0          # Уровень активации (0.0-1.0)
    propagation_depth: int = 0             # Глубина от источника
    source_nodes: Set[str]                 # Узлы-источники активации
    metadata: Dict[str, Any]               # Дополнительные данные
    
    def is_active(self, threshold: float = 0.1) -> bool:
        """Проверка активности узла."""
        pass
```

## Быстрое создание компонентов

### Фабричные методы для простого использования

```python
from neurograph.propagation import (
    create_default_engine, create_default_config,
    create_fast_config, create_precise_config,
    PropagationFactory
)

# Создание движка с настройками по умолчанию
engine = create_default_engine(graph)

# Создание различных конфигураций
default_config = create_default_config()      # Сбалансированные настройки
fast_config = create_fast_config()            # Быстрое выполнение
precise_config = create_precise_config()      # Точное распространение

# Создание через фабрику
engine = PropagationFactory.create_engine("spreading_activation", graph=graph)
config = PropagationFactory.create_config(
    activation_function=ActivationFunction.SIGMOID,
    propagation_mode=PropagationMode.BIDIRECTIONAL
)
```

### Конструктор конфигураций (Builder Pattern)

```python
from neurograph.propagation import PropagationConfigBuilder

config = PropagationConfigBuilder().reset() \
    .set_performance_mode("balanced") \
    .set_activation_function(ActivationFunction.SIGMOID, steepness=2.0) \
    .set_propagation_mode(PropagationMode.SPREADING) \
    .set_iterations(150, convergence_threshold=0.001) \
    .set_activation_limits(threshold=0.15, max_nodes=500) \
    .set_lateral_inhibition(True, strength=0.3, radius=2) \
    .build()
```

## Режимы распространения

### PropagationMode - Алгоритмы распространения

```python
class PropagationMode(Enum):
    SPREADING = "spreading"          # Расходящееся от источников
    FOCUSING = "focusing"            # Сходящееся к центральным узлам  
    BIDIRECTIONAL = "bidirectional" # Двунаправленное
    CONSTRAINED = "constrained"     # Ограниченное по условиям
```

**Когда использовать каждый режим:**

- **SPREADING** - Исследование связанных концептов, ассоциативный поиск
- **FOCUSING** - Поиск центральных/важных концептов, концентрация внимания
- **BIDIRECTIONAL** - Поиск семантических связей между концептами
- **CONSTRAINED** - Целенаправленный поиск с ограничениями

## Интеграция с другими модулями

### 1. Интеграция с SemGraph (Семантический граф)

```python
# Пример использования с семантическим графом
from neurograph.semgraph import SemGraphFactory
from neurograph.propagation import create_default_engine, create_default_config

# Создание или получение графа
graph = SemGraphFactory.create("memory_efficient")
graph.add_node("apple", type="concept", category="fruit")
graph.add_node("fruit", type="concept", category="food")
graph.add_edge("apple", "fruit", "is_a", weight=0.9)

# Настройка распространения
engine = create_default_engine(graph)
config = create_default_config()

# Активация концепта "apple"
result = engine.propagate({"apple": 1.0}, config)

# Получение связанных концептов
related_concepts = result.get_most_activated_nodes(top_n=10)
for concept_id, activation_level in related_concepts:
    print(f"{concept_id}: {activation_level:.3f}")
```

### 2. Интеграция с Memory (Система памяти)

```python
# Пример интеграции с памятью для поиска релевантных воспоминаний
from neurograph.propagation import integrate_with_memory

def find_related_memories(memory_module, query_concepts: List[str]):
    # Создаем граф из структуры памяти
    memory_graph = memory_module.get_semantic_graph()
    
    # Настраиваем распространение для поиска в памяти
    engine = create_default_engine(memory_graph)
    config = create_fast_config()  # Быстрый поиск для интерактивности
    
    # Нормализуем входящие концепты
    initial_activations = {concept: 1.0 / len(query_concepts) 
                          for concept in query_concepts 
                          if memory_graph.has_node(concept)}
    
    # Выполняем распространение
    result = engine.propagate(initial_activations, config)
    
    # Интегрируем результаты с памятью
    integrate_with_memory(result, memory_module)
    
    # Возвращаем релевантные воспоминания
    return [
        memory_module.get_memory_by_concept(node_id)
        for node_id, activation in result.activated_nodes.items()
        if activation.activation_level > 0.3
    ]
```

### 3. Интеграция с Processor (Нейросимволический процессор)

```python
# Пример использования для активации релевантных правил
from neurograph.propagation import integrate_with_processor

def activate_relevant_rules(processor_module, context_facts: Dict[str, Any]):
    # Получаем граф правил и фактов
    knowledge_graph = processor_module.get_knowledge_graph()
    
    # Находим концепты, связанные с фактами из контекста
    context_concepts = processor_module.extract_concepts_from_facts(context_facts)
    
    # Настраиваем распространение для поиска правил
    engine = create_default_engine(knowledge_graph)
    config = PropagationConfigBuilder().reset() \
        .set_propagation_mode(PropagationMode.FOCUSING) \
        .set_filters(node_types=["rule", "inference_pattern"]) \
        .build()
    
    # Выполняем распространение от концептов контекста
    result = engine.propagate(context_concepts, config)
    
    # Создаем контекст для процессора
    processing_context = integrate_with_processor(result, processor_module)
    
    # Возвращаем активированные правила
    return [
        processor_module.get_rule(node_id)
        for node_id, activation in result.activated_nodes.items()
        if activation.activation_level > 0.4 and 
           knowledge_graph.get_node(node_id).get("type") == "rule"
    ]
```

## Готовые сценарии использования

### 1. Исследование концептов

```python
from neurograph.propagation import scenario_concept_exploration

# Исследование концептов вокруг заданного понятия
exploration_result = scenario_concept_exploration(
    graph=knowledge_graph,
    start_concept="artificial_intelligence",
    exploration_depth=3
)

print(f"Найдено связанных концептов: {len(exploration_result['related_concepts'])}")
for concept_info in exploration_result['related_concepts'][:5]:
    print(f"- {concept_info['concept']}: {concept_info['relevance']:.3f}")
```

### 2. Активация знаний

```python
from neurograph.propagation import scenario_knowledge_activation

# Активация знаний по набору концептов
activation_result = scenario_knowledge_activation(
    graph=knowledge_graph,
    query_concepts=["machine_learning", "neural_networks", "deep_learning"],
    focus_mode=True
)

primary_knowledge = activation_result['activated_knowledge']['primary']
connections = activation_result['activated_knowledge']['connections']
insights = activation_result['activated_knowledge']['insights']
```

### 3. Семантическая близость

```python
from neurograph.propagation import scenario_semantic_similarity

# Оценка семантической близости между концептами
similarity_result = scenario_semantic_similarity(
    graph=knowledge_graph,
    concept1="dog",
    concept2="cat",
    max_depth=4
)

print(f"Общая близость: {similarity_result['overall_similarity']:.3f}")
print(f"Уровень связи: {similarity_result['analysis']['similarity_level']}")
```

## Функции активации и затухания

### Доступные функции активации

```python
class ActivationFunction(Enum):
    LINEAR = "linear"        # Линейная функция
    SIGMOID = "sigmoid"      # Сигмоида (по умолчанию)
    TANH = "tanh"           # Гиперболический тангенс
    RELU = "relu"           # Rectified Linear Unit
    SOFTMAX = "softmax"     # Softmax (для нормализации)
    THRESHOLD = "threshold"  # Пороговая функция
    GAUSSIAN = "gaussian"    # Гауссовская функция
```

### Доступные функции затухания

```python
class DecayFunction(Enum):
    EXPONENTIAL = "exponential"  # Экспоненциальное затухание (по умолчанию)
    LINEAR = "linear"            # Линейное затухание
    LOGARITHMIC = "logarithmic"  # Логарифмическое затухание
    POWER = "power"              # Степенное затухание
    STEP = "step"                # Ступенчатое затухание
    NONE = "none"                # Без затухания
```

## Визуализация результатов

```python
from neurograph.propagation import PropagationFactory

# Создание визуализатора
visualizer = PropagationFactory.create_visualizer()

# Статическая визуализация
visualizer.visualize_propagation(
    result=propagation_result,
    graph=knowledge_graph,
    save_path="propagation_result.png",
    show_animation=False
)

# Анимированная визуализация процесса
visualizer.visualize_propagation(
    result=propagation_result,
    graph=knowledge_graph,
    save_path="propagation_animation.gif",
    show_animation=True
)

# Тепловая карта активации
current_activations = {node_id: activation.activation_level 
                      for node_id, activation in propagation_result.activated_nodes.items()}
visualizer.create_activation_heatmap(
    activations=current_activations,
    graph=knowledge_graph,
    save_path="activation_heatmap.png"
)
```

## Диагностика и отладка

### Анализ конфигурации

```python
from neurograph.propagation import PropagationDiagnostics

diagnostics = PropagationDiagnostics()

# Анализ конфигурации на потенциальные проблемы
config_analysis = diagnostics.analyze_config(config)
print(f"Рейтинг производительности: {config_analysis['performance_rating']}")
print(f"Использование памяти: {config_analysis['memory_usage_estimate']}")

if config_analysis['warnings']:
    print("Предупреждения:")
    for warning in config_analysis['warnings']:
        print(f"- {warning}")

# Оценка времени выполнения
time_estimate = diagnostics.estimate_execution_time(config, graph_size=1000)
print(f"Ожидаемое время: {time_estimate['estimated_seconds']:.2f} секунд")
```

### Отладочное выполнение

```python
from neurograph.propagation import debug_propagation

# Детальный анализ процесса распространения
debug_info = debug_propagation(
    graph=knowledge_graph,
    initial_nodes={"concept1": 1.0, "concept2": 0.5},
    config=config
)

print("Статистика выполнения:")
print(f"- Успех: {debug_info['execution_results']['success']}")
print(f"- Время: {debug_info['execution_results']['processing_time']:.3f}с")
print(f"- Итерации: {debug_info['execution_results']['iterations_used']}")
print(f"- Сходимость: {debug_info['execution_results']['convergence_achieved']}")
```

## Рекомендации по производительности

### 1. Выбор режима производительности

```python
# Для интерактивных приложений
fast_config = PropagationConfigBuilder().reset() \
    .set_performance_mode("fast") \
    .build()

# Для точного анализа
precise_config = PropagationConfigBuilder().reset() \
    .set_performance_mode("precise") \
    .build()

# Для исследований
research_config = PropagationConfigBuilder().reset() \
    .set_performance_mode("balanced") \
    .set_iterations(500, convergence_threshold=0.0001) \
    .build()
```

### 2. Оптимизация для больших графов

```python
# Ограничение области поиска
constrained_config = PropagationConfigBuilder().reset() \
    .set_propagation_mode(PropagationMode.CONSTRAINED) \
    .set_depth_limit(3) \
    .set_activation_limits(threshold=0.3, max_nodes=100) \
    .set_filters(edge_types=["is_a", "related_to"]) \
    .build()
```

### 3. Настройка для специфических задач

```python
# Для поиска ассоциаций
association_config = PropagationConfigBuilder().reset() \
    .set_propagation_mode(PropagationMode.SPREADING) \
    .set_lateral_inhibition(False) \
    .set_activation_function(ActivationFunction.SIGMOID, steepness=1.5) \
    .build()

# Для концентрации внимания
focus_config = PropagationConfigBuilder().reset() \
    .set_propagation_mode(PropagationMode.FOCUSING) \
    .set_lateral_inhibition(True, strength=0.4) \
    .set_activation_function(ActivationFunction.GAUSSIAN) \
    .build()
```

## Обработка ошибок

```python
from neurograph.propagation import (
    PropagationError, GraphNotSetError, 
    InvalidConfigurationError, ConvergenceError
)

try:
    result = engine.propagate(initial_nodes, config)
    
    if not result.success:
        print(f"Ошибка распространения: {result.error_message}")
        return None
        
except GraphNotSetError:
    print("Граф не установлен в движке")
    engine.set_graph(knowledge_graph)
    
except InvalidConfigurationError as e:
    print(f"Некорректная конфигурация: {e.message}")
    if e.config_errors:
        for error in e.config_errors:
            print(f"- {error}")
            
except ConvergenceError as e:
    print(f"Проблемы со сходимостью: {e.message}")
    print(f"Использовано итераций: {e.iterations_used}")
    
except PropagationError as e:
    print(f"Общая ошибка распространения: {e.message}")
    if e.details:
        print(f"Детали: {e.details}")
```

## Заключение

Модуль `propagation` предоставляет мощный и гибкий инструмент для работы с активацией в графах знаний. Основные принципы интеграции:

1. **Используйте фабричные методы** для быстрого создания компонентов
2. **Настраивайте конфигурацию** под конкретные задачи через Builder
3. **Обрабатывайте результаты** через стандартные интерфейсы
4. **Интегрируйтесь** с другими модулями через готовые функции
5. **Мониторьте производительность** через диагностические утилиты

Модуль спроектирован для простой интеграции с другими компонентами системы NeuroGraph, предоставляя богатый набор возможностей для моделирования когнитивных процессов и семантического поиска.