# NeuroGraph Modules Documentation

Полная документация для модулей Core, SemGraph, ContextVec и Memory системы NeuroGraph.

## Содержание

1. [Введение](#введение)
2. [Core Module](#core-module)
3. [SemGraph Module](#semgraph-module)
4. [ContextVec Module](#contextvec-module)
5. [Memory Module](#memory-module)
6. [Практические примеры](#практические-примеры)

---

## Введение

NeuroGraph - это модульная нейросимволическая система для создания персонального ассистента. Система состоит из взаимосвязанных модулей, каждый из которых отвечает за определенную функциональность.

### Архитектурная схема

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     NLP     │    │  Processor  │    │ Propagation │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌─────────────┐
                    │ Integration │
                    │             │
                    └─────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SemGraph  │    │ ContextVec  │    │   Memory    │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌─────────────┐
                    │    Core     │
                    │             │
                    └─────────────┘
```

### Принципы работы

- **Модульность**: Каждый модуль имеет четкую ответственность
- **Слабая связанность**: Модули взаимодействуют через интерфейсы
- **Событийность**: Асинхронное взаимодействие через систему событий
- **Конфигурируемость**: Все настройки через файлы конфигурации

---

## Core Module

Базовый модуль, предоставляющий инфраструктуру для всех остальных компонентов системы.

### Основные компоненты

#### 1. Система компонентов

**Файл:** `neurograph/core/component.py`

Базовые классы для всех компонентов системы:

```python
from neurograph.core.component import Component, Configurable

class MyComponent(Component, Configurable):
    def __init__(self, component_id: str):
        super().__init__(component_id)
        self.config = {}
    
    def initialize(self) -> bool:
        self.logger.info("Инициализация компонента")
        return True
    
    def shutdown(self) -> bool:
        self.logger.info("Завершение работы компонента")
        return True
    
    def configure(self, config: Dict[str, Any]) -> bool:
        self.config = config
        return True
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
```

**Ключевые методы:**
- `initialize()` - инициализация компонента
- `shutdown()` - корректное завершение работы
- `configure()` - настройка через конфигурацию
- `get_config()` - получение текущей конфигурации

#### 2. Система конфигурации

**Файл:** `neurograph/core/config.py`

Управление настройками компонентов:

```python
from neurograph.core.config import Configuration

# Создание конфигурации
config = Configuration({
    "database": {
        "host": "localhost",
        "port": 5432
    },
    "memory": {
        "stm_capacity": 100,
        "ltm_capacity": 10000
    }
})

# Получение значений
db_host = config.get("database.host")  # "localhost"
stm_size = config.get("memory.stm_capacity", 50)  # 100

# Установка значений
config.set("database.host", "remote.server")

# Сохранение и загрузка
config.save_to_file("config.json")
config2 = Configuration.load_from_file("config.json")

# Объединение конфигураций
merged = merge_configs(config, config2)

# Конфигурация из переменных окружения
env_config = from_env("NEUROGRAPH_")
```

**Особенности:**
- Точечная нотация для доступа к вложенным значениям
- Автоматическое создание промежуточных объектов
- Слияние конфигураций с приоритетами

#### 3. Система логирования

**Файл:** `neurograph/core/logging.py`

Централизованное логирование для всех компонентов:

```python
from neurograph.core.logging import get_logger, setup_logging

# Настройка логирования
setup_logging(
    level="INFO",
    log_file="neurograph.log",
    rotation="10 MB",
    retention="1 week"
)

# Получение логгера с контекстом
logger = get_logger("my_module", user_id="123", session="abc")

# Использование
logger.info("Обработка начата")
logger.warning("Внимание: низкая память")
logger.error("Ошибка обработки")
logger.debug("Отладочная информация")
```

**Возможности:**
- Контекстное логирование
- Ротация файлов
- Цветной вывод в консоль
- Настраиваемые форматы

#### 4. Система событий

**Файл:** `neurograph/core/events.py`

Асинхронное взаимодействие между компонентами:

```python
from neurograph.core.events import subscribe, publish, unsubscribe

# Подписка на события
def handle_memory_full(data):
    print(f"Память переполнена: {data['memory_level']}")

subscription_id = subscribe("memory.full", handle_memory_full)

# Публикация события
publish("memory.full", {
    "memory_level": "STM",
    "current_size": 150,
    "max_size": 100
})

# Отписка от события
unsubscribe("memory.full", subscription_id)

# Проверка наличия подписчиков
if has_subscribers("memory.full"):
    publish("memory.full", data)
```

**Типы событий в системе:**
- `memory.*` - события памяти
- `graph.*` - события графа
- `processor.*` - события процессора
- `system.*` - системные события

#### 5. Система кеширования

**Файл:** `neurograph/core/cache.py`

Кеширование для повышения производительности:

```python
from neurograph.core.cache import global_cache, cached, Cache

# Использование глобального кеша
global_cache.set("user_123", user_data, ttl=300)  # 5 минут
user = global_cache.get("user_123")

# Декоратор для функций
@cached(ttl=60)  # 1 минута
def expensive_computation(param1, param2):
    # Тяжелые вычисления
    return result

# Собственный кеш
my_cache = Cache(max_size=1000, ttl=600)  # 10 минут

# Статистика кеша
stats = global_cache.stats()
print(f"Размер кеша: {stats['size']}")
print(f"Попаданий: {stats['hit_count']}")
```

#### 6. Реестр компонентов

**Файл:** `neurograph/core/utils/registry.py`

Динамическая регистрация и создание компонентов:

```python
from neurograph.core.utils.registry import Registry

# Создание реестра
processors = Registry[IProcessor]("processors")

# Регистрация компонентов
processors.register("rule_based", RuleBasedProcessor)
processors.register("neural", NeuralProcessor)

# Создание экземпляров
processor = processors.create("rule_based", config=my_config)

# Декораторная регистрация
@processors.decorator("hybrid")
class HybridProcessor(IProcessor):
    pass

# Получение списка
available = processors.get_names()  # ["rule_based", "neural", "hybrid"]
```

#### 7. Обработка ошибок

**Файл:** `neurograph/core/errors.py`

Типизированные исключения:

```python
from neurograph.core.errors import (
    NeuroGraphError, ConfigurationError, ComponentError,
    ValidationError, GraphError, VectorError
)

# Базовое исключение
try:
    risky_operation()
except NeuroGraphError as e:
    print(f"Ошибка: {e.message}")
    print(f"Детали: {e.details}")

# Специфичные исключения
try:
    process_config(invalid_config)
except ConfigurationError as e:
    handle_config_error(e)

try:
    graph.get_node("nonexistent")
except NodeNotFoundError as e:
    handle_missing_node(e)
```

#### 8. Мониторинг ресурсов

**Файл:** `neurograph/core/resources.py`

Отслеживание использования системных ресурсов:

```python
from neurograph.core.resources import (
    start_resource_monitoring, 
    get_resource_usage,
    stop_resource_monitoring
)

# Запуск мониторинга
start_resource_monitoring(check_interval=5.0)

# Получение статистики
usage = get_resource_usage()
print(f"CPU: {usage['cpu_percent']}%")
print(f"Память: {usage['memory_rss'] / 1024 / 1024:.1f} MB")
print(f"Потоки: {usage['thread_count']}")

# Остановка мониторинга
stop_resource_monitoring()
```

### Использование Core в других модулях

```python
# Базовый шаблон для нового модуля
from neurograph.core import Component, Configuration
from neurograph.core.logging import get_logger
from neurograph.core.events import subscribe, publish
from neurograph.core.cache import cached

class MyModule(Component):
    def __init__(self, module_id: str, config: Configuration):
        super().__init__(module_id)
        self.config = config
        
        # Подписка на события
        subscribe("system.shutdown", self._handle_shutdown)
    
    @cached(ttl=300)
    def expensive_operation(self, data):
        self.logger.info("Выполнение операции")
        # ... логика ...
        publish("module.operation_completed", {"data": result})
        return result
    
    def _handle_shutdown(self, data):
        self.logger.info("Получен сигнал завершения")
        self.shutdown()
```

---

## SemGraph Module

Модуль семантического графа для хранения и поиска знаний в виде узлов и связей.

### Основные концепции

#### Граф знаний
Семантический граф представляет знания в виде триплетов:
- **Узел (Node)** - сущность (человек, понятие, объект)
- **Ребро (Edge)** - отношение между сущностями
- **Атрибуты** - свойства узлов и ребер

Пример:
```
(Python) --[written_in]--> (Programming Language)
(Django) --[framework_for]--> (Python)
(FastAPI) --[alternative_to]--> (Django)
```

### Интерфейсы

**Файл:** `neurograph/semgraph/base.py`

Основной интерфейс `ISemGraph`:

```python
from neurograph.semgraph.base import ISemGraph

class MyGraphImplementation(ISemGraph):
    def add_node(self, node_id: str, **attributes) -> None:
        pass
    
    def add_edge(self, source: str, target: str, 
                edge_type: str = "default", weight: float = 1.0, 
                **attributes) -> None:
        pass
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    # ... остальные методы
```

### Фабрика графов

**Файл:** `neurograph/semgraph/factory.py`

```python
from neurograph.semgraph import SemGraphFactory

# Создание графа в памяти
graph = SemGraphFactory.create("memory_efficient")

# Создание персистентного графа
persistent_graph = SemGraphFactory.create("persistent", 
                                        file_path="knowledge.json",
                                        auto_save_interval=300.0)
```

**Доступные типы:**
- `memory_efficient` - граф в памяти (быстрый)
- `persistent` - граф с сохранением на диск

### Базовые операции с графом

#### Добавление узлов и ребер

```python
# Добавление узлов с атрибутами
graph.add_node("python", 
               type="programming_language",
               description="High-level programming language",
               year_created=1991,
               popularity="high")

graph.add_node("django", 
               type="framework",
               description="Web framework for Python",
               license="BSD")

# Добавление ребер с весами и атрибутами
graph.add_edge("django", "python", "written_in", 
               weight=1.0,
               confidence=0.95,
               source="official_docs")

graph.add_edge("python", "programming_language", "is_a",
               weight=1.0)
```

#### Поиск и навигация

```python
# Проверка существования
if graph.has_node("python"):
    print("Узел Python существует")

if graph.has_edge("django", "python", "written_in"):
    print("Django написан на Python")

# Получение узлов и атрибутов
python_node = graph.get_node("python")
print(f"Тип: {python_node['type']}")
print(f"Описание: {python_node['description']}")

# Получение ребра и его атрибутов
edge = graph.get_edge("django", "python", "written_in")
print(f"Уверенность: {edge['confidence']}")

# Поиск соседей
neighbors = graph.get_neighbors("python")
frameworks = graph.get_neighbors("python", edge_type="written_in")

# Получение веса ребра
weight = graph.get_edge_weight("django", "python", "written_in")

# Обновление веса
graph.update_edge_weight("django", "python", 0.9, "written_in")
```

#### Получение всех элементов

```python
# Все узлы
all_nodes = graph.get_all_nodes()
print(f"Всего узлов: {len(all_nodes)}")

# Все ребра
all_edges = graph.get_all_edges()
for source, target, edge_type in all_edges:
    print(f"{source} --[{edge_type}]--> {target}")
```

### Поиск по шаблонам

**Файл:** `neurograph/semgraph/query/pattern.py`

#### Простые шаблоны

```python
from neurograph.semgraph.query.pattern import PatternMatcher, Pattern

matcher = PatternMatcher(graph)

# Поиск всех фреймворков для Python
pattern = Pattern(predicate="written_in", object="python")
frameworks = matcher.match(pattern)

# Поиск всех языков программирования
pattern = Pattern(predicate="is_a", object="programming_language")
languages = matcher.match(pattern)

# Поиск всех связей Django
pattern = Pattern(subject="django")
django_relations = matcher.match(pattern)
```

#### Поиск с регулярными выражениями

```python
# Поиск узлов, начинающихся с "py"
results = matcher.match_with_regex(subject_pattern=r"^py.*")

# Поиск отношений типа "is_*"
results = matcher.match_with_regex(predicate_pattern=r"is_.*")

# Поиск целей, содержащих "language"
results = matcher.match_with_regex(object_pattern=r".*language.*")
```

#### Поиск по атрибутам узлов

```python
# Поиск языков программирования высокой популярности
results = matcher.match_with_attributes(
    subject_attrs={"type": "programming_language", "popularity": "high"}
)

# Поиск фреймворков с лицензией BSD
results = matcher.match_with_attributes(
    subject_attrs={"type": "framework", "license": "BSD"},
    edge_type="written_in"
)
```

### Поиск путей

**Файл:** `neurograph/semgraph/query/path.py`

#### Кратчайшие пути

```python
from neurograph.semgraph.query.path import PathFinder

path_finder = PathFinder(graph)

# Кратчайший путь между узлами
path = path_finder.find_shortest_path("django", "programming_language")
for source, target, edge_type in path:
    print(f"{source} --[{edge_type}]--> {target}")

# Кратчайший путь с учетом весов (алгоритм Дейкстры)
weighted_path = path_finder.find_weighted_shortest_path("django", "programming_language")

# Все пути с ограничением глубины
all_paths = path_finder.find_all_paths("django", "programming_language", max_depth=3)
print(f"Найдено {len(all_paths)} путей")
```

#### Фильтрация по типам ребер

```python
# Поиск только по определенным типам отношений
path = path_finder.find_shortest_path("django", "python", 
                                    edge_types=["written_in", "uses"])

# Исключение определенных типов
all_edge_types = ["written_in", "uses", "similar_to", "alternative_to"]
allowed_types = [t for t in all_edge_types if t != "alternative_to"]
path = path_finder.find_shortest_path("django", "python", 
                                    edge_types=allowed_types)
```

### Расширенный поиск

**Файл:** `neurograph/semgraph/query/search.py`

#### Поиск по атрибутам

```python
from neurograph.semgraph.query.search import GraphSearcher

searcher = GraphSearcher(graph)

# Поиск по значению атрибута
high_pop_languages = searcher.search_by_attribute(
    "popularity", "high", 
    node_types=["programming_language"]
)

# Поиск по регулярному выражению
web_frameworks = searcher.search_by_regex(
    "description", r".*web.*framework.*",
    node_types=["framework"]
)
```

#### Поиск по расстоянию

```python
# Узлы в радиусе 2 ребер от Python
nearby = searcher.search_by_distance("python", max_distance=2)
for node_id, distance in nearby.items():
    print(f"{node_id}: расстояние {distance}")

# Только через определенные типы ребер
related = searcher.search_by_distance("python", max_distance=2, 
                                     edge_types=["written_in", "uses"])
```

#### Поиск похожих узлов

```python
# Узлы с похожими связями (сходство Жаккара)
similar = searcher.search_nodes_with_similar_connections("django", min_similarity=0.3)
for node_id, similarity in similar:
    print(f"{node_id}: сходство {similarity:.2f}")
```

#### Поиск подграфов по шаблону

```python
# Поиск паттерна "фреймворк написан на языке"
pattern = {
    "nodes": [
        {"id": "framework", "type": "framework"},
        {"id": "language", "type": "programming_language"}
    ],
    "edges": [
        {"source": "framework", "target": "language", "type": "written_in"}
    ]
}

matches = searcher.search_subgraph_pattern(pattern)
for match in matches:
    print(f"Фреймворк: {match['framework']}, Язык: {match['language']}")
```

### Семантический поиск

```python
from neurograph.semgraph.query.search import SemanticSearcher

# Требует провайдер векторных представлений
semantic_searcher = SemanticSearcher(graph, vectors_provider)

# Поиск узлов по семантической близости
results = semantic_searcher.search_by_text_similarity(
    "web development framework", 
    attribute_name="description",
    top_n=5
)

for node_id, similarity in results:
    print(f"{node_id}: сходство {similarity:.2f}")
```

### Анализ графа

**Файл:** `neurograph/semgraph/analysis/metrics.py`

#### Базовые метрики

```python
from neurograph.semgraph.analysis.metrics import GraphMetrics

metrics = GraphMetrics(graph)

# Размер графа
node_count = metrics.get_node_count()
edge_count = metrics.get_edge_count()
print(f"Граф: {node_count} узлов, {edge_count} ребер")

# Степени узлов
degrees = metrics.get_node_degrees()
for node_id, degree_info in degrees.items():
    print(f"{node_id}: входящие={degree_info['in']}, "
          f"исходящие={degree_info['out']}, всего={degree_info['total']}")
```

#### Центральность узлов

```python
# Наиболее центральные узлы (PageRank)
central = metrics.get_central_nodes(top_n=10)
for node_id, centrality in central:
    print(f"{node_id}: центральность {centrality:.3f}")
```

#### Кратчайшие пути

```python
# Пути от одного узла до нескольких
targets = ["django", "flask", "fastapi"]
paths = metrics.get_shortest_paths("python", targets)

for target, path in paths.items():
    if path:
        print(f"Путь до {target}: {len(path)} ребер")
    else:
        print(f"Путь до {target}: не найден")
```

#### Связные компоненты и сообщества

```python
# Связные компоненты
components = metrics.get_connected_components()
print(f"Связных компонент: {len(components)}")
for i, component in enumerate(components):
    print(f"Компонента {i+1}: {len(component)} узлов")

# Сообщества (требует library community или использует NetworkX)
communities = metrics.get_communities(resolution=1.0)
print(f"Сообществ: {len(communities)}")
```

#### Полная статистика

```python
# Все метрики сразу
all_metrics = metrics.compute_all_metrics()
print(f"Плотность графа: {all_metrics['density']:.3f}")
print(f"Средняя степень: {all_metrics['average_degree']:.1f}")
print(f"Связных компонент: {all_metrics['connected_components']}")
```

### Визуализация

**Файл:** `neurograph/semgraph/visualization/visualizer.py`

#### Базовая визуализация

```python
from neurograph.semgraph.visualization.visualizer import GraphVisualizer

visualizer = GraphVisualizer(graph)

# Простая визуализация
visualizer.visualize(output_path="graph.png", show=True)

# Настройка размера и фигуры
visualizer.visualize(
    output_path="large_graph.png",
    show=False,
    node_size=800,
    figsize=(16, 12)
)
```

#### Визуализация подграфов

```python
# Подграф конкретных узлов
important_nodes = ["python", "django", "flask", "fastapi"]
visualizer.visualize_subgraph(
    important_nodes,
    output_path="frameworks.png",
    include_neighbors=True,  # Включить соседей
    node_size=600
)
```

#### Экспорт в различные форматы

```python
# GraphML для Gephi, Cytoscape
visualizer.save_as_graphml("graph.graphml")

# GEXF для Gephi
visualizer.save_as_gexf("graph.gexf")

# JSON для Cytoscape.js
visualizer.export_to_cytoscape("graph.json")
```

### Индексирование HNSW

**Файл:** `neurograph/semgraph/index/hnsw.py`

Быстрый поиск ближайших соседей:

```python
from neurograph.semgraph.index.hnsw import HNSWIndex
import numpy as np

# Создание индекса
index = HNSWIndex(dim=384, max_elements=10000)

# Добавление элементов
for node_id in graph.get_all_nodes():
    # Векторное представление узла
    vector = get_node_embedding(node_id)  # ваша функция
    index.add_item(node_id, vector)

# Поиск ближайших соседей
query_vector = get_query_embedding("machine learning")
similar_nodes = index.search(query_vector, k=10)

for node_id, similarity in similar_nodes:
    print(f"{node_id}: сходство {similarity:.3f}")

# Сохранение и загрузка индекса
index.save("node_index")
loaded_index = HNSWIndex.load("node_index")
```

### Работа с персистентным графом

```python
# Создание персистентного графа
persistent_graph = SemGraphFactory.create("persistent", 
                                        file_path="knowledge.json",
                                        auto_save_interval=300.0)  # автосохранение каждые 5 минут

# Работа как с обычным графом
persistent_graph.add_node("new_concept", type="concept")

# Принудительное сохранение
persistent_graph.save_now()

# Перезагрузка из файла (отмена несохраненных изменений)
persistent_graph.reload()

# Корректное закрытие
persistent_graph.close()
```

### Сериализация и экспорт

```python
from neurograph.semgraph.impl.memory_graph import MemoryEfficientSemGraph

# Сериализация в словарь
data = graph.serialize()

# Создание из сериализованных данных
new_graph = MemoryEfficientSemGraph.deserialize(data)

# JSON экспорт/импорт
json_str = graph.to_json()
graph_from_json = MemoryEfficientSemGraph.from_json(json_str)

# Сохранение/загрузка файлов
graph.save("my_graph.json")
loaded_graph = MemoryEfficientSemGraph.load("my_graph.json")

# Объединение графов
graph1.merge(graph2)  # graph2 объединяется с graph1
```

---

## ContextVec Module

Модуль для работы с векторными представлениями (эмбеддингами) текстов, слов и понятий.

### Основные концепции

Векторные представления позволяют:
- Преобразовывать текст в числовые векторы
- Вычислять семантическую близость
- Выполнять быстрый поиск похожих элементов
- Кластеризовать похожие понятия

### Интерфейсы

**Файл:** `neurograph/contextvec/base.py`

Основной интерфейс `IContextVectors`:

```python
from neurograph.contextvec.base import IContextVectors
import numpy as np

class MyVectorImplementation(IContextVectors):
    def create_vector(self, key: str, vector: np.ndarray) -> bool:
        pass
    
    def get_vector(self, key: str) -> Optional[np.ndarray]:
        pass
    
    def similarity(self, key1: str, key2: str) -> Optional[float]:
        pass
    
    def get_most_similar(self, key: str, top_n: int = 5) -> List[Tuple[str, float]]:
        pass
    
    # ... остальные методы
```

### Фабрика векторных представлений

**Файл:** `neurograph/contextvec/factory.py`

```python
from neurograph.contextvec import ContextVectorsFactory

# Статические векторы (простые, быстрые)
static_vectors = ContextVectorsFactory.create("static", vector_size=100)

# Динамические векторы (с индексацией и обновлением)
dynamic_vectors = ContextVectorsFactory.create("dynamic", 
                                              vector_size=384, 
                                              use_indexing=True)
```

### Статические векторные представления

**Файл:** `neurograph/contextvec/impl/static.py`

Простое хранилище векторов в памяти:

```python
from neurograph.contextvec.impl.static import StaticContextVectors
import numpy as np

# Создание хранилища
vectors = StaticContextVectors(vector_size=100)

# Добавление векторов
word_vector = np.random.random(100)
vectors.create_vector("machine_learning", word_vector)

phrase_vector = np.random.random(100)
vectors.create_vector("artificial_intelligence", phrase_vector)

# Получение векторов
ml_vector = vectors.get_vector("machine_learning")
if ml_vector is not None:
    print(f"Размерность вектора: {ml_vector.shape}")

# Вычисление сходства
similarity = vectors.similarity("machine_learning", "artificial_intelligence")
print(f"Сходство: {similarity:.3f}")

# Поиск похожих
similar = vectors.get_most_similar("machine_learning", top_n=5)
for word, score in similar:
    print(f"{word}: {score:.3f}")

# Проверка наличия
if vectors.has_key("deep_learning"):
    print("Вектор deep_learning существует")

# Получение всех ключей
all_keys = vectors.get_all_keys()
print(f"Всего векторов: {len(all_keys)}")

# Удаление вектора
vectors.remove_vector("old_concept")
```

### Динамические векторные представления

**Файл:** `neurograph/contextvec/impl/dynamic.py`

Продвинутое хранилище с индексацией и обновлением:

```python
from neurograph.contextvec.impl.dynamic import DynamicContextVectors
import numpy as np

# Создание с индексацией HNSW
vectors = DynamicContextVectors(vector_size=384, use_indexing=True)

# Добавление векторов
concepts = ["машинное обучение", "глубокое обучение", "нейронные сети"]
for concept in concepts:
    vector = np.random.random(384)  # В реальности - эмбеддинг
    vectors.create_vector(concept, vector)

# Быстрый поиск похожих (через HNSW индекс)
similar = vectors.get_most_similar("машинное обучение", top_n=10)

# Обновление существующего вектора
new_vector = np.random.random(384)
vectors.update_vector("машинное обучение", new_vector, learning_rate=0.1)

# Усреднение векторов
concepts_to_average = ["машинное обучение", "глубокое обучение"]
averaged = vectors.average_vectors(concepts_to_average)
if averaged is not None:
    vectors.create_vector("ИИ_концепции", averaged)

# Сохранение и загрузка
vectors.save("my_vectors.json")
loaded_vectors = DynamicContextVectors.load("my_vectors.json")
```

**Особенности динамических векторов:**
- Автоматическая индексация для быстрого поиска
- Потокобезопасность
- Инкрементальное обновление векторов
- Усреднение векторов
- Сохранение/загрузка состояния

### Адаптеры для внешних моделей

#### Word2Vec адаптер

**Файл:** `neurograph/contextvec/adapters/word2vec.py`

```python
from neurograph.contextvec.adapters.word2vec import Word2VecAdapter

# Загрузка предобученной модели Word2Vec
adapter = Word2VecAdapter("path/to/word2vec.bin")

# Получение размерности
vector_size = adapter.get_vector_size()
print(f"Размерность векторов: {vector_size}")

# Кодирование отдельного текста
text = "машинное обучение и искусственный интеллект"
vector = adapter.encode(text, normalize=True)
print(f"Вектор текста: {vector.shape}")

# Пакетное кодирование
texts = [
    "обработка естественного языка",
    "компьютерное зрение", 
    "глубокое обучение"
]
vectors = adapter.encode_batch(texts, normalize=True)
print(f"Матрица векторов: {vectors.shape}")

# Поиск похожих слов (если доступен словарь)
similar_words = adapter.get_most_similar("программирование", top_n=10)
for word, similarity in similar_words:
    print(f"{word}: {similarity:.3f}")
```

**Поддерживаемые форматы:**
- `.bin` - бинарный формат Word2Vec
- `.vec` - текстовый формат Word2Vec
- Модели gensim

#### Sentence Transformers адаптер

**Файл:** `neurograph/contextvec/adapters/sentence.py`

```python
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter

# Инициализация с предобученной моделью
adapter = SentenceTransformerAdapter("all-MiniLM-L6-v2")

# Получение размерности
vector_size = adapter.get_vector_size()  # 384 для all-MiniLM-L6-v2

# Кодирование предложений
sentence = "NeuroGraph - это система для обработки знаний"
vector = adapter.encode(sentence, normalize=True)

# Пакетное кодирование с настройками
sentences = [
    "Граф знаний содержит узлы и связи",
    "Векторные представления позволяют измерять близость",
    "Память имеет несколько уровней"
]
vectors = adapter.encode_batch(sentences, 
                              batch_size=16, 
                              normalize=True)

print(f"Закодировано {len(sentences)} предложений")
print(f"Размер каждого вектора: {vectors[0].shape}")
```

**Популярные модели:**
- `all-MiniLM-L6-v2` - быстрая, 384 измерения
- `all-mpnet-base-v2` - качественная, 768 измерений
- `paraphrase-multilingual-MiniLM-L12-v2` - многоязычная

### Легковесные модели

**Файл:** `neurograph/contextvec/models/lightweight.py`

Модели, не требующие предобучения:

#### Хеширующий векторизатор

```python
from neurograph.contextvec.models.lightweight import HashingVectorizer

# Создание векторизатора
vectorizer = HashingVectorizer(
    vector_size=100,
    lowercase=True,
    ngram_range=(1, 2)  # униграммы и биграммы
)

# Векторизация текста
text = "машинное обучение и анализ данных"
vector = vectorizer.transform(text)
print(f"Хеш-вектор: {vector.shape}")

# Пакетная векторизация
texts = [
    "обработка естественного языка",
    "компьютерное зрение",
    "глубокое обучение"
]
vectors = vectorizer.transform_batch(texts)
print(f"Матрица векторов: {vectors.shape}")
```

**Особенности:**
- Не требует предобучения
- Фиксированная размерность
- Поддержка n-грамм
- Нормализация векторов

#### Случайная проекция

```python
from neurograph.contextvec.models.lightweight import RandomProjection

# Снижение размерности с 1000 до 100
projection = RandomProjection(input_dim=1000, output_dim=100, seed=42)

# Проекция одного вектора
high_dim_vector = np.random.random(1000)
low_dim_vector = projection.transform(high_dim_vector)
print(f"Исходная размерность: {high_dim_vector.shape}")
print(f"Новая размерность: {low_dim_vector.shape}")

# Пакетная проекция
high_dim_vectors = np.random.random((50, 1000))
low_dim_vectors = projection.transform_batch(high_dim_vectors)
print(f"Пакет: {high_dim_vectors.shape} -> {low_dim_vectors.shape}")

# Сохранение и загрузка модели
projection.save("projection_model.npz")
loaded_projection = RandomProjection.load("projection_model.npz")
```

### Интеграция с другими модулями

#### Связь с SemGraph

```python
from neurograph.semgraph import SemGraphFactory
from neurograph.contextvec import ContextVectorsFactory

# Создание компонентов
graph = SemGraphFactory.create("memory_efficient")
vectors = ContextVectorsFactory.create("dynamic", vector_size=384)

# Адаптер для создания эмбеддингов
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter
encoder = SentenceTransformerAdapter("all-MiniLM-L6-v2")

# Синхронное добавление узлов и векторов
def add_concept_with_vector(concept_id: str, description: str, **attributes):
    # Добавляем узел в граф
    graph.add_node(concept_id, description=description, **attributes)
    
    # Создаем векторное представление
    vector = encoder.encode(description)
    vectors.create_vector(concept_id, vector)
    
    return concept_id

# Использование
concept_id = add_concept_with_vector(
    "machine_learning",
    "Подраздел искусственного интеллекта, изучающий алгоритмы обучения",
    type="concept",
    domain="AI"
)

# Семантический поиск концепций
def find_similar_concepts(query: str, top_n: int = 5):
    query_vector = encoder.encode(query)
    
    # Находим похожие векторы
    similar_keys = []
    for key in vectors.get_all_keys():
        key_vector = vectors.get_vector(key)
        if key_vector is not None:
            # Вычисляем косинусное сходство
            similarity = np.dot(query_vector, key_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(key_vector)
            )
            similar_keys.append((key, similarity))
    
    # Сортируем по убыванию сходства
    similar_keys.sort(key=lambda x: x[1], reverse=True)
    
    # Возвращаем информацию из графа
    results = []
    for key, similarity in similar_keys[:top_n]:
        node_info = graph.get_node(key)
        if node_info:
            results.append({
                "concept_id": key,
                "similarity": similarity,
                "description": node_info.get("description", ""),
                "type": node_info.get("type", "")
            })
    
    return results

# Поиск
results = find_similar_concepts("алгоритмы обучения", top_n=3)
for result in results:
    print(f"{result['concept_id']}: {result['similarity']:.3f}")
    print(f"  {result['description']}")
```

#### Связь с Memory

```python
from neurograph.memory import create_default_biomorphic_memory, MemoryItem

# Создание памяти с семантическим индексом
memory = create_default_biomorphic_memory(use_semantic_indexing=True)

# Адаптер для векторизации
encoder = SentenceTransformerAdapter("all-MiniLM-L6-v2")

# Функция для добавления знаний в память
def add_knowledge_to_memory(text: str, content_type: str = "knowledge"):
    # Создаем векторное представление
    embedding = encoder.encode(text)
    
    # Создаем элемент памяти
    memory_item = MemoryItem(text, embedding, content_type)
    
    # Добавляем в память
    item_id = memory.add(memory_item)
    
    return item_id

# Добавление знаний
knowledge_texts = [
    "Python - это высокоуровневый язык программирования",
    "Django - веб-фреймворк для Python",
    "Машинное обучение использует алгоритмы для анализа данных"
]

item_ids = []
for text in knowledge_texts:
    item_id = add_knowledge_to_memory(text)
    item_ids.append(item_id)

# Семантический поиск в памяти
def search_memory_semantically(query: str, limit: int = 5):
    # Создаем вектор запроса
    query_vector = encoder.encode(query)
    
    # Ищем в памяти
    results = memory.search(query_vector, limit=limit)
    
    # Получаем детали найденных элементов
    detailed_results = []
    for item_id, similarity in results:
        item = memory.get(item_id)
        if item:
            detailed_results.append({
                "item_id": item_id,
                "content": item.content,
                "similarity": similarity,
                "access_count": item.access_count,
                "content_type": item.content_type
            })
    
    return detailed_results

# Поиск знаний
search_results = search_memory_semantically("веб-разработка на Python")
for result in search_results:
    print(f"Сходство: {result['similarity']:.3f}")
    print(f"Содержание: {result['content']}")
    print(f"Обращений: {result['access_count']}")
    print("---")
```

### Практические применения

#### Кластеризация концепций

```python
from sklearn.cluster import KMeans
import numpy as np

def cluster_concepts(concept_ids: List[str], n_clusters: int = 3):
    # Получаем векторы для концепций
    concept_vectors = []
    valid_concepts = []
    
    for concept_id in concept_ids:
        vector = vectors.get_vector(concept_id)
        if vector is not None:
            concept_vectors.append(vector)
            valid_concepts.append(concept_id)
    
    if len(concept_vectors) < n_clusters:
        return {}
    
    # Выполняем кластеризацию
    X = np.array(concept_vectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Группируем по кластерам
    clusters = {}
    for concept, label in zip(valid_concepts, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(concept)
    
    return clusters

# Кластеризация
concepts = ["python", "java", "django", "spring", "tensorflow", "pytorch"]
clusters = cluster_concepts(concepts, n_clusters=3)

for cluster_id, cluster_concepts in clusters.items():
    print(f"Кластер {cluster_id}: {', '.join(cluster_concepts)}")
```

#### Автоматическое тегирование

```python
def auto_tag_text(text: str, tag_vectors: Dict[str, np.ndarray], threshold: float = 0.7):
    # Векторизируем текст
    text_vector = encoder.encode(text)
    
    # Находим релевантные теги
    relevant_tags = []
    for tag, tag_vector in tag_vectors.items():
        similarity = np.dot(text_vector, tag_vector) / (
            np.linalg.norm(text_vector) * np.linalg.norm(tag_vector)
        )
        
        if similarity >= threshold:
            relevant_tags.append((tag, similarity))
    
    # Сортируем по релевантности
    relevant_tags.sort(key=lambda x: x[1], reverse=True)
    return relevant_tags

# Подготовка тегов
tag_descriptions = {
    "programming": "создание программного обеспечения и алгоритмов",
    "web_development": "разработка веб-сайтов и веб-приложений",
    "machine_learning": "алгоритмы машинного обучения и ИИ",
    "data_science": "анализ данных и статистика"
}

tag_vectors = {}
for tag, description in tag_descriptions.items():
    tag_vectors[tag] = encoder.encode(description)

# Автоматическое тегирование
text = "Создание веб-приложения на Django с использованием Python"
tags = auto_tag_text(text, tag_vectors, threshold=0.3)

print(f"Текст: {text}")
print("Релевантные теги:")
for tag, score in tags:
    print(f"  {tag}: {score:.3f}")
```

---

## Memory Module

Модуль биоморфной памяти, имитирующий многоуровневую организацию человеческой памяти.

### Основные концепции

#### Архитектура памяти

Система реализует трёхуровневую модель памяти:

```
Working Memory (Рабочая память)
    ↓ ← активные элементы для обработки
STM (Кратковременная память) 
    ↓ ← консолидация через время/важность
LTM (Долговременная память)
    ↓ ← забывание по кривой Эббингауза
[FORGOTTEN] (Забытые элементы)
```

**Характеристики уровней:**
- **Working Memory**: 7±2 элемента, мгновенный доступ
- **STM**: 50-200 элементов, быстрый доступ, ограниченное время хранения
- **LTM**: 1000-50000 элементов, медленный доступ, долгосрочное хранение

### Базовые интерфейсы

**Файл:** `neurograph/memory/base.py`

#### Элемент памяти

```python
from neurograph.memory.base import MemoryItem
import numpy as np
import time

# Создание элемента памяти
content = "Python - это язык программирования"
embedding = np.random.random(384)  # Векторное представление

memory_item = MemoryItem(
    content=content,
    embedding=embedding,
    content_type="knowledge",
    metadata={
        "source": "documentation",
        "topic": "programming",
        "language": "ru"
    }
)

# Автоматически устанавливаются:
print(f"Создан: {memory_item.created_at}")
print(f"Последний доступ: {memory_item.last_accessed_at}")
print(f"Количество обращений: {memory_item.access_count}")

# Отметка доступа к элементу
memory_item.access()
print(f"Обращений после доступа: {memory_item.access_count}")
```

#### Интерфейс памяти

```python
from neurograph.memory.base import IMemory

class MyMemoryImplementation(IMemory):
    def add(self, item: MemoryItem) -> str:
        # Добавление элемента, возврат ID
        pass
    
    def get(self, item_id: str) -> Optional[MemoryItem]:
        # Получение элемента по ID
        pass
    
    def search(self, query: Union[str, np.ndarray], limit: int = 10) -> List[Tuple[str, float]]:
        # Поиск похожих элементов
        pass
    
    def remove(self, item_id: str) -> bool:
        # Удаление элемента
        pass
    
    def clear(self) -> None:
        # Очистка памяти
        pass
    
    def size(self) -> int:
        # Размер памяти
        pass
```

### Фабрика памяти

**Файл:** `neurograph/memory/factory.py`

#### Создание экземпляров памяти

```python
from neurograph.memory.factory import (
    MemoryFactory,
    create_default_biomorphic_memory,
    create_lightweight_memory,
    create_high_performance_memory
)

# Через фабрику
memory = MemoryFactory.create("biomorphic", 
                             stm_capacity=100,
                             ltm_capacity=10000)

# Готовые конфигурации
default_memory = create_default_biomorphic_memory()

lightweight_memory = create_lightweight_memory(
    stm_capacity=25,
    ltm_capacity=500
)

performance_memory = create_high_performance_memory(
    stm_capacity=300,
    ltm_capacity=100000
)

# Из конфигурации
config = {
    "type": "biomorphic",
    "stm_capacity": 150,
    "ltm_capacity": 15000,
    "use_semantic_indexing": True,
    "auto_consolidation": True
}
configured_memory = MemoryFactory.create_from_config(config)
```

### Биоморфная память

**Файл:** `neurograph/memory/impl/biomorphic.py`

#### Основное использование

```python
from neurograph.memory import create_default_biomorphic_memory, MemoryItem
import numpy as np

# Создание памяти
memory = create_default_biomorphic_memory(
    stm_capacity=100,           # Размер STM
    ltm_capacity=10000,         # Размер LTM  
    use_semantic_indexing=True, # Семантический поиск
    auto_consolidation=True,    # Автоконсолидация
    consolidation_interval=300.0 # Интервал консолидации (секунды)
)

# Добавление знаний
knowledge_items = [
    "Искусственный интеллект - область компьютерных наук",
    "Машинное обучение - подраздел ИИ",
    "Глубокое обучение использует нейронные сети",
    "Python популярен для разработки ИИ"
]

item_ids = []
for knowledge in knowledge_items:
    # Создаем векторное представление (в реальности через encoder)
    embedding = np.random.random(384)
    
    # Создаем элемент памяти
    item = MemoryItem(knowledge, embedding, "knowledge")
    
    # Добавляем в память
    item_id = memory.add(item)
    item_ids.append(item_id)
    
    print(f"Добавлено: {item_id}")

print(f"Всего элементов в памяти: {memory.size()}")
```

#### Поиск и доступ

```python
# Получение элемента по ID
item = memory.get(item_ids[0])
if item:
    print(f"Содержание: {item.content}")
    print(f"Тип: {item.content_type}")
    print(f"Обращений: {item.access_count}")

# Семантический поиск
query_vector = np.random.random(384)  # В реальности - эмбеддинг запроса
results = memory.search(query_vector, limit=5)

print("Результаты поиска:")
for item_id, similarity in results:
    item = memory.get(item_id)
    if item:
        print(f"  {similarity:.3f}: {item.content[:50]}...")

# Текстовый поиск (если есть встроенный энкодер)
text_results = memory.search("машинное обучение", limit=3)
```

#### Статистика и мониторинг

```python
# Подробная статистика памяти
stats = memory.get_memory_statistics()

print("=== Уровни памяти ===")
for level, info in stats["memory_levels"].items():
    print(f"{level.upper()}:")
    print(f"  Размер: {info['size']}/{info['capacity']}")
    print(f"  Загрузка: {info['pressure']:.1%}")

print("\n=== Производительность ===")
perf = stats["performance"]
print(f"Добавлено элементов: {perf['items_added']}")
print(f"Консолидировано: {perf['items_consolidated']}")
print(f"Забыто: {perf['items_forgotten']}")
print(f"Обращений: {perf['items_accessed']}")

print("\n=== Эффективность ===")
eff = stats["memory_efficiency"]
print(f"Cache hit rate: {eff['cache_hit_rate']:.1%}")
print(f"Consolidation rate: {eff['consolidation_rate']:.1%}")
print(f"Forgetting rate: {eff['forgetting_rate']:.1%}")

print("\n=== Консолидация ===")
consolidation = stats["consolidation"]
print(f"Всего консолидированных: {consolidation['consolidation']['total_consolidated']}")
print(f"Всего забытых: {consolidation['consolidation']['total_forgotten']}")
```

#### Принудительные операции

```python
# Принудительная консолидация
consolidation_result = memory.force_consolidation()
print("Результат консолидации:")
print(f"  STM до: {consolidation_result['before']['stm_size']}")
print(f"  STM после: {consolidation_result['after']['stm_size']}")
print(f"  Консолидировано: {consolidation_result['consolidated']}")

# Оптимизация памяти
optimization_result = memory.optimize_memory()
print("Действия оптимизации:")
for action in optimization_result["actions_taken"]:
    print(f"  - {action}")
```

#### Анализ содержимого

```python
# Недавние элементы (за последние 24 часа)
recent_items = memory.get_recent_items(hours=24.0)
print(f"Недавних элементов: {len(recent_items)}")

# Недавние элементы по уровням памяти
stm_recent = memory.get_recent_items(hours=1.0, memory_level="stm")
ltm_recent = memory.get_recent_items(hours=24.0, memory_level="ltm")

print(f"STM за час: {len(stm_recent)}")
print(f"LTM за день: {len(ltm_recent)}")

# Наиболее используемые элементы
most_accessed = memory.get_most_accessed_items(limit=10)
print("Самые популярные элементы:")
for item, access_count in most_accessed:
    print(f"  {access_count} обращений: {item.content[:40]}...")
```

### Стратегии консолидации и забывания

**Файл:** `neurograph/memory/strategies.py`

#### Стратегии консолидации

```python
from neurograph.memory.strategies import (
    TimeBasedConsolidation,
    ImportanceBasedConsolidation, 
    SemanticClusteringConsolidation,
    AdaptiveConsolidation
)

# Консолидация по времени
time_strategy = TimeBasedConsolidation(
    min_age_seconds=300.0,  # 5 минут
    max_stm_size=100
)

# Консолидация по важности
importance_strategy = ImportanceBasedConsolidation(
    importance_threshold=0.7,
    access_weight=0.3,      # Вес фактора доступа
    recency_weight=0.4,     # Вес фактора свежести
    content_weight=0.3      # Вес фактора содержания
)

# Семантическая кластеризация
semantic_strategy = SemanticClusteringConsolidation(
    similarity_threshold=0.8,
    min_cluster_size=3
)

# Адаптивная стратегия (комбинирует несколько)
adaptive_strategy = AdaptiveConsolidation(
    strategies=[time_strategy, importance_strategy],
    weights=[0.6, 0.4]
)
```

#### Стратегии забывания

```python
from neurograph.memory.strategies import (
    EbbinghausBasedForgetting,
    LeastRecentlyUsedForgetting
)

# Забывание по кривой Эббингауза
ebbinghaus_strategy = EbbinghausBasedForgetting(
    base_retention=0.1,     # Базовое удержание
    decay_rate=0.693,       # Скорость забывания (ln(2))
    access_boost=2.0        # Усиление при доступе
)

# LRU забывание
lru_strategy = LeastRecentlyUsedForgetting(
    max_capacity=1000
)
```

#### Монитор давления памяти

```python
from neurograph.memory.strategies import MemoryPressureMonitor

monitor = MemoryPressureMonitor(
    low_pressure_threshold=0.7,   # 70% заполнения
    high_pressure_threshold=0.9   # 90% заполнения
)

# Вычисление давления
current_pressure = monitor.get_memory_pressure(
    current_items=85,
    max_capacity=100
)

print(f"Давление памяти: {current_pressure:.1%}")

if monitor.should_be_aggressive(current_pressure):
    print("Требуется агрессивное забывание")
elif monitor.should_be_conservative(current_pressure):
    print("Можно быть консервативным")
```

### Система консолидации

**Файл:** `neurograph/memory/consolidation.py`

#### Менеджер консолидации

```python
from neurograph.memory.consolidation import ConsolidationManager
from neurograph.memory.strategies import TimeBasedConsolidation, EbbinghausBasedForgetting

# Создание менеджера
consolidation_strategy = TimeBasedConsolidation()
forgetting_strategy = EbbinghausBasedForgetting()

manager = ConsolidationManager(
    consolidation_strategy=consolidation_strategy,
    forgetting_strategy=forgetting_strategy,
    auto_consolidation_interval=60.0,  # 1 минута
    enable_background_processing=True
)

# Статистика менеджера
stats = manager.get_statistics()
print(f"Всего консолидировано: {stats['total_consolidated']}")
print(f"Всего забыто: {stats['total_forgotten']}")
print(f"Последняя консолидация: {stats['last_consolidation']}")
```

#### Логгер переходов

```python
from neurograph.memory.consolidation import TransitionLogger, MemoryTransition

# Создание логгера
logger = TransitionLogger(max_history=1000)

# Логирование перехода
transition = MemoryTransition(
    item_id="item_123",
    source_level="STM", 
    target_level="LTM",
    reason="time_based_consolidation"
)
logger.log_transition(transition)

# Анализ переходов
transitions_for_item = logger.get_transitions_by_item("item_123")
recent_transitions = logger.get_recent_transitions(hours=24.0)

# Статистика переходов
transition_stats = logger.get_statistics()
print(f"Всего переходов: {transition_stats['total_transitions']}")
print(f"Консолидаций: {transition_stats['consolidations']}")
print(f"Забываний: {transition_stats['forgettings']}")
```

### События памяти

Система памяти генерирует события для интеграции с другими модулями:

```python
from neurograph.core.events import subscribe

# Подписка на события памяти
def handle_memory_item_added(data):
    print(f"Добавлен элемент: {data['item_id']}")
    print(f"Уровень памяти: {data['memory_level']}")

def handle_consolidation_completed(data):
    print(f"Консолидация завершена:")
    print(f"  Консолидировано: {data['consolidated_count']}")
    print(f"  Забыто: {data['forgotten_count']}")
    print(f"  Давление STM: {data['stm_pressure']:.2f}")

def handle_memory_pressure_high(data):
    print(f"Высокое давление памяти в {data['memory_type']}: {data['pressure']:.1%}")

# Подписки
subscribe("memory.item_added", handle_memory_item_added)
subscribe("memory.consolidation_completed", handle_consolidation_completed)
subscribe("memory.memory_pressure_high", handle_memory_pressure_high)
```

**Основные события:**
- `memory.item_added` - добавлен элемент
- `memory.item_removed` - удален элемент
- `memory.consolidation_started` - начата консолидация
- `memory.consolidation_completed` - завершена консолидация
- `memory.memory_pressure_high` - высокое давление памяти
- `memory.transition_logged` - зарегистрирован переход
- `memory.cleared` - память очищена
- `memory.shutdown` - завершение работы

### Экспорт и анализ

#### Дамп памяти для анализа

```python
# Экспорт полного дампа памяти
memory_dump = memory.export_memory_dump()

print("=== Метаданные ===")
print(f"Версия: {memory_dump['metadata']['version']}")
print(f"Тип памяти: {memory_dump['metadata']['memory_type']}")

print("\n=== Конфигурация ===")
config = memory_dump['configuration']
print(f"STM capacity: {config['stm_capacity']}")
print(f"LTM capacity: {config['ltm_capacity']}")
print(f"Семантическое индексирование: {config['semantic_indexing']}")

print("\n=== Содержимое ===")
items = memory_dump['items']
print(f"Working Memory: {len(items['working_memory'])} элементов")
print(f"STM: {len(items['stm'])} элементов")
print(f"LTM: {len(items['ltm'])} элементов")

# Примеры элементов из STM
print("\nПримеры из STM:")
for item in items['stm'][:3]:
    print(f"  {item['id']}: {item['content']}")
    print(f"    Создан: {item['created_at']}")
    print(f"    Обращений: {item['access_count']}")
```

### Практические применения

#### Система вопросов и ответов

```python
import time
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter

class QASystem:
    def __init__(self):
        self.memory = create_default_biomorphic_memory()
        self.encoder = SentenceTransformerAdapter("all-MiniLM-L6-v2")
    
    def learn_fact(self, fact: str, source: str = "user"):
        """Изучение нового факта"""
        embedding = self.encoder.encode(fact)
        item = MemoryItem(fact, embedding, "fact", 
                         metadata={"source": source, "learned_at": time.time()})
        
        item_id = self.memory.add(item)
        print(f"Изучен факт: {fact}")
        return item_id
    
    def answer_question(self, question: str, top_k: int = 3):
        """Ответ на вопрос на основе изученных фактов"""
        question_embedding = self.encoder.encode(question)
        
        # Поиск релевантных фактов
        results = self.memory.search(question_embedding, limit=top_k)
        
        if not results:
            return "Я не знаю ответа на этот вопрос."
        
        # Формирование ответа на основе найденных фактов
        relevant_facts = []
        for item_id, similarity in results:
            if similarity > 0.3:  # Порог релевантности
                item = self.memory.get(item_id)
                if item:
                    relevant_facts.append(item.content)
        
        if relevant_facts:
            answer = "На основе того, что я знаю:\n" + "\n".join(f"• {fact}" for fact in relevant_facts)
            return answer
        else:
            return "Не нашел достаточно релевантной информации."
    
    def get_memory_status(self):
        """Статус памяти системы"""
        stats = self.memory.get_memory_statistics()
        return {
            "total_facts": self.memory.size(),
            "stm_facts": stats['memory_levels']['stm']['size'],
            "ltm_facts": stats['memory_levels']['ltm']['size'],
            "cache_hit_rate": stats['memory_efficiency']['cache_hit_rate']
        }

# Использование системы Q&A
qa_system = QASystem()

# Обучение фактам
facts = [
    "Python был создан Гвидо ван Россумом в 1991 году",
    "Django - это веб-фреймворк для Python",
    "FastAPI - современный веб-фреймворк для создания API",
    "Machine Learning - раздел искусственного интеллекта",
    "NumPy используется для научных вычислений в Python"
]

for fact in facts:
    qa_system.learn_fact(fact)

# Ответы на вопросы
questions = [
    "Кто создал Python?",
    "Что такое Django?", 
    "Какие фреймворки есть для Python?",
    "Что используется для научных вычислений?"
]

for question in questions:
    print(f"\nВопрос: {question}")
    answer = qa_system.answer_question(question)
    print(f"Ответ: {answer}")

# Статус памяти
status = qa_system.get_memory_status()
print(f"\nСтатус памяти:")
print(f"Всего фактов: {status['total_facts']}")
print(f"В STM: {status['stm_facts']}")
print(f"В LTM: {status['ltm_facts']}")
print(f"Cache hit rate: {status['cache_hit_rate']:.1%}")
```

#### Персональный ассистент знаний

```python
class PersonalKnowledgeAssistant:
    def __init__(self):
        # Высокопроизводительная память
        self.memory = create_high_performance_memory(
            stm_capacity=200,
            ltm_capacity=50000,
            consolidation_interval=120.0
        )
        self.encoder = SentenceTransformerAdapter("all-MiniLM-L6-v2")
        
        # Подписка на события для логирования
        subscribe("memory.consolidation_completed", self._log_consolidation)
    
    def _log_consolidation(self, data):
        print(f"[Система] Консолидация: {data['consolidated_count']} → LTM, "
              f"{data['forgotten_count']} забыто")
    
    def add_knowledge(self, content: str, category: str = "general", 
                     source: str = "user", importance: float = 1.0):
        """Добавление знания с категоризацией"""
        embedding = self.encoder.encode(content)
        
        item = MemoryItem(
            content=content,
            embedding=embedding,
            content_type="knowledge",
            metadata={
                "category": category,
                "source": source,
                "importance": importance,
                "keywords": self._extract_keywords(content)
            }
        )
        
        item_id = self.memory.add(item)
        return item_id
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Простое извлечение ключевых слов"""
        # В реальности здесь был бы NLP процессор
        words = text.lower().split()
        # Фильтруем стоп-слова и короткие слова
        stop_words = {"и", "в", "на", "с", "для", "это", "то", "как", "что"}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:5]  # Топ-5 ключевых слов
    
    def search_knowledge(self, query: str, category: str = None, limit: int = 5):
        """Поиск знаний с фильтрацией по категории"""
        query_embedding = self.encoder.encode(query)
        results = self.memory.search(query_embedding, limit=limit * 2)  # Берем больше для фильтрации
        
        filtered_results = []
        for item_id, similarity in results:
            item = self.memory.get(item_id)
            if item and similarity > 0.2:  # Порог релевантности
                # Фильтр по категории
                if category is None or item.metadata.get("category") == category:
                    filtered_results.append({
                        "content": item.content,
                        "similarity": similarity,
                        "category": item.metadata.get("category", "unknown"),
                        "source": item.metadata.get("source", "unknown"),
                        "importance": item.metadata.get("importance", 1.0),
                        "keywords": item.metadata.get("keywords", [])
                    })
            
            if len(filtered_results) >= limit:
                break
        
        return filtered_results
    
    def get_categories(self):
        """Получение всех категорий знаний"""
        categories = {}
        
        # Анализируем недавние элементы
        recent_items = self.memory.get_recent_items(hours=24*7)  # За неделю
        
        for item in recent_items:
            category = item.metadata.get("category", "unknown")
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        return sorted(categories.items(), key=lambda x: x[1], reverse=True)
    
    def get_trending_topics(self, hours: int = 24):
        """Популярные темы за период"""
        recent_items = self.memory.get_recent_items(hours=hours)
        
        keyword_freq = {}
        for item in recent_items:
            keywords = item.metadata.get("keywords", [])
            for keyword in keywords:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Топ-10 ключевых слов
        trending = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return trending
    
    def suggest_related(self, content: str, limit: int = 3):
        """Предложение связанных знаний"""
        # Находим похожие знания
        related = self.search_knowledge(content, limit=limit)
        
        suggestions = []
        for item in related:
            if item["similarity"] > 0.4:  # Высокое сходство
                suggestions.append({
                    "content": item["content"],
                    "relevance": item["similarity"],
                    "category": item["category"]
                })
        
        return suggestions

# Использование персонального ассистента
assistant = PersonalKnowledgeAssistant()

# Добавление знаний по категориям
programming_knowledge = [
    ("Python поддерживает объектно-ориентированное программирование", "programming"),
    ("Git используется для контроля версий", "tools"),
    ("REST API - архитектурный стиль для веб-сервисов", "web_development"),
    ("Docker помогает контейнеризации приложений", "devops")
]

for content, category in programming_knowledge:
    assistant.add_knowledge(content, category=category, importance=0.8)

# Поиск знаний
print("=== Поиск по программированию ===")
results = assistant.search_knowledge("объектно-ориентированное программирование")
for result in results:
    print(f"Сходство: {result['similarity']:.3f}")
    print(f"Категория: {result['category']}")
    print(f"Содержание: {result['content']}")
    print("---")

# Категории знаний
print("\n=== Категории знаний ===")
categories = assistant.get_categories()
for category, count in categories:
    print(f"{category}: {count} знаний")

# Популярные темы
print("\n=== Популярные темы (24 часа) ===")
trending = assistant.get_trending_topics(hours=24)
for keyword, freq in trending:
    print(f"{keyword}: {freq} упоминаний")

# Связанные знания
print("\n=== Связанные знания ===")
suggestions = assistant.suggest_related("веб-разработка")
for suggestion in suggestions:
    print(f"Релевантность: {suggestion['relevance']:.3f}")
    print(f"Содержание: {suggestion['content']}")
```

---

## Практические примеры

### Пример 1: Создание системы знаний

Полный пример создания системы знаний, объединяющей все модули:

```python
from neurograph.core import Configuration
from neurograph.core.logging import setup_logging, get_logger
from neurograph.core.events import subscribe, publish
from neurograph.semgraph import SemGraphFactory
from neurograph.contextvec import ContextVectorsFactory
from neurograph.memory import create_default_biomorphic_memory, MemoryItem
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter
import numpy as np

class KnowledgeSystem:
    def __init__(self, config_path: str = None):
        # Настройка логирования
        setup_logging(level="INFO", log_file="knowledge_system.log")
        self.logger = get_logger("knowledge_system")
        
        # Загрузка конфигурации
        if config_path:
            self.config = Configuration.load_from_file(config_path)
        else:
            self.config = Configuration({
                "graph": {"type": "memory_efficient"},
                "vectors": {"type": "dynamic", "vector_size": 384},
                "memory": {"stm_capacity": 100, "ltm_capacity": 10000}
            })
        
        # Инициализация компонентов
        self._initialize_components()
        
        # Подписка на события
        self._setup_event_handlers()
    
    def _initialize_components(self):
        """Инициализация всех компонентов"""
        # Семантический граф
        self.graph = SemGraphFactory.create(
            self.config.get("graph.type", "memory_efficient")
        )
        
        # Векторные представления
        self.vectors = ContextVectorsFactory.create(
            self.config.get("vectors.type", "dynamic"),
            vector_size=self.config.get("vectors.vector_size", 384),
            use_indexing=True
        )
        
        # Биоморфная память
        self.memory = create_default_biomorphic_memory(
            stm_capacity=self.config.get("memory.stm_capacity", 100),
            ltm_capacity=self.config.get("memory.ltm_capacity", 10000)
        )
        
        # Адаптер для создания эмбеддингов
        self.encoder = SentenceTransformerAdapter("all-MiniLM-L6-v2")
        
        self.logger.info("Все компоненты инициализированы")
    
    def _setup_event_handlers(self):
        """Настройка обработчиков событий"""
        subscribe("memory.consolidation_completed", self._handle_consolidation)
        subscribe("graph.node_added", self._handle_graph_update)
    
    def _handle_consolidation(self, data):
        """Обработка событий консолидации памяти"""
        self.logger.info(f"Консолидация: {data['consolidated_count']} → LTM")
    
    def _handle_graph_update(self, data):
        """Обработка обновлений графа"""
        self.logger.debug(f"Граф обновлен: {data}")
    
    def add_knowledge(self, text: str, concept_type: str = "concept", 
                     metadata: dict = None) -> dict:
        """Добавление знания во все компоненты системы"""
        knowledge_id = f"knowledge_{hash(text) % 1000000}"
        
        # 1. Создание векторного представления
        embedding = self.encoder.encode(text)
        
        # 2. Добавление в граф знаний
        self.graph.add_node(
            knowledge_id,
            text=text,
            type=concept_type,
            **(metadata or {})
        )
        
        # 3. Добавление в векторные представления
        self.vectors.create_vector(knowledge_id, embedding)
        
        # 4. Добавление в память
        memory_item = MemoryItem(
            content=text,
            embedding=embedding,
            content_type="knowledge",
            metadata={"concept_type": concept_type, **(metadata or {})}
        )
        memory_id = self.memory.add(memory_item)
        
        # 5. Связывание компонентов через метаданные
        self.graph.add_node(knowledge_id, memory_id=memory_id)
        
        self.logger.info(f"Добавлено знание: {knowledge_id}")
        
        # Публикация события
        publish("knowledge.added", {
            "knowledge_id": knowledge_id,
            "text": text,
            "concept_type": concept_type
        })
        
        return {
            "knowledge_id": knowledge_id,
            "memory_id": memory_id,
            "graph_node": knowledge_id
        }
    
    def add_relation(self, source_text: str, target_text: str, 
                    relation_type: str, confidence: float = 1.0):
        """Добавление отношения между знаниями"""
        # Находим или создаем узлы
        source_id = self._find_or_create_concept(source_text)
        target_id = self._find_or_create_concept(target_text)
        
        # Добавляем связь в граф
        self.graph.add_edge(
            source_id, target_id, relation_type,
            weight=confidence,
            confidence=confidence
        )
        
        self.logger.info(f"Добавлена связь: {source_text} --[{relation_type}]--> {target_text}")
        
        return {"source": source_id, "target": target_id, "relation": relation_type}
    
    def _find_or_create_concept(self, text: str) -> str:
        """Поиск существующего концепта или создание нового"""
        # Поиск в векторных представлениях
        embedding = self.encoder.encode(text)
        
        # Ищем в векторах похожие концепты
        similar = self.vectors.get_most_similar_vector(embedding, top_n=1)
        
        if similar and similar[0][1] > 0.9:  # Очень высокое сходство
            return similar[0][0]
        else:
            # Создаем новый концепт
            result = self.add_knowledge(text)
            return result["knowledge_id"]
    
    def search_knowledge(self, query: str, limit: int = 10) -> list:
        """Поиск знаний по всем компонентам"""
        query_embedding = self.encoder.encode(query)
        
        results = []
        
        # 1. Поиск в памяти (семантический)
        memory_results = self.memory.search(query_embedding, limit=limit)
        
        for memory_id, similarity in memory_results:
            memory_item = self.memory.get(memory_id)
            if memory_item:
                # 2. Поиск связанной информации в графе
                graph_info = self._find_graph_info_by_memory_id(memory_id)
                
                results.append({
                    "content": memory_item.content,
                    "similarity": similarity,
                    "memory_id": memory_id,
                    "graph_info": graph_info,
                    "access_count": memory_item.access_count,
                    "content_type": memory_item.content_type
                })
        
        return results
    
    def _find_graph_info_by_memory_id(self, memory_id: str) -> dict:
        """Поиск информации в графе по ID памяти"""
        # Поиск узлов графа с соответствующим memory_id
        for node_id in self.graph.get_all_nodes():
            node_data = self.graph.get_node(node_id)
            if node_data and node_data.get("memory_id") == memory_id:
                # Получаем связи узла
                neighbors = self.graph.get_neighbors(node_id)
                relations = []
                
                for neighbor in neighbors:
                    edge = self.graph.get_edge(node_id, neighbor)
                    if edge:
                        relations.append({
                            "target": neighbor,
                            "relation_type": edge.get("type", "related"),
                            "confidence": edge.get("confidence", 1.0)
                        })
                
                return {
                    "node_id": node_id,
                    "node_data": node_data,
                    "relations": relations
                }
        
        return {}
    
    def get_related_concepts(self, concept_text: str, max_depth: int = 2) -> list:
        """Получение связанных концептов через граф"""
        # Находим концепт в системе
        concept_id = self._find_concept_by_text(concept_text)
        if not concept_id:
            return []
        
        from neurograph.semgraph.query.search import GraphSearcher
        searcher = GraphSearcher(self.graph)
        
        # Поиск на расстоянии
        related_nodes = searcher.search_by_distance(concept_id, max_distance=max_depth)
        
        related_concepts = []
        for node_id, distance in related_nodes.items():
            if node_id != concept_id:
                node_data = self.graph.get_node(node_id)
                if node_data:
                    related_concepts.append({
                        "concept_id": node_id,
                        "text": node_data.get("text", ""),
                        "type": node_data.get("type", ""),
                        "distance": distance
                    })
        
        return sorted(related_concepts, key=lambda x: x["distance"])
    
    def _find_concept_by_text(self, text: str) -> str:
        """Поиск концепта по тексту"""
        embedding = self.encoder.encode(text)
        similar = self.vectors.get_most_similar_vector(embedding, top_n=1)
        
        if similar and similar[0][1] > 0.8:
            return similar[0][0]
        return None
    
    def get_system_statistics(self) -> dict:
        """Статистика всей системы"""
        memory_stats = self.memory.get_memory_statistics()
        
        return {
            "graph": {
                "nodes": len(self.graph.get_all_nodes()),
                "edges": len(self.graph.get_all_edges())
            },
            "vectors": {
                "total_vectors": len(self.vectors.get_all_keys())
            },
            "memory": {
                "total_items": self.memory.size(),
                "stm_items": memory_stats["memory_levels"]["stm"]["size"],
                "ltm_items": memory_stats["memory_levels"]["ltm"]["size"],
                "cache_hit_rate": memory_stats["memory_efficiency"]["cache_hit_rate"]
            }
        }

# Использование системы знаний
def main():
    # Создание системы
    knowledge_system = KnowledgeSystem()
    
    # Добавление знаний
    knowledge_system.add_knowledge(
        "Python - высокоуровневый язык программирования",
        concept_type="programming_language",
        metadata={"domain": "computer_science", "popularity": "high"}
    )
    
    knowledge_system.add_knowledge(
        "Django - веб-фреймворк для Python",
        concept_type="framework",
        metadata={"domain": "web_development", "language": "python"}
    )
    
    # Добавление связей
    knowledge_system.add_relation(
        "Django", "Python", "implemented_in", confidence=0.95
    )
    
    # Поиск знаний
    print("=== Поиск: веб-разработка ===")
    results = knowledge_system.search_knowledge("веб-разработка", limit=3)
    for result in results:
        print(f"Сходство: {result['similarity']:.3f}")
        print(f"Содержание: {result['content']}")
        if result['graph_info']:
            print(f"Связи: {len(result['graph_info'].get('relations', []))}")
        print("---")
    
    # Связанные концепты
    print("\n=== Связанные с Python концепты ===")
    related = knowledge_system.get_related_concepts("Python", max_depth=2)
    for concept in related:
        print(f"Расстояние: {concept['distance']}")
        print(f"Концепт: {concept['text']}")
        print(f"Тип: {concept['type']}")
        print("---")
    
    # Статистика системы
    print("\n=== Статистика системы ===")
    stats = knowledge_system.get_system_statistics()
    print(f"Граф: {stats['graph']['nodes']} узлов, {stats['graph']['edges']} связей")
    print(f"Векторы: {stats['vectors']['total_vectors']} представлений")
    print(f"Память: {stats['memory']['total_items']} элементов")
    print(f"  STM: {stats['memory']['stm_items']}")
    print(f"  LTM: {stats['memory']['ltm_items']}")
    print(f"  Cache hit rate: {stats['memory']['cache_hit_rate']:.1%}")

if __name__ == "__main__":
    main()
```

### Пример 2: Обработка документов

Система для обработки и индексации документов:

```python
class DocumentProcessor:
    def __init__(self):
        self.knowledge_system = KnowledgeSystem()
        self.logger = get_logger("document_processor")
    
    def process_document(self, document_text: str, document_id: str, 
                        metadata: dict = None) -> dict:
        """Обработка документа и извлечение знаний"""
        # Разбиение на предложения (упрощенное)
        sentences = self._split_sentences(document_text)
        
        processed_sentences = []
        document_concepts = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:  # Пропускаем короткие предложения
                # Добавляем как знание
                knowledge_info = self.knowledge_system.add_knowledge(
                    sentence,
                    concept_type="fact",
                    metadata={
                        "document_id": document_id,
                        "sentence_index": i,
                        "document_metadata": metadata or {}
                    }
                )
                
                processed_sentences.append({
                    "sentence": sentence,
                    "knowledge_id": knowledge_info["knowledge_id"],
                    "index": i
                })
                
                document_concepts.append(knowledge_info["knowledge_id"])
        
        # Создание узла документа в графе
        self.knowledge_system.graph.add_node(
            document_id,
            type="document",
            title=metadata.get("title", "Unknown Document"),
            author=metadata.get("author", "Unknown"),
            concepts_count=len(document_concepts),
            **(metadata or {})
        )
        
        # Связывание документа с концептами
        for concept_id in document_concepts:
            self.knowledge_system.graph.add_edge(
                document_id, concept_id, "contains",
                weight=1.0 / len(document_concepts)  # Вес обратно пропорционален количеству
            )
        
        self.logger.info(f"Обработан документ {document_id}: {len(processed_sentences)} предложений")
        
        return {
            "document_id": document_id,
            "sentences_processed": len(processed_sentences),
            "concepts_extracted": len(document_concepts),
            "processed_sentences": processed_sentences
        }
    
    def _split_sentences(self, text: str) -> list:
        """Простое разбиение на предложения"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def search_in_documents(self, query: str, document_filter: dict = None) -> list:
        """Поиск в обработанных документах"""
        # Поиск знаний в системе
        results = self.knowledge_system.search_knowledge(query, limit=20)
        
        document_results = {}
        
        for result in results:
            # Получаем информацию о документе из метаданных
            memory_item = self.knowledge_system.memory.get(result["memory_id"])
            if memory_item and memory_item.metadata:
                doc_id = memory_item.metadata.get("document_id")
                
                if doc_id:
                    # Фильтрация по документам
                    if document_filter:
                        doc_node = self.knowledge_system.graph.get_node(doc_id)
                        if not self._matches_filter(doc_node, document_filter):
                            continue
                    
                    if doc_id not in document_results:
                        document_results[doc_id] = {
                            "document_id": doc_id,
                            "matches": [],
                            "max_similarity": 0.0,
                            "total_matches": 0
                        }
                    
                    document_results[doc_id]["matches"].append({
                        "content": result["content"],
                        "similarity": result["similarity"],
                        "sentence_index": memory_item.metadata.get("sentence_index", 0)
                    })
                    
                    document_results[doc_id]["max_similarity"] = max(
                        document_results[doc_id]["max_similarity"],
                        result["similarity"]
                    )
                    document_results[doc_id]["total_matches"] += 1
        
        # Сортируем документы по релевантности
        sorted_docs = sorted(
            document_results.values(),
            key=lambda x: (x["max_similarity"], x["total_matches"]),
            reverse=True
        )
        
        return sorted_docs
    
    def _matches_filter(self, doc_node: dict, document_filter: dict) -> bool:
        """Проверка соответствия документа фильтру"""
        if not doc_node:
            return False
        
        for key, value in document_filter.items():
            if key not in doc_node or doc_node[key] != value:
                return False
        return True
    
    def get_document_summary(self, document_id: str) -> dict:
        """Получение сводки по документу"""
        doc_node = self.knowledge_system.graph.get_node(document_id)
        if not doc_node:
            return {}
        
        # Получаем все концепты документа
        concept_neighbors = self.knowledge_system.graph.get_neighbors(document_id, edge_type="contains")
        
        concepts_info = []
        for concept_id in concept_neighbors:
            concept_node = self.knowledge_system.graph.get_node(concept_id)
            if concept_node:
                concepts_info.append({
                    "concept_id": concept_id,
                    "text": concept_node.get("text", ""),
                    "type": concept_node.get("type", "")
                })
        
        return {
            "document_info": doc_node,
            "concepts_count": len(concepts_info),
            "concepts": concepts_info[:10]  # Первые 10 концептов
        }

# Использование обработчика документов
def demo_document_processing():
    processor = DocumentProcessor()
    
    # Пример документа
    document_text = """
    Искусственный интеллект (ИИ) — это область компьютерных наук, которая занимается созданием умных машин.
    Машинное обучение является подразделом ИИ. Глубокое обучение использует нейронные сети для решения сложных задач.
    Python широко используется в разработке ИИ благодаря своим библиотекам. TensorFlow и PyTorch — популярные фреймворки для глубокого обучения.
    Обработка естественного языка (NLP) позволяет компьютерам понимать человеческий язык.
    """
    
    # Обработка документа
    result = processor.process_document(
        document_text,
        "ai_intro_doc",
        metadata={
            "title": "Введение в ИИ",
            "author": "AI Researcher", 
            "topic": "artificial_intelligence",
            "language": "ru"
        }
    )
    
    print(f"Обработан документ: {result['sentences_processed']} предложений")
    print(f"Извлечено концептов: {result['concepts_extracted']}")
    
    # Поиск в документах
    print("\n=== Поиск: машинное обучение ===")
    search_results = processor.search_in_documents("машинное обучение")
    
    for doc_result in search_results:
        print(f"Документ: {doc_result['document_id']}")
        print(f"Макс. сходство: {doc_result['max_similarity']:.3f}")
        print(f"Совпадений: {doc_result['total_matches']}")
        
        for match in doc_result['matches'][:2]:  # Первые 2 совпадения
            print(f"  Сходство: {match['similarity']:.3f}")
            print(f"  Текст: {match['content']}")
        print("---")
    
    # Сводка по документу
    print("\n=== Сводка по документу ===")
    summary = processor.get_document_summary("ai_intro_doc")
    if summary:
        print(f"Название: {summary['document_info'].get('title', 'N/A')}")
        print(f"Автор: {summary['document_info'].get('author', 'N/A')}")
        print(f"Концептов: {summary['concepts_count']}")
        
        print("Основные концепты:")
        for concept in summary['concepts'][:5]:
            print(f"  - {concept['text'][:60]}...")

demo_document_processing()
```

### Пример 3: Конфигурация и развертывание

Полный пример настройки системы через конфигурационные файлы:

```python
# config.json
{
    "system": {
        "name": "PersonalAssistant",
        "version": "1.0.0",
        "environment": "development"
    },
    "logging": {
        "level": "INFO",
        "file": "assistant.log",
        "rotation": "10 MB",
        "retention": "1 week"
    },
    "core": {
        "cache": {
            "max_size": 1000,
            "ttl": 300
        },
        "events": {
            "max_handlers": 100
        }
    },
    "semgraph": {
        "type": "persistent",
        "file_path": "knowledge_graph.json",
        "auto_save_interval": 300.0,
        "visualization": {
            "default_node_size": 500,
            "default_figsize": [12, 8]
        }
    },
    "contextvec": {
        "type": "dynamic",
        "vector_size": 384,
        "use_indexing": true,
        "adapters": {
            "sentence_transformer": {
                "model": "all-MiniLM-L6-v2",
                "cache_folder": "./models/"
            }
        }
    },
    "memory": {
        "type": "biomorphic",
        "stm_capacity": 150,
        "ltm_capacity": 20000,
        "working_capacity": 7,
        "use_semantic_indexing": true,
        "auto_consolidation": true,
        "consolidation_interval": 240.0,
        "strategies": {
            "consolidation": {
                "time_based": {
                    "min_age_seconds": 300.0,
                    "weight": 0.6
                },
                "importance_based": {
                    "threshold": 0.7,
                    "weight": 0.4
                }
            },
            "forgetting": {
                "ebbinghaus": {
                    "base_retention": 0.1,
                    "decay_rate": 0.693,
                    "access_boost": 2.0
                }
            }
        }
    },
    "performance": {
        "resource_monitoring": {
            "enabled": true,
            "check_interval": 5.0
        },
        "profiling": {
            "enabled": false,
            "output_file": "profile.prof"
        }
    }
}

# deployment.py
class ConfigurableKnowledgeSystem:
    def __init__(self, config_path: str):
        # Загрузка конфигурации
        self.config = Configuration.load_from_file(config_path)
        
        # Настройка логирования
        log_config = self.config.get("logging", {})
        setup_logging(
            level=log_config.get("level", "INFO"),
            log_file=log_config.get("file"),
            rotation=log_config.get("rotation", "10 MB"),
            retention=log_config.get("retention", "1 week")
        )
        
        self.logger = get_logger("configurable_system")
        self.logger.info(f"Запуск системы {self.config.get('system.name', 'NeuroGraph')}")
        
        # Инициализация мониторинга ресурсов
        if self.config.get("performance.resource_monitoring.enabled", False):
            from neurograph.core.resources import start_resource_monitoring
            interval = self.config.get("performance.resource_monitoring.check_interval", 5.0)
            start_resource_monitoring(interval)
            self.logger.info("Мониторинг ресурсов включен")
        
        # Инициализация компонентов
        self._initialize_components()
    
    def _initialize_components(self):
        """Инициализация компонентов согласно конфигурации"""
        
        # Семантический граф
        graph_config = self.config.get("semgraph", {})
        if graph_config.get("type") == "persistent":
            self.graph = SemGraphFactory.create(
                "persistent",
                file_path=graph_config.get("file_path", "graph.json"),
                auto_save_interval=graph_config.get("auto_save_interval", 300.0)
            )
        else:
            self.graph = SemGraphFactory.create("memory_efficient")
        
        # Векторные представления
        vec_config = self.config.get("contextvec", {})
        self.vectors = ContextVectorsFactory.create(
            vec_config.get("type", "dynamic"),
            vector_size=vec_config.get("vector_size", 384),
            use_indexing=vec_config.get("use_indexing", True)
        )
        
        # Биоморфная память с настройками стратегий
        memory_config = self.config.get("memory", {})
        self.memory = self._create_configured_memory(memory_config)
        
        # Адаптер векторизации
        adapter_config = vec_config.get("adapters.sentence_transformer", {})
        model_name = adapter_config.get("model", "all-MiniLM-L6-v2")
        
        try:
            from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter
            self.encoder = SentenceTransformerAdapter(model_name)
            self.logger.info(f"Инициализирован энкодер: {model_name}")
        except ImportError:
            self.logger.warning("SentenceTransformers недоступен, используется заглушка")
            self.encoder = None
        
        self.logger.info("Все компоненты инициализированы")
    
    def _create_configured_memory(self, memory_config: dict):
        """Создание памяти с настроенными стратегиями"""
        from neurograph.memory.impl.biomorphic import BiomorphicMemory
        from neurograph.memory.strategies import (
            TimeBasedConsolidation, ImportanceBasedConsolidation,
            AdaptiveConsolidation, EbbinghausBasedForgetting
        )
        
        # Создание стратегий консолидации
        strategies_config = memory_config.get("strategies", {})
        consolidation_strategies = []
        weights = []
        
        if "time_based" in strategies_config.get("consolidation", {}):
            time_config = strategies_config["consolidation"]["time_based"]
            strategy = TimeBasedConsolidation(
                min_age_seconds=time_config.get("min_age_seconds", 300.0)
            )
            consolidation_strategies.append(strategy)
            weights.append(time_config.get("weight", 0.5))
        
        if "importance_based" in strategies_config.get("consolidation", {}):
            imp_config = strategies_config["consolidation"]["importance_based"]
            strategy = ImportanceBasedConsolidation(
                importance_threshold=imp_config.get("threshold", 0.7)
            )
            consolidation_strategies.append(strategy)
            weights.append(imp_config.get("weight", 0.5))
        
        # Адаптивная стратегия
        if consolidation_strategies:
            consolidation_strategy = AdaptiveConsolidation(consolidation_strategies, weights)
        else:
            consolidation_strategy = TimeBasedConsolidation()
        
        # Стратегия забывания
        forgetting_config = strategies_config.get("forgetting.ebbinghaus", {})
        forgetting_strategy = EbbinghausBasedForgetting(
            base_retention=forgetting_config.get("base_retention", 0.1),
            decay_rate=forgetting_config.get("decay_rate", 0.693),
            access_boost=forgetting_config.get("access_boost", 2.0)
        )
        
        # Создание памяти
        memory = BiomorphicMemory(
            stm_capacity=memory_config.get("stm_capacity", 100),
            ltm_capacity=memory_config.get("ltm_capacity", 10000),
            use_semantic_indexing=memory_config.get("use_semantic_indexing", True),
            auto_consolidation=memory_config.get("auto_consolidation", True),
            consolidation_interval=memory_config.get("consolidation_interval", 300.0)
        )
        
        return memory
    
    def get_health_status(self) -> dict:
        """Проверка состояния системы"""
        from neurograph.core.resources import get_resource_usage
        
        # Системные ресурсы
        resources = get_resource_usage()
        
        # Статистика компонентов
        memory_stats = self.memory.get_memory_statistics()
        
        # Статистика графа
        graph_stats = {
            "nodes": len(self.graph.get_all_nodes()),
            "edges": len(self.graph.get_all_edges())
        }
        
        # Статистика векторов
        vector_stats = {
            "total_vectors": len(self.vectors.get_all_keys())
        }
        
        health_status = {
            "system": {
                "name": self.config.get("system.name"),
                "version": self.config.get("system.version"),
                "environment": self.config.get("system.environment"),
                "uptime": "N/A"  # Можно добавить отслеживание времени работы
            },
            "resources": {
                "cpu_percent": resources.get("cpu_percent", 0),
                "memory_mb": resources.get("memory_rss", 0) / 1024 / 1024,
                "thread_count": resources.get("thread_count", 0)
            },
            "components": {
                "graph": {
                    "status": "healthy",
                    "nodes": graph_stats["nodes"],
                    "edges": graph_stats["edges"]
                },
                "vectors": {
                    "status": "healthy",
                    "total_vectors": vector_stats["total_vectors"]
                },
                "memory": {
                    "status": "healthy",
                    "total_items": self.memory.size(),
                    "stm_pressure": memory_stats["memory_levels"]["stm"]["pressure"],
                    "ltm_pressure": memory_stats["memory_levels"]["ltm"]["pressure"],
                    "cache_hit_rate": memory_stats["memory_efficiency"]["cache_hit_rate"]
                }
            }
        }
        
        # Определение общего статуса
        if (health_status["resources"]["cpu_percent"] > 90 or 
            health_status["components"]["memory"]["stm_pressure"] > 0.95):
            health_status["overall_status"] = "warning"
        else:
            health_status["overall_status"] = "healthy"
        
        return health_status
    
    def shutdown(self):
        """Корректное завершение работы системы"""
        self.logger.info("Начало завершения работы системы")
        
        # Остановка мониторинга ресурсов
        try:
            from neurograph.core.resources import stop_resource_monitoring
            stop_resource_monitoring()
        except:
            pass
        
        # Завершение работы памяти (принудительная консолидация)
        try:
            self.memory.shutdown()
        except:
            pass
        
        # Сохранение графа (если персистентный)
        try:
            if hasattr(self.graph, 'save_now'):
                self.graph.save_now()
        except:
            pass
        
        self.logger.info("Система завершила работу")

# Использование конфигурируемой системы
def main():
    import signal
    import sys
    
    # Создание системы
    system = ConfigurableKnowledgeSystem("config.json")
    
    # Обработчик сигнала завершения
    def signal_handler(sig, frame):
        print("\nПолучен сигнал завершения...")
        system.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Демонстрация работы
        print("Система запущена. Добавление тестовых данных...")
        
        # Добавление знаний
        test_knowledge = [
            "Квантовые компьютеры используют квантовые биты",
            "Блокчейн обеспечивает децентрализованное хранение данных",
            "Нейронные сети имитируют работу человеческого мозга"
        ]
        
        for knowledge in test_knowledge:
            if system.encoder:
                embedding = system.encoder.encode(knowledge)
                from neurograph.memory.base import MemoryItem
                item = MemoryItem(knowledge, embedding, "knowledge")
                system.memory.add(item)
        
        # Статус здоровья
        health = system.get_health_status()
        print(f"\n=== Статус системы: {health['overall_status'].upper()} ===")
        print(f"CPU: {health['resources']['cpu_percent']:.1f}%")
        print(f"Память: {health['resources']['memory_mb']:.1f} MB")
        print(f"Граф: {health['components']['graph']['nodes']} узлов")
        print(f"Память: {health['components']['memory']['total_items']} элементов")
        print(f"Cache hit rate: {health['components']['memory']['cache_hit_rate']:.1%}")
        
        # Поддержание работы системы
        print("\nСистема работает. Нажмите Ctrl+C для завершения.")
        while True:
            import time
            time.sleep(10)
            
            # Периодический вывод статуса
            health = system.get_health_status()
            if health['overall_status'] == 'warning':
                print(f"[WARNING] Система перегружена: CPU {health['resources']['cpu_percent']:.1f}%")
    
    except KeyboardInterrupt:
        pass
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()
```

---

## Заключение

Данная документация описывает четыре готовых модуля системы NeuroGraph:

1. **Core** - Базовая инфраструктура с компонентами, конфигурацией, логированием, событиями и кешированием
2. **SemGraph** - Семантический граф с поиском, анализом и визуализацией
3. **ContextVec** - Векторные представления с адаптерами и легковесными моделями  
4. **Memory** - Биоморфная память с консолидацией и стратегиями забывания

### Ключевые возможности:

- **Полная интеграция** всех модулей через события и общие интерфейсы
- **Конфигурируемость** через файлы настроек
- **Масштабируемость** благодаря эффективным алгоритмам и индексам
- **Мониторинг** производительности и ресурсов
- **Биоморфная архитектура** памяти, имитирующая человеческую память

### Готовность к использованию:

Все описанные модули полностью готовы к использованию и могут служить основой для разработки остальных компонентов системы (NLP, Processor, Propagation, Integration).

Разработчикам других модулей достаточно следовать представленным интерфейсам и паттернам для обеспечения совместимости и интеграции с существующими компонентами.