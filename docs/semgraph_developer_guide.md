# Модуль SemGraph - Документация для разработчиков

## Обзор

Модуль `neurograph.semgraph` предоставляет мощную и гибкую систему для работы с семантическими графами знаний. Он является основой для хранения и организации структурированных знаний в системе NeuroGraph.

### Ключевые особенности

- **Гибкая архитектура** с поддержкой различных реализаций (в памяти, персистентные)
- **Эффективное индексирование** для быстрого поиска (HNSW, FAISS)
- **Мощная система запросов** с поддержкой шаблонов и путей
- **Визуализация графов** с экспортом в различные форматы
- **Интеграция с векторными представлениями** для семантического поиска

## Быстрый старт

```python
from neurograph.semgraph import SemGraphFactory
from neurograph.semgraph.query import PatternMatcher, PathFinder

# Создание графа
graph = SemGraphFactory.create("memory_efficient")

# Добавление узлов и ребер
graph.add_node("python", type="language", popularity="high")
graph.add_node("programming", type="activity")
graph.add_edge("python", "programming", "used_for", weight=0.9)

# Поиск соседей
neighbors = graph.get_neighbors("python")

# Поиск путей
path_finder = PathFinder(graph)
path = path_finder.find_shortest_path("python", "programming")
```

## Архитектура модуля

### Структура директории

```
semgraph/
├── __init__.py          # Основной API
├── base.py              # Базовые интерфейсы и классы
├── factory.py           # Фабрика для создания графов
├── impl/                # Конкретные реализации
│   ├── memory_graph.py     # Граф в памяти
│   └── persistent_graph.py # Персистентный граф
├── index/               # Индексирование
│   └── hnsw.py             # HNSW индекс
├── query/               # Система запросов
│   ├── pattern.py          # Поиск по шаблонам
│   ├── path.py             # Поиск путей
│   └── search.py           # Расширенный поиск
├── analysis/            # Анализ графов
│   └── metrics.py          # Метрики и статистика
├── neural/              # Нейронные компоненты
│   └── embedding.py        # Векторные представления
└── visualization/       # Визуализация
    └── visualizer.py       # Инструменты визуализации
```

## Основные компоненты

### 1. Базовые интерфейсы (base.py)

#### ISemGraph

Основной интерфейс для всех реализаций семантического графа:

```python
from neurograph.semgraph.base import ISemGraph

class ISemGraph(ABC):
    @abstractmethod
    def add_node(self, node_id: str, **attributes) -> None:
        """Добавляет узел с атрибутами"""
        
    @abstractmethod
    def add_edge(self, source: str, target: str, edge_type: str = "default", 
                weight: float = 1.0, **attributes) -> None:
        """Добавляет направленное ребро"""
        
    @abstractmethod
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Возвращает соседей узла"""
```

#### Node и Edge

Классы для представления узлов и ребер:

```python
from neurograph.semgraph.base import Node, Edge

# Создание узла
node = Node("concept_1", type="entity", importance=0.8)

# Создание ребра
edge = Edge("concept_1", "concept_2", "related_to", weight=0.7)
```

### 2. Реализации графов

#### MemoryEfficientSemGraph

Оптимизированная по памяти реализация на основе NetworkX:

```python
from neurograph.semgraph.impl.memory_graph import MemoryEfficientSemGraph

graph = MemoryEfficientSemGraph()

# Основные операции
graph.add_node("python", type="language")
graph.add_edge("python", "programming", "used_for")

# Сериализация
graph.save("my_graph.json")
loaded_graph = MemoryEfficientSemGraph.load("my_graph.json")

# Слияние графов
other_graph = MemoryEfficientSemGraph()
graph.merge(other_graph)
```

#### PersistentSemGraph

Граф с автоматическим сохранением на диск:

```python
from neurograph.semgraph.impl.persistent_graph import PersistentSemGraph

# Автосохранение каждые 5 минут
graph = PersistentSemGraph("graph.json", auto_save_interval=300.0)

# Работает как обычный граф, но автоматически сохраняется
graph.add_node("concept", data="value")

# Принудительное сохранение
graph.save_now()

# Перезагрузка из файла
graph.reload()
```

### 3. Система запросов

#### PatternMatcher - Поиск по шаблонам

```python
from neurograph.semgraph.query.pattern import Pattern, PatternMatcher

matcher = PatternMatcher(graph)

# Простой шаблон: все связи типа "used_for"
pattern = Pattern(subject=None, predicate="used_for", object=None)
results = matcher.match(pattern)

# Поиск с регулярными выражениями
results = matcher.match_with_regex(
    subject_pattern=r"python.*",
    predicate_pattern="used_for"
)

# Поиск по атрибутам узлов
results = matcher.match_with_attributes(
    subject_attrs={"type": "language"},
    edge_type="used_for"
)
```

#### PathFinder - Поиск путей

```python
from neurograph.semgraph.query.path import PathFinder

path_finder = PathFinder(graph)

# Кратчайший путь
path = path_finder.find_shortest_path("python", "ai")

# Все пути (ограниченные по глубине)
paths = path_finder.find_all_paths("python", "ai", max_depth=3)

# Взвешенный кратчайший путь (алгоритм Дейкстры)
weighted_path = path_finder.find_weighted_shortest_path("python", "ai")
```

#### GraphSearcher - Расширенный поиск

```python
from neurograph.semgraph.query.search import GraphSearcher

searcher = GraphSearcher(graph)

# Поиск по атрибутам
nodes = searcher.search_by_attribute("type", "language")

# Поиск по регулярному выражению
nodes = searcher.search_by_regex("name", r"python.*")

# Поиск по расстоянию
nearby = searcher.search_by_distance("python", max_distance=2)

# Поиск похожих по связям
similar = searcher.search_nodes_with_similar_connections("python")

# Поиск подграфов по паттерну
pattern = {
    "nodes": [
        {"id": "A", "type": "language"},
        {"id": "B", "type": "activity"}
    ],
    "edges": [
        {"source": "A", "target": "B", "type": "used_for"}
    ]
}
matches = searcher.search_subgraph_pattern(pattern)
```

### 4. Индексирование

#### HNSW Index для векторного поиска

```python
from neurograph.semgraph.index.hnsw import HNSWIndex
import numpy as np

# Создание индекса
index = HNSWIndex(dim=384, max_elements=10000)

# Добавление векторов
vector = np.random.random(384)
index.add_item("concept_1", vector)

# Поиск похожих
query_vector = np.random.random(384)
similar_items = index.search(query_vector, k=5)

# Сохранение и загрузка
index.save("my_index")
loaded_index = HNSWIndex.load("my_index")
```

### 5. Анализ и метрики

#### GraphMetrics - Анализ структуры графа

```python
from neurograph.semgraph.analysis.metrics import GraphMetrics

metrics = GraphMetrics(graph)

# Базовые метрики
node_count = metrics.get_node_count()
edge_count = metrics.get_edge_count()

# Степени узлов
degrees = metrics.get_node_degrees()

# Центральные узлы (PageRank)
central_nodes = metrics.get_central_nodes(top_n=10)

# Кратчайшие пути
paths = metrics.get_shortest_paths("python", ["ai", "ml", "data"])

# Связные компоненты
components = metrics.get_connected_components()

# Сообщества (если доступна библиотека community)
communities = metrics.get_communities()

# Все метрики сразу
all_metrics = metrics.compute_all_metrics()
```

### 6. Векторные представления

#### GraphEmbedding - Обучение эмбеддингов

```python
from neurograph.semgraph.neural.embedding import GraphEmbedding

# Создание модели эмбеддингов
embedding = GraphEmbedding(graph, embedding_dim=100)

# Инициализация случайных эмбеддингов
embedding.initialize_random_embeddings(seed=42)

# Обучение на структуре графа (TransE)
embedding.train_embeddings(
    num_iterations=1000,
    learning_rate=0.01,
    negative_samples=5
)

# Получение эмбеддингов
node_embedding = embedding.get_node_embedding("python")
edge_type_embedding = embedding.get_edge_type_embedding("used_for")

# Сохранение и загрузка
embedding.save("embeddings.json")
loaded_embedding = GraphEmbedding.load("embeddings.json", graph)
```

### 7. Визуализация

#### GraphVisualizer - Визуализация графов

```python
from neurograph.semgraph.visualization.visualizer import GraphVisualizer

visualizer = GraphVisualizer(graph)

# Базовая визуализация
visualizer.visualize(output_path="graph.png", show=True)

# Визуализация подграфа
node_ids = ["python", "programming", "ai"]
visualizer.visualize_subgraph(
    node_ids, 
    include_neighbors=True,
    output_path="subgraph.png"
)

# Экспорт в различные форматы
visualizer.save_as_graphml("graph.graphml")
visualizer.save_as_gexf("graph.gexf")
visualizer.export_to_cytoscape("graph_cytoscape.json")
```

## Паттерны использования

### 1. Создание и наполнение графа

```python
from neurograph.semgraph import SemGraphFactory

# Создание графа
graph = SemGraphFactory.create("memory_efficient")

# Добавление концептов программирования
concepts = [
    ("python", {"type": "language", "paradigm": "multi"}),
    ("java", {"type": "language", "paradigm": "oop"}),
    ("programming", {"type": "activity", "domain": "cs"}),
    ("machine_learning", {"type": "field", "domain": "ai"})
]

for concept_id, attrs in concepts:
    graph.add_node(concept_id, **attrs)

# Добавление связей
relations = [
    ("python", "programming", "used_for", 0.9),
    ("java", "programming", "used_for", 0.8),
    ("python", "machine_learning", "popular_in", 0.9)
]

for src, tgt, rel_type, weight in relations:
    graph.add_edge(src, tgt, rel_type, weight=weight)
```

### 2. Комплексный поиск и анализ

```python
from neurograph.semgraph.query import PatternMatcher, PathFinder
from neurograph.semgraph.analysis.metrics import GraphMetrics

# Поиск всех языков программирования
pattern = Pattern(subject=None, predicate=None, object=None)
matcher = PatternMatcher(graph)

programming_languages = []
for node_id in graph.get_all_nodes():
    node_data = graph.get_node(node_id)
    if node_data and node_data.get("type") == "language":
        programming_languages.append(node_id)

# Анализ связей между языками
path_finder = PathFinder(graph)
metrics = GraphMetrics(graph)

for i, lang1 in enumerate(programming_languages):
    for lang2 in programming_languages[i+1:]:
        path = path_finder.find_shortest_path(lang1, lang2)
        if path:
            print(f"Путь {lang1} -> {lang2}: {len(path)} шагов")

# Поиск центральных концептов
central_concepts = metrics.get_central_nodes(top_n=5)
print("Наиболее важные концепты:", central_concepts)
```

### 3. Работа с большими графами

```python
from neurograph.semgraph.impl.persistent_graph import PersistentSemGraph
from neurograph.semgraph.index.hnsw import HNSWIndex

# Создание персистентного графа для больших данных
large_graph = PersistentSemGraph(
    "large_graph.json", 
    auto_save_interval=600  # автосохранение каждые 10 минут
)

# Создание индекса для быстрого поиска
vector_index = HNSWIndex(dim=384, max_elements=100000)

# Массовое добавление данных
for i in range(10000):
    node_id = f"concept_{i}"
    large_graph.add_node(node_id, type="auto_generated")
    
    # Добавление в векторный индекс
    vector = np.random.random(384)
    vector_index.add_item(node_id, vector)
    
    if i % 1000 == 0:
        print(f"Добавлено {i} узлов")

# Сохранение индекса
vector_index.save("large_index")
```

### 4. Интеграция с внешними данными

```python
import json
from neurograph.semgraph import SemGraphFactory

def load_from_knowledge_base(kb_file: str):
    """Загрузка данных из внешней базы знаний"""
    
    graph = SemGraphFactory.create("memory_efficient")
    
    with open(kb_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Загрузка сущностей
    for entity in data.get('entities', []):
        graph.add_node(
            entity['id'],
            name=entity['name'],
            type=entity['type'],
            description=entity.get('description', ''),
            **entity.get('attributes', {})
        )
    
    # Загрузка отношений
    for relation in data.get('relations', []):
        graph.add_edge(
            relation['source'],
            relation['target'],
            relation['type'],
            weight=relation.get('confidence', 1.0),
            **relation.get('attributes', {})
        )
    
    return graph

# Использование
graph = load_from_knowledge_base("external_kb.json")
```

## Продвинутые возможности

### 1. Пользовательские реализации графа

```python
from neurograph.semgraph.base import ISemGraph
from neurograph.semgraph.factory import graph_registry

class CustomSemGraph(ISemGraph):
    """Пользовательская реализация графа"""
    
    def __init__(self, backend="redis"):
        self.backend = backend
        # Инициализация пользовательского бэкенда
    
    def add_node(self, node_id: str, **attributes) -> None:
        # Пользовательская логика добавления узла
        pass
    
    # Реализация остальных методов интерфейса
    # ...

# Регистрация пользовательской реализации
graph_registry.register("custom", CustomSemGraph)

# Использование через фабрику
custom_graph = SemGraphFactory.create("custom", backend="redis")
```

### 2. Пользовательские индексы

```python
from neurograph.semgraph.index.hnsw import HNSWIndex

class MultiModalIndex(HNSWIndex):
    """Индекс для мультимодальных данных"""
    
    def __init__(self, text_dim: int, image_dim: int, **kwargs):
        super().__init__(dim=text_dim + image_dim, **kwargs)
        self.text_dim = text_dim
        self.image_dim = image_dim
    
    def add_multimodal_item(self, item_id: str, text_vector: np.ndarray, 
                           image_vector: np.ndarray):
        combined_vector = np.concatenate([text_vector, image_vector])
        self.add_item(item_id, combined_vector)
    
    def search_by_text(self, text_vector: np.ndarray, k: int = 10):
        # Поиск только по текстовой части
        padded_vector = np.concatenate([
            text_vector, 
            np.zeros(self.image_dim)
        ])
        return self.search(padded_vector, k)
```

### 3. Кастомные метрики

```python
from neurograph.semgraph.analysis.metrics import GraphMetrics

class ExtendedGraphMetrics(GraphMetrics):
    """Расширенные метрики графа"""
    
    def get_semantic_density(self) -> float:
        """Вычисление семантической плотности"""
        total_nodes = self.get_node_count()
        total_edges = self.get_edge_count()
        
        if total_nodes <= 1:
            return 0.0
        
        max_possible_edges = total_nodes * (total_nodes - 1)
        return total_edges / max_possible_edges
    
    def get_type_distribution(self) -> Dict[str, int]:
        """Распределение узлов по типам"""
        type_counts = {}
        
        for node_id in self.graph.get_all_nodes():
            node_data = self.graph.get_node(node_id) or {}
            node_type = node_data.get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        return type_counts
    
    def get_clustering_coefficient(self) -> float:
        """Коэффициент кластеризации"""
        try:
            return nx.average_clustering(self.nx_graph.to_undirected())
        except:
            return 0.0
```

## Рекомендации по производительности

### 1. Выбор реализации графа

- **MemoryEfficientSemGraph**: для небольших и средних графов (< 100k узлов)
- **PersistentSemGraph**: для долгоживущих приложений с автосохранением
- **Пользовательские реализации**: для специфических требований (Redis, PostgreSQL, etc.)

### 2. Оптимизация запросов

```python
# ✅ Хорошо: используйте индексы для частых запросов
index = HNSWIndex(dim=384)
# Добавляйте векторы узлов в индекс

# ✅ Хорошо: кешируйте результаты поиска путей
path_cache = {}
def cached_shortest_path(src, tgt):
    key = (src, tgt)
    if key not in path_cache:
        path_cache[key] = path_finder.find_shortest_path(src, tgt)
    return path_cache[key]

# ❌ Плохо: повторные линейные поиски
for node in graph.get_all_nodes():  # O(n)
    if graph.get_node(node).get("type") == "language":  # O(1)
        # ...

# ✅ Хорошо: предварительная индексация по типам
language_nodes = searcher.search_by_attribute("type", "language")
```

### 3. Управление памятью

```python
# Для больших графов используйте генераторы
def iterate_large_graph(graph):
    for node_id in graph.get_all_nodes():
        yield node_id, graph.get_node(node_id)

# Периодическая очистка кешей
if len(path_cache) > 10000:
    path_cache.clear()

# Используйте персистентное хранение для временных данных
temp_graph = PersistentSemGraph("temp_graph.json")
```

## Тестирование

### Базовый набор тестов

```python
import pytest
from neurograph.semgraph import SemGraphFactory

class TestSemGraph:
    
    @pytest.fixture
    def graph(self):
        return SemGraphFactory.create("memory_efficient")
    
    def test_add_node(self, graph):
        graph.add_node("test_node", type="test")
        assert graph.has_node("test_node")
        
        node_data = graph.get_node("test_node")
        assert node_data["type"] == "test"
    
    def test_add_edge(self, graph):
        graph.add_node("node1")
        graph.add_node("node2")
        graph.add_edge("node1", "node2", "test_relation", weight=0.8)
        
        assert graph.has_edge("node1", "node2", "test_relation")
        
        edge_data = graph.get_edge("node1", "node2", "test_relation")
        assert edge_data["weight"] == 0.8
    
    def test_neighbors(self, graph):
        graph.add_node("center")
        graph.add_node("neighbor1")
        graph.add_node("neighbor2")
        
        graph.add_edge("center", "neighbor1", "rel1")
        graph.add_edge("center", "neighbor2", "rel2")
        
        neighbors = graph.get_neighbors("center")
        assert len(neighbors) == 2
        assert "neighbor1" in neighbors
        assert "neighbor2" in neighbors
    
    def test_serialization(self, graph, tmp_path):
        graph.add_node("test", data="value")
        
        file_path = tmp_path / "test_graph.json"
        graph.save(str(file_path))
        
        loaded_graph = SemGraphFactory.create("memory_efficient")
        loaded_graph = type(graph).load(str(file_path))
        
        assert loaded_graph.has_node("test")
        assert loaded_graph.get_node("test")["data"] == "value"
```

## Заключение

Модуль SemGraph предоставляет мощную и гибкую основу для работы с семантическими графами в системе NeuroGraph. Его модульная архитектура позволяет легко расширять функциональность, а богатый набор инструментов покрывает большинство задач работы с графами знаний.

### Основные преимущества:
- **Гибкость**: поддержка различных типов графов и индексов
- **Производительность**: оптимизированные алгоритмы и структуры данных  
- **Масштабируемость**: от простых экспериментов до промышленных решений
- **Интеграция**: легкая интеграция с другими модулями NeuroGraph

### Следующие шаги:
1. Изучите примеры в `examples/` для конкретных задач
2. Рассмотрите интеграцию с модулем `propagation` для распространения активации
3. Экспериментируйте с векторными представлениями для семантического поиска
4. Создайте пользовательские реализации для специфических требований

Для получения дополнительной информации обратитесь к документации по архитектуре проекта и примерам использования в соответствующих директориях.