# Документация разработчика: Модуль Memory

## Обзор

Модуль Memory предоставляет многоуровневую биоморфную память для хранения и управления информацией в системе NeuroGraph. Он имитирует работу человеческой памяти с различными уровнями хранения и механизмами консолидации.

## Архитектура

### Основные компоненты

```
memory/
├── base.py              # Базовые интерфейсы и классы
├── impl/                # Конкретные реализации
│   └── biomorphic.py    # Биоморфная память
├── strategies.py        # Стратегии консолидации и забывания
├── consolidation.py     # Механизмы консолидации памяти
└── factory.py           # Фабрика для создания экземпляров
```

### Ключевые интерфейсы

```python
class IMemory(ABC):
    """Базовый интерфейс для системы памяти."""
    
    @abstractmethod
    def add(self, item: MemoryItem) -> str:
        """Добавляет элемент в память и возвращает его ID."""
        pass
        
    @abstractmethod
    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Возвращает элемент памяти по ID."""
        pass
        
    @abstractmethod
    def search(self, query: Union[str, np.ndarray], limit: int = 10) -> List[Tuple[str, float]]:
        """Ищет элементы, похожие на запрос."""
        pass
```

## Биоморфная память

### Концепция

BiomorphicMemory реализует трёхуровневую модель памяти:

1. **Working Memory** - активные элементы для обработки (7±2 элемента)
2. **STM (Short-Term Memory)** - кратковременная память (по умолчанию 100 элементов)
3. **LTM (Long-Term Memory)** - долговременная память (по умолчанию 10,000 элементов)

### Создание и настройка

```python
from neurograph.memory import create_default_biomorphic_memory, create_lightweight_memory

# Стандартная конфигурация
memory = create_default_biomorphic_memory(
    stm_capacity=100,
    ltm_capacity=10000,
    use_semantic_indexing=True,
    auto_consolidation=True
)

# Облегченная версия для ограниченных ресурсов
lightweight_memory = create_lightweight_memory(
    stm_capacity=50,
    ltm_capacity=1000
)

# Высокопроизводительная версия
from neurograph.memory import create_high_performance_memory
performance_memory = create_high_performance_memory(
    stm_capacity=200,
    ltm_capacity=50000
)
```

### Работа с элементами памяти

```python
import numpy as np
from neurograph.memory import MemoryItem

# Создание элемента памяти
item = MemoryItem(
    content="Яблоки - это фрукты красного или зеленого цвета",
    embedding=np.random.random(384).astype(np.float32),
    content_type="text",
    metadata={
        "importance": 0.8,
        "source": "user_input",
        "topic": "food"
    }
)

# Добавление в память
item_id = memory.add(item)
print(f"Добавлен элемент с ID: {item_id}")

# Получение элемента
retrieved_item = memory.get(item_id)
if retrieved_item:
    print(f"Содержимое: {retrieved_item.content}")
    print(f"Количество обращений: {retrieved_item.access_count}")
```

### Поиск в памяти

```python
# Поиск по векторному представлению
query_vector = np.random.random(384).astype(np.float32)
results = memory.search(query_vector, limit=5)

for item_id, similarity_score in results:
    item = memory.get(item_id)
    print(f"ID: {item_id}, Сходство: {similarity_score:.3f}")
    print(f"Содержимое: {item.content[:50]}...")

# Поиск по тексту (требует интеграции с ContextVec)
text_results = memory.search("фрукты", limit=3)
```

## Стратегии консолидации

### Временная консолидация

```python
from neurograph.memory.strategies import TimeBasedConsolidation

# Консолидация на основе времени
time_strategy = TimeBasedConsolidation(
    min_age_seconds=300.0,  # 5 минут
    max_stm_size=100
)

# Использование в кастомной конфигурации
from neurograph.memory.impl.biomorphic import BiomorphicMemory

memory = BiomorphicMemory(
    stm_capacity=80,
    ltm_capacity=5000
)
# Стратегия устанавливается автоматически, но можно настроить
```

### Консолидация по важности

```python
from neurograph.memory.strategies import ImportanceBasedConsolidation

importance_strategy = ImportanceBasedConsolidation(
    importance_threshold=0.7,  # Консолидировать элементы с важностью > 0.7
    access_weight=0.3,         # Вес количества обращений
    recency_weight=0.4,        # Вес свежести
    content_weight=0.3         # Вес содержания
)
```

### Адаптивная консолидация

```python
from neurograph.memory.strategies import AdaptiveConsolidation

# Комбинирование нескольких стратегий
adaptive_strategy = AdaptiveConsolidation(
    strategies=[time_strategy, importance_strategy],
    weights=[0.6, 0.4]  # Временная стратегия важнее
)
```

## Стратегии забывания

### Кривая забывания Эббингауза

```python
from neurograph.memory.strategies import EbbinghausBasedForgetting

ebbinghaus_forgetting = EbbinghausBasedForgetting(
    base_retention=0.1,     # Базовый коэффициент удержания
    decay_rate=0.693,       # Скорость забывания (ln(2) для полураспада в 1 день)
    access_boost=2.0        # Усиление при каждом доступе
)
```

### LRU (Least Recently Used)

```python
from neurograph.memory.strategies import LeastRecentlyUsedForgetting

lru_forgetting = LeastRecentlyUsedForgetting(
    max_capacity=1000
)
```

## Механизмы консолидации

### Автоматическая консолидация

```python
# Консолидация происходит автоматически в фоновом режиме
memory = create_default_biomorphic_memory(
    auto_consolidation=True,
    consolidation_interval=300.0  # Каждые 5 минут
)

# Принудительная консолидация
consolidation_stats = memory.force_consolidation()
print(f"Переведено в LTM: {consolidation_stats['consolidated']} элементов")
```

### Мониторинг консолидации

```python
# Получение статистики консолидации
stats = memory.get_memory_statistics()

print("Уровни памяти:")
for level, info in stats["memory_levels"].items():
    print(f"  {level}: {info['size']}/{info['capacity']} (давление: {info['pressure']:.2f})")

print(f"Производительность:")
print(f"  Процент попаданий в кэш: {stats['memory_efficiency']['cache_hit_rate']:.2f}")
print(f"  Уровень консолидации: {stats['memory_efficiency']['consolidation_rate']:.2f}")
```

## Продвинутое использование

### Работа с метаданными

```python
# Поиск недавних элементов
recent_items = memory.get_recent_items(hours=24.0, memory_level="stm")
for item in recent_items[:5]:
    print(f"Создан: {item.created_at}, Содержимое: {item.content[:30]}...")

# Получение наиболее используемых элементов
most_accessed = memory.get_most_accessed_items(limit=10)
for item, access_count in most_accessed:
    print(f"Обращений: {access_count}, Содержимое: {item.content[:30]}...")
```

### Оптимизация памяти

```python
# Ручная оптимизация
optimization_stats = memory.optimize_memory()
print("Выполненные действия:")
for action in optimization_stats["actions_taken"]:
    print(f"  - {action}")

# Экспорт дампа для анализа
memory_dump = memory.export_memory_dump()
print(f"Общее количество элементов: {memory_dump['statistics']['total_items']}")
```

### События и мониторинг

```python
from neurograph.core.events import subscribe

def on_item_added(data):
    print(f"Добавлен элемент {data['item_id']} в {data['memory_level']}")

def on_consolidation_completed(data):
    print(f"Консолидация завершена: {data['consolidated_count']} элементов переведено в LTM")

# Подписка на события
subscribe("memory.item_added", on_item_added)
subscribe("memory.consolidation_completed", on_consolidation_completed)
```

## Кастомизация и расширение

### Создание собственной стратегии консолидации

```python
from neurograph.memory.strategies import ConsolidationStrategy

class CustomConsolidationStrategy(ConsolidationStrategy):
    def __init__(self, custom_threshold: float = 0.5):
        self.custom_threshold = custom_threshold
    
    def should_consolidate(self, stm_items, ltm_items):
        """Кастомная логика консолидации."""
        candidates = []
        for item_id, item in stm_items.items():
            # Ваша логика определения кандидатов для консолидации
            if self._calculate_consolidation_score(item) > self.custom_threshold:
                candidates.append(item_id)
        return candidates
    
    def _calculate_consolidation_score(self, item):
        # Ваш алгоритм оценки элемента
        return item.access_count * 0.3 + len(item.content) * 0.0001
```

### Создание кастомной реализации памяти

```python
from neurograph.memory.base import IMemory

class CustomMemory(IMemory):
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._items = {}
    
    def add(self, item: MemoryItem) -> str:
        if len(self._items) >= self.capacity:
            # Ваша логика управления переполнением
            self._evict_items()
        
        item_id = str(uuid.uuid4())
        item.id = item_id
        self._items[item_id] = item
        return item_id
    
    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)
    
    def search(self, query, limit=10):
        # Ваша логика поиска
        return []
    
    # Остальные методы интерфейса...
```

### Регистрация кастомной реализации

```python
from neurograph.memory.factory import MemoryFactory

# Регистрация новой реализации
MemoryFactory.register_implementation("custom", CustomMemory)

# Использование через фабрику
custom_memory = MemoryFactory.create("custom", capacity=500)
```

## Интеграция с другими модулями

### С ContextVec

```python
from neurograph.contextvec import ContextVectorsFactory

# Создание векторизатора
vectors = ContextVectorsFactory.create("dynamic")

# Создание элемента с автоматической векторизацией
def create_memory_item_with_embedding(text: str, metadata: dict = None):
    # В реальной интеграции это будет происходить автоматически
    embedding = vectors.encode(text)  # Предполагаемый метод
    
    return MemoryItem(
        content=text,
        embedding=embedding,
        content_type="text",
        metadata=metadata or {}
    )

item = create_memory_item_with_embedding(
    "Машинное обучение - это подраздел искусственного интеллекта",
    {"category": "AI", "importance": 0.9}
)
memory.add(item)
```

### С SemGraph

```python
# Интеграция с семантическим графом (концептуальный пример)
def add_memory_to_graph(memory_item, graph):
    """Добавляет элемент памяти в семантический граф."""
    # Извлечение сущностей из содержимого
    entities = extract_entities(memory_item.content)
    
    for entity in entities:
        # Добавление узла в граф
        graph.add_node(entity["text"], type=entity["type"])
        
        # Связывание с элементом памяти
        graph.add_edge(
            memory_item.id, 
            entity["text"], 
            "contains_entity",
            weight=entity["confidence"]
        )
```

## Лучшие практики

### 1. Управление ресурсами

```python
# Мониторинг использования памяти
def monitor_memory_usage(memory):
    stats = memory.get_memory_statistics()
    
    for level, info in stats["memory_levels"].items():
        pressure = info["pressure"]
        if pressure > 0.9:
            print(f"ВНИМАНИЕ: Высокое давление в {level}: {pressure:.2f}")
        elif pressure > 0.7:
            print(f"Предупреждение: Умеренное давление в {level}: {pressure:.2f}")

# Регулярный мониторинг
import threading
import time

def periodic_monitoring(memory, interval=60):
    while True:
        monitor_memory_usage(memory)
        time.sleep(interval)

# Запуск в отдельном потоке
monitoring_thread = threading.Thread(
    target=periodic_monitoring, 
    args=(memory, 30), 
    daemon=True
)
monitoring_thread.start()
```

### 2. Оптимизация производительности

```python
# Пакетное добавление элементов
def add_items_efficiently(memory, items_data):
    """Эффективное добавление множества элементов."""
    item_ids = []
    
    for content, embedding, metadata in items_data:
        item = MemoryItem(
            content=content,
            embedding=embedding,
            metadata=metadata
        )
        item_id = memory.add(item)
        item_ids.append(item_id)
        
        # Периодическая консолидация при больших объемах
        if len(item_ids) % 100 == 0:
            memory.force_consolidation()
    
    return item_ids

# Использование
items_data = [
    ("Текст 1", np.random.random(384), {"category": "A"}),
    ("Текст 2", np.random.random(384), {"category": "B"}),
    # ... множество элементов
]
item_ids = add_items_efficiently(memory, items_data)
```

### 3. Обработка ошибок

```python
from neurograph.core.errors import NeuroGraphError

try:
    item_id = memory.add(memory_item)
    retrieved_item = memory.get(item_id)
    
except NeuroGraphError as e:
    logger.error(f"Ошибка в работе с памятью: {e}")
    # Специфическая обработка ошибок NeuroGraph
    
except Exception as e:
    logger.error(f"Неожиданная ошибка: {e}")
    # Общая обработка ошибок
```

### 4. Тестирование

```python
import pytest
from neurograph.memory import create_lightweight_memory

def test_memory_basic_operations():
    """Тест базовых операций с памятью."""
    memory = create_lightweight_memory()
    
    # Создание тестового элемента
    item = MemoryItem(
        content="Тестовый контент",
        embedding=np.random.random(384).astype(np.float32),
        metadata={"test": True}
    )
    
    # Тестирование добавления
    item_id = memory.add(item)
    assert isinstance(item_id, str)
    assert len(item_id) > 0
    
    # Тестирование получения
    retrieved = memory.get(item_id)
    assert retrieved is not None
    assert retrieved.content == "Тестовый контент"
    assert retrieved.metadata["test"] is True
    
    # Тестирование поиска
    results = memory.search(item.embedding, limit=1)
    assert len(results) > 0
    assert results[0][0] == item_id

def test_memory_consolidation():
    """Тест механизма консолидации."""
    memory = create_lightweight_memory(stm_capacity=5)
    
    # Добавляем элементы, превышающие вместимость STM
    item_ids = []
    for i in range(10):
        item = MemoryItem(
            content=f"Элемент {i}",
            embedding=np.random.random(384).astype(np.float32)
        )
        item_id = memory.add(item)
        item_ids.append(item_id)
    
    # Принудительная консолидация
    consolidation_stats = memory.force_consolidation()
    
    # Проверяем, что консолидация произошла
    assert consolidation_stats["consolidated"] > 0
    
    # Проверяем, что элементы все еще доступны
    for item_id in item_ids:
        item = memory.get(item_id)
        assert item is not None

if __name__ == "__main__":
    test_memory_basic_operations()
    test_memory_consolidation()
    print("Все тесты прошли успешно!")
```

## Диагностика и отладка

### Логирование

```python
from neurograph.core.logging import setup_logging, get_logger

# Настройка детального логирования для модуля памяти
setup_logging(level="DEBUG", log_file="memory_debug.log")
logger = get_logger("memory.debug")

# В коде модуля будут выводиться подробные логи
memory = create_default_biomorphic_memory()
logger.info("Память инициализирована")
```

### Профилирование

```python
import cProfile
import pstats

def profile_memory_operations():
    """Профилирование операций с памятью."""
    memory = create_default_biomorphic_memory()
    
    # Создание тестовых данных
    items = []
    for i in range(1000):
        item = MemoryItem(
            content=f"Тестовый элемент {i}",
            embedding=np.random.random(384).astype(np.float32)
        )
        items.append(item)
    
    # Профилирование добавления
    pr = cProfile.Profile()
    pr.enable()
    
    for item in items:
        memory.add(item)
    
    pr.disable()
    
    # Анализ результатов
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Топ 10 самых медленных функций

# profile_memory_operations()
```

## Заключение

Модуль Memory предоставляет мощную и гибкую систему для управления памятью в NeuroGraph. Биоморфная архитектура с автоматической консолидацией и адаптивными стратегиями забывания обеспечивает эффективное использование ресурсов при сохранении важной информации.

Ключевые преимущества:
- **Биологически вдохновленная архитектура** с реалистичным моделированием работы памяти
- **Автоматическая консолидация** с настраиваемыми стратегиями
- **Высокая производительность** благодаря многоуровневой организации
- **Гибкость настройки** для различных сценариев использования
- **Rich monitoring** и диагностические возможности

Для получения дополнительной информации обратитесь к исходному коду модуля и примерам использования в тестах.