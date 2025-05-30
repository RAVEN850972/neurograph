# NeuroGraph NLP Module - Complete Documentation

Полная документация модуля обработки естественного языка системы NeuroGraph для разработчиков других модулей.

## Содержание

1. [Обзор и назначение](#обзор-и-назначение)
2. [Архитектура модуля](#архитектура-модуля)
3. [Публичные интерфейсы](#публичные-интерфейсы)
4. [Основные классы данных](#основные-классы-данных)
5. [Фабрика и конфигурация](#фабрика-и-конфигурация)
6. [Интеграция с другими модулями](#интеграция-с-другими-модулями)
7. [Примеры использования](#примеры-использования)
8. [События и уведомления](#события-и-уведомления)
9. [Производительность и масштабирование](#производительность-и-масштабирование)
10. [Расширение модуля](#расширение-модуля)
11. [Обработка ошибок](#обработка-ошибок)
12. [Конфигурация и настройка](#конфигурация-и-настройка)

---

## Обзор и назначение

### Что делает модуль NLP

Модуль NLP отвечает за обработку естественного языка в системе NeuroGraph и предоставляет следующие возможности:

- **Токенизация**: Разбиение текста на токены с лингвистическими атрибутами
- **Извлечение сущностей**: Поиск именованных сущностей (люди, организации, места, концепты)
- **Извлечение отношений**: Поиск семантических связей между сущностями
- **Генерация текста**: Создание ответов и текстов по шаблонам
- **Извлечение знаний**: Структурирование информации из неструктурированного текста

### Место в архитектуре NeuroGraph

```
┌─────────────────────────────────────────────────────────┐
│                    Integration Layer                    │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     NLP     │◄──►│  Processor  │◄──►│ Propagation │
│   (Этот)    │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SemGraph  │    │ ContextVec  │    │   Memory    │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Входящие данные:**
- Необработанный текст
- Запросы пользователей
- Документы для анализа

**Исходящие данные:**
- Структурированные знания для SemGraph
- Векторные представления для ContextVec
- Факты и концепты для Memory
- Логические утверждения для Processor

---

## Архитектура модуля

### Структура файлов

```
neurograph/nlp/
├── __init__.py                 # Публичный API модуля
├── base.py                     # Базовые интерфейсы и классы данных
├── processor.py                # Основные реализации процессоров
├── factory.py                  # Фабрика и конфигурация
├── tokenization.py             # Токенизаторы
├── entity_extraction.py        # Извлечение сущностей
├── relation_extraction.py      # Извлечение отношений
├── text_generation.py          # Генерация текста
├── semantic_processor.py       # Семантический анализ
└── examples/
    └── basic_usage.py          # Примеры использования
```

### Компонентная архитектура

```
┌─────────────────────────────────────────────────┐
│              NLP Module Interface               │
│                (INLProcessor)                   │
└─────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────┐
│              Core Components                    │
├─────────────┬─────────────┬─────────────────────┤
│ Tokenizer   │ Entity      │ Relation            │
│             │ Extractor   │ Extractor           │
├─────────────┼─────────────┼─────────────────────┤
│ Text        │ Semantic    │ Configuration       │
│ Generator   │ Processor   │ Manager             │
└─────────────┴─────────────┴─────────────────────┘
                        │
┌─────────────────────────────────────────────────┐
│              Implementation Layer               │
├─────────────┬─────────────┬─────────────────────┤
│ Simple      │ spaCy       │ Hybrid              │
│ Rules       │ ML-based    │ Combined            │
└─────────────┴─────────────┴─────────────────────┘
```

---

## Публичные интерфейсы

### Основной интерфейс: INLProcessor

Это главный интерфейс для всех NLP процессоров в системе:

```python
from neurograph.nlp import INLProcessor, ProcessingResult

class INLProcessor(ABC):
    @abstractmethod
    def process_text(self, text: str, 
                    extract_entities: bool = True,
                    extract_relations: bool = True,
                    analyze_sentiment: bool = False) -> ProcessingResult:
        """Полная обработка текста с извлечением сущностей и отношений."""
        pass
    
    @abstractmethod
    def extract_knowledge(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение структурированных знаний из текста."""
        pass
    
    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """Нормализация текста."""
        pass
    
    @abstractmethod
    def get_language(self, text: str) -> str:
        """Определение языка текста."""
        pass
```

### Дополнительные интерфейсы

```python
# Токенизация
class ITokenizer(ABC):
    def tokenize(self, text: str) -> List[Token]
    def sent_tokenize(self, text: str) -> List[str]

# Извлечение сущностей
class IEntityExtractor(ABC):
    def extract_entities(self, text: str) -> List[Entity]
    def extract_entities_from_tokens(self, tokens: List[Token]) -> List[Entity]

# Извлечение отношений
class IRelationExtractor(ABC):
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]
    def extract_relations_from_sentence(self, sentence: Sentence) -> List[Relation]

# Генерация текста
class ITextGenerator(ABC):
    def generate_text(self, template: str, params: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> str
    def generate_response(self, query: str, knowledge: Dict[str, Any],
                         style: Optional[str] = None) -> str
```

---

## Основные классы данных

### ProcessingResult - Результат обработки

```python
@dataclass
class ProcessingResult:
    original_text: str              # Исходный текст
    sentences: List[Sentence]       # Разбитый на предложения
    entities: List[Entity]          # Найденные сущности
    relations: List[Relation]       # Найденные отношения
    tokens: List[Token]             # Токены
    processing_time: float          # Время обработки
    language: Optional[str]         # Определенный язык
    metadata: Optional[Dict[str, Any]]  # Дополнительные данные
```

### Entity - Сущность

```python
@dataclass
class Entity:
    text: str                       # Текст сущности
    entity_type: EntityType         # Тип сущности
    start: int                      # Начальная позиция
    end: int                        # Конечная позиция
    confidence: float = 1.0         # Уверенность (0.0-1.0)
    metadata: Optional[Dict[str, Any]] = None  # Метаданные

# Типы сущностей
class EntityType(Enum):
    PERSON = "PERSON"               # Люди
    ORGANIZATION = "ORG"            # Организации
    LOCATION = "LOC"                # Места
    DATE = "DATE"                   # Даты
    TIME = "TIME"                   # Время
    MONEY = "MONEY"                 # Денежные суммы
    CONCEPT = "CONCEPT"             # Концепты
    FACT = "FACT"                   # Факты
    UNKNOWN = "UNKNOWN"             # Неизвестный тип
```

### Relation - Отношение

```python
@dataclass
class Relation:
    subject: Entity                 # Субъект отношения
    predicate: RelationType         # Тип отношения
    object: Entity                  # Объект отношения
    confidence: float = 1.0         # Уверенность
    text_span: Optional[str] = None # Текст, откуда извлечено
    metadata: Optional[Dict[str, Any]] = None

# Типы отношений
class RelationType(Enum):
    IS_A = "is_a"                   # "является"
    HAS = "has"                     # "имеет"
    LOCATED_IN = "located_in"       # "находится в"
    WORKS_FOR = "works_for"         # "работает в"
    CREATED_BY = "created_by"       # "создано"
    USES = "uses"                   # "использует"
    RELATED_TO = "related_to"       # "связано с"
    CAUSES = "causes"               # "вызывает"
    PART_OF = "part_of"             # "часть от"
    UNKNOWN = "unknown"             # Неизвестный тип
```

### Token - Токен

```python
@dataclass
class Token:
    text: str                       # Текст токена
    lemma: str                      # Лемма
    pos: str                        # Часть речи
    tag: str                        # Подробный тег
    is_alpha: bool                  # Алфавитный символ
    is_stop: bool                   # Стоп-слово
    is_punct: bool                  # Пунктуация
    start: int                      # Начальная позиция
    end: int                        # Конечная позиция
    dependency: Optional[str] = None    # Синтаксическая зависимость
    head: Optional[int] = None          # Индекс главного токена
```

### Sentence - Предложение

```python
@dataclass
class Sentence:
    text: str                       # Текст предложения
    tokens: List[Token]             # Токены предложения
    start: int                      # Начальная позиция
    end: int                        # Конечная позиция
    entities: Optional[List[Entity]] = None     # Сущности в предложении
    relations: Optional[List[Relation]] = None  # Отношения в предложении
    sentiment: Optional[float] = None           # Тональность (-1 до 1)
```

---

## Фабрика и конфигурация

### NLPFactory - Создание процессоров

```python
from neurograph.nlp import NLPFactory

# Создание стандартных процессоров
processor = NLPFactory.create("standard", language="ru", use_spacy=True)
lightweight = NLPFactory.create("lightweight", language="ru")
advanced = NLPFactory.create("advanced", language="ru", use_spacy=True)

# Создание из конфигурации
config = {"type": "standard", "params": {"language": "ru"}}
processor = NLPFactory.create_from_config(config)

# Получение доступных типов
available = NLPFactory.get_available_processors()
# Результат: ["standard", "lightweight", "advanced"]
```

### Готовые фабричные функции

```python
from neurograph.nlp import (
    create_default_processor,
    create_lightweight_processor,
    create_high_performance_processor
)

# Процессор по умолчанию
default_proc = create_default_processor(language="ru", use_spacy=True)

# Облегченный для ограниченных ресурсов
light_proc = create_lightweight_processor(language="ru")

# Высокопроизводительный
perf_proc = create_high_performance_processor(language="ru")
```

### NLPConfiguration - Управление настройками

```python
from neurograph.nlp import NLPConfiguration

# Создание конфигурации
config = NLPConfiguration()

# Обновление настроек
config.update_config({
    "processor": {
        "type": "advanced",
        "language": "ru"
    },
    "entity_extraction": {
        "confidence_threshold": 0.8
    }
})

# Валидация
is_valid = config.validate_config()

# Создание процессора из конфигурации
processor = config.create_processor()

# Сохранение/загрузка
config.save_to_file("nlp_config.json")
loaded_config = NLPConfiguration.load_from_file("nlp_config.json")
```

### Предустановленные конфигурации

```python
from neurograph.nlp import (
    get_development_config,
    get_production_config,
    get_minimal_config
)

# Конфигурация для разработки (быстрая, простая)
dev_config = get_development_config()

# Конфигурация для продакшена (полная функциональность)
prod_config = get_production_config()

# Минимальная конфигурация (ограниченные ресурсы)
min_config = get_minimal_config()
```

---

## Интеграция с другими модулями

### Интеграция с SemGraph

NLP модуль подготавливает данные для графа знаний:

```python
def integrate_with_semgraph(nlp_result: ProcessingResult, graph):
    """Интеграция результатов NLP с SemGraph."""
    
    # Добавление сущностей как узлов
    for entity in nlp_result.entities:
        graph.add_node(
            entity.text,
            type=entity.entity_type.value,
            confidence=entity.confidence,
            source_text=nlp_result.original_text
        )
    
    # Добавление отношений как ребер
    for relation in nlp_result.relations:
        graph.add_edge(
            relation.subject.text,
            relation.object.text,
            relation.predicate.value,
            weight=relation.confidence,
            source_text=relation.text_span
        )
```

**Формат данных для SemGraph:**
```python
# Структура узла
{
    "id": "entity_text",
    "type": "PERSON|ORG|LOCATION|CONCEPT|etc",
    "confidence": 0.95,
    "source_text": "исходный текст",
    "metadata": {"extractor": "spacy", "method": "ner"}
}

# Структура ребра
{
    "source": "subject_entity",
    "target": "object_entity", 
    "relation": "is_a|has|located_in|etc",
    "weight": 0.87,
    "source_text": "фрагмент текста",
    "metadata": {"extractor": "pattern_based"}
}
```

### Интеграция с ContextVec

NLP создает текстовые представления для векторизации:

```python
def integrate_with_contextvec(nlp_result: ProcessingResult, vectors, encoder):
    """Интеграция результатов NLP с ContextVec."""
    
    # Векторизация сущностей
    for entity in nlp_result.entities:
        if entity.entity_type in [EntityType.CONCEPT, EntityType.FACT]:
            vector = encoder.encode(entity.text)
            vectors.create_vector(entity.text, vector)
    
    # Векторизация отношений
    for relation in nlp_result.relations:
        relation_text = f"{relation.subject.text} {relation.predicate.value} {relation.object.text}"
        vector = encoder.encode(relation_text)
        vectors.create_vector(f"rel_{hash(relation_text)}", vector)
    
    # Векторизация предложений
    for sentence in nlp_result.sentences:
        if len(sentence.entities) > 0:  # Только предложения с сущностями
            vector = encoder.encode(sentence.text)
            vectors.create_vector(f"sent_{hash(sentence.text)}", vector)
```

**Формат данных для ContextVec:**
```python
# Векторные элементы
{
    "key": "entity_text|relation_id|sentence_id",
    "vector": np.array([0.1, 0.2, ...]),  # Размерность 384/768/etc
    "metadata": {
        "type": "entity|relation|sentence",
        "confidence": 0.95,
        "source": "nlp_extraction"
    }
}
```

### Интеграция с Memory

NLP подготавливает элементы для биоморфной памяти:

```python
def integrate_with_memory(nlp_result: ProcessingResult, memory, encoder):
    """Интеграция результатов NLP с Memory."""
    
    from neurograph.memory.base import MemoryItem
    
    # Добавление фактов в память
    for relation in nlp_result.relations:
        fact_text = f"{relation.subject.text} {relation.predicate.value} {relation.object.text}"
        embedding = encoder.encode(fact_text)
        
        memory_item = MemoryItem(
            content=fact_text,
            embedding=embedding,
            content_type="fact",
            metadata={
                "confidence": relation.confidence,
                "subject": relation.subject.text,
                "predicate": relation.predicate.value,
                "object": relation.object.text,
                "source_text": nlp_result.original_text
            }
        )
        
        memory.add(memory_item)
    
    # Добавление концептов
    for entity in nlp_result.entities:
        if entity.entity_type == EntityType.CONCEPT:
            embedding = encoder.encode(entity.text)
            
            memory_item = MemoryItem(
                content=entity.text,
                embedding=embedding,
                content_type="concept",
                metadata={
                    "entity_type": entity.entity_type.value,
                    "confidence": entity.confidence
                }
            )
            
            memory.add(memory_item)
```

**Формат данных для Memory:**
```python
# Элемент памяти
{
    "content": "текстовое содержимое",
    "embedding": np.array([...]),
    "content_type": "fact|concept|definition|instruction",
    "metadata": {
        "confidence": 0.95,
        "source": "nlp",
        "entities": ["entity1", "entity2"],
        "relations": [{"subj": "A", "pred": "is_a", "obj": "B"}]
    }
}
```

### Интеграция с Processor

NLP подготавливает логические утверждения:

```python
def integrate_with_processor(nlp_result: ProcessingResult, processor):
    """Интеграция результатов NLP с Processor."""
    
    from neurograph.processor import SymbolicRule, ProcessingContext
    
    # Создание правил из отношений
    for relation in nlp_result.relations:
        if relation.confidence > 0.8:  # Только высоконадежные
            condition = f"{relation.subject.text} является {relation.object.text}"
            action = f"derive {relation.subject.text} имеет_свойства_от {relation.object.text}"
            
            rule = SymbolicRule(
                condition=condition,
                action=action,
                confidence=relation.confidence,
                metadata={"source": "nlp_extraction"}
            )
            
            processor.add_rule(rule)
    
    # Создание контекста с фактами
    context = ProcessingContext()
    for relation in nlp_result.relations:
        fact_key = f"{relation.subject.text}_{relation.predicate.value}_{relation.object.text}"
        context.add_fact(fact_key, True, relation.confidence)
    
    return context
```

**Формат данных для Processor:**
```python
# Символьное правило
{
    "condition": "Python является языком_программирования",
    "action": "derive Python используется_для программирования",
    "confidence": 0.95,
    "metadata": {"source": "nlp", "text_span": "Python - это язык..."}
}

# Факт в контексте
{
    "fact_key": "Python_is_a_programming_language",
    "value": True,
    "confidence": 0.95
}
```

---

## Примеры использования

### Базовое использование

```python
from neurograph.nlp import quick_process, quick_extract_knowledge

# Быстрая обработка текста
text = "Python создан Гвидо ван Россумом в 1991 году."
result = quick_process(text, language="ru")

print(f"Найдено сущностей: {len(result.entities)}")
print(f"Найдено отношений: {len(result.relations)}")

# Извлечение знаний
knowledge = quick_extract_knowledge(text)
for item in knowledge:
    if item['type'] == 'relation':
        print(f"{item['subject']['text']} -> {item['predicate']} -> {item['object']['text']}")
```

### Создание и настройка процессора

```python
from neurograph.nlp import NLPFactory, NLPConfiguration

# Создание процессора с настройками
processor = NLPFactory.create(
    "advanced",
    language="ru",
    use_spacy=True,
    cache_results=True
)

# Обработка с дополнительными параметрами
result = processor.process_text(
    text,
    extract_entities=True,
    extract_relations=True,
    analyze_sentiment=True
)

# Получение статистики
stats = processor.get_statistics()
print(f"Обработано текстов: {stats['processing_stats']['texts_processed']}")
```

### Интеграция с другими модулями

```python
# Полная интеграция NLP с другими модулями
from neurograph.semgraph import SemGraphFactory
from neurograph.contextvec import ContextVectorsFactory
from neurograph.memory import create_default_biomorphic_memory

# Создание компонентов
graph = SemGraphFactory.create("memory_efficient")
vectors = ContextVectorsFactory.create("dynamic", vector_size=384)
memory = create_default_biomorphic_memory()
nlp_processor = NLPFactory.create("standard", language="ru")

# Обработка текста
text = "TensorFlow - это библиотека машинного обучения от Google."
nlp_result = nlp_processor.process_text(text)

# Интеграция с графом
for entity in nlp_result.entities:
    graph.add_node(entity.text, type=entity.entity_type.value)

for relation in nlp_result.relations:
    graph.add_edge(
        relation.subject.text,
        relation.object.text,
        relation.predicate.value
    )

# Интеграция с векторами (если есть encoder)
# for entity in nlp_result.entities:
#     vector = encoder.encode(entity.text)
#     vectors.create_vector(entity.text, vector)

print(f"Добавлено в граф: {len(nlp_result.entities)} узлов, {len(nlp_result.relations)} связей")
```

### Пакетная обработка

```python
from neurograph.nlp import NLPManager, get_production_config

# Создание менеджера для пакетной обработки
config = get_production_config()
manager = NLPManager(config)

# Список текстов для обработки
texts = [
    "Python - высокоуровневый язык программирования.",
    "Django - веб-фреймворк для Python.",
    "TensorFlow используется для машинного обучения."
]

# Пакетная обработка
results = manager.batch_process_texts(texts, extract_entities=True)

# Анализ результатов
total_entities = sum(len(result.entities) for result in results if result)
total_relations = sum(len(result.relations) for result in results if result)

print(f"Обработано {len(texts)} текстов")
print(f"Найдено {total_entities} сущностей и {total_relations} отношений")

# Статистика менеджера
stats = manager.get_statistics()
print(f"Среднее время обработки: {stats['manager_stats']['avg_processing_time']:.3f}с")
```

### Создание пользовательского процессора

```python
from neurograph.nlp import register_nlp_processor, INLProcessor, ProcessingResult

@register_nlp_processor("custom")
class CustomNLProcessor(INLProcessor):
    """Пользовательский процессор с дополнительной логикой."""
    
    def __init__(self, custom_param: str = "default"):
        self.custom_param = custom_param
        self.logger = get_logger("custom_nlp")
    
    def process_text(self, text: str, **kwargs) -> ProcessingResult:
        import time
        start_time = time.time()
        
        # Пользовательская логика обработки
        tokens = self._custom_tokenize(text)
        entities = self._custom_extract_entities(text)
        relations = self._custom_extract_relations(text, entities)
        
        return ProcessingResult(
            original_text=text,
            sentences=[],
            entities=entities,
            relations=relations,
            tokens=tokens,
            processing_time=time.time() - start_time,
            metadata={"processor": "custom", "param": self.custom_param}
        )
    
    def extract_knowledge(self, text: str) -> list:
        # Пользовательская логика извлечения знаний
        return [{"type": "custom", "text": text, "custom_param": self.custom_param}]
    
    def normalize_text(self, text: str) -> str:
        return text.strip().lower()
    
    def get_language(self, text: str) -> str:
        return "unknown"
    
    def _custom_tokenize(self, text: str):
        # Пользовательская токенизация
        return []
    
    def _custom_extract_entities(self, text: str):
        # Пользовательское извлечение сущностей
        return []
    
    def _custom_extract_relations(self, text: str, entities):
        # Пользовательское извлечение отношений
        return []

# Использование пользовательского процессора
custom_processor = NLPFactory.create("custom", custom_param="test_value")
result = custom_processor.process_text("Тестовый текст")
```

---

## События и уведомления

NLP модуль интегрируется с системой событий NeuroGraph для уведомления других модулей:

### Публикуемые события

```python
from neurograph.core.events import publish

# События обработки текста
publish("nlp.text_processed", {
    "text_length": len(text),
    "entities_count": len(entities),
    "relations_count": len(relations),
    "processing_time": processing_time,
    "language": detected_language
})

# События извлечения знаний
publish("nlp.knowledge_extracted", {
    "knowledge_items": len(knowledge_items),
    "source_text": text[:100],  # Первые 100 символов
    "extraction_method": "hybrid"
})

# События ошибок
publish("nlp.processing_error", {
    "error_type": type(error).__name__,
    "error_message": str(error),
    "text_length": len(text),
    "processor_type": processor_type
})
```

### Подписка на события

```python
from neurograph.core.events import subscribe

def handle_nlp_results(data):
    """Обработчик результатов NLP для других модулей."""
    
    if data["entities_count"] > 5:
        # Много сущностей - важный текст
        priority_processing(data)
    
    if data["language"] != "ru":
        # Неожиданный язык
        language_analysis(data)

def handle_knowledge_extraction(data):
    """Обработчик извлеченных знаний."""
    
    # Автоматическое добавление в граф знаний
    auto_add_to_graph(data["knowledge_items"])

# Подписка на события
subscribe("nlp.text_processed", handle_nlp_results)
subscribe("nlp.knowledge_extracted", handle_knowledge_extraction)
```

### Интеграционные события

Для интеграции с другими модулями NLP может подписываться на их события:

```python
# Подписка на события памяти
def handle_memory_consolidation(data):
    """Реакция на консолидацию памяти."""
    if data["consolidated_count"] > 10:
        # Много элементов консолидировано - возможно нужен анализ паттернов
        analyze_consolidation_patterns(data)

def handle_graph_update(data):
    """Реакция на обновления графа знаний."""
    if data["operation"] == "node_added":
        # Новый узел - возможно нужна дополнительная обработка
        enrich_node_with_nlp(data["node_id"])

# Подписки на события других модулей
subscribe("memory.consolidation_completed", handle_memory_consolidation)
subscribe("graph.node_added", handle_graph_update)
subscribe("processor.rule_executed", handle_rule_execution)
```

---

## Производительность и масштабирование

### Метрики производительности

NLP модуль отслеживает ключевые метрики для мониторинга и оптимизации:

```python
def get_performance_metrics(processor):
    """Получение метрик производительности NLP."""
    
    stats = processor.get_statistics()
    
    return {
        "throughput": {
            "texts_per_second": stats["texts_processed"] / stats["total_uptime"],
            "tokens_per_second": stats["tokens_processed"] / stats["total_uptime"],
            "entities_per_text": stats["entities_extracted"] / max(1, stats["texts_processed"])
        },
        "latency": {
            "avg_processing_time": stats["avg_processing_time"],
            "p95_processing_time": stats.get("p95_processing_time", 0),
            "p99_processing_time": stats.get("p99_processing_time", 0)
        },
        "accuracy": {
            "entity_confidence_avg": stats.get("avg_entity_confidence", 0),
            "relation_confidence_avg": stats.get("avg_relation_confidence", 0),
            "language_detection_accuracy": stats.get("language_accuracy", 0)
        },
        "resource_usage": {
            "memory_usage_mb": stats.get("memory_usage", 0),
            "cpu_usage_percent": stats.get("cpu_usage", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0)
        }
    }
```

### Стратегии кеширования

```python
from neurograph.core.cache import cached

class OptimizedNLProcessor:
    """Оптимизированный процессор с кешированием."""
    
    @cached(ttl=300, key_func=lambda text: hash(text.lower().strip()))
    def process_text_cached(self, text: str, **kwargs):
        """Кешированная обработка текста."""
        return self._process_text_internal(text, **kwargs)
    
    @cached(ttl=600)
    def extract_entities_cached(self, text: str):
        """Кешированное извлечение сущностей."""
        return self.entity_extractor.extract_entities(text)
    
    def invalidate_cache_for_text(self, text: str):
        """Инвалидация кеша для конкретного текста."""
        cache_key = hash(text.lower().strip())
        # Логика инвалидации кеша
```

### Параллельная обработка

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

class ParallelNLPManager:
    """Менеджер для параллельной обработки NLP."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.processor_pool = []
    
    def process_texts_parallel(self, texts: List[str], **kwargs):
        """Параллельная обработка списка текстов."""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Создаем задачи
            futures = [
                executor.submit(self._process_single_text, text, **kwargs)
                for text in texts
            ]
            
            # Собираем результаты
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 сек таймаут
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
            
            return results
    
    async def process_texts_async(self, texts: List[str], **kwargs):
        """Асинхронная обработка текстов."""
        
        tasks = [
            asyncio.create_task(self._process_text_async(text, **kwargs))
            for text in texts
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def process_large_document(self, document: str, chunk_size: int = 1000):
        """Обработка больших документов по частям."""
        
        # Разбиваем документ на части
        chunks = [
            document[i:i+chunk_size] 
            for i in range(0, len(document), chunk_size)
        ]
        
        # Обрабатываем части параллельно
        chunk_results = self.process_texts_parallel(chunks)
        
        # Объединяем результаты
        return self._merge_chunk_results(chunk_results)
```

### Оптимизация памяти

```python
class MemoryOptimizedProcessor:
    """Процессор, оптимизированный по памяти."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self._result_cache = {}
        self._cache_access_order = []
    
    def process_with_memory_limit(self, text: str, **kwargs):
        """Обработка с контролем памяти."""
        
        # Проверяем кеш
        cache_key = self._get_cache_key(text, kwargs)
        if cache_key in self._result_cache:
            self._update_cache_access(cache_key)
            return self._result_cache[cache_key]
        
        # Обрабатываем
        result = self._process_text_internal(text, **kwargs)
        
        # Добавляем в кеш с учетом лимита
        self._add_to_cache(cache_key, result)
        
        return result
    
    def _add_to_cache(self, key: str, result):
        """Добавление в кеш с вытеснением старых элементов."""
        
        if len(self._result_cache) >= self.max_cache_size:
            # Удаляем самый старый элемент
            oldest_key = self._cache_access_order.pop(0)
            del self._result_cache[oldest_key]
        
        self._result_cache[key] = result
        self._cache_access_order.append(key)
    
    def clear_cache(self):
        """Очистка кеша для освобождения памяти."""
        self._result_cache.clear()
        self._cache_access_order.clear()
```

---

## Расширение модуля

### Добавление новых типов сущностей

```python
from neurograph.nlp.base import EntityType
from enum import Enum

# Расширение перечисления типов сущностей
class ExtendedEntityType(EntityType):
    PRODUCT = "PRODUCT"              # Товары/продукты
    EVENT = "EVENT"                  # События
    TECHNOLOGY = "TECHNOLOGY"        # Технологии
    METRIC = "METRIC"                # Метрики/показатели
    PROCESS = "PROCESS"              # Процессы

# Пользовательский экстрактор с новыми типами
class ExtendedEntityExtractor(IEntityExtractor):
    
    def __init__(self):
        self.product_patterns = [
            r'\b(iPhone|Android|Windows|Linux)\b',
            r'\b[А-ЯЁ][а-яё]+\s+(версии|v\.)\s+\d+\.\d+\b'
        ]
        
        self.technology_patterns = [
            r'\b(blockchain|машинное\s+обучение|нейронные\s+сети)\b',
            r'\b[А-ЯЁ][а-яё]+\s+(технология|алгоритм|протокол)\b'
        ]
    
    def extract_entities(self, text: str) -> List[Entity]:
        entities = []
        
        # Поиск продуктов
        for pattern in self.product_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    entity_type=ExtendedEntityType.PRODUCT,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Поиск технологий
        for pattern in self.technology_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    entity_type=ExtendedEntityType.TECHNOLOGY,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))
        
        return entities
```

### Добавление новых типов отношений

```python
from neurograph.nlp.base import RelationType

class ExtendedRelationType(RelationType):
    COMPETES_WITH = "competes_with"      # Конкурирует с
    DEPENDS_ON = "depends_on"            # Зависит от
    REPLACES = "replaces"                # Заменяет
    COMPATIBLE_WITH = "compatible_with"   # Совместим с
    REQUIRES = "requires"                 # Требует

# Пользовательский экстрактор отношений
class ExtendedRelationExtractor(IRelationExtractor):
    
    def __init__(self):
        self.competition_patterns = [
            (r'(.+?)\s+конкурирует\s+с\s+(.+)', ExtendedRelationType.COMPETES_WITH),
            (r'(.+?)\s+против\s+(.+)', ExtendedRelationType.COMPETES_WITH)
        ]
        
        self.dependency_patterns = [
            (r'(.+?)\s+зависит\s+от\s+(.+)', ExtendedRelationType.DEPENDS_ON),
            (r'(.+?)\s+требует\s+(.+)', ExtendedRelationType.REQUIRES)
        ]
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        relations = []
        
        all_patterns = self.competition_patterns + self.dependency_patterns
        
        for pattern, relation_type in all_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if len(match.groups()) >= 2:
                    subject_text = match.group(1).strip()
                    object_text = match.group(2).strip()
                    
                    # Создаем временные сущности
                    subject = Entity(
                        text=subject_text,
                        entity_type=EntityType.CONCEPT,
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.7
                    )
                    
                    obj = Entity(
                        text=object_text,
                        entity_type=EntityType.CONCEPT,
                        start=match.start(2),
                        end=match.end(2),
                        confidence=0.7
                    )
                    
                    relations.append(Relation(
                        subject=subject,
                        predicate=relation_type,
                        object=obj,
                        confidence=0.8,
                        text_span=match.group()
                    ))
        
        return relations
```

### Создание специализированных процессоров

```python
from neurograph.nlp import register_nlp_processor

@register_nlp_processor("domain_specific")
class DomainSpecificProcessor(INLProcessor):
    """Процессор для специфической предметной области."""
    
    def __init__(self, domain: str = "general", **kwargs):
        self.domain = domain
        self.logger = get_logger(f"domain_nlp_{domain}")
        
        # Загружаем специфические для домена компоненты
        self.domain_tokenizer = self._load_domain_tokenizer()
        self.domain_entities = self._load_domain_entities()
        self.domain_relations = self._load_domain_relations()
        
        # Специфические для домена словари
        self.domain_vocabulary = self._load_domain_vocabulary()
        self.domain_patterns = self._load_domain_patterns()
    
    def process_text(self, text: str, **kwargs) -> ProcessingResult:
        """Обработка с учетом специфики домена."""
        
        # Предобработка для домена
        normalized_text = self._domain_normalize(text)
        
        # Токенизация с учетом домена
        tokens = self.domain_tokenizer.tokenize(normalized_text)
        
        # Извлечение сущностей для домена
        entities = self.domain_entities.extract_entities(normalized_text)
        
        # Извлечение отношений для домена
        relations = self.domain_relations.extract_relations(normalized_text, entities)
        
        # Дополнительная обработка для домена
        enhanced_entities = self._enhance_entities_for_domain(entities)
        enhanced_relations = self._enhance_relations_for_domain(relations)
        
        return ProcessingResult(
            original_text=text,
            sentences=self._create_sentences(normalized_text, tokens),
            entities=enhanced_entities,
            relations=enhanced_relations,
            tokens=tokens,
            processing_time=time.time() - start_time,
            language=self.get_language(text),
            metadata={
                "domain": self.domain,
                "processor": "domain_specific"
            }
        )
    
    def _load_domain_tokenizer(self):
        """Загрузка токенизатора для домена."""
        # Логика загрузки специфического токенизатора
        pass
    
    def _domain_normalize(self, text: str) -> str:
        """Нормализация текста для домена."""
        # Специфическая для домена нормализация
        if self.domain == "medical":
            return self._medical_normalize(text)
        elif self.domain == "legal":
            return self._legal_normalize(text)
        else:
            return text
```

### Интеграция с внешними API

```python
class ExternalAPIProcessor(INLProcessor):
    """Процессор с интеграцией внешних API."""
    
    def __init__(self, api_config: Dict[str, str]):
        self.api_config = api_config
        self.fallback_processor = create_lightweight_processor()
    
    async def process_with_external_api(self, text: str) -> ProcessingResult:
        """Обработка с использованием внешнего API."""
        
        try:
            # Попытка использования внешнего API
            external_result = await self._call_external_api(text)
            return self._convert_external_result(external_result)
            
        except Exception as e:
            self.logger.warning(f"Внешний API недоступен: {e}, используем fallback")
            return self.fallback_processor.process_text(text)
    
    async def _call_external_api(self, text: str):
        """Вызов внешнего API."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_config["endpoint"],
                json={"text": text},
                headers={"Authorization": f"Bearer {self.api_config['api_key']}"}
            ) as response:
                return await response.json()
    
    def _convert_external_result(self, external_result) -> ProcessingResult:
        """Конвертация результата внешнего API в наш формат."""
        # Логика конвертации
        pass
```

---

## Обработка ошибок

### Типы ошибок модуля

```python
from neurograph.core.errors import NeuroGraphError

class NLPError(NeuroGraphError):
    """Базовая ошибка NLP модуля."""
    pass

class TokenizationError(NLPError):
    """Ошибка токенизации."""
    pass

class EntityExtractionError(NLPError):
    """Ошибка извлечения сущностей."""
    pass

class RelationExtractionError(NLPError):
    """Ошибка извлечения отношений."""
    pass

class TextGenerationError(NLPError):
    """Ошибка генерации текста."""
    pass

class ConfigurationError(NLPError):
    """Ошибка конфигурации NLP."""
    pass

class ModelLoadError(NLPError):
    """Ошибка загрузки модели."""
    pass
```

### Стратегии обработки ошибок

```python
class RobustNLProcessor:
    """Процессор с надежной обработкой ошибок."""
    
    def __init__(self):
        self.fallback_processors = [
            ("spacy", self._create_spacy_processor),
            ("rules", self._create_rule_processor),
            ("simple", self._create_simple_processor)
        ]
        self.current_processor = None
        self.error_counts = {}
    
    def process_text(self, text: str, **kwargs) -> ProcessingResult:
        """Обработка с fallback стратегией."""
        
        for processor_name, processor_factory in self.fallback_processors:
            try:
                if self.current_processor is None:
                    self.current_processor = processor_factory()
                
                result = self.current_processor.process_text(text, **kwargs)
                
                # Успешная обработка - сбрасываем счетчик ошибок
                self.error_counts[processor_name] = 0
                return result
                
            except Exception as e:
                self.error_counts[processor_name] = self.error_counts.get(processor_name, 0) + 1
                
                self.logger.warning(
                    f"Ошибка в процессоре {processor_name}: {e}. "
                    f"Ошибок подряд: {self.error_counts[processor_name]}"
                )
                
                # Если слишком много ошибок - переключаемся на следующий
                if self.error_counts[processor_name] >= 3:
                    self.current_processor = None
                    continue
                
                # Для единичных ошибок - пробуем еще раз
                if self.error_counts[processor_name] == 1:
                    try:
                        return self.current_processor.process_text(text, **kwargs)
                    except:
                        self.current_processor = None
                        continue
        
        # Если все процессоры не работают - возвращаем минимальный результат
        return self._create_minimal_result(text)
    
    def _create_minimal_result(self, text: str) -> ProcessingResult:
        """Создание минимального результата при критических ошибках."""
        return ProcessingResult(
            original_text=text,
            sentences=[],
            entities=[],
            relations=[],
            tokens=[],
            processing_time=0.0,
            metadata={"error": "all_processors_failed", "status": "minimal"}
        )
```

### Валидация входных данных

```python
def validate_text_input(text: str) -> None:
    """Валидация входного текста."""
    
    if text is None:
        raise ValueError("Текст не может быть None")
    
    if not isinstance(text, str):
        raise TypeError(f"Ожидается строка, получен {type(text)}")
    
    if len(text) == 0:
        raise ValueError("Текст не может быть пустым")
    
    if len(text) > 100000:  # 100KB лимит
        raise ValueError(f"Текст слишком длинный: {len(text)} символов (макс: 100000)")
    
    # Проверка на подозрительные символы
    suspicious_chars = len([c for c in text if ord(c) > 65535])
    if suspicious_chars > len(text) * 0.1:
        raise ValueError("Текст содержит слишком много необычных символов")

def validate_processing_params(**kwargs) -> None:
    """Валидация параметров обработки."""
    
    extract_entities = kwargs.get("extract_entities", True)
    extract_relations = kwargs.get("extract_relations", True)
    analyze_sentiment = kwargs.get("analyze_sentiment", False)
    
    if not isinstance(extract_entities, bool):
        raise TypeError("extract_entities должен быть bool")
    
    if not isinstance(extract_relations, bool):
        raise TypeError("extract_relations должен быть bool")
    
    if not isinstance(analyze_sentiment, bool):
        raise TypeError("analyze_sentiment должен быть bool")
    
    # Логическая валидация
    if extract_relations and not extract_entities:
        raise ValueError("Нельзя извлекать отношения без извлечения сущностей")
```

### Мониторинг ошибок

```python
class ErrorMonitor:
    """Мониторинг ошибок NLP модуля."""
    
    def __init__(self, alert_threshold: int = 10):
        self.alert_threshold = alert_threshold
        self.error_log = []
        self.error_stats = {}
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Логирование ошибки с контекстом."""
        
        error_entry = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        self.error_log.append(error_entry)
        
        # Обновляем статистику
        error_type = error_entry["error_type"]
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1
        
        # Проверяем пороги для алертов
        if self.error_stats[error_type] >= self.alert_threshold:
            self._send_alert(error_type, self.error_stats[error_type])
    
    def _send_alert(self, error_type: str, count: int):
        """Отправка алерта о критическом количестве ошибок."""
        
        from neurograph.core.events import publish
        
        publish("nlp.error_threshold_exceeded", {
            "error_type": error_type,
            "error_count": count,
            "threshold": self.alert_threshold,
            "recent_errors": self.get_recent_errors(error_type, limit=5)
        })
    
    def get_recent_errors(self, error_type: str = None, limit: int = 10) -> List[Dict]:
        """Получение списка недавних ошибок."""
        
        filtered_errors = self.error_log
        
        if error_type:
            filtered_errors = [
                error for error in self.error_log 
                if error["error_type"] == error_type
            ]
        
        return sorted(filtered_errors, key=lambda x: x["timestamp"], reverse=True)[:limit]
```

---

## Конфигурация и настройка

### Структура конфигурации

```json
{
  "processor": {
    "type": "standard|lightweight|advanced|custom",
    "language": "ru|en|mixed",
    "use_spacy": true,
    "cache_results": true,
    "tokenizer_type": "simple|spacy|subword"
  },
  "tokenization": {
    "language": "ru",
    "model_name": "ru_core_news_sm",
    "subword_vocab_size": 10000,
    "min_token_length": 2
  },
  "entity_extraction": {
    "use_spacy": true,
    "use_rules": true,
    "confidence_threshold": 0.5,
    "max_entities_per_text": 100,
    "custom_patterns": true
  },
  "relation_extraction": {
    "use_spacy": true,
    "use_patterns": true,
    "use_rules": true,
    "confidence_threshold": 0.5,
    "max_relations_per_text": 50
  },
  "text_generation": {
    "use_templates": true,
    "use_markov": false,
    "use_rules": true,
    "default_style": "formal|casual|scientific",
    "max_length": 500
  },
  "semantic_processing": {
    "enabled": true,
    "vector_provider": "sentence_transformers",
    "similarity_threshold": 0.7,
    "max_keywords": 20
  },
  "advanced_features": {
    "custom_patterns": true,
    "sentiment_analysis": false,
    "language_detection": true,
    "topic_modeling": false
  },
  "performance": {
    "cache_enabled": true,
    "cache_ttl": 300,
    "parallel_processing": false,
    "batch_size": 10,
    "max_workers": 4,
    "timeout_seconds": 30
  },
  "error_handling": {
    "fallback_enabled": true,
    "retry_attempts": 3,
    "error_threshold": 10,
    "log_errors": true
  },
  "integration": {
    "semgraph": {
      "auto_add_entities": true,
      "auto_add_relations": true,
      "confidence_threshold": 0.7
    },
    "memory": {
      "auto_add_facts": true,
      "fact_confidence_threshold": 0.8
    },
    "contextvec": {
      "auto_vectorize": true,
      "vector_entities": true,
      "vector_relations": true
    }
  }
}
```

### Примеры конфигураций для разных сценариев

#### Конфигурация для разработки

```json
{
  "processor": {
    "type": "lightweight",
    "language": "ru",
    "use_spacy": false,
    "cache_results": false
  },
  "entity_extraction": {
    "use_spacy": false,
    "use_rules": true,
    "confidence_threshold": 0.3
  },
  "performance": {
    "cache_enabled": false,
    "parallel_processing": false,
    "timeout_seconds": 10
  },
  "error_handling": {
    "fallback_enabled": true,
    "retry_attempts": 1,
    "log_errors": true
  }
}
```

#### Конфигурация для продакшена

```json
{
  "processor": {
    "type": "advanced",
    "language": "ru",
    "use_spacy": true,
    "cache_results": true
  },
  "entity_extraction": {
    "use_spacy": true,
    "use_rules": true,
    "confidence_threshold": 0.7
  },
  "performance": {
    "cache_enabled": true,
    "cache_ttl": 600,
    "parallel_processing": true,
    "batch_size": 20,
    "max_workers": 8
  },
  "error_handling": {
    "fallback_enabled": true,
    "retry_attempts": 3,
    "error_threshold": 20
  }
}
```

#### Конфигурация для ограниченных ресурсов

```json
{
  "processor": {
    "type": "lightweight",
    "language": "ru",
    "use_spacy": false,
    "cache_results": false
  },
  "entity_extraction": {
    "use_spacy": false,
    "use_rules": true,
    "confidence_threshold": 0.5,
    "max_entities_per_text": 20
  },
  "relation_extraction": {
    "use_spacy": false,
    "use_patterns": false,
    "use_rules": true,
    "max_relations_per_text": 10
  },
  "performance": {
    "cache_enabled": false,
    "parallel_processing": false,
    "batch_size": 1,
    "timeout_seconds": 5
  }
}
```

### Управление конфигурацией в коде

```python
from neurograph.nlp import NLPConfiguration

# Создание и настройка конфигурации
config = NLPConfiguration()

# Программное изменение настроек
config.update_config({
    "processor": {
        "language": "en",  # Переключаем на английский
        "use_spacy": True
    },
    "performance": {
        "parallel_processing": True,
        "max_workers": 6
    }
})

# Создание процессора с обновленной конфигурацией
processor = config.create_processor()

# Экспорт конфигурации
config.save_to_file("custom_nlp_config.json")

# Создание конфигурации для специфических задач
def create_medical_config():
    """Конфигурация для медицинских текстов."""
    config = NLPConfiguration()
    config.update_config({
        "processor": {"type": "advanced"},
        "entity_extraction": {
            "confidence_threshold": 0.9,  # Высокая точность для медицины
            "custom_patterns": True
        },
        "semantic_processing": {
            "vector_provider": "biobert",  # Специализированная модель
            "similarity_threshold": 0.8
        }
    })
    return config

def create_social_media_config():
    """Конфигурация для социальных сетей."""
    config = NLPConfiguration()
    config.update_config({
        "processor": {"type": "lightweight"},
        "text_generation": {
            "default_style": "casual",
            "max_length": 280  # Лимит Twitter
        },
        "advanced_features": {
            "sentiment_analysis": True,  # Важно для соцсетей
            "language_detection": True   # Смешанные языки
        },
        "performance": {
            "parallel_processing": True,
            "batch_size": 50  # Много коротких текстов
        }
    })
    return config
```

---

## Заключение

### Ключевые точки интеграции для разработчиков других модулей

1. **Входные данные для NLP:**
   - Необработанный текст (строки)
   - Документы любого размера
   - Запросы пользователей
   - Метаданные контекста

2. **Выходные данные из NLP:**
   - `ProcessingResult` с полной структурированной информацией
   - Списки `Entity` для добавления в граф знаний
   - Списки `Relation` для создания связей
   - Структурированные знания для памяти

3. **События для подписки:**
   - `nlp.text_processed` - завершение обработки текста
   - `nlp.knowledge_extracted` - извлечение знаний
   - `nlp.processing_error` - ошибки обработки

4. **Конфигурационные параметры:**
   - Типы процессоров и их настройки
   - Пороги уверенности для фильтрации
   - Параметры производительности

### Рекомендации по использованию

1. **Для модуля SemGraph:**
   - Используйте `nlp_result.entities` для создания узлов
   - Используйте `nlp_result.relations` для создания ребер
   - Фильтруйте по `confidence` для качества данных

2. **Для модуля ContextVec:**
   - Векторизуйте тексты сущностей и отношений
   - Используйте предложения с высокой плотностью сущностей
   - Кешируйте векторы для производительности

3. **Для модуля Memory:**
   - Добавляйте факты из отношений как `MemoryItem`
   - Используйте метаданные для контекста
   - Учитывайте уверенность для приоритизации

4. **Для модуля Processor:**
   - Конвертируйте отношения в логические правила
   - Используйте высоконадежные факты для вывода
   - Создавайте контексты с извлеченными фактами

### Примеры интеграционного кода

```python
# Полная интеграция NLP с другими модулями
class NeuroGraphNLPIntegrator:
    """Интегратор NLP с другими модулями системы."""
    
    def __init__(self, graph, vectors, memory, processor, encoder):
        self.graph = graph
        self.vectors = vectors
        self.memory = memory
        self.processor = processor
        self.encoder = encoder
        self.nlp = create_default_processor()
    
    def process_and_integrate(self, text: str) -> Dict[str, Any]:
        """Полная обработка и интеграция с всеми модулями."""
        
        # 1. NLP обработка
        nlp_result = self.nlp.process_text(text)
        
        # 2. Интеграция с SemGraph
        graph_stats = self._integrate_with_graph(nlp_result)
        
        # 3. Интеграция с ContextVec
        vector_stats = self._integrate_with_vectors(nlp_result)
        
        # 4. Интеграция с Memory
        memory_stats = self._integrate_with_memory(nlp_result)
        
        # 5. Интеграция с Processor
        processor_stats = self._integrate_with_processor(nlp_result)
        
        return {
            "nlp_result": nlp_result,
            "integration_stats": {
                "graph": graph_stats,
                "vectors": vector_stats,
                "memory": memory_stats,
                "processor": processor_stats
            }
        }
    
    def _integrate_with_graph(self, nlp_result: ProcessingResult) -> Dict:
        """Интеграция с графом знаний."""
        nodes_added = 0
        edges_added = 0
        
        # Добавляем сущности как узлы
        for entity in nlp_result.entities:
            if entity.confidence > 0.7:  # Фильтр по уверенности
                self.graph.add_node(
                    entity.text,
                    type=entity.entity_type.value,
                    confidence=entity.confidence,
                    source="nlp_extraction"
                )
                nodes_added += 1
        
        # Добавляем отношения как ребра
        for relation in nlp_result.relations:
            if relation.confidence > 0.6:
                self.graph.add_edge(
                    relation.subject.text,
                    relation.object.text,
                    relation.predicate.value,
                    weight=relation.confidence
                )
                edges_added += 1
        
        return {"nodes_added": nodes_added, "edges_added": edges_added}
    
    def _integrate_with_vectors(self, nlp_result: ProcessingResult) -> Dict:
        """Интеграция с векторными представлениями."""
        vectors_created = 0
        
        # Векторизуем сущности
        for entity in nlp_result.entities:
            if entity.entity_type in [EntityType.CONCEPT, EntityType.FACT]:
                vector = self.encoder.encode(entity.text)
                self.vectors.create_vector(entity.text, vector)
                vectors_created += 1
        
        # Векторизуем важные предложения
        for sentence in nlp_result.sentences:
            if len(sentence.entities) >= 2:  # Предложения с несколькими сущностями
                vector = self.encoder.encode(sentence.text)
                sentence_id = f"sent_{hash(sentence.text)}"
                self.vectors.create_vector(sentence_id, vector)
                vectors_created += 1
        
        return {"vectors_created": vectors_created}
    
    def _integrate_with_memory(self, nlp_result: ProcessingResult) -> Dict:
        """Интеграция с памятью."""
        items_added = 0
        
        # Добавляем факты из отношений
        for relation in nlp_result.relations:
            if relation.confidence > 0.8:
                fact_text = f"{relation.subject.text} {relation.predicate.value} {relation.object.text}"
                embedding = self.encoder.encode(fact_text)
                
                memory_item = MemoryItem(
                    content=fact_text,
                    embedding=embedding,
                    content_type="fact",
                    metadata={
                        "confidence": relation.confidence,
                        "source": "nlp_relation"
                    }
                )
                
                self.memory.add(memory_item)
                items_added += 1
        
        return {"items_added": items_added}
    
    def _integrate_with_processor(self, nlp_result: ProcessingResult) -> Dict:
        """Интеграция с процессором логического вывода."""
        rules_added = 0
        
        # Создаем правила из высоконадежных отношений
        for relation in nlp_result.relations:
            if relation.confidence > 0.9 and relation.predicate == RelationType.IS_A:
                from neurograph.processor.base import SymbolicRule
                
                condition = f"{relation.subject.text} exists"
                action = f"derive {relation.subject.text} has_type {relation.object.text}"
                
                rule = SymbolicRule(
                    condition=condition,
                    action=action,
                    confidence=relation.confidence
                )
                
                self.processor.add_rule(rule)
                rules_added += 1
        
        return {"rules_added": rules_added}

# Использование интегратора
integrator = NeuroGraphNLPIntegrator(graph, vectors, memory, processor, encoder)

text = "TensorFlow - это библиотека машинного обучения, созданная Google."
result = integrator.process_and_integrate(text)

print("Результат интеграции:")
for module, stats in result["integration_stats"].items():
    print(f"  {module}: {stats}")
```

### Тестирование интеграции

```python
import unittest
from neurograph.nlp import quick_process
from neurograph.semgraph import SemGraphFactory

class TestNLPIntegration(unittest.TestCase):
    """Тесты интеграции NLP с другими модулями."""
    
    def setUp(self):
        self.graph = SemGraphFactory.create("memory_efficient")
        self.test_text = "Python создан Гвидо ван Россумом в 1991 году."
    
    def test_nlp_to_semgraph_integration(self):
        """Тест интеграции NLP -> SemGraph."""
        
        # Обрабатываем текст
        nlp_result = quick_process(self.test_text)
        
        # Добавляем в граф
        for entity in nlp_result.entities:
            self.graph.add_node(entity.text, type=entity.entity_type.value)
        
        for relation in nlp_result.relations:
            self.graph.add_edge(
                relation.subject.text,
                relation.object.text,
                relation.predicate.value
            )
        
        # Проверяем результат
        nodes = self.graph.get_all_nodes()
        edges = self.graph.get_all_edges()
        
        self.assertGreater(len(nodes), 0, "Должны быть добавлены узлы")
        self.assertGreater(len(edges), 0, "Должны быть добавлены связи")
        
        # Проверяем конкретные ожидаемые сущности
        self.assertIn("Python", nodes)
        self.assertIn("Гвидо ван Россум", nodes)
    
    def test_data_format_compatibility(self):
        """Тест совместимости форматов данных."""
        
        nlp_result = quick_process(self.test_text)
        
        # Проверяем, что данные имеют ожидаемый формат
        self.assertIsInstance(nlp_result.entities, list)
        self.assertIsInstance(nlp_result.relations, list)
        
        for entity in nlp_result.entities:
            self.assertHasattr(entity, 'text')
            self.assertHasattr(entity, 'entity_type')
            self.assertHasattr(entity, 'confidence')
        
        for relation in nlp_result.relations:
            self.assertHasattr(relation, 'subject')
            self.assertHasattr(relation, 'predicate')
            self.assertHasattr(relation, 'object')
            self.assertHasattr(relation, 'confidence')

if __name__ == "__main__":
    unittest.main()
```

---

### Метрики интеграции

Для мониторинга качества интеграции рекомендуется отслеживать следующие метрики:

```python
class IntegrationMetrics:
    """Метрики интеграции NLP с другими модулями."""
    
    def __init__(self):
        self.metrics = {
            "nlp_processing": {
                "texts_processed": 0,
                "avg_entities_per_text": 0,
                "avg_relations_per_text": 0,
                "avg_confidence": 0
            },
            "graph_integration": {
                "nodes_added": 0,
                "edges_added": 0,
                "duplicate_nodes_skipped": 0,
                "low_confidence_filtered": 0
            },
            "memory_integration": {
                "facts_added": 0,
                "concepts_added": 0,
                "avg_importance": 0
            },
            "vector_integration": {
                "vectors_created": 0,
                "avg_vector_similarity": 0
            }
        }
    
    def update_nlp_metrics(self, nlp_result: ProcessingResult):
        """Обновление метрик NLP обработки."""
        self.metrics["nlp_processing"]["texts_processed"] += 1
        # ... обновление других метрик
    
    def get_integration_health(self) -> Dict[str, str]:
        """Оценка здоровья интеграции."""
        health = {}
        
        # Проверяем качество извлечения
        avg_entities = self.metrics["nlp_processing"]["avg_entities_per_text"]
        if avg_entities > 3:
            health["entity_extraction"] = "good"
        elif avg_entities > 1:
            health["entity_extraction"] = "fair"
        else:
            health["entity_extraction"] = "poor"
        
        # Проверяем эффективность интеграции с графом
        nodes_added = self.metrics["graph_integration"]["nodes_added"]
        texts_processed = self.metrics["nlp_processing"]["texts_processed"]
        
        if texts_processed > 0:
            integration_ratio = nodes_added / texts_processed
            if integration_ratio > 2:
                health["graph_integration"] = "excellent"
            elif integration_ratio > 1:
                health["graph_integration"] = "good"
            else:
                health["graph_integration"] = "needs_improvement"
        
        return health
```

Эта документация предоставляет разработчикам других модулей всю необходимую информацию для интеграции с модулем NLP, включая интерфейсы, форматы данных, примеры кода и рекомендации по использованию.