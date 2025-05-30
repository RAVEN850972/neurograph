# NeuroGraph Processor Module - Complete Documentation

Полная документация модуля нейросимволического процессора системы NeuroGraph для разработчиков других модулей.

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

### Что делает модуль Processor

Модуль Processor реализует нейросимволический процессор для логического вывода и рассуждений в системе NeuroGraph. Он отвечает за:

- **Символические правила**: Создание и управление правилами вида "если-то"
- **Логический вывод**: Прямой и обратный вывод с контролем глубины
- **Сопоставление шаблонов**: Поиск и применение релевантных правил
- **Объяснения**: Генерация понятных объяснений для выводов
- **Интеграция с графом**: Использование структуры графа знаний для рассуждений

### Место в архитектуре NeuroGraph

```
┌─────────────────────────────────────────────────────────┐
│                    Integration Layer                    │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     NLP     │◄──►│  Processor  │◄──►│ Propagation │
│             │    │   (Этот)    │    │             │
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
- Факты и знания из других модулей
- Символические правила
- Контексты для вывода
- Запросы на рассуждение

**Исходящие данные:**
- Новые выведенные факты
- Объяснения логических цепочек
- Результаты применения правил
- Метрики уверенности

---

## Архитектура модуля

### Структура файлов

```
neurograph/processor/
├── __init__.py                     # Публичный API модуля
├── base.py                         # Базовые интерфейсы и классы данных
├── factory.py                      # Фабрика и конфигурация
├── utils.py                        # Утилиты и вспомогательные классы
├── impl/
│   ├── __init__.py
│   ├── pattern_matching.py         # Процессор сопоставления шаблонов
│   └── graph_based.py              # Процессор на основе графа
├── examples/
│   └── basic_usage.py              # Примеры использования
└── tests/                          # Тесты модуля
```

### Компонентная архитектура

```
┌─────────────────────────────────────────────────┐
│             Processor Interface                 │
│            (INeuroSymbolicProcessor)            │
└─────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────┐
│              Core Components                    │
├─────────────┬─────────────┬─────────────────────┤
│ Symbolic    │ Pattern     │ Derivation          │
│ Rules       │ Matcher     │ Engine              │
├─────────────┼─────────────┼─────────────────────┤
│ Processing  │ Explanation │ Statistics          │
│ Context     │ Generator   │ Tracker             │
└─────────────┴─────────────┴─────────────────────┘
                        │
┌─────────────────────────────────────────────────┐
│              Implementation Layer               │
├─────────────┬─────────────┬─────────────────────┤
│ Pattern     │ Graph       │ Hybrid              │
│ Matching    │ Based       │ Combined            │
└─────────────┴─────────────┴─────────────────────┘
```

---

## Публичные интерфейсы

### Основной интерфейс: INeuroSymbolicProcessor

Это главный интерфейс для всех процессоров в системе:

```python
from neurograph.processor import INeuroSymbolicProcessor, ProcessingContext, DerivationResult

class INeuroSymbolicProcessor(ABC):
    @abstractmethod
    def add_rule(self, rule: SymbolicRule) -> str:
        """Добавляет правило в базу знаний и возвращает его ID."""
        pass
    
    @abstractmethod
    def remove_rule(self, rule_id: str) -> bool:
        """Удаляет правило из базы знаний."""
        pass
    
    @abstractmethod
    def execute_rule(self, rule_id: str, context: ProcessingContext) -> DerivationResult:
        """Выполняет указанное правило в заданном контексте."""
        pass
    
    @abstractmethod
    def derive(self, context: ProcessingContext, depth: int = 1) -> DerivationResult:
        """Производит логический вывод на основе правил и контекста."""
        pass
    
    @abstractmethod
    def find_relevant_rules(self, context: ProcessingContext) -> List[str]:
        """Находит правила, релевантные заданному контексту."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику процессора."""
        pass
```

### Дополнительные интерфейсы

```python
# Управление правилами
class IRuleManager(ABC):
    def validate_rule(self, rule: SymbolicRule) -> Tuple[bool, Optional[str]]
    def update_rule(self, rule_id: str, **attributes) -> bool
    def get_all_rules(self) -> List[SymbolicRule]
    def clear_rules(self) -> None

# Объяснение выводов
class IExplanationGenerator(ABC):
    def generate_explanation(self, result: DerivationResult) -> str
    def explain_derivation(self, derivation_path: List[str]) -> List[ExplanationStep]

# Оптимизация производительности
class IPerformanceOptimizer(ABC):
    def optimize_rules(self, rules: List[SymbolicRule]) -> List[SymbolicRule]
    def get_performance_metrics(self) -> Dict[str, Any]
```

---

## Основные классы данных

### SymbolicRule - Символическое правило

```python
@dataclass
class SymbolicRule:
    condition: str                          # Условие правила
    action: str                             # Действие правила
    rule_type: RuleType = RuleType.SYMBOLIC # Тип правила
    action_type: ActionType = ActionType.DERIVE # Тип действия
    weight: float = 1.0                     # Вес правила (0.0-10.0)
    confidence: float = 1.0                 # Уверенность (0.0-1.0)
    priority: int = 0                       # Приоритет (больше = выше)
    metadata: Dict[str, Any] = field(default_factory=dict) # Метаданные
    
    # Служебные поля
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0                    # Количество использований
    last_used: Optional[datetime] = None    # Последнее использование

# Типы правил
class RuleType(Enum):
    SYMBOLIC = "symbolic"       # Символическое правило
    PATTERN = "pattern"         # Шаблонное правило
    CONDITIONAL = "conditional" # Условное правило
    INFERENCE = "inference"     # Правило вывода

# Типы действий
class ActionType(Enum):
    ASSERT = "assert"           # Утверждение факта
    RETRACT = "retract"         # Отзыв факта
    DERIVE = "derive"           # Вывод нового факта
    QUERY = "query"             # Запрос
    EXECUTE = "execute"         # Выполнение функции
```

### ProcessingContext - Контекст обработки

```python
@dataclass
class ProcessingContext:
    facts: Dict[str, Any] = field(default_factory=dict)         # Факты
    variables: Dict[str, Any] = field(default_factory=dict)     # Переменные
    query_params: Dict[str, Any] = field(default_factory=dict) # Параметры запроса
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_depth: int = 5                      # Максимальная глубина вывода
    confidence_threshold: float = 0.5       # Порог уверенности
    
    def add_fact(self, key: str, value: Any, confidence: float = 1.0):
        """Добавление факта в контекст."""
        self.facts[key] = {
            "value": value,
            "confidence": confidence,
            "added_at": datetime.now()
        }
    
    def get_fact(self, key: str) -> Optional[Any]:
        """Получение факта из контекста."""
        if key in self.facts:
            return self.facts[key]["value"]
        return None
    
    def has_fact(self, key: str) -> bool:
        """Проверка наличия факта."""
        return key in self.facts
    
    def copy(self) -> "ProcessingContext":
        """Создание копии контекста."""
        # Возвращает глубокую копию контекста
```

### DerivationResult - Результат логического вывода

```python
@dataclass
class DerivationResult:
    success: bool                                           # Успешность вывода
    derived_facts: Dict[str, Any] = field(default_factory=dict) # Выведенные факты
    confidence: float = 0.0                                 # Общая уверенность
    explanation: List[ExplanationStep] = field(default_factory=list) # Объяснение
    rules_used: List[str] = field(default_factory=list)    # Использованные правила
    processing_time: float = 0.0                           # Время обработки
    error_message: Optional[str] = None                     # Сообщение об ошибке
    
    def add_derived_fact(self, key: str, value: Any, confidence: float = 1.0):
        """Добавление выведенного факта."""
        self.derived_facts[key] = {
            "value": value,
            "confidence": confidence
        }
    
    def add_explanation_step(self, step: ExplanationStep):
        """Добавление шага объяснения."""
        self.explanation.append(step)
```

### ExplanationStep - Шаг объяснения

```python
@dataclass
class ExplanationStep:
    step_number: int                        # Номер шага
    rule_id: str                           # ID использованного правила
    rule_description: str                  # Описание правила
    input_facts: Dict[str, Any]            # Входные факты
    output_facts: Dict[str, Any]           # Выходные факты
    confidence: float                      # Уверенность шага
    reasoning: str                         # Текстовое объяснение
```

---

## Фабрика и конфигурация

### ProcessorFactory - Создание процессоров

```python
from neurograph.processor import ProcessorFactory

# Создание стандартных процессоров
processor = ProcessorFactory.create("pattern_matching", 
                                   confidence_threshold=0.7,
                                   max_depth=5)

graph_processor = ProcessorFactory.create("graph_based",
                                         graph_provider=graph,
                                         use_graph_structure=True)

# Создание из конфигурации
config = {"type": "pattern_matching", "config": {"confidence_threshold": 0.6}}
processor = ProcessorFactory.create_from_config(config)

# Получение доступных типов
available = ProcessorFactory.get_available_types()
# Результат: ["pattern_matching", "graph_based"]
```

### Готовые фабричные функции

```python
from neurograph.processor import (
    create_default_processor,
    create_high_performance_processor,
    create_graph_processor
)

# Процессор по умолчанию
default_proc = create_default_processor(confidence_threshold=0.5)

# Высокопроизводительный процессор
perf_proc = create_high_performance_processor(
    confidence_threshold=0.3,
    max_depth=10,
    enable_explanations=False,
    parallel_processing=True
)

# Процессор с интеграцией графа
graph_proc = create_graph_processor(graph_provider=my_graph)
```

### Регистрация пользовательских процессоров

```python
from neurograph.processor import ProcessorFactory

# Регистрация нового типа процессора
ProcessorFactory.register_processor("custom", MyCustomProcessor)

# Использование
custom_proc = ProcessorFactory.create("custom", custom_param="value")
```

---

## Интеграция с другими модулями

### Интеграция с SemGraph

Processor использует граф знаний для структурированного вывода:

```python
def integrate_with_semgraph(processor, graph):
    """Интеграция процессора с графом знаний."""
    
    # Создание правил из структуры графа
    for node_id in graph.get_all_nodes():
        node_data = graph.get_node(node_id)
        if node_data and node_data.get('type') == 'concept':
            
            # Создание правила наследования
            neighbors = graph.get_neighbors(node_id, edge_type='is_a')
            for parent in neighbors:
                rule = SymbolicRule(
                    condition=f"{node_id} является экземпляром",
                    action=f"derive {node_id} имеет_свойства_от {parent}",
                    confidence=0.9,
                    metadata={"source": "graph_structure"}
                )
                processor.add_rule(rule)
    
    # Создание контекста из графа
    context = ProcessingContext()
    for source, target, edge_type in graph.get_all_edges():
        fact_key = f"{source}_{edge_type}_{target}"
        edge_data = graph.get_edge(source, target, edge_type)
        confidence = edge_data.get('weight', 1.0) if edge_data else 1.0
        context.add_fact(fact_key, True, confidence)
    
    return context
```

**Формат данных для SemGraph:**
```python
# Правила добавляются как специальные узлы
{
    "node_id": "rule_12345",
    "type": "rule",
    "condition": "собака является животным",
    "action": "derive собака нуждается в пище",
    "confidence": 0.9,
    "rule_type": "symbolic"
}

# Выводы добавляются как новые ребра
{
    "source": "собака",
    "target": "пища", 
    "edge_type": "нуждается_в",
    "weight": 0.8,
    "derived": True,
    "rule_source": "rule_12345"
}
```

### Интеграция с Memory

Processor работает с фактами из биоморфной памяти:

```python
def integrate_with_memory(processor, memory):
    """Интеграция процессора с памятью."""
    
    # Извлечение фактов из памяти для контекста
    context = ProcessingContext()
    
    # Получаем недавние элементы памяти
    recent_items = memory.get_recent_items(hours=24.0)
    
    for item in recent_items:
        if item.content_type == "fact":
            # Преобразуем содержимое в факт
            fact_key = f"memory_fact_{item.id}"
            context.add_fact(
                fact_key, 
                item.content, 
                confidence=item.metadata.get('confidence', 1.0)
            )
        
        elif item.content_type == "rule":
            # Добавляем правило в процессор
            rule = SymbolicRule(
                condition=item.metadata.get('condition', ''),
                action=item.metadata.get('action', ''),
                confidence=item.metadata.get('confidence', 0.8),
                metadata={"source": "memory", "memory_id": item.id}
            )
            processor.add_rule(rule)
    
    # Выполняем вывод
    result = processor.derive(context, depth=3)
    
    # Сохраняем новые выводы в память
    for fact_key, fact_data in result.derived_facts.items():
        from neurograph.memory.base import MemoryItem
        import numpy as np
        
        # Создаем векторное представление (если есть encoder)
        embedding = np.random.random(384)  # Заглушка
        
        memory_item = MemoryItem(
            content=str(fact_data["value"]),
            embedding=embedding,
            content_type="derived_fact",
            metadata={
                "confidence": fact_data["confidence"],
                "derived_from": result.rules_used,
                "processor_session": context.session_id
            }
        )
        
        memory.add(memory_item)
    
    return result
```

**Формат данных для Memory:**
```python
# Элемент памяти с правилом
{
    "content": "правило логического вывода",
    "embedding": np.array([...]),
    "content_type": "rule",
    "metadata": {
        "condition": "собака является животным",
        "action": "derive собака нуждается в пище",
        "confidence": 0.9,
        "rule_type": "symbolic"
    }
}

# Элемент памяти с выведенным фактом
{
    "content": "собака нуждается в пище",
    "embedding": np.array([...]),
    "content_type": "derived_fact",
    "metadata": {
        "confidence": 0.8,
        "derived_from": ["rule_123", "rule_456"],
        "derivation_path": ["step1", "step2"]
    }
}
```

### Интеграция с ContextVec

Processor использует векторные представления для семантического сопоставления:

```python
def integrate_with_contextvec(processor, vectors, encoder):
    """Интеграция процессора с векторными представлениями."""
    
    # Создание семантических правил
    semantic_rules = []
    
    # Находим похожие концепты через векторы
    all_keys = vectors.get_all_keys()
    
    for key in all_keys:
        # Находим семантически похожие концепты
        similar = vectors.get_most_similar(key, top_n=5)
        
        for similar_key, similarity in similar:
            if similarity > 0.8:  # Высокое сходство
                # Создаем правило семантического наследования
                rule = SymbolicRule(
                    condition=f"{key} семантически_похож на {similar_key}",
                    action=f"derive {key} может_наследовать_свойства {similar_key}",
                    confidence=similarity,
                    metadata={
                        "source": "semantic_similarity",
                        "similarity_score": similarity
                    }
                )
                semantic_rules.append(rule)
    
    # Добавляем семантические правила в процессор
    for rule in semantic_rules:
        processor.add_rule(rule)
    
    # Семантическое сопоставление фактов
    def semantic_fact_matching(fact_text: str, threshold: float = 0.7) -> List[str]:
        """Поиск семантически похожих фактов."""
        
        if encoder:
            fact_vector = encoder.encode(fact_text)
            # Поиск похожих векторов
            similar_vectors = vectors.get_most_similar_vector(fact_vector, top_n=10)
            
            matching_facts = []
            for key, similarity in similar_vectors:
                if similarity >= threshold:
                    matching_facts.append(key)
            
            return matching_facts
        
        return []
    
    # Добавляем семантическое сопоставление в процессор
    processor._semantic_matcher = semantic_fact_matching
    
    return semantic_rules
```

**Формат данных для ContextVec:**
```python
# Векторное представление правила
{
    "key": "rule_semantic_12345",
    "vector": np.array([0.1, 0.2, ...]),  # Размерность 384/768/etc
    "metadata": {
        "type": "rule",
        "condition_vector": np.array([...]),
        "action_vector": np.array([...]),
        "confidence": 0.9
    }
}

# Векторное представление факта
{
    "key": "fact_derived_67890", 
    "vector": np.array([0.3, 0.4, ...]),
    "metadata": {
        "type": "derived_fact",
        "fact_text": "собака нуждается в пище",
        "confidence": 0.8,
        "source_rule": "rule_12345"
    }
}
```

### Интеграция с NLP

Processor получает структурированные данные от NLP модуля:

```python
def integrate_with_nlp(processor, nlp_result):
    """Интеграция процессора с результатами NLP."""
    
    # Создание правил из отношений NLP
    nlp_rules = []
    
    for relation in nlp_result.relations:
        if relation.confidence > 0.8:  # Высоконадежные отношения
            
            if relation.predicate.value == "is_a":
                # Правило типизации
                rule = SymbolicRule(
                    condition=f"{relation.subject.text} существует",
                    action=f"derive {relation.subject.text} является {relation.object.text}",
                    confidence=relation.confidence,
                    metadata={
                        "source": "nlp_extraction",
                        "relation_type": relation.predicate.value,
                        "text_span": relation.text_span
                    }
                )
                nlp_rules.append(rule)
            
            elif relation.predicate.value == "has":
                # Правило свойств
                rule = SymbolicRule(
                    condition=f"{relation.subject.text} является {relation.object.text}",
                    action=f"derive {relation.subject.text} имеет_свойство {relation.object.text}",
                    confidence=relation.confidence,
                    metadata={"source": "nlp_extraction"}
                )
                nlp_rules.append(rule)
    
    # Добавление правил в процессор
    for rule in nlp_rules:
        processor.add_rule(rule)
    
    # Создание контекста из сущностей NLP
    context = ProcessingContext()
    
    for entity in nlp_result.entities:
        fact_key = f"entity_{entity.text}_{entity.entity_type.value}"
        context.add_fact(fact_key, True, entity.confidence)
    
    for relation in nlp_result.relations:
        fact_key = f"{relation.subject.text}_{relation.predicate.value}_{relation.object.text}"
        context.add_fact(fact_key, True, relation.confidence)
    
    return context, nlp_rules
```

**Формат данных от NLP:**
```python
# Правило из NLP отношения
{
    "condition": "Python является языком_программирования",
    "action": "derive Python используется_для программирования",
    "confidence": 0.95,
    "metadata": {
        "source": "nlp",
        "text_span": "Python - это язык программирования",
        "entity_types": ["TECHNOLOGY", "CONCEPT"]
    }
}

# Факт из NLP сущности
{
    "fact_key": "entity_Python_TECHNOLOGY",
    "value": True,
    "confidence": 0.9,
    "metadata": {
        "entity_type": "TECHNOLOGY",
        "text_position": [0, 6]
    }
}
```

---

## Примеры использования

### Базовое использование

```python
from neurograph.processor import ProcessorFactory, SymbolicRule, ProcessingContext

# Создание процессора
processor = ProcessorFactory.create("pattern_matching",
                                   confidence_threshold=0.7,
                                   enable_explanations=True)

# Добавление правил
rules = [
    SymbolicRule(
        condition="собака является животным",
        action="derive собака является живой",
        confidence=1.0
    ),
    SymbolicRule(
        condition="собака является живой",
        action="derive собака нуждается в пище",
        confidence=0.9
    ),
    SymbolicRule(
        condition="кот является животным", 
        action="derive кот является живой",
        confidence=1.0
    )
]

# Добавляем правила
rule_ids = []
for rule in rules:
    rule_id = processor.add_rule(rule)
    rule_ids.append(rule_id)

print(f"Добавлено {len(rule_ids)} правил")

# Создание контекста с фактами
context = ProcessingContext()
context.add_fact("собака_является_животным", True, 0.95)
context.add_fact("кот_является_животным", True, 0.9)

# Выполнение логического вывода
result = processor.derive(context, depth=3)

print(f"Успех: {result.success}")
print(f"Уверенность: {result.confidence:.2f}")
print(f"Выведенные факты: {len(result.derived_facts)}")

# Просмотр выведенных фактов
for fact_key, fact_data in result.derived_facts.items():
    print(f"  {fact_key}: {fact_data['value']} (уверенность: {fact_data['confidence']:.2f})")

# Просмотр объяснения
print("\nОбъяснение вывода:")
for step in result.explanation:
    print(f"Шаг {step.step_number}: {step.reasoning}")
    print(f"  Правило: {step.rule_description}")
    print(f"  Уверенность: {step.confidence:.2f}")
```

### Работа с медицинскими правилами

```python
# Создание медицинской экспертной системы
medical_processor = ProcessorFactory.create("pattern_matching",
                                           confidence_threshold=0.8,
                                           max_depth=5)

# Медицинские правила
medical_rules = [
    SymbolicRule(
        condition="пациент имеет свойство высокая_температура",
        action="derive пациент имеет свойство лихорадка",
        confidence=0.9,
        metadata={"domain": "medical", "category": "symptoms"}
    ),
    SymbolicRule(
        condition="пациент имеет свойство кашель И пациент имеет свойство лихорадка",
        action="derive пациент возможно_имеет простуда",
        confidence=0.7,
        metadata={"domain": "medical", "category": "diagnosis"}
    ),
    SymbolicRule(
        condition="пациент имеет свойство одышка И пациент имеет свойство лихорадка",
        action="derive пациент требует медицинское_внимание",
        confidence=0.85,
        metadata={"domain": "medical", "category": "recommendation"}
    )
]

# Добавление правил
for rule in medical_rules:
    medical_processor.add_rule(rule)

# Симптомы пациента
patient_context = ProcessingContext()
patient_context.add_fact("пациент_имеет_высокая_температура", True, 0.9)
patient_context.add_fact("пациент_имеет_кашель", True, 0.8)
patient_context.add_fact("пациент_имеет_одышка", True, 0.7)

# Диагностика
diagnosis_result = medical_processor.derive(patient_context, depth=3)

print("=== Медицинская диагностика ===")
for fact_key, fact_data in diagnosis_result.derived_facts.items():
    print(f"  {fact_key}: {fact_data['confidence']:.2f}")

# Получение рекомендаций
recommendations = [
    fact for fact in diagnosis_result.derived_facts.keys() 
    if "требует" in fact or "рекомендует" in fact
]
print(f"Рекомендации: {recommendations}")
```

### Интеграция с графом знаний

```python
from neurograph.semgraph import SemGraphFactory

# Создание графа и процессора
graph = SemGraphFactory.create("memory_efficient")
graph_processor = ProcessorFactory.create("graph_based",
                                          graph_provider=graph,
                                          use_graph_structure=True)

# Построение графа знаний
knowledge_data = [
    ("Python", "is_a", "programming_language"),
    ("Django", "is_a", "web_framework"),
    ("Django", "written_in", "Python"),
    ("Flask", "is_a", "web_framework"),
    ("Flask", "written_in", "Python"),
    ("programming_language", "is_a", "technology"),
    ("web_framework", "is_a", "software_tool")
]

for subject, predicate, obj in knowledge_data:
    graph.add_node(subject, type="concept")
    graph.add_node(obj, type="concept")
    graph.add_edge(subject, obj, predicate, weight=1.0)

# Добавление правил наследования
inheritance_rules = [
    SymbolicRule(
        condition="X written_in Y И Y is_a programming_language",
        action="derive X requires Y",
        confidence=0.9
    ),
    SymbolicRule(
        condition="X is_a Y И Y is_a Z",
        action="derive X is_a Z", 
        confidence=0.8
    ),
    SymbolicRule(
        condition="X is_a web_framework",
        action="derive X used_for web_development",
        confidence=0.85
    )
]

for rule in inheritance_rules:
    graph_processor.add_rule(rule)

# Создание контекста из графа
graph_context = ProcessingContext()
for source, target, edge_type in graph.get_all_edges():
    fact_key = f"{source}_{edge_type}_{target}"
    graph_context.add_fact(fact_key, True, 1.0)

# Логический вывод на основе графа
graph_result = graph_processor.derive(graph_context, depth=4)

print("=== Вывод на основе графа знаний ===")
print(f"Использовано правил: {len(graph_result.rules_used)}")
print("Новые выводы:")
for fact_key, fact_data in graph_result.derived_facts.items():
    print(f"  {fact_key}")

# Объяснение вывода
print("\nЦепочка рассуждений:")
for step in graph_result.explanation:
    print(f"  {step.step_number}. {step.reasoning}")
```

### Пакетная обработка правил

```python
from neurograph.processor.utils import RuleManager, RuleTemplateEngine

# Создание менеджера правил
rule_manager = RuleManager()

# Создание коллекций правил по доменам
rule_manager.create_collection("animals", "Правила о животных", "biology")
rule_manager.create_collection("technology", "Правила о технологиях", "IT")

# Шаблонный движок для генерации правил
template_engine = RuleTemplateEngine()

# Добавление шаблонов
template_engine.add_template(
    "inheritance",
    condition="{child} является {parent}",
    action="derive {child} наследует_свойства {parent}"
)

template_engine.add_template(
    "composition", 
    condition="{part} часть_от {whole}",
    action="derive {whole} содержит {part}"
)

# Добавление значений переменных
template_engine.add_variable_values("child", ["собака", "кот", "птица"])
template_engine.add_variable_values("parent", ["животное", "млекопитающее"]) 
template_engine.add_variable_values("part", ["колесо", "двигатель", "руль"])
template_engine.add_variable_values("whole", ["автомобиль", "велосипед"])

# Генерация правил из шаблонов
inheritance_rules = template_engine.generate_all_combinations("inheritance", confidence=0.9)
composition_rules = template_engine.generate_all_combinations("composition", confidence=0.8)

# Добавление в коллекции
for rule in inheritance_rules:
    rule_manager.add_rule_to_collection("animals", rule)

for rule in composition_rules:
    rule_manager.add_rule_to_collection("technology", rule)

# Фильтрация правил
high_confidence_rules = rule_manager.filter_rules(
    "animals", 
    min_confidence=0.85,
    keywords=["животное", "млекопитающее"]
)

print(f"Правил высокой уверенности: {len(high_confidence_rules)}")

# Экспорт коллекции
rule_manager.export_collection("animals", "animal_rules.json")

# Создание процессора с правилами из коллекции
batch_processor = ProcessorFactory.create("pattern_matching")

animal_rules = rule_manager.get_collection("animals")
for rule in animal_rules:
    batch_processor.add_rule(rule)

print(f"Загружено правил в процессор: {len(animal_rules)}")
```

---

## События и уведомления

Processor модуль интегрируется с системой событий NeuroGraph:

### Публикуемые события

```python
from neurograph.core.events import publish

# События правил
publish("processor.rule_added", {
    "rule_id": rule.id,
    "rule_type": rule.rule_type.value,
    "confidence": rule.confidence,
    "domain": rule.metadata.get("domain", "general")
})

publish("processor.rule_executed", {
    "rule_id": rule_id,
    "execution_time": execution_time,
    "success": result.success,
    "confidence": result.confidence,
    "facts_derived": len(result.derived_facts)
})

# События вывода
publish("processor.derivation_completed", {
    "session_id": context.session_id,
    "depth_used": actual_depth,
    "rules_fired": len(result.rules_used),
    "facts_derived": len(result.derived_facts),
    "processing_time": result.processing_time,
    "confidence": result.confidence
})

# События ошибок
publish("processor.execution_error", {
    "error_type": type(error).__name__,
    "rule_id": rule_id,
    "context_size": len(context.facts),
    "error_message": str(error)
})
```

### Подписка на события

```python
from neurograph.core.events import subscribe

def handle_rule_execution(data):
    """Обработчик выполнения правил для мониторинга."""
    
    if data["success"] and data["confidence"] > 0.9:
        # Высококачественный вывод
        high_quality_inference(data)
    
    if data["execution_time"] > 1.0:
        # Медленное выполнение правила
        performance_alert(data["rule_id"], data["execution_time"])

def handle_derivation_completed(data):
    """Обработчик завершения вывода."""
    
    if data["facts_derived"] > 10:
        # Много новых фактов - возможно стоит обновить память
        update_memory_with_facts(data)
    
    if data["confidence"] < 0.5:
        # Низкая уверенность - требует проверки
        review_low_confidence_derivation(data)

# Подписка на события
subscribe("processor.rule_executed", handle_rule_execution)
subscribe("processor.derivation_completed", handle_derivation_completed)
```

### Интеграционные события

Для интеграции с другими модулями Processor подписывается на их события:

```python
# Подписка на события других модулей
def handle_graph_node_added(data):
    """Реакция на добавление узла в граф."""
    if data["node_type"] == "concept":
        # Новый концепт - создаем правила классификации
        create_classification_rules(data["node_id"])

def handle_memory_consolidation(data):
    """Реакция на консолидацию памяти."""
    if data["consolidated_count"] > 5:
        # Много консолидированных элементов - обновляем правила
        update_rules_from_memory(data)

def handle_nlp_knowledge_extracted(data):
    """Реакция на извлечение знаний NLP."""
    if len(data["relations"]) > 3:
        # Много отношений - создаем правила
        create_rules_from_nlp(data)

# Подписки на события других модулей
subscribe("graph.node_added", handle_graph_node_added)
subscribe("memory.consolidation_completed", handle_memory_consolidation)
subscribe("nlp.knowledge_extracted", handle_nlp_knowledge_extracted)
```

---

## Производительность и масштабирование

### Метрики производительности

Processor модуль отслеживает ключевые метрики:

```python
def get_performance_metrics(processor):
    """Получение метрик производительности Processor."""
    
    stats = processor.get_statistics()
    
    return {
        "throughput": {
            "rules_per_second": stats["rules_executed"] / stats["total_uptime"],
            "derivations_per_minute": stats["derivations_performed"] * 60 / stats["total_uptime"],
            "facts_per_derivation": stats["total_facts_derived"] / max(1, stats["derivations_performed"])
        },
        "latency": {
            "avg_rule_execution_time": stats["average_execution_time"],
            "avg_derivation_time": stats["total_execution_time"] / max(1, stats["derivations_performed"]),
            "p95_execution_time": stats.get("p95_execution_time", 0),
            "p99_execution_time": stats.get("p99_execution_time", 0)
        },
        "accuracy": {
            "avg_rule_confidence": stats.get("avg_rule_confidence", 0),
            "avg_derivation_confidence": stats.get("avg_derivation_confidence", 0),
            "successful_derivations_rate": stats.get("success_rate", 0)
        },
        "resource_usage": {
            "rules_count": stats["rules_count"],
            "active_contexts": stats.get("active_contexts", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0),
            "memory_usage_mb": stats.get("memory_usage", 0)
        }
    }
```

### Оптимизация правил

```python
from neurograph.processor.utils import RuleOptimizer, PerformanceProfiler

class OptimizedProcessor:
    """Процессор с оптимизацией производительности."""
    
    def __init__(self, base_processor):
        self.base_processor = base_processor
        self.optimizer = RuleOptimizer()
        self.profiler = PerformanceProfiler()
        
        # Периодическая оптимизация
        self.optimization_interval = 100  # Каждые 100 операций
        self.operation_count = 0
    
    def derive(self, context: ProcessingContext, depth: int = 1):
        """Вывод с профилированием и оптимизацией."""
        
        self.profiler.start_profiling()
        
        try:
            result = self.base_processor.derive(context, depth)
            
            # Записываем использование правил
            for rule_id in result.rules_used:
                self.profiler.record_rule_usage(rule_id)
            
            # Записываем обращения к фактам
            for fact_key in context.facts.keys():
                self.profiler.record_fact_access(fact_key)
            
            return result
            
        finally:
            self.profiler.stop_profiling()
            self.operation_count += 1
            
            # Периодическая оптимизация
            if self.operation_count % self.optimization_interval == 0:
                self._optimize_rules()
    
    def _optimize_rules(self):
        """Оптимизация правил на основе статистики использования."""
        
        current_rules = self.base_processor.get_all_rules()
        usage_stats = dict(self.profiler.rule_usage)
        
        # Оптимизация
        optimized_rules = self.optimizer.optimize_rules(current_rules, usage_stats)
        
        # Замена правил
        self.base_processor.clear_rules()
        for rule in optimized_rules:
            self.base_processor.add_rule(rule)
        
        print(f"Оптимизировано {len(current_rules)} -> {len(optimized_rules)} правил")
    
    def get_performance_report(self):
        """Отчет о производительности."""
        return self.profiler.get_performance_report()
```

### Параллельная обработка

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

class ParallelProcessor:
    """Процессор с параллельной обработкой."""
    
    def __init__(self, base_processor, max_workers: int = 4):
        self.base_processor = base_processor
        self.max_workers = max_workers
    
    def derive_batch(self, contexts: List[ProcessingContext], depth: int = 1):
        """Параллельная обработка списка контекстов."""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Создаем задачи
            futures = [
                executor.submit(self._derive_single, context, depth)
                for context in contexts
            ]
            
            # Собираем результаты
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 сек таймаут
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "success": False})
            
            return results
    
    def _derive_single(self, context: ProcessingContext, depth: int):
        """Обработка одного контекста."""
        try:
            return self.base_processor.derive(context, depth)
        except Exception as e:
            return DerivationResult(
                success=False,
                error_message=str(e)
            )
    
    async def derive_async(self, context: ProcessingContext, depth: int = 1):
        """Асинхронная обработка."""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.base_processor.derive, 
            context, 
            depth
        )
    
    def derive_with_timeout(self, context: ProcessingContext, 
                           depth: int = 1, timeout: float = 10.0):
        """Обработка с таймаутом."""
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.base_processor.derive, context, depth)
            
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                return DerivationResult(
                    success=False,
                    error_message=f"Превышен таймаут {timeout} секунд"
                )
```

### Кеширование результатов

```python
from neurograph.core.cache import cached
import hashlib

class CachedProcessor:
    """Процессор с кешированием результатов."""
    
    def __init__(self, base_processor, cache_ttl: int = 300):
        self.base_processor = base_processor
        self.cache_ttl = cache_ttl
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def _get_context_hash(self, context: ProcessingContext, depth: int) -> str:
        """Создание хеша контекста для кеширования."""
        
        # Сериализуем факты и параметры
        facts_str = str(sorted(context.facts.items()))
        params_str = f"depth={depth},threshold={context.confidence_threshold}"
        
        # Создаем хеш
        content = f"{facts_str}|{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @cached(ttl=300)  # 5 минут
    def derive_cached(self, context: ProcessingContext, depth: int = 1):
        """Кешированный вывод."""
        
        # Проверяем, изменились ли правила с последнего кеширования
        current_rules_hash = self._get_rules_hash()
        
        cache_key = self._get_context_hash(context, depth)
        cached_result = self._get_from_cache(cache_key, current_rules_hash)
        
        if cached_result:
            self.cache_stats["hits"] += 1
            return cached_result
        
        # Выполняем вывод
        result = self.base_processor.derive(context, depth)
        
        # Кешируем результат
        self._store_in_cache(cache_key, result, current_rules_hash)
        self.cache_stats["misses"] += 1
        
        return result
    
    def _get_rules_hash(self) -> str:
        """Хеш текущих правил для инвалидации кеша."""
        rules = self.base_processor.get_all_rules()
        rules_content = "|".join([f"{r.id}:{r.condition}:{r.action}" for r in rules])
        return hashlib.md5(rules_content.encode()).hexdigest()
    
    def get_cache_statistics(self):
        """Статистика кеширования."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / max(1, total)
        
        return {
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "total_requests": total
        }
```

---

## Расширение модуля

### Создание собственного процессора

```python
from neurograph.processor import INeuroSymbolicProcessor, ProcessorFactory

class CustomDomainProcessor(INeuroSymbolicProcessor):
    """Процессор для специфической предметной области."""
    
    def __init__(self, domain: str = "general", **kwargs):
        self.domain = domain
        self.logger = get_logger(f"domain_processor_{domain}")
        
        # Доменно-специфичные компоненты
        self.domain_rules = {}
        self.domain_patterns = self._load_domain_patterns()
        self.domain_vocabulary = self._load_domain_vocabulary()
        
        # Базовые параметры
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        self.max_depth = kwargs.get("max_depth", 5)
        
        # Статистика
        self._stats = {
            "domain_rules_applied": 0,
            "domain_specific_derivations": 0,
            "cross_domain_inferences": 0
        }
    
    def add_rule(self, rule: SymbolicRule) -> str:
        """Добавление правила с доменной специфичностью."""
        
        # Проверяем принадлежность к домену
        rule_domain = rule.metadata.get("domain", "general")
        
        if rule_domain == self.domain or rule_domain == "general":
            # Применяем доменную обработку правила
            processed_rule = self._process_domain_rule(rule)
            self.domain_rules[processed_rule.id] = processed_rule
            
            self.logger.info(f"Добавлено доменное правило {processed_rule.id}")
            return processed_rule.id
        else:
            # Правило не подходит для домена
            self.logger.warning(f"Правило {rule.id} не подходит для домена {self.domain}")
            raise ValueError(f"Правило не соответствует домену {self.domain}")
    
    def derive(self, context: ProcessingContext, depth: int = 1) -> DerivationResult:
        """Вывод с учетом доменной специфики."""
        
        # Предобработка контекста для домена
        domain_context = self._preprocess_context_for_domain(context)
        
        # Выполнение доменного вывода
        result = self._perform_domain_derivation(domain_context, depth)
        
        # Постобработка результатов
        final_result = self._postprocess_domain_result(result)
        
        self._stats["domain_specific_derivations"] += 1
        
        return final_result
    
    def _process_domain_rule(self, rule: SymbolicRule) -> SymbolicRule:
        """Обработка правила для домена."""
        
        if self.domain == "medical":
            return self._process_medical_rule(rule)
        elif self.domain == "legal":
            return self._process_legal_rule(rule)
        elif self.domain == "technical":
            return self._process_technical_rule(rule)
        else:
            return rule
    
    def _process_medical_rule(self, rule: SymbolicRule) -> SymbolicRule:
        """Обработка медицинского правила."""
        
        # Проверяем медицинскую валидность
        if not self._is_valid_medical_rule(rule):
            rule.confidence *= 0.7  # Снижаем уверенность
        
        # Добавляем медицинские метаданные
        rule.metadata.update({
            "medical_category": self._classify_medical_rule(rule),
            "safety_level": self._assess_medical_safety(rule),
            "evidence_level": "expert_system"
        })
        
        return rule
    
    def _preprocess_context_for_domain(self, context: ProcessingContext) -> ProcessingContext:
        """Предобработка контекста для домена."""
        
        domain_context = context.copy()
        
        # Фильтрация фактов по релевантности домену
        relevant_facts = {}
        for fact_key, fact_data in context.facts.items():
            if self._is_fact_relevant_to_domain(fact_key):
                # Нормализация факта для домена
                normalized_key = self._normalize_fact_for_domain(fact_key)
                relevant_facts[normalized_key] = fact_data
        
        domain_context.facts = relevant_facts
        
        # Добавление доменных параметров
        domain_context.query_params.update({
            "domain": self.domain,
            "domain_vocabulary": True,
            "cross_domain_inference": False
        })
        
        return domain_context
    
    def find_relevant_rules(self, context: ProcessingContext) -> List[str]:
        """Поиск релевантных правил с учетом домена."""
        
        relevant_rules = []
        
        for rule_id, rule in self.domain_rules.items():
            # Базовая релевантность
            if self._is_rule_applicable(rule, context):
                relevance_score = self._calculate_domain_relevance(rule, context)
                
                if relevance_score >= self.confidence_threshold:
                    relevant_rules.append((rule_id, relevance_score))
        
        # Сортируем по релевантности
        relevant_rules.sort(key=lambda x: x[1], reverse=True)
        
        return [rule_id for rule_id, _ in relevant_rules]
    
    def _calculate_domain_relevance(self, rule: SymbolicRule, context: ProcessingContext) -> float:
        """Расчет доменной релевантности правила."""
        
        base_relevance = rule.confidence
        
        # Бонус за доменную специфичность
        if rule.metadata.get("domain") == self.domain:
            base_relevance *= 1.2
        
        # Проверка доменного словаря
        condition_words = rule.condition.lower().split()
        domain_words_in_condition = sum(
            1 for word in condition_words 
            if word in self.domain_vocabulary
        )
        
        vocabulary_bonus = min(0.3, domain_words_in_condition * 0.1)
        
        return min(1.0, base_relevance + vocabulary_bonus)
    
    # Методы для загрузки доменных данных
    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Загрузка паттернов для домена."""
        if self.domain == "medical":
            return {
                "symptoms": [r"пациент\s+имеет\s+(.+)", r"симптом\s+(.+)"],
                "diagnosis": [r"диагноз\s+(.+)", r"заболевание\s+(.+)"],
                "treatment": [r"лечение\s+(.+)", r"терапия\s+(.+)"]
            }
        elif self.domain == "legal":
            return {
                "law": [r"статья\s+(\d+)", r"закон\s+(.+)"],
                "procedure": [r"процедура\s+(.+)", r"порядок\s+(.+)"]
            }
        else:
            return {}
    
    def _load_domain_vocabulary(self) -> Set[str]:
        """Загрузка словаря предметной области."""
        if self.domain == "medical":
            return {
                "пациент", "симптом", "диагноз", "лечение", "терапия",
                "болезнь", "заболевание", "синдром", "препарат"
            }
        elif self.domain == "legal":
            return {
                "закон", "статья", "право", "обязанность", "ответственность",
                "процедура", "решение", "постановление"
            }
        else:
            return set()
    
    # Остальные методы интерфейса...
    def remove_rule(self, rule_id: str) -> bool:
        if rule_id in self.domain_rules:
            del self.domain_rules[rule_id]
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[SymbolicRule]:
        return self.domain_rules.get(rule_id)
    
    def execute_rule(self, rule_id: str, context: ProcessingContext) -> DerivationResult:
        # Реализация выполнения правила
        pass
    
    def update_rule(self, rule_id: str, **attributes) -> bool:
        if rule_id in self.domain_rules:
            rule = self.domain_rules[rule_id]
            for attr, value in attributes.items():
                if hasattr(rule, attr):
                    setattr(rule, attr, value)
            return True
        return False
    
    def get_all_rules(self) -> List[SymbolicRule]:
        return list(self.domain_rules.values())
    
    def validate_rule(self, rule: SymbolicRule) -> Tuple[bool, Optional[str]]:
        # Доменная валидация
        if not self._is_valid_domain_rule(rule):
            return False, f"Правило не соответствует домену {self.domain}"
        return True, None
    
    def clear_rules(self) -> None:
        self.domain_rules.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        base_stats = {
            "rules_count": len(self.domain_rules),
            "domain": self.domain,
            "domain_rules_applied": self._stats["domain_rules_applied"],
            "domain_specific_derivations": self._stats["domain_specific_derivations"]
        }
        return base_stats

# Регистрация в фабрике
ProcessorFactory.register_processor("domain_specific", CustomDomainProcessor)

# Использование
medical_processor = ProcessorFactory.create("domain_specific", 
                                           domain="medical",
                                           confidence_threshold=0.8)
```

### Добавление новых типов правил

```python
from neurograph.processor.base import RuleType, ActionType
from enum import Enum

# Расширение типов правил
class ExtendedRuleType(RuleType):
    PROBABILISTIC = "probabilistic"        # Вероятностное правило
    TEMPORAL = "temporal"                  # Временное правило
    FUZZY = "fuzzy"                        # Нечеткое правило
    META = "meta"                          # Мета-правило

class ExtendedActionType(ActionType):
    UPDATE = "update"                      # Обновление факта
    SCHEDULE = "schedule"                  # Планирование действия
    NOTIFY = "notify"                      # Уведомление
    AGGREGATE = "aggregate"                # Агрегация данных

# Расширенное правило с дополнительными свойствами
@dataclass
class ExtendedSymbolicRule(SymbolicRule):
    probability: float = 1.0               # Вероятность (для probabilistic)
    temporal_constraint: Optional[str] = None  # Временные ограничения
    fuzzy_degree: float = 1.0              # Степень нечеткости
    meta_level: int = 0                    # Уровень мета-правила
    
    def to_dict(self) -> Dict[str, Any]:
        """Расширенная сериализация."""
        base_dict = super().to_dict()
        base_dict.update({
            "probability": self.probability,
            "temporal_constraint": self.temporal_constraint,
            "fuzzy_degree": self.fuzzy_degree,
            "meta_level": self.meta_level
        })
        return base_dict

# Процессор для расширенных правил
class ExtendedProcessor(INeuroSymbolicProcessor):
    """Процессор с поддержкой расширенных типов правил."""
    
    def __init__(self, **kwargs):
        self.extended_rules: Dict[str, ExtendedSymbolicRule] = {}
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        self.temporal_engine = TemporalReasoningEngine()
        self.fuzzy_engine = FuzzyReasoningEngine()
        
    def add_rule(self, rule: ExtendedSymbolicRule) -> str:
        """Добавление расширенного правила."""
        
        # Валидация расширенных свойств
        if rule.rule_type == ExtendedRuleType.PROBABILISTIC:
            if not 0 <= rule.probability <= 1:
                raise ValueError("Вероятность должна быть между 0 и 1")
        
        if rule.rule_type == ExtendedRuleType.TEMPORAL:
            if not rule.temporal_constraint:
                raise ValueError("Временное правило должно иметь ограничения")
        
        if rule.rule_type == ExtendedRuleType.FUZZY:
            if not 0 <= rule.fuzzy_degree <= 1:
                raise ValueError("Степень нечеткости должна быть между 0 и 1")
        
        self.extended_rules[rule.id] = rule
        return rule.id
    
    def execute_rule(self, rule_id: str, context: ProcessingContext) -> DerivationResult:
        """Выполнение расширенного правила."""
        
        rule = self.extended_rules.get(rule_id)
        if not rule:
            raise RuleExecutionError(f"Правило {rule_id} не найдено", rule_id, context)
        
        # Выбор движка в зависимости от типа правила
        if rule.rule_type == ExtendedRuleType.PROBABILISTIC:
            return self._execute_probabilistic_rule(rule, context)
        elif rule.rule_type == ExtendedRuleType.TEMPORAL:
            return self._execute_temporal_rule(rule, context)
        elif rule.rule_type == ExtendedRuleType.FUZZY:
            return self._execute_fuzzy_rule(rule, context)
        elif rule.rule_type == ExtendedRuleType.META:
            return self._execute_meta_rule(rule, context)
        else:
            # Стандартное выполнение
            return self._execute_standard_rule(rule, context)
    
    def _execute_probabilistic_rule(self, rule: ExtendedSymbolicRule, 
                                   context: ProcessingContext) -> DerivationResult:
        """Выполнение вероятностного правила."""
        
        import random
        
        # Проверяем вероятность выполнения
        if random.random() > rule.probability:
            return DerivationResult(success=False, 
                                  error_message="Вероятностное правило не сработало")
        
        # Выполняем с учетом вероятности в уверенности
        result = self._execute_standard_rule(rule, context)
        if result.success:
            result.confidence *= rule.probability
        
        return result
    
    def _execute_temporal_rule(self, rule: ExtendedSymbolicRule, 
                              context: ProcessingContext) -> DerivationResult:
        """Выполнение временного правила."""
        
        # Проверяем временные ограничения
        if not self.temporal_engine.check_temporal_constraint(
            rule.temporal_constraint, context
        ):
            return DerivationResult(success=False,
                                  error_message="Временные ограничения не выполнены")
        
        # Выполняем правило с временным контекстом
        temporal_context = self.temporal_engine.add_temporal_facts(context)
        return self._execute_standard_rule(rule, temporal_context)
    
    def _execute_fuzzy_rule(self, rule: ExtendedSymbolicRule, 
                           context: ProcessingContext) -> DerivationResult:
        """Выполнение нечеткого правила."""
        
        # Нечеткое сопоставление условий
        fuzzy_match = self.fuzzy_engine.fuzzy_match(rule.condition, context)
        
        if fuzzy_match < self.confidence_threshold:
            return DerivationResult(success=False,
                                  error_message="Нечеткое сопоставление не удалось")
        
        # Выполняем с нечеткой уверенностью
        result = self._execute_standard_rule(rule, context)
        if result.success:
            result.confidence *= fuzzy_match * rule.fuzzy_degree
        
        return result


class TemporalReasoningEngine:
    """Движок для временных рассуждений."""
    
    def check_temporal_constraint(self, constraint: str, context: ProcessingContext) -> bool:
        """Проверка временных ограничений."""
        
        # Простые временные ограничения
        if "ALWAYS" in constraint:
            return True
        elif "NEVER" in constraint:
            return False
        elif "BEFORE" in constraint:
            return self._check_before_constraint(constraint, context)
        elif "AFTER" in constraint:
            return self._check_after_constraint(constraint, context)
        elif "DURING" in constraint:
            return self._check_during_constraint(constraint, context)
        
        return True
    
    def add_temporal_facts(self, context: ProcessingContext) -> ProcessingContext:
        """Добавление временных фактов в контекст."""
        
        temporal_context = context.copy()
        current_time = datetime.now()
        
        # Добавляем временные факты
        temporal_context.add_fact("current_time", current_time.isoformat(), 1.0)
        temporal_context.add_fact("current_hour", current_time.hour, 1.0)
        temporal_context.add_fact("current_day", current_time.weekday(), 1.0)
        
        return temporal_context


class FuzzyReasoningEngine:
    """Движок для нечетких рассуждений."""
    
    def fuzzy_match(self, condition: str, context: ProcessingContext) -> float:
        """Нечеткое сопоставление условий."""
        
        # Простое нечеткое сопоставление на основе сходства строк
        condition_words = set(condition.lower().split())
        
        max_similarity = 0.0
        for fact_key in context.facts.keys():
            fact_words = set(fact_key.lower().replace("_", " ").split())
            
            # Коэффициент Жаккара
            intersection = len(condition_words.intersection(fact_words))
            union = len(condition_words.union(fact_words))
            
            if union > 0:
                similarity = intersection / union
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
```

### Интеграция с внешними системами

```python
class ExternalSystemProcessor(INeuroSymbolicProcessor):
    """Процессор с интеграцией внешних систем."""
    
    def __init__(self, external_apis: Dict[str, str], **kwargs):
        self.external_apis = external_apis
        self.fallback_processor = create_default_processor(**kwargs)
        self.api_cache = {}
    
    async def derive_with_external_knowledge(self, context: ProcessingContext, 
                                           depth: int = 1) -> DerivationResult:
        """Вывод с использованием внешних знаний."""
        
        # Обогащение контекста внешними данными
        enriched_context = await self._enrich_context_with_external_data(context)
        
        # Выполнение вывода на обогащенном контексте
        result = self.fallback_processor.derive(enriched_context, depth)
        
        # Валидация результатов через внешние системы
        validated_result = await self._validate_with_external_systems(result)
        
        return validated_result
    
    async def _enrich_context_with_external_data(self, context: ProcessingContext) -> ProcessingContext:
        """Обогащение контекста данными из внешних систем."""
        
        enriched_context = context.copy()
        
        # Извлекаем сущности из контекста
        entities = self._extract_entities_from_context(context)
        
        # Запрашиваем данные для каждой сущности
        for entity in entities:
            external_data = await self._fetch_external_data(entity)
            
            if external_data:
                # Добавляем внешние данные как факты
                for fact_key, fact_value in external_data.items():
                    enriched_context.add_fact(
                        f"external_{entity}_{fact_key}",
                        fact_value,
                        confidence=0.8  # Внешние данные имеют высокую, но не полную уверенность
                    )
        
        return enriched_context
    
    async def _fetch_external_data(self, entity: str) -> Dict[str, Any]:
        """Получение данных о сущности из внешних API."""
        
        # Проверяем кеш
        if entity in self.api_cache:
            cache_time, data = self.api_cache[entity]
            if time.time() - cache_time < 3600:  # Кеш на час
                return data
        
        external_data = {}
        
        # Запросы к различным API
        if "wikipedia" in self.external_apis:
            wiki_data = await self._query_wikipedia(entity)
            external_data.update(wiki_data)
        
        if "wikidata" in self.external_apis:
            wikidata_data = await self._query_wikidata(entity)
            external_data.update(wikidata_data)
        
        if "knowledge_graph" in self.external_apis:
            kg_data = await self._query_knowledge_graph(entity)
            external_data.update(kg_data)
        
        # Кешируем результат
        self.api_cache[entity] = (time.time(), external_data)
        
        return external_data
    
    async def _query_wikipedia(self, entity: str) -> Dict[str, Any]:
        """Запрос к Wikipedia API."""
        
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "description": data.get("extract", ""),
                            "source": "wikipedia",
                            "type": data.get("type", "unknown")
                        }
        except Exception as e:
            self.logger.warning(f"Wikipedia API error: {e}")
        
        return {}
    
    async def _validate_with_external_systems(self, result: DerivationResult) -> DerivationResult:
        """Валидация результатов через внешние системы."""
        
        validated_result = result
        
        # Проверяем каждый выведенный факт
        for fact_key, fact_data in result.derived_facts.items():
            
            # Извлекаем информацию для валидации
            validation_query = self._create_validation_query(fact_key, fact_data)
            
            # Валидируем через внешние источники
            validation_score = await self._external_validation(validation_query)
            
            # Корректируем уверенность на основе валидации
            if validation_score >= 0:
                fact_data["confidence"] *= (0.5 + validation_score * 0.5)
                fact_data["external_validation"] = validation_score
        
        return validated_result
    
    def _create_validation_query(self, fact_key: str, fact_data: Dict) -> str:
        """Создание запроса для валидации факта."""
        
        # Простое преобразование ключа факта в запрос
        query_parts = fact_key.split("_")
        if len(query_parts) >= 3:
            subject, relation, obj = query_parts[0], query_parts[1], "_".join(query_parts[2:])
            return f"{subject} {relation} {obj}"
        
        return fact_key.replace("_", " ")


# Пример использования расширенного процессора
def example_extended_usage():
    """Пример использования расширенного процессора."""
    
    # Создание процессора с внешними API
    external_apis = {
        "wikipedia": "https://en.wikipedia.org/api/rest_v1/",
        "wikidata": "https://www.wikidata.org/w/api.php"
    }
    
    extended_proc = ExternalSystemProcessor(external_apis)
    
    # Добавление расширенных правил
    probabilistic_rule = ExtendedSymbolicRule(
        condition="weather is rainy",
        action="derive people carry umbrellas",
        rule_type=ExtendedRuleType.PROBABILISTIC,
        probability=0.7,  # 70% вероятность
        confidence=0.9
    )
    
    temporal_rule = ExtendedSymbolicRule(
        condition="person is at work",
        action="derive person is busy",
        rule_type=ExtendedRuleType.TEMPORAL,
        temporal_constraint="DURING working_hours",
        confidence=0.8
    )
    
    fuzzy_rule = ExtendedSymbolicRule(
        condition="temperature high",
        action="derive weather warm",
        rule_type=ExtendedRuleType.FUZZY,
        fuzzy_degree=0.8,
        confidence=0.7
    )
    
    # Контекст с внешними сущностями
    context = ProcessingContext()
    context.add_fact("entity_Einstein", True, 1.0)
    context.add_fact("entity_relativity", True, 1.0)
    context.add_fact("weather_condition", "rainy", 0.9)
    
    # Асинхронный вывод с внешними данными
    import asyncio
    
    async def run_extended_inference():
        result = await extended_proc.derive_with_external_knowledge(context, depth=3)
        
        print("=== Расширенный вывод ===")
        print(f"Успех: {result.success}")
        print(f"Обогащенных фактов: {len(result.derived_facts)}")
        
        for fact_key, fact_data in result.derived_facts.items():
            external_val = fact_data.get("external_validation", "N/A")
            print(f"  {fact_key}: {fact_data['confidence']:.2f} (внешняя валидация: {external_val})")
    
    # Запуск
    # asyncio.run(run_extended_inference())
```

---

## Обработка ошибок

### Типы ошибок модуля

```python
from neurograph.core.errors import NeuroGraphError

class ProcessorError(NeuroGraphError):
    """Базовая ошибка процессора."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PROCESSOR_ERROR"
        self.details = details or {}

class RuleValidationError(ProcessorError):
    """Ошибка валидации правил."""
    
    def __init__(self, message: str, rule_id: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "RULE_VALIDATION_ERROR", details)
        self.rule_id = rule_id

class RuleExecutionError(ProcessorError):
    """Ошибка выполнения правил."""
    
    def __init__(self, message: str, rule_id: str = None, context: ProcessingContext = None):
        super().__init__(message, "RULE_EXECUTION_ERROR")
        self.rule_id = rule_id
        self.context = context

class DerivationError(ProcessorError):
    """Ошибка логического вывода."""
    
    def __init__(self, message: str, context: ProcessingContext = None, 
                 partial_result: DerivationResult = None):
        super().__init__(message, "DERIVATION_ERROR")
        self.context = context
        self.partial_result = partial_result

class PerformanceError(ProcessorError):
    """Ошибка производительности."""
    
    def __init__(self, message: str, performance_data: Dict[str, Any] = None):
        super().__init__(message, "PERFORMANCE_ERROR")
        self.performance_data = performance_data or {}
```

### Стратегии обработки ошибок

```python
class RobustProcessor:
    """Процессор с надежной обработкой ошибок."""
    
    def __init__(self, base_processor, **kwargs):
        self.base_processor = base_processor
        self.error_recovery_enabled = kwargs.get("error_recovery", True)
        self.fallback_strategies = [
            self._simple_fallback,
            self._rule_based_fallback,
            self._minimal_fallback
        ]
        self.error_counts = {}
        self.max_consecutive_errors = kwargs.get("max_consecutive_errors", 5)
    
    def derive(self, context: ProcessingContext, depth: int = 1) -> DerivationResult:
        """Вывод с обработкой ошибок и восстановлением."""
        
        try:
            # Основная попытка
            result = self.base_processor.derive(context, depth)
            
            # Сброс счетчика ошибок при успехе
            self.error_counts["consecutive"] = 0
            
            return result
            
        except RuleExecutionError as e:
            self.logger.warning(f"Ошибка выполнения правила {e.rule_id}: {e}")
            return self._handle_rule_execution_error(e, context, depth)
            
        except DerivationError as e:
            self.logger.warning(f"Ошибка вывода: {e}")
            return self._handle_derivation_error(e, context, depth)
            
        except PerformanceError as e:
            self.logger.warning(f"Ошибка производительности: {e}")
            return self._handle_performance_error(e, context, depth)
            
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка: {e}")
            return self._handle_unexpected_error(e, context, depth)
    
    def _handle_rule_execution_error(self, error: RuleExecutionError, 
                                   context: ProcessingContext, depth: int) -> DerivationResult:
        """Обработка ошибок выполнения правил."""
        
        if not self.error_recovery_enabled:
            raise error
        
        # Удаляем проблемное правило временно
        if error.rule_id:
            problematic_rule = self.base_processor.get_rule(error.rule_id)
            if problematic_rule:
                self.base_processor.remove_rule(error.rule_id)
                self.logger.info(f"Временно удалено проблемное правило {error.rule_id}")
        
        # Повторная попытка без проблемного правила
        try:
            result = self.base_processor.derive(context, depth)
            result.error_message = f"Частичный успех: исключено правило {error.rule_id}"
            return result
            
        except Exception as retry_error:
            self.logger.error(f"Повторная попытка не удалась: {retry_error}")
            return self._apply_fallback_strategy(context, depth, error)
    
    def _handle_derivation_error(self, error: DerivationError, 
                               context: ProcessingContext, depth: int) -> DerivationResult:
        """Обработка ошибок логического вывода."""
        
        # Проверяем частичный результат
        if error.partial_result and error.partial_result.derived_facts:
            self.logger.info("Используем частичный результат")
            error.partial_result.success = True  # Помечаем как частично успешный
            error.partial_result.error_message = f"Частичный результат: {error.message}"
            return error.partial_result
        
        # Упрощение задачи
        simplified_context = self._simplify_context(context)
        reduced_depth = max(1, depth - 1)
        
        try:
            result = self.base_processor.derive(simplified_context, reduced_depth)
            result.error_message = f"Упрощенный вывод: {error.message}"
            return result
            
        except Exception:
            return self._apply_fallback_strategy(context, depth, error)
    
    def _handle_performance_error(self, error: PerformanceError, 
                                context: ProcessingContext, depth: int) -> DerivationResult:
        """Обработка ошибок производительности."""
        
        # Уменьшение нагрузки
        optimized_context = self._optimize_context_for_performance(context)
        reduced_depth = min(depth, 2)  # Ограничиваем глубину
        
        # Установка жестких лимитов времени
        start_time = time.time()
        timeout = 5.0  # 5 секунд максимум
        
        try:
            result = self.base_processor.derive(optimized_context, reduced_depth)
            
            execution_time = time.time() - start_time
            if execution_time > timeout:
                raise PerformanceError(f"Превышен таймаут: {execution_time:.2f}с")
            
            result.error_message = f"Оптимизированный вывод: {error.message}"
            return result
            
        except Exception:
            return self._create_minimal_result(context, error)
    
    def _handle_unexpected_error(self, error: Exception, 
                               context: ProcessingContext, depth: int) -> DerivationResult:
        """Обработка неожиданных ошибок."""
        
        self.error_counts["consecutive"] = self.error_counts.get("consecutive", 0) + 1
        
        if self.error_counts["consecutive"] >= self.max_consecutive_errors:
            # Критическое состояние - переходим в безопасный режим
            self.logger.critical(f"Критическое количество ошибок: {self.error_counts['consecutive']}")
            return self._enter_safe_mode(context, error)
        
        # Применяем стратегии восстановления
        return self._apply_fallback_strategy(context, depth, error)
    
    def _apply_fallback_strategy(self, context: ProcessingContext, 
                               depth: int, original_error: Exception) -> DerivationResult:
        """Применение стратегий восстановления."""
        
        for i, strategy in enumerate(self.fallback_strategies):
            try:
                self.logger.info(f"Применение стратегии восстановления {i+1}")
                result = strategy(context, depth)
                result.error_message = f"Восстановление {i+1}: {original_error}"
                return result
                
            except Exception as strategy_error:
                self.logger.warning(f"Стратегия {i+1} не удалась: {strategy_error}")
                continue
        
        # Все стратегии не сработали
        return self._create_error_result(context, original_error)
    
    def _simple_fallback(self, context: ProcessingContext, depth: int) -> DerivationResult:
        """Простая стратегия восстановления."""
        
        # Используем только базовые правила с высокой уверенностью
        high_confidence_rules = [
            rule for rule in self.base_processor.get_all_rules()
            if rule.confidence >= 0.9
        ]
        
        if not high_confidence_rules:
            raise ProcessorError("Нет правил высокой уверенности")
        
        # Создаем временный процессор только с надежными правилами
        temp_processor = create_default_processor()
        for rule in high_confidence_rules[:5]:  # Максимум 5 правил
            temp_processor.add_rule(rule)
        
        return temp_processor.derive(context, min(depth, 2))
    
    def _rule_based_fallback(self, context: ProcessingContext, depth: int) -> DerivationResult:
        """Стратегия на основе правил."""
        
        # Применяем только прямые правила (без цепочек)
        direct_rules = [
            rule for rule in self.base_processor.get_all_rules()
            if "И" not in rule.condition and "ИЛИ" not in rule.condition
        ]
        
        result = DerivationResult(success=True)
        
        for rule in direct_rules[:3]:  # Максимум 3 правила
            try:
                rule_result = self.base_processor.execute_rule(rule.id, context)
                if rule_result.success:
                    result.derived_facts.update(rule_result.derived_facts)
                    result.rules_used.append(rule.id)
            except:
                continue
        
        if result.derived_facts:
            result.confidence = 0.5  # Низкая уверенность для fallback
            return result
        else:
            raise ProcessorError("Не удалось применить прямые правила")
    
    def _minimal_fallback(self, context: ProcessingContext, depth: int) -> DerivationResult:
        """Минимальная стратегия восстановления."""
        
        # Возвращаем хотя бы один факт на основе входного контекста
        result = DerivationResult(success=True)
        
        if context.facts:
            # Берем самый надежный факт и делаем простой вывод
            most_confident_fact = max(
                context.facts.items(),
                key=lambda x: x[1].get("confidence", 0)
            )
            
            fact_key, fact_data = most_confident_fact
            
            # Простое правило идентичности
            result.add_derived_fact(
                f"confirmed_{fact_key}",
                fact_data["value"],
                fact_data.get("confidence", 1.0) * 0.3  # Сильно снижаем уверенность
            )
            
            result.confidence = 0.3
            result.explanation = [ExplanationStep(
                step_number=1,
                rule_id="fallback_identity",
                rule_description="Fallback identity rule",
                input_facts={fact_key: fact_data["value"]},
                output_facts={f"confirmed_{fact_key}": fact_data["value"]},
                confidence=0.3,
                reasoning="Минимальное восстановление на основе входных данных"
            )]
            
            return result
        
        # Если даже это не работает
        raise ProcessorError("Невозможно выполнить минимальное восстановление")
    
    def _create_error_result(self, context: ProcessingContext, error: Exception) -> DerivationResult:
        """Создание результата с ошибкой."""
        
        return DerivationResult(
            success=False,
            error_message=f"Все стратегии восстановления не удались: {error}",
            processing_time=0.0
        )
    
    def _enter_safe_mode(self, context: ProcessingContext, error: Exception) -> DerivationResult:
        """Переход в безопасный режим."""
        
        self.logger.critical("Переход процессора в безопасный режим")
        
        # Отключаем все сложные функции
        return DerivationResult(
            success=False,
            error_message="Процессор в безопасном режиме после критических ошибок",
            processing_time=0.0
        )
```

### Валидация и мониторинг

```python
from neurograph.processor.utils import RuleValidator

class ValidatedProcessor:
    """Процессор с валидацией и мониторингом."""
    
    def __init__(self, base_processor):
        self.base_processor = base_processor
        self.validator = RuleValidator()
        self.error_monitor = ErrorMonitor()
        
    def add_rule(self, rule: SymbolicRule) -> str:
        """Добавление правила с валидацией."""
        
        try:
            # Валидация правила
            existing_rules = self.base_processor.get_all_rules()
            is_valid, errors = self.validator.validate_rule(rule, existing_rules)
            
            if not is_valid:
                error_msg = f"Правило не прошло валидацию: {'; '.join(errors)}"
                raise RuleValidationError(error_msg, rule.id)
            
            # Добавление правила
            return self.base_processor.add_rule(rule)
            
        except Exception as e:
            # Мониторинг ошибки
            self.error_monitor.log_error(e, {
                "operation": "add_rule",
                "rule_id": rule.id,
                "rule_condition": rule.condition,
                "rule_action": rule.action
            })
            raise
    
    def derive(self, context: ProcessingContext, depth: int = 1) -> DerivationResult:
        """Вывод с мониторингом ошибок."""
        
        try:
            # Валидация контекста
            self._validate_context(context)
            
            # Выполнение вывода
            result = self.base_processor.derive(context, depth)
            
            return result
            
        except Exception as e:
            # Детальное логирование ошибки
            self.error_monitor.log_error(e, {
                "operation": "derive",
                "context_facts_count": len(context.facts),
                "depth": depth,
                "session_id": context.session_id,
                "confidence_threshold": context.confidence_threshold
            })
            raise
    
    def _validate_context(self, context: ProcessingContext):
        """Валидация контекста обработки."""
        
        if not context.facts:
            raise ValueError("Контекст не содержит фактов")
        
        # Проверка размера контекста
        if len(context.facts) > 1000:
            raise PerformanceError(f"Слишком большой контекст: {len(context.facts)} фактов")
        
        # Проверка валидности фактов
        for fact_key, fact_data in context.facts.items():
            if not isinstance(fact_data, dict):
                raise ValueError(f"Неверный формат факта: {fact_key}")
            
            if "confidence" in fact_data:
                confidence = fact_data["confidence"]
                if not 0 <= confidence <= 1:
                    raise ValueError(f"Неверная уверенность факта {fact_key}: {confidence}")


class ErrorMonitor:
    """Мониторинг ошибок процессора."""
    
    def __init__(self, alert_threshold: int = 10):
        self.alert_threshold = alert_threshold
        self.error_log = []
        self.error_stats = {}
        self.alert_sent = set()
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Логирование ошибки с контекстом."""
        
        error_entry = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "stack_trace": traceback.format_exc() if hasattr(traceback, 'format_exc') else None
        }
        
        self.error_log.append(error_entry)
        
        # Обновляем статистику
        error_type = error_entry["error_type"]
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1
        
        # Проверяем пороги для алертов
        if (self.error_stats[error_type] >= self.alert_threshold and 
            error_type not in self.alert_sent):
            self._send_alert(error_type, self.error_stats[error_type])
            self.alert_sent.add(error_type)
    
    def _send_alert(self, error_type: str, count: int):
        """Отправка алерта о критическом количестве ошибок."""
        
        from neurograph.core.events import publish
        
        publish("processor.error_threshold_exceeded", {
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
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Сводка по ошибкам."""
        
        total_errors = len(self.error_log)
        
        if total_errors == 0:
            return {"total_errors": 0, "status": "healthy"}
        
        # Анализ последних 24 часов
        recent_cutoff = time.time() - 86400  # 24 часа
        recent_errors = [e for e in self.error_log if e["timestamp"] > recent_cutoff]
        
        # Топ ошибок
        error_frequency = {}
        for error in recent_errors:
            error_type = error["error_type"]
            error_frequency[error_type] = error_frequency.get(error_type, 0) + 1
        
        top_errors = sorted(error_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_errors": total_errors,
            "recent_errors_24h": len(recent_errors),
            "top_error_types": top_errors,
            "alert_status": len(self.alert_sent),
            "status": "critical" if len(recent_errors) > 50 else "warning" if len(recent_errors) > 10 else "stable"
        }
```

---

## Конфигурация и настройка

### Структура конфигурации

```json
{
  "processor": {
    "type": "pattern_matching|graph_based|custom",
    "confidence_threshold": 0.5,
    "max_depth": 5,
    "enable_explanations": true,
    "cache_rules": true,
    "parallel_processing": false
  },
  "pattern_matching": {
    "rule_indexing": true,
    "cache_results": true,
    "optimization_enabled": true,
    "optimization_interval": 100
  },
  "graph_based": {
    "use_graph_structure": true,
    "path_search_limit": 100,
    "transitive_inference": true,
    "graph_cache_enabled": true
  },
  "rules": {
    "validation_enabled": true,
    "auto_optimization": true,
    "max_rules_per_processor": 10000,
    "rule_priority_enabled": true
  },
  "performance": {
    "timeout_seconds": 30,
    "max_workers": 4,
    "batch_size": 10,
    "memory_limit_mb": 512,
    "profiling_enabled": false
  },
  "error_handling": {
    "error_recovery": true,
    "max_consecutive_errors": 5,
    "fallback_strategies": ["simple", "rule_based", "minimal"],
    "alert_threshold": 10
  },
  "integration": {
    "graph_integration": {
      "auto_create_rules": true,
      "rule_confidence_threshold": 0.7
    },
    "memory_integration": {
      "auto_add_derived_facts": true,
      "fact_confidence_threshold": 0.8
    },
    "nlp_integration": {
      "auto_create_rules_from_relations": true,
      "relation_confidence_threshold": 0.8
    }
  },
  "advanced": {
    "probabilistic_reasoning": false,
    "temporal_reasoning": false,
    "fuzzy_reasoning": false,
    "meta_reasoning": false,
    "external_validation": false
  }
}
```

### Примеры конфигураций для разных сценариев

#### Конфигурация для разработки

```json
{
  "processor": {
    "type": "pattern_matching",
    "confidence_threshold": 0.3,
    "max_depth": 3,
    "enable_explanations": true,
    "cache_rules": false
  },
  "rules": {
    "validation_enabled": true,
    "max_rules_per_processor": 100
  },
  "performance": {
    "timeout_seconds": 10,
    "profiling_enabled": true
  },
  "error_handling": {
    "error_recovery": true,
    "max_consecutive_errors": 3
  }
}
```

#### Конфигурация для продакшена

```json
{
  "processor": {
    "type": "graph_based",
    "confidence_threshold": 0.7,
    "max_depth": 10,
    "enable_explanations": false,
    "cache_rules": true,
    "parallel_processing": true
  },
  "graph_based": {
    "use_graph_structure": true,
    "path_search_limit": 500,
    "transitive_inference": true
  },
  "performance": {
    "timeout_seconds": 60,
    "max_workers": 8,
    "batch_size": 50,
    "memory_limit_mb": 2048
  },
  "error_handling": {
    "error_recovery": true,
    "max_consecutive_errors": 10,
    "alert_threshold": 20
  }
}
```

#### Конфигурация для ограниченных ресурсов

```json
{
  "processor": {
    "type": "pattern_matching",
    "confidence_threshold": 0.8,
    "max_depth": 2,
    "enable_explanations": false,
    "cache_rules": false
  },
  "rules": {
    "max_rules_per_processor": 50,
    "auto_optimization": false
  },
  "performance": {
    "timeout_seconds": 5,
    "max_workers": 1,
    "batch_size": 1,
    "memory_limit_mb": 64
  }
}
```

### Управление конфигурацией в коде

```python
from neurograph.processor import ProcessorFactory
from neurograph.core import Configuration

# Программная конфигурация
config = Configuration({
    "processor": {
        "type": "pattern_matching",
        "confidence_threshold": 0.6,
        "enable_explanations": True
    },
    "rules": {
        "validation_enabled": True,
        "auto_optimization": True
    }
})

# Создание процессора из конфигурации
processor = ProcessorFactory.create_from_config(config.get("processor"))

# Динамическое изменение настроек
processor.update_configuration({
    "confidence_threshold": 0.8,
    "max_depth": 7
})

# Конфигурация для специальных случаев
def create_medical_processor_config():
    """Конфигурация для медицинских правил."""
    return {
        "processor": {
            "type": "pattern_matching",
            "confidence_threshold": 0.9,  # Высокая точность
            "max_depth": 5,
            "enable_explanations": True   # Важно для медицины
        },
        "rules": {
            "validation_enabled": True,
            "domain_validation": "medical"
        },
        "error_handling": {
            "error_recovery": False,  # Строгий режим
            "max_consecutive_errors": 1
        }
    }

def create_realtime_processor_config():
    """Конфигурация для реального времени."""
    return {
        "processor": {
            "type": "pattern_matching",
            "confidence_threshold": 0.5,
            "max_depth": 3,
            "enable_explanations": False,  # Экономим время
            "parallel_processing": True
        },
        "performance": {
            "timeout_seconds": 1,  # Жесткий лимит
            "cache_rules": True,
            "optimization_enabled": True
        }
    }

def create_research_processor_config():
    """Конфигурация для исследований."""
    return {
        "processor": {
            "type": "graph_based",
            "confidence_threshold": 0.3,  # Исследуем все
            "max_depth": 15,
            "enable_explanations": True
        },
        "advanced": {
            "probabilistic_reasoning": True,
            "temporal_reasoning": True,
            "fuzzy_reasoning": True,
            "meta_reasoning": True
        },
        "performance": {
            "timeout_seconds": 300,  # 5 минут
            "profiling_enabled": True
        }
    }

# Использование конфигураций
medical_config = create_medical_processor_config()
medical_processor = ProcessorFactory.create_from_config(medical_config)

realtime_config = create_realtime_processor_config()
realtime_processor = ProcessorFactory.create_from_config(realtime_config)
```

### Конфигурационные валидаторы

```python
class ProcessorConfigValidator:
    """Валидатор конфигурации процессора."""
    
    def __init__(self):
        self.required_fields = {
            "processor": ["type", "confidence_threshold"],
            "performance": ["timeout_seconds"],
            "error_handling": ["error_recovery"]
        }
        
        self.valid_ranges = {
            "confidence_threshold": (0.0, 1.0),
            "max_depth": (1, 50),
            "timeout_seconds": (1, 3600),
            "max_workers": (1, 16)
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Валидация конфигурации."""
        
        errors = []
        
        # Проверка обязательных полей
        for section, fields in self.required_fields.items():
            if section not in config:
                errors.append(f"Отсутствует секция: {section}")
                continue
            
            section_config = config[section]
            for field in fields:
                if field not in section_config:
                    errors.append(f"Отсутствует поле: {section}.{field}")
        
        # Проверка диапазонов значений
        for field, (min_val, max_val) in self.valid_ranges.items():
            value = self._get_nested_value(config, field)
            if value is not None:
                if not min_val <= value <= max_val:
                    errors.append(f"Значение {field}={value} вне диапазона [{min_val}, {max_val}]")
        
        # Логическая валидация
        logical_errors = self._validate_logical_constraints(config)
        errors.extend(logical_errors)
        
        return len(errors) == 0, errors
    
    def _get_nested_value(self, config: Dict[str, Any], field: str):
        """Получение значения из вложенной структуры."""
        
        for section in config.values():
            if isinstance(section, dict) and field in section:
                return section[field]
        return None
    
    def _validate_logical_constraints(self, config: Dict[str, Any]) -> List[str]:
        """Валидация логических ограничений."""
        
        errors = []
        
        # Проверка совместимости настроек
        processor_config = config.get("processor", {})
        performance_config = config.get("performance", {})
        
        # Если включена параллельная обработка, должно быть больше 1 воркера
        if (processor_config.get("parallel_processing", False) and 
            performance_config.get("max_workers", 1) <= 1):
            errors.append("Параллельная обработка требует max_workers > 1")
        
        # Если отключены объяснения, то не стоит включать профилирование
        if (not processor_config.get("enable_explanations", True) and 
            performance_config.get("profiling_enabled", False)):
            errors.append("Профилирование без объяснений неэффективно")
        
        # Проверка продвинутых функций
        advanced_config = config.get("advanced", {})
        if any(advanced_config.values()):
            if processor_config.get("type") == "pattern_matching":
                errors.append("Продвинутые функции требуют graph_based процессор")
        
        return errors

# Использование валидатора
validator = ProcessorConfigValidator()

config_to_validate = {
    "processor": {
        "type": "pattern_matching",
        "confidence_threshold": 0.7,
        "parallel_processing": True  # Потенциальная проблема
    },
    "performance": {
        "timeout_seconds": 30,
        "max_workers": 1  # Проблема: параллельная обработка с 1 воркером
    },
    "error_handling": {
        "error_recovery": True
    }
}

is_valid, errors = validator.validate_config(config_to_validate)

if not is_valid:
    print("Ошибки конфигурации:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Конфигурация валидна")
```

---

## Заключение

### Ключевые точки интеграции для разработчиков других модулей

1. **Входные данные для Processor:**
   - `ProcessingContext` с фактами и переменными
   - `SymbolicRule` для добавления правил логического вывода
   - Структурированные данные из NLP (отношения → правила)
   - Узлы и связи из SemGraph (граф → правила наследования)

2. **Выходные данные из Processor:**
   - `DerivationResult` с выведенными фактами и объяснениями
   - Новые логические утверждения для добавления в граф
   - Выведенные факты для сохранения в памяти
   - События для уведомления других модулей

3. **События для подписки:**
   - `processor.rule_executed` - выполнение правила
   - `processor.derivation_completed` - завершение логического вывода
   - `processor.error_threshold_exceeded` - критические ошибки

4. **Конфигурационные параметры:**
   - Типы процессоров и их специфичные настройки
   - Пороги уверенности для фильтрации результатов
   - Параметры производительности и масштабирования

### Рекомендации по использованию

1. **Для модуля NLP:**
   - Конвертируйте извлеченные отношения в `SymbolicRule`
   - Используйте высокие пороги уверенности (>0.8) для создания правил
   - Добавляйте метаданные с информацией об источнике извлечения

2. **Для модуля SemGraph:**
   - Создавайте правила наследования из иерархии графа  
   - Используйте структуру графа для транзитивных выводов
   - Добавляйте выведенные факты как новые ребра графа

3. **Для модуля Memory:**
   - Сохраняйте выведенные факты как элементы памяти
   - Используйте цепочки вывода для объяснений
   - Консолидируйте правила на основе частоты использования

4. **Для модуля ContextVec:**
   - Векторизуйте правила для семантического поиска
   - Используйте векторное сходство для нечеткого сопоставления
   - Кешируйте векторы часто используемых правил

### Примеры интеграционного кода

```python
# Полная интеграция Processor с другими модулями
class NeuroGraphProcessorIntegrator:
    """Интегратор Processor с другими модулями системы."""
    
    def __init__(self, processor, graph, vectors, memory, nlp):
        self.processor = processor
        self.graph = graph
        self.vectors = vectors
        self.memory = memory
        self.nlp = nlp
    
    def process_text_and_reason(self, text: str) -> Dict[str, Any]:
        """Полная обработка: NLP → правила → вывод → сохранение."""
        
        # 1. NLP обработка
        nlp_result = self.nlp.process_text(text)
        
        # 2. Создание правил из NLP
        rules_created = self._create_rules_from_nlp(nlp_result)
        
        # 3. Создание контекста из NLP и графа
        context = self._create_context_from_sources(nlp_result)
        
        # 4. Логический вывод
        derivation_result = self.processor.derive(context, depth=5)
        
        # 5. Сохранение результатов
        storage_result = self._store_results(derivation_result, nlp_result)
        
        return {
            "nlp_result": nlp_result,
            "rules_created": rules_created,
            "derivation_result": derivation_result,
            "storage_result": storage_result
        }
    
    def _create_rules_from_nlp(self, nlp_result) -> List[str]:
        """Создание правил из результатов NLP."""
        
        rule_ids = []
        
        for relation in nlp_result.relations:
            if relation.confidence > 0.8:
                if relation.predicate.value == "is_a":
                    # Правило классификации
                    rule = SymbolicRule(
                        condition=f"{relation.subject.text} существует",
                        action=f"derive {relation.subject.text} принадлежит_классу {relation.object.text}",
                        confidence=relation.confidence,
                        metadata={
                            "source": "nlp_relation",
                            "relation_type": relation.predicate.value
                        }
                    )
                    rule_id = self.processor.add_rule(rule)
                    rule_ids.append(rule_id)
        
        return rule_ids
    
    def _create_context_from_sources(self, nlp_result) -> ProcessingContext:
        """Создание контекста из NLP и графа."""
        
        context = ProcessingContext()
        
        # Факты из NLP
        for entity in nlp_result.entities:
            fact_key = f"entity_{entity.text}_{entity.entity_type.value}"
            context.add_fact(fact_key, True, entity.confidence)
        
        # Факты из графа
        for source, target, edge_type in self.graph.get_all_edges():
            fact_key = f"graph_{source}_{edge_type}_{target}"
            edge_data = self.graph.get_edge(source, target, edge_type)
            confidence = edge_data.get('weight', 1.0) if edge_data else 1.0
            context.add_fact(fact_key, True, confidence)
        
        return context
    
    def _store_results(self, derivation_result: DerivationResult, nlp_result) -> Dict[str, int]:
        """Сохранение результатов во все модули."""
        
        storage_stats = {
            "graph_nodes_added": 0,
            "graph_edges_added": 0,
            "memory_items_added": 0,
            "vectors_created": 0
        }
        
        # Сохранение в граф
        for fact_key, fact_data in derivation_result.derived_facts.items():
            if self._is_relational_fact(fact_key):
                parts = fact_key.split('_')
                if len(parts) >= 3:
                    subject, relation, obj = parts[0], parts[1], '_'.join(parts[2:])
                    
                    # Добавляем узлы
                    self.graph.add_node(subject, type="derived_entity")
                    self.graph.add_node(obj, type="derived_entity")
                    storage_stats["graph_nodes_added"] += 2
                    
                    # Добавляем связь
                    self.graph.add_edge(subject, obj, relation, 
                                      weight=fact_data["confidence"],
                                      derived=True)
                    storage_stats["graph_edges_added"] += 1
        
        # Сохранение в память
        for fact_key, fact_data in derivation_result.derived_facts.items():
            if hasattr(self, 'encoder'):  # Если есть энкодер
                from neurograph.memory.base import MemoryItem
                import numpy as np
                
                fact_text = str(fact_data["value"])
                embedding = np.random.random(384)  # Заглушка для векторизации
                
                memory_item = MemoryItem(
                    content=fact_text,
                    embedding=embedding,
                    content_type="derived_fact",
                    metadata={
                        "confidence": fact_data["confidence"],
                        "derivation_rules": derivation_result.rules_used,
                        "source": "processor_derivation"
                    }
                )
                
                self.memory.add(memory_item)
                storage_stats["memory_items_added"] += 1
        
        return storage_stats

# Пример использования интегратора
integrator = NeuroGraphProcessorIntegrator(processor, graph, vectors, memory, nlp)

text = "Собаки являются млекопитающими. Млекопитающие являются животными."
result = integrator.process_text_and_reason(text)

print("Результат интеграции:")
print(f"  Правил создано: {len(result['rules_created'])}")
print(f"  Фактов выведено: {len(result['derivation_result'].derived_facts)}")
print(f"  Узлов добавлено в граф: {result['storage_result']['graph_nodes_added']}")
print(f"  Элементов добавлено в память: {result['storage_result']['memory_items_added']}")
```

### Тестирование интеграции

```python
import unittest
from neurograph.processor import ProcessorFactory, SymbolicRule, ProcessingContext

class TestProcessorIntegration(unittest.TestCase):
    """Тесты интеграции Processor с другими модулями."""
    
    def setUp(self):
        self.processor = ProcessorFactory.create("pattern_matching")
        self.test_rule = SymbolicRule(
            condition="собака является животным",
            action="derive собака является живой",
            confidence=0.9
        )
    
    def test_rule_execution_integration(self):
        """Тест интеграции выполнения правил."""
        
        # Добавляем правило
        rule_id = self.processor.add_rule(self.test_rule)
        
        # Создаем контекст
        context = ProcessingContext()
        context.add_fact("собака_является_животным", True, 0.95)
        
        # Выполняем вывод
        result = self.processor.derive(context, depth=2)
        
        # Проверяем результат
        self.assertTrue(result.success)
        self.assertGreater(len(result.derived_facts), 0)
        self.assertIn(rule_id, result.rules_used)
        
    def test_data_format_compatibility(self):
        """Тест совместимости форматов данных."""
        
        rule_id = self.processor.add_rule(self.test_rule)
        
        context = ProcessingContext()
        context.add_fact("test_fact", "test_value", 0.8)
        
        result = self.processor.derive(context, depth=1)
        
        # Проверяем формат результата
        self.assertIsInstance(result.derived_facts, dict)
        self.assertIsInstance(result.explanation, list)
        self.assertIsInstance(result.rules_used, list)
        self.assertIsInstance(result.confidence, float)
        
        # Проверяем формат фактов
        for fact_key, fact_data in result.derived_facts.items():
            self.assertIsInstance(fact_key, str)
            self.assertIsInstance(fact_data, dict)
            self.assertIn("value", fact_data)
            self.assertIn("confidence", fact_data)
    
    def test_performance_under_load(self):
        """Тест производительности при нагрузке."""
        
        # Добавляем много правил
        rule_ids = []
        for i in range(100):
            rule = SymbolicRule(
                condition=f"entity_{i} является типом_{i//10}",
                action=f"derive entity_{i} имеет_свойство свойство_{i//10}",
                confidence=0.7
            )
            rule_id = self.processor.add_rule(rule)
            rule_ids.append(rule_id)
        
        # Создаем большой контекст
        context = ProcessingContext()
        for i in range(50):
            context.add_fact(f"entity_{i}_является_типом_{i//10}", True, 0.8)
        
        # Измеряем время выполнения
        import time
        start_time = time.time()
        
        result = self.processor.derive(context, depth=3)
        
        execution_time = time.time() - start_time
        
        # Проверяем, что выполнение завершилось за разумное время
        self.assertLess(execution_time, 10.0)  # Менее 10 секунд
        self.assertTrue(result.success)

if __name__ == "__main__":
    unittest.main()
```

### Метрики интеграции

Для мониторинга качества интеграции рекомендуется отслеживать:

```python
class ProcessorIntegrationMetrics:
    """Метрики интеграции Processor с другими модулями."""
    
    def __init__(self):
        self.metrics = {
            "rule_creation": {
                "from_nlp": 0,
                "from_graph": 0,
                "from_memory": 0,
                "total": 0
            },
            "derivation_performance": {
                "successful_derivations": 0,
                "failed_derivations": 0,
                "avg_facts_per_derivation": 0,
                "avg_confidence": 0
            },
            "integration_quality": {
                "facts_added_to_graph": 0,
                "facts_added_to_memory": 0,
                "cross_module_validations": 0,
                "validation_success_rate": 0
            }
        }
    
    def update_rule_creation(self, source: str):
        """Обновление метрик создания правил."""
        self.metrics["rule_creation"][source] += 1
        self.metrics["rule_creation"]["total"] += 1
    
    def update_derivation_performance(self, result: DerivationResult):
        """Обновление метрик производительности вывода."""
        if result.success:
            self.metrics["derivation_performance"]["avg_facts_per_derivation"] = (
                (self.metrics["derivation_performance"]["avg_facts_per_derivation"] * (total_derivations - 1) + facts_count) / total_derivations
            )
            
            self.metrics["derivation_performance"]["avg_confidence"] = (
                (self.metrics["derivation_performance"]["avg_confidence"] * (total_derivations - 1) + result.confidence) / total_derivations
            )
        else:
            self.metrics["derivation_performance"]["failed_derivations"] += 1
    
    def get_integration_health(self) -> Dict[str, str]:
        """Оценка здоровья интеграции."""
        
        health = {}
        
        # Проверяем создание правил
        total_rules = self.metrics["rule_creation"]["total"]
        if total_rules > 50:
            health["rule_creation"] = "excellent"
        elif total_rules > 10:
            health["rule_creation"] = "good"
        else:
            health["rule_creation"] = "needs_improvement"
        
        # Проверяем успешность выводов
        successful = self.metrics["derivation_performance"]["successful_derivations"]
        failed = self.metrics["derivation_performance"]["failed_derivations"]
        
        if failed == 0 and successful > 0:
            health["derivation_success"] = "excellent"
        elif successful > failed * 3:
            health["derivation_success"] = "good"
        else:
            health["derivation_success"] = "needs_attention"
        
        # Проверяем качество выводов
        avg_confidence = self.metrics["derivation_performance"]["avg_confidence"]
        if avg_confidence > 0.8:
            health["confidence_quality"] = "high"
        elif avg_confidence > 0.6:
            health["confidence_quality"] = "medium"
        else:
            health["confidence_quality"] = "low"
        
        return health
    
    def generate_report(self) -> str:
        """Генерация отчета по интеграции."""
        
        health = self.get_integration_health()
        
        report = """
=== Отчет по интеграции Processor модуля ===

Создание правил:
  Из NLP: {from_nlp}
  Из Graph: {from_graph}  
  Из Memory: {from_memory}
  Всего: {total}
  Статус: {rule_status}

Производительность вывода:
  Успешных: {successful}
  Неудачных: {failed}
  Среднее фактов на вывод: {avg_facts:.1f}
  Средняя уверенность: {avg_conf:.2f}
  Статус: {deriv_status}

Качество интеграции:
  Фактов добавлено в граф: {graph_facts}
  Фактов добавлено в память: {memory_facts}
  Общий статус уверенности: {conf_status}

Рекомендации:
{recommendations}
        """.format(
            from_nlp=self.metrics["rule_creation"]["from_nlp"],
            from_graph=self.metrics["rule_creation"]["from_graph"],
            from_memory=self.metrics["rule_creation"]["from_memory"],
            total=self.metrics["rule_creation"]["total"],
            rule_status=health.get("rule_creation", "unknown"),
            
            successful=self.metrics["derivation_performance"]["successful_derivations"],
            failed=self.metrics["derivation_performance"]["failed_derivations"],
            avg_facts=self.metrics["derivation_performance"]["avg_facts_per_derivation"],
            avg_conf=self.metrics["derivation_performance"]["avg_confidence"],
            deriv_status=health.get("derivation_success", "unknown"),
            
            graph_facts=self.metrics["integration_quality"]["facts_added_to_graph"],
            memory_facts=self.metrics["integration_quality"]["facts_added_to_memory"],
            conf_status=health.get("confidence_quality", "unknown"),
            
            recommendations=self._generate_recommendations(health)
        )
        
        return report
    
    def _generate_recommendations(self, health: Dict[str, str]) -> str:
        """Генерация рекомендаций на основе состояния."""
        
        recommendations = []
        
        if health.get("rule_creation") == "needs_improvement":
            recommendations.append("- Увеличить количество источников для создания правил")
            recommendations.append("- Проверить качество интеграции с NLP модулем")
        
        if health.get("derivation_success") == "needs_attention":
            recommendations.append("- Проверить валидность правил")
            recommendations.append("- Снизить пороги уверенности для экспериментов")
        
        if health.get("confidence_quality") == "low":
            recommendations.append("- Повысить качество входных данных")
            recommendations.append("- Настроить более строгие правила валидации")
        
        if not recommendations:
            recommendations.append("- Система работает стабильно, продолжайте мониторинг")
        
        return "\n".join(recommendations)


# Пример использования метрик
def example_metrics_usage():
    """Пример использования системы метрик."""
    
    metrics = ProcessorIntegrationMetrics()
    
    # Симуляция работы системы
    for i in range(20):
        # Создание правил из разных источников
        if i % 3 == 0:
            metrics.update_rule_creation("from_nlp")
        elif i % 3 == 1:
            metrics.update_rule_creation("from_graph")
        else:
            metrics.update_rule_creation("from_memory")
        
        # Симуляция результатов вывода
        mock_result = DerivationResult(
            success=True if i % 5 != 0 else False,  # 80% успешных
            derived_facts={f"fact_{j}": {"value": True, "confidence": 0.8} 
                          for j in range(i % 3 + 1)},
            confidence=0.7 + (i % 3) * 0.1
        )
        
        metrics.update_derivation_performance(mock_result)
    
    # Генерация отчета
    print(metrics.generate_report())
    
    # Проверка здоровья системы
    health = metrics.get_integration_health()
    print(f"\nЗдоровье системы: {health}")

# example_metrics_usage()
```

---

## Заключение для разработчиков

Модуль Processor является центральным компонентом системы NeuroGraph для логического вывода и рассуждений. Данная документация предоставляет все необходимые инструменты для интеграции с модулем:

### Ключевые преимущества модуля:

1. **Гибкость**: Поддержка различных типов правил и стратегий вывода
2. **Масштабируемость**: Оптимизация производительности и параллельная обработка  
3. **Надежность**: Комплексная обработка ошибок и стратегии восстановления
4. **Интегрируемость**: Готовые паттерны для связи с другими модулями
5. **Конфигурируемость**: Гибкая настройка под различные сценарии использования

### Для успешной интеграции рекомендуется:

1. **Изучить интерфейсы**: Особенно `INeuroSymbolicProcessor`, `ProcessingContext` и `DerivationResult`
2. **Следовать форматам данных**: Использовать рекомендованные структуры для фактов и правил
3. **Подписаться на события**: Для получения уведомлений о выводах и ошибках
4. **Настроить валидацию**: Обеспечить корректность передаваемых данных
5. **Мониторить производительность**: Отслеживать метрики интеграции и качества

### Точки расширения:

1. **Собственные процессоры**: Создание специализированных реализаций для доменов
2. **Новые типы правил**: Расширение поддержки логических конструкций
3. **Внешние интеграции**: Подключение внешних систем рассуждений
4. **Оптимизации**: Улучшение алгоритмов и производительности

Модуль готов к продуктивному использованию и легко интегрируется с остальными компонентами системы NeuroGraph, обеспечивая мощные возможности логического вывода для создания интеллектуального персонального ассистента.successful_derivations"] += 1
            
            # Обновляем средние значения
            facts_count = len(result.derived_facts)
            total_derivations = (self.metrics["derivation_performance"]["successful_derivations"] + 
                               self.metrics["derivation_performance"]["failed_derivations"])
            
            self.metrics["derivation_performance"]["