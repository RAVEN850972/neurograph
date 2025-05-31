# Модуль Processor - Документация для разработчиков

## 📋 Обзор

Модуль `neurograph.processor` реализует нейросимволический процессор для логического вывода и рассуждений. Он является ключевым компонентом системы NeuroGraph, обеспечивающим интеллектуальную обработку символических правил и выполнение логических выводов.

## 🏗️ Архитектура

### Основные компоненты:

```
processor/
├── __init__.py          # Публичный API модуля
├── base.py              # Базовые интерфейсы и классы данных
├── factory.py           # Фабрика для создания процессоров
├── utils.py             # Утилиты и вспомогательные классы
└── impl/                # Конкретные реализации
    ├── pattern_matching.py  # Процессор сопоставления шаблонов
    └── graph_based.py       # Процессор на основе графа знаний
```

### Паттерны проектирования:
- **Strategy Pattern** - различные стратегии логического вывода
- **Factory Pattern** - создание процессоров разных типов
- **Template Method** - общая структура обработки с кастомизацией
- **Observer Pattern** - отслеживание выполнения правил

## 🔧 Базовые интерфейсы

### INeuroSymbolicProcessor

Главный интерфейс модуля, определяющий контракт для всех процессоров:

```python
from neurograph.processor import INeuroSymbolicProcessor, SymbolicRule, ProcessingContext

class MyCustomProcessor(INeuroSymbolicProcessor):
    def add_rule(self, rule: SymbolicRule) -> str:
        """Добавление правила в базу знаний"""
        pass
    
    def derive(self, context: ProcessingContext, depth: int = 1) -> DerivationResult:
        """Выполнение логического вывода"""
        pass
    
    # ... другие методы интерфейса
```

### Ключевые методы:

| Метод | Описание | Возвращает |
|-------|----------|------------|
| `add_rule()` | Добавляет правило в базу знаний | ID правила |
| `execute_rule()` | Выполняет конкретное правило | Результат выполнения |
| `derive()` | Производит логический вывод | Полный результат с объяснениями |
| `find_relevant_rules()` | Находит применимые правила | Список ID правил |
| `validate_rule()` | Проверяет корректность правила | (bool, error_message) |

## 📊 Структуры данных

### SymbolicRule

Представляет символическое правило для логического вывода:

```python
from neurograph.processor import SymbolicRule, RuleType, ActionType

rule = SymbolicRule(
    condition="собака является млекопитающим",
    action="derive собака является животным",
    rule_type=RuleType.SYMBOLIC,
    action_type=ActionType.DERIVE,
    confidence=0.95,
    weight=1.0,
    priority=0,
    metadata={"domain": "biology"}
)
```

**Поля:**
- `condition` - условие срабатывания правила
- `action` - действие при выполнении условия
- `confidence` - уверенность в правиле (0.0-1.0)
- `weight` - вес правила для приоритизации
- `priority` - приоритет выполнения
- `metadata` - дополнительные метаданные

### ProcessingContext

Контекст для выполнения логического вывода:

```python
from neurograph.processor import ProcessingContext

context = ProcessingContext()

# Добавление фактов
context.add_fact("собака_существует", True, confidence=1.0)
context.add_fact("собака_млекопитающее", True, confidence=0.9)

# Работа с переменными
context.set_variable("animal", "собака")
animal = context.get_variable("animal")

# Проверка фактов
if context.has_fact("собака_существует"):
    print("Факт найден!")
```

### DerivationResult

Результат логического вывода с детальной информацией:

```python
result = processor.derive(context, depth=3)

print(f"Успех: {result.success}")
print(f"Уверенность: {result.confidence}")
print(f"Время: {result.processing_time}с")

# Выведенные факты
for fact_key, fact_data in result.derived_facts.items():
    print(f"Факт: {fact_key} = {fact_data['value']}")

# Объяснение вывода
for step in result.explanation:
    print(f"Шаг {step.step_number}: {step.reasoning}")
```

## 🚀 Быстрый старт

### 1. Создание процессора

```python
from neurograph.processor import ProcessorFactory

# Создание процессора сопоставления шаблонов
processor = ProcessorFactory.create("pattern_matching", 
                                   confidence_threshold=0.5,
                                   enable_explanations=True)

# Или процессора на основе графа
graph_processor = ProcessorFactory.create("graph_based",
                                         graph_provider=my_graph,
                                         use_graph_structure=True)
```

### 2. Добавление правил

```python
from neurograph.processor import SymbolicRule

# Простое правило
rule1 = SymbolicRule(
    condition="X является млекопитающим",
    action="derive X является животным",
    confidence=0.9
)

# Правило с условной логикой
rule2 = SymbolicRule(
    condition="животное И живое",
    action="derive животное нуждается в заботе",
    confidence=0.8
)

# Добавление в процессор
rule1_id = processor.add_rule(rule1)
rule2_id = processor.add_rule(rule2)
```

### 3. Выполнение вывода

```python
from neurograph.processor import ProcessingContext

# Создание контекста
context = ProcessingContext()
context.add_fact("собака_млекопитающее", True, 1.0)
context.add_fact("собака_живая", True, 1.0)

# Логический вывод
result = processor.derive(context, depth=2)

if result.success:
    print("Выводы:")
    for fact, data in result.derived_facts.items():
        print(f"- {fact}: {data['value']} (уверенность: {data['confidence']})")
    
    print("\nОбъяснение:")
    for step in result.explanation:
        print(f"- {step.rule_description} → {step.reasoning}")
```

## 🎯 Реализации процессоров

### PatternMatchingProcessor

Процессор на основе сопоставления шаблонов с развитой системой парсинга условий:

**Особенности:**
- Поддержка различных типов условий и действий
- Индексация правил для быстрого поиска
- Кеширование результатов
- Параллельная обработка (опционально)

**Поддерживаемые шаблоны:**

```python
# Базовые отношения
"X является Y"
"X имеет свойство Y" 
"X связан с Y"

# Логические операторы
"X И Y"
"X ИЛИ Y"
"НЕ X"

# Переменные и функции
"?X является Y"
"exists(?X)"
"count(X) > 5"
```

**Пример использования:**

```python
processor = ProcessorFactory.create("pattern_matching",
                                   confidence_threshold=0.5,
                                   enable_explanations=True,
                                   cache_rules=True)

# Правило с переменной
rule = SymbolicRule(
    condition="?animal является млекопитающим",
    action="assert ?animal является животным"
)
```

### GraphBasedProcessor

Процессор, интегрированный с графом знаний для использования его структуры:

**Особенности:**
- Использование структуры графа для вывода
- Поиск транзитивных отношений
- Распространение активации по графу
- Кеширование путей в графе

**Пример использования:**

```python
from neurograph.semgraph import SemGraphFactory

# Создание графа
graph = SemGraphFactory.create("memory_efficient")
graph.add_node("собака", type="animal")
graph.add_node("млекопитающее", type="class")
graph.add_edge("собака", "млекопитающее", "is_a")

# Создание процессора с графом
processor = ProcessorFactory.create("graph_based",
                                   graph_provider=graph,
                                   use_graph_structure=True,
                                   path_search_limit=100)
```

## 🔧 Расширение функциональности

### Создание пользовательского процессора

```python
from neurograph.processor.base import INeuroSymbolicProcessor

class MyCustomProcessor(INeuroSymbolicProcessor):
    def __init__(self, custom_param: str):
        self.custom_param = custom_param
        self._rules = {}
    
    def add_rule(self, rule: SymbolicRule) -> str:
        # Ваша логика добавления правил
        self._rules[rule.id] = rule
        return rule.id
    
    def derive(self, context: ProcessingContext, depth: int = 1) -> DerivationResult:
        # Ваша логика вывода
        result = DerivationResult(success=True)
        
        # Выполнение пользовательской логики
        for rule_id, rule in self._rules.items():
            if self._check_condition(rule, context):
                self._execute_action(rule, context, result)
        
        return result
    
    # Реализация остальных методов...

# Регистрация в фабрике
ProcessorFactory.register_processor("my_custom", MyCustomProcessor)
```

### Создание пользовательских правил

```python
class ConditionalRule(SymbolicRule):
    """Правило с условной логикой."""
    
    def __init__(self, if_condition: str, then_action: str, else_action: str = None, **kwargs):
        super().__init__(
            condition=if_condition,
            action=then_action,
            **kwargs
        )
        self.else_action = else_action
    
    def evaluate(self, context: ProcessingContext) -> tuple[bool, str]:
        """Оценка правила с возвращением соответствующего действия."""
        # Логика оценки условия
        condition_met = self._evaluate_condition(context)
        
        if condition_met:
            return True, self.action
        elif self.else_action:
            return True, self.else_action
        else:
            return False, ""
```

## 🧪 Тестирование

### Тестирование процессора

```python
import pytest
from neurograph.processor import ProcessorFactory, SymbolicRule, ProcessingContext

def test_basic_inference():
    """Тест базового логического вывода."""
    processor = ProcessorFactory.create("pattern_matching")
    
    # Добавление правила
    rule = SymbolicRule(
        condition="собака является млекопитающим",
        action="derive собака является животным"
    )
    processor.add_rule(rule)
    
    # Создание контекста
    context = ProcessingContext()
    context.add_fact("собака_является_млекопитающим", True)
    
    # Выполнение вывода
    result = processor.derive(context)
    
    # Проверки
    assert result.success
    assert len(result.derived_facts) > 0
    assert "собака_является_животным" in result.derived_facts

def test_rule_validation():
    """Тест валидации правил."""
    processor = ProcessorFactory.create("pattern_matching")
    
    # Невалидное правило
    invalid_rule = SymbolicRule(condition="", action="test")
    
    is_valid, error = processor.validate_rule(invalid_rule)
    assert not is_valid
    assert "пустым" in error.lower()

def test_performance():
    """Тест производительности."""
    processor = ProcessorFactory.create("pattern_matching")
    
    # Добавление множества правил
    for i in range(100):
        rule = SymbolicRule(
            condition=f"test_{i} является объектом",
            action=f"derive test_{i} существует"
        )
        processor.add_rule(rule)
    
    # Тест времени вывода
    context = ProcessingContext()
    context.add_fact("test_50_является_объектом", True)
    
    import time
    start = time.time()
    result = processor.derive(context)
    duration = time.time() - start
    
    assert result.success
    assert duration < 1.0  # Должно выполняться быстро
```

### Заглушки для тестирования

```python
class ProcessorStub(INeuroSymbolicProcessor):
    """Заглушка процессора для тестирования."""
    
    def __init__(self):
        self.rules_added = []
        self.derive_calls = []
    
    def add_rule(self, rule: SymbolicRule) -> str:
        self.rules_added.append(rule)
        return f"stub_rule_{len(self.rules_added)}"
    
    def derive(self, context: ProcessingContext, depth: int = 1) -> DerivationResult:
        self.derive_calls.append((context, depth))
        
        # Возвращаем предсказуемый результат
        result = DerivationResult(success=True, confidence=1.0)
        result.add_derived_fact("stub_fact", "stub_value", 1.0)
        return result
    
    # Остальные методы с минимальной реализацией...
```

## 📊 Мониторинг и отладка

### Получение статистики

```python
# Статистика процессора
stats = processor.get_statistics()
print(f"Правил добавлено: {stats['rules_added']}")
print(f"Правил выполнено: {stats['rules_executed']}")
print(f"Среднее время выполнения: {stats['average_execution_time']:.3f}с")
print(f"Процент попаданий в кеш: {stats['cache_hit_rate']:.1%}")
```

### Отладка правил

```python
# Проверка применимости правил
relevant_rules = processor.find_relevant_rules(context)
print(f"Найдено релевантных правил: {len(relevant_rules)}")

for rule_id in relevant_rules:
    rule = processor.get_rule(rule_id)
    print(f"- {rule.condition} → {rule.action} (уверенность: {rule.confidence})")

# Валидация всех правил
all_rules = processor.get_all_rules()
for rule in all_rules:
    is_valid, error = processor.validate_rule(rule)
    if not is_valid:
        print(f"Невалидное правило {rule.id}: {error}")
```

### Детальное объяснение вывода

```python
result = processor.derive(context, depth=3)

if result.success and result.explanation:
    print("Детальное объяснение вывода:")
    for step in result.explanation:
        print(f"\nШаг {step.step_number}:")
        print(f"  Правило: {step.rule_description}")
        print(f"  Входные факты: {step.input_facts}")
        print(f"  Выходные факты: {step.output_facts}")
        print(f"  Уверенность: {step.confidence:.2f}")
        print(f"  Рассуждение: {step.reasoning}")
```

## ⚙️ Конфигурация и оптимизация

### Настройка производительности

```python
# Высокопроизводительная конфигурация
processor = ProcessorFactory.create("pattern_matching",
    confidence_threshold=0.3,        # Более низкий порог
    max_depth=10,                    # Больше глубина
    enable_explanations=False,       # Отключить объяснения
    cache_rules=True,                # Включить кеширование
    parallel_processing=True,        # Параллельная обработка
    rule_indexing=True               # Индексация правил
)
```

### Настройка для отладки

```python
# Конфигурация для отладки
debug_processor = ProcessorFactory.create("pattern_matching",
    confidence_threshold=0.1,        # Низкий порог для всех правил
    max_depth=1,                     # Малая глубина для простоты
    enable_explanations=True,        # Детальные объяснения
    cache_rules=False,               # Отключить кеш для точности
    parallel_processing=False        # Последовательная обработка
)
```

## 🔗 Интеграция с другими модулями

### Работа с графом знаний

```python
from neurograph.semgraph import SemGraphFactory

# Создание интегрированной системы
graph = SemGraphFactory.create("memory_efficient")
processor = ProcessorFactory.create("graph_based", graph_provider=graph)

# Граф предоставляет дополнительные факты для вывода
context = ProcessingContext()
result = processor.derive(context)  # Использует структуру графа
```

### Работа с памятью

```python
from neurograph.memory import MemoryFactory

memory = MemoryFactory.create("biomorphic")
processor = ProcessorFactory.create("pattern_matching")

# Извлечение фактов из памяти
recent_memories = memory.get_recent_items(hours=24)
context = ProcessingContext()

for memory_item in recent_memories:
    # Преобразование воспоминаний в факты
    fact_key = f"memory_{memory_item.id}"
    context.add_fact(fact_key, memory_item.content)

result = processor.derive(context)
```

### Работа с векторными представлениями

```python
from neurograph.contextvec import ContextVectorsFactory

vectors = ContextVectorsFactory.create("dynamic")
processor = ProcessorFactory.create("pattern_matching")

# Семантический поиск правил
def find_semantic_rules(query_text: str, processor, vectors):
    query_vector = vectors.create_vector(query_text)
    similar_concepts = vectors.get_most_similar(query_text, top_n=10)
    
    # Поиск правил, связанных с похожими концептами
    relevant_rules = []
    for concept, similarity in similar_concepts:
        rules = processor.find_relevant_rules_by_concept(concept)
        relevant_rules.extend(rules)
    
    return relevant_rules
```

## 📚 Лучшие практики

### 1. Проектирование правил

```python
# ✅ Хорошо: специфичные, точные правила
rule = SymbolicRule(
    condition="собака И домашнее_животное",
    action="derive собака нуждается_в_уходе",
    confidence=0.9
)

# ❌ Плохо: слишком общие правила
rule = SymbolicRule(
    condition="животное",
    action="derive что-то",
    confidence=0.5
)
```

### 2. Управление уверенностью

```python
# Градация уверенности
CONFIDENCE_HIGH = 0.9      # Факты
CONFIDENCE_MEDIUM = 0.7    # Вероятные выводы
CONFIDENCE_LOW = 0.5       # Предположения
CONFIDENCE_GUESS = 0.3     # Догадки

rule = SymbolicRule(
    condition="температура > 37",
    action="derive возможна_лихорадка",
    confidence=CONFIDENCE_MEDIUM  # Не факт, но вероятно
)
```

### 3. Организация правил

```python
from neurograph.processor.utils import RuleManager

# Группировка правил по доменам
rule_manager = RuleManager()
rule_manager.create_collection("medical", "Медицинские правила")
rule_manager.create_collection("biological", "Биологические правила")

# Добавление правил в соответствующие коллекции
medical_rule = SymbolicRule(...)
rule_manager.add_rule_to_collection("medical", medical_rule)
```

### 4. Оптимизация производительности

```python
# Используйте профилировщик
from neurograph.processor.utils import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.start_profiling()

result = processor.derive(context)

execution_time = profiler.stop_profiling()
report = profiler.get_performance_report()

print(f"Время выполнения: {execution_time:.3f}с")
print(f"Наиболее используемые правила: {report['rule_usage']['most_used_rules']}")
```

## 🚨 Распространенные ошибки

### 1. Циклические зависимости в правилах

```python
# ❌ Проблема: циклические правила
rule1 = SymbolicRule(condition="A", action="derive B")
rule2 = SymbolicRule(condition="B", action="derive A")  # Цикл!

# ✅ Решение: используйте валидатор
validator = RuleValidator()
is_valid, errors = validator.validate_rule(rule2, existing_rules=[rule1])
if not is_valid:
    print(f"Ошибки валидации: {errors}")
```

### 2. Неэффективные правила

```python
# ❌ Проблема: слишком общие условия
rule = SymbolicRule(condition="X", action="derive Y")  # Срабатывает всегда

# ✅ Решение: специфичные условия
rule = SymbolicRule(
    condition="X является млекопитающим И X имеет_шерсть",
    action="derive X нуждается_в_груминге"
)
```

### 3. Неправильная обработка уверенности

```python
# ❌ Проблема: игнорирование уверенности
result = processor.derive(context)
for fact in result.derived_facts:
    print(fact)  # Не учитываем confidence

# ✅ Решение: фильтрация по уверенности
MINIMUM_CONFIDENCE = 0.7
for fact_key, fact_data in result.derived_facts.items():
    if fact_data['confidence'] >= MINIMUM_CONFIDENCE:
        print(f"{fact_key}: {fact_data['value']} ({fact_data['confidence']:.2f})")
```

---

## 📞 Поддержка

Для получения помощи по модулю Processor:

1. Изучите примеры в `neurograph/processor/examples/`
2. Запустите тесты: `python -m pytest neurograph/processor/tests/`
3. Проверьте логи для диагностики проблем
4. Используйте статистику процессора для мониторинга производительности

**Команда разработки NeuroGraph**