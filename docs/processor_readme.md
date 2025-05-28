# NeuroGraph Processor Module

Модуль нейросимволического процессора для логического вывода и рассуждений в системе NeuroGraph.

## Обзор

Модуль Processor реализует механизмы символического и гибридного логического вывода, позволяя системе делать заключения на основе правил и фактов. Он сочетает классические подходы экспертных систем с современными методами обработки знаний.

## Ключевые особенности

- **Символические правила** - поддержка правил вида "если-то" с уверенностью
- **Логический вывод** - прямой и обратный вывод с контролем глубины
- **Интеграция с графом** - использование структуры графа знаний для вывода
- **Объяснения** - генерация понятных объяснений для выводов
- **Производительность** - кеширование, индексация и оптимизация
- **Гибкость** - поддержка различных типов правил и стратегий

## Архитектура

```
processor/
├── base.py                     # Базовые интерфейсы и классы
├── factory.py                  # Фабрика процессоров
├── impl/
│   ├── pattern_matching.py    # Процессор сопоставления шаблонов
│   └── graph_based.py          # Процессор на основе графа
├── examples/
│   └── basic_usage.py          # Примеры использования
└── tests/                      # Тесты модуля
```

## Основные компоненты

### SymbolicRule
Представляет символическое правило для логического вывода:

```python
from neurograph.processor import SymbolicRule

rule = SymbolicRule(
    condition="собака является животным",
    action="derive собака нуждается в пище",
    confidence=0.9,
    priority=1
)
```

### ProcessingContext
Контекст для хранения фактов и переменных:

```python
from neurograph.processor import ProcessingContext

context = ProcessingContext()
context.add_fact("собака_is_a_животным", True, 0.9)
context.set_variable("X", "собака")
```

### INeuroSymbolicProcessor
Базовый интерфейс для всех процессоров:

```python
from neurograph.processor import INeuroSymbolicProcessor

class MyProcessor(INeuroSymbolicProcessor):
    def derive(self, context, depth=1):
        # Реализация логического вывода
        pass
```

## Быстрый старт

### Создание процессора

```python
from neurograph.processor import ProcessorFactory

# Создание базового процессора
processor = ProcessorFactory.create("pattern_matching")

# Создание с настройками
processor = ProcessorFactory.create("pattern_matching",
                                  confidence_threshold=0.7,
                                  max_depth=5,
                                  enable_explanations=True)
```

### Добавление правил

```python
from neurograph.processor import SymbolicRule

# Простое правило
rule = SymbolicRule(
    condition="собака является животным",
    action="derive собака нуждается в пище",
    confidence=0.9
)

rule_id = processor.add_rule(rule)
```

### Выполнение логического вывода

```python
from neurograph.processor import ProcessingContext

# Создание контекста с фактами
context = ProcessingContext()
context.add_fact("собака_is_a_животным", True, 0.95)

# Выполнение вывода
result = processor.derive(context, depth=2)

print(f"Успех: {result.success}")
print(f"Уверенность: {result.confidence}")
print(f"Выведенные факты: {list(result.derived_facts.keys())}")

# Объяснение
for step in result.explanation:
    print(f"Шаг: {step.reasoning}")
```

## Типы процессоров

### PatternMatchingProcessor
Процессор на основе сопоставления шаблонов:

```python
processor = ProcessorFactory.create("pattern_matching",
                                  confidence_threshold=0.5,
                                  enable_explanations=True,
                                  cache_rules=True)
```

**Особенности:**
- Быстрая обработка простых правил
- Поддержка логических операторов (И, ИЛИ, НЕ)
- Индексация правил для производительности
- Кеширование результатов

### GraphBasedProcessor
Процессор с интеграцией графа знаний:

```python
# Требует провайдер графа знаний
processor = ProcessorFactory.create("graph_based",
                                  graph_provider=graph,
                                  use_graph_structure=True,
                                  path_search_limit=100)
```

**Особенности:**
- Использование структуры графа для вывода
- Поиск транзитивных отношений
- Кеширование путей в графе
- Семантические связи между узлами

## Паттерны правил

### Базовые шаблоны

```python
# Отношение "является"
"собака является животным"
"X является Y"

# Свойства
"собака имеет свойство лай"
"X имеет свойство Y"

# Связи
"собака связан с хозяином"
"X связан с Y через Z"
```

### Логические операторы

```python
# Логическое И
"A И B"

# Логическое ИЛИ  
"A ИЛИ B"

# Отрицание
"НЕ A"
```

### Действия

```python
# Утверждение фактов
"assert собака является животным"

# Вывод новых фактов
"derive собака нуждается в пище"

# Выполнение функций
"execute print(результат)"
```

## Примеры использования

### Медицинская диагностика

```python
# Медицинские правила
rules = [
    SymbolicRule(
        condition="пациент имеет свойство температура",
        action="derive пациент имеет свойство лихорадка",
        confidence=0.8
    ),
    SymbolicRule(
        condition="пациент имеет свойство кашель",
        action="derive пациент имеет свойство респираторные_симптомы",
        confidence=0.7
    )
]

for rule in rules:
    processor.add_rule(rule)

# Симптомы пациента
context = ProcessingContext()
context.add_fact("пациент_has_температура", True, 0.9)
context.add_fact("пациент_has_кашель", True, 0.8)

# Диагностика
result = processor.derive(context, depth=2)
```

### Классификация животных

```python
# Правила классификации
hierarchy_rules = [
    SymbolicRule(
        condition="собака является млекопитающим",
        action="derive собака является животным",
        confidence=1.0
    ),
    SymbolicRule(
        condition="млекопитающее является животным",
        action="derive млекопитающее является живым",
        confidence=1.0
    ),
    SymbolicRule(
        condition="живое дышит",
        action="derive живое потребляет кислород",
        confidence=0.9
    )
]
```

### Бизнес-правила

```python
# Правила для обработки заказов
business_rules = [
    SymbolicRule(
        condition="клиент имеет свойство VIP",
        action="derive заказ получает приоритет_высокий",
        confidence=1.0
    ),
    SymbolicRule(
        condition="заказ имеет приоритет_высокий",
        action="derive заказ обрабатывается в_первую_очередь",
        confidence=0.9
    )
]
```

## Управление правилами

### Экспорт и импорт

```python
# Экспорт правил
exported_rules = processor.export_rules()

# Сохранение в файл
import json
with open("rules.json", "w") as f:
    json.dump(exported_rules, f, ensure_ascii=False, indent=2)

# Загрузка из файла
with open("rules.json", "r") as f:
    loaded_rules = json.load(f)

# Импорт правил
processor.clear_rules()
imported_count = processor.import_rules(loaded_rules)
```

### Обновление правил

```python
# Обновление уверенности и приоритета
processor.update_rule(rule_id, 
                     confidence=0.95, 
                     priority=10)

# Получение правила
rule = processor.get_rule(rule_id)
print(f"Уверенность: {rule.confidence}")
print(f"Использований: {rule.usage_count}")
```

## Оптимизация производительности

### Индексация правил

```python
processor = ProcessorFactory.create("pattern_matching",
                                  rule_indexing=True,  # Включить индексацию
                                  cache_rules=True)    # Включить кеширование
```

### Параллельная обработка

```python
processor = ProcessorFactory.create("pattern_matching",
                                  parallel_processing=True,
                                  confidence_threshold=0.3)
```

### Мониторинг производительности

```python
# Получение статистики
stats = processor.get_statistics()

print(f"Правил в базе: {stats['rules_count']}")
print(f"Правил выполнено: {stats['rules_executed']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Среднее время выполнения: {stats['average_execution_time']:.3f}с")
```

## Интеграция с другими модулями

### С модулем Memory

```python
# Факты из памяти
memory_facts = memory.search("животные", limit=10)

context = ProcessingContext()
for item_id, similarity in memory_facts:
    memory_item = memory.get(item_id)
    if memory_item:
        fact_key = f"fact_{item_id}"
        context.add_fact(fact_key, memory_item.content, similarity)

result = processor.derive(context)
```

### С модулем SemGraph

```python
# Использование GraphBasedProcessor с реальным графом
from neurograph.semgraph import SemGraphFactory

graph = SemGraphFactory.create("memory_efficient")
processor = ProcessorFactory.create("graph_based", 
                                  graph_provider=graph)

# Правила будут использовать структуру графа
result = processor.derive(context, depth=3)
```

### С модулем NLP

```python
# Извлечение правил из текста
def extract_rules_from_text(text):
    # Простой парсер правил из естественного языка
    if "если" in text and "то" in text:
        parts = text.split("то")
        condition = parts[0].replace("если", "").strip()
        action = f"derive {parts[1].strip()}"
        
        return SymbolicRule(condition=condition, action=action)
    return None

# Использование
rule_text = "если собака является животным то собака нуждается в пище"
rule = extract_rules_from_text(rule_text)
if rule:
    processor.add_rule(rule)
```

## Обработка ошибок

```python
from neurograph.processor.base import (
    ProcessorError,
    RuleValidationError,
    RuleExecutionError,
    DerivationError
)

try:
    # Добавление невалидного правила
    invalid_rule = SymbolicRule(condition="", action="действие")
    processor.add_rule(invalid_rule)
    
except RuleValidationError as e:
    print(f"Ошибка валидации: {e.message}")
    
except ProcessorError as e:
    print(f"Общая ошибка процессора: {e.message}")
```

## Тестирование

```python
# Запуск тестов
python -m pytest neurograph/processor/tests/ -v

# Запуск конкретного теста
python -m pytest neurograph/processor/tests/test_pattern_matching.py::TestPatternMatchingProcessor::test_derive_single_step -v

# Запуск с покрытием
python -m pytest neurograph/processor/tests/ --cov=neurograph.processor --cov-report=html
```

## Конфигурация

### Через конфигурационный файл

```json
{
  "processor": {
    "type": "pattern_matching",
    "config": {
      "confidence_threshold": 0.6,
      "max_depth": 7,
      "enable_explanations": true,
      "cache_rules": true,
      "rule_indexing": true,
      "parallel_processing": false
    }
  }
}
```

```python
# Загрузка из конфигурации
import json

with open("config.json") as f:
    config = json.load(f)

processor = ProcessorFactory.create_from_config(config["processor"])
```

### Готовые конфигурации

```python
from neurograph.processor.factory import (
    create_default_processor,
    create_high_performance_processor
)

# Базовая конфигурация
processor = create_default_processor()

# Высокопроизводительная конфигурация
fast_processor = create_high_performance_processor()
```

## Расширение

### Создание собственного процессора

```python
from neurograph.processor.base import INeuroSymbolicProcessor

class CustomProcessor(INeuroSymbolicProcessor):
    def __init__(self, custom_param=None):
        self.custom_param = custom_param
        self._rules = {}
    
    def derive(self, context, depth=1):
        # Ваша логика вывода
        result = DerivationResult(success=True)
        return result
    
    # Реализация остальных методов интерфейса
    ...

# Регистрация в фабрике
ProcessorFactory.register_processor("custom", CustomProcessor)

# Использование
processor = ProcessorFactory.create("custom", custom_param="value")
```

## Лучшие практики

1. **Используйте ясные имена** для условий и действий правил
2. **Группируйте правила** по доменам и приоритетам
3. **Тестируйте цепочки вывода** различной глубины
4. **Мониторьте производительность** при большом количестве правил
5. **Документируйте сложные правила** через метаданные
6. **Валидируйте правила** перед добавлением в продакшн
7. **Используйте объяснения** для отладки и презентации результатов

## Ограничения

- Глубина вывода ограничена настройкой `max_depth`
- Производительность зависит от количества правил и фактов
- Текущая реализация не поддерживает вероятностную логику
- Парсер условий и действий поддерживает ограниченный набор шаблонов

## Дальнейшее развитие

Планируемые улучшения:
- Поддержка темпоральной логики
- Нечеткая логика и вероятностные рассуждения
- Машинное обучение правил из примеров
- Распределенная обработка правил
- Визуализация процесса вывода
- Интеграция с внешними системами рассуждений

## Вклад в проект

Для участия в разработке модуля Processor:

1. Изучите существующую архитектуру
2. Добавьте тесты для новой функциональности
3. Следуйте стандартам кодирования проекта
4. Обновите документацию при изменениях API
5. Создайте pull request с подробным описанием изменений

## Лицензия

Модуль Processor является частью проекта NeuroGraph и распространяется под той же лицензией.