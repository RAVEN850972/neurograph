# NeuroGraph Integration Module

Модуль интеграции системы NeuroGraph - связующий слой между всеми компонентами.

## Основные возможности

- 🔗 **Унифицированный интерфейс** для работы со всеми модулями
- ⚡ **Конвейеры обработки** для различных типов задач
- 🔄 **Адаптеры интеграции** между компонентами
- 📊 **Мониторинг и метрики** системы
- ⚙️ **Гибкая конфигурация** под различные сценарии

## Быстрый старт

```python
from neurograph.integration import create_default_engine

# Создание движка
engine = create_default_engine()

# Обработка текста
response = engine.process_text("Python - это язык программирования")
print(response.primary_response)

# Запрос к системе знаний
response = engine.query("Что такое Python?")
print(response.primary_response)

# Обучение системы
response = engine.learn("Django - веб-фреймворк для Python")

# Завершение работы
engine.shutdown()
```

## Компоненты модуля

### 1. Движок NeuroGraph (engine.py)
Основной движок системы, координирующий работу всех компонентов.

### 2. Конвейеры обработки (pipelines.py)
- **TextProcessingPipeline** - обработка произвольного текста
- **QueryProcessingPipeline** - обработка запросов к знаниям
- **LearningPipeline** - обучение системы
- **InferencePipeline** - логический вывод

### 3. Адаптеры (adapters.py)
- **GraphMemoryAdapter** - между графом и памятью
- **VectorProcessorAdapter** - между векторами и процессором
- **NLPGraphAdapter** - между NLP и графом
- **MemoryProcessorAdapter** - между памятью и процессором

### 4. Утилиты (utils.py)
- **IntegrationMetrics** - сбор метрик
- **HealthChecker** - проверка здоровья компонентов
- **ComponentMonitor** - мониторинг в реальном времени

## Конфигурации

### Стандартная конфигурация
```python
from neurograph.integration import create_default_engine
engine = create_default_engine()
```

### Облегченная конфигурация
```python
from neurograph.integration import create_lightweight_engine
engine = create_lightweight_engine()  # Для ограниченных ресурсов
```

### Исследовательская конфигурация
```python
from neurograph.integration import create_research_engine
engine = create_research_engine()  # Максимальная функциональность
```

### Пользовательская конфигурация
```python
from neurograph.integration import IntegrationConfig, EngineFactory

config = IntegrationConfig(
    engine_name="my_engine",
    components={
        "nlp": {"params": {"language": "en"}},
        "memory": {"params": {"stm_capacity": 200}}
    },
    max_concurrent_requests=20
)

engine = EngineFactory.create("default", config)
```

## Мониторинг системы

```python
from neurograph.integration import ComponentMonitor, HealthChecker

# Создание монитора
monitor = ComponentMonitor()
monitor.start_monitoring(engine.provider)

# Проверка здоровья
health_checker = HealthChecker()
health = health_checker.check_all_components(engine.provider)
print(health_checker.get_health_summary())

# Получение метрик
report = monitor.get_monitoring_report()
dashboard_data = monitor.get_dashboard_data()
```

## Обработка различных типов запросов

### Обучение системы
```python
request = ProcessingRequest(
    content="Новая информация для изучения",
    request_type="learning",
    enable_graph_reasoning=True,
    enable_memory_search=True
)
response = engine.process_request(request)
```

### Поиск в знаниях
```python
request = ProcessingRequest(
    content="Что я знаю о машинном обучении?",
    request_type="query",
    response_format="conversational",
    max_results=5
)
response = engine.process_request(request)
```

### Логический вывод
```python
request = ProcessingRequest(
    content="Если Python - язык программирования, то что можно с ним делать?",
    request_type="inference",
    explanation_level="detailed"
)
response = engine.process_request(request)
```

## Интеграция компонентов

Модуль обеспечивает прозрачную интеграцию между всеми компонентами:

1. **NLP → SemGraph**: Сущности и отношения → узлы и ребра графа
2. **SemGraph → Memory**: Знания графа → элементы памяти
3. **Memory → Processor**: Факты из памяти → контекст для вывода
4. **Processor → Propagation**: Выводы → активация связанных концептов
5. **All → ContextVec**: Векторизация для семантического поиска

## Производительность

- **Многопоточность**: Параллельная обработка запросов
- **Кеширование**: Умное кеширование результатов
- **Мониторинг**: Отслеживание производительности в реальном времени
- **Оптимизация**: Автоматическая настройка под нагрузку

## Расширяемость

Модуль легко расширяется новыми компонентами:

```python
# Регистрация нового типа движка
EngineFactory.register_engine("my_engine", MyCustomEngine)

# Добавление нового конвейера
class CustomPipeline(BasePipeline):
    def process(self, request, provider):
        # Ваша логика
        pass

# Создание пользовательского адаптера
class CustomAdapter(BaseAdapter):
    def adapt(self, source_data, target_format):
        # Ваша адаптация
        pass
```

## Примеры использования

См. файлы в директории `examples/`:
- `basic_usage.py` - базовые примеры
- `advanced_integration.py` - продвинутая интеграция
- `performance_testing.py` - тесты производительности

## Развертывание

### Разработка
```python
from neurograph.integration.config import IntegrationConfigManager

config_manager = IntegrationConfigManager()
dev_config = config_manager.create_template_config("default")
```

### Продакшн
```python
prod_config = config_manager.create_template_config("production")
engine = EngineFactory.create_from_config(prod_config)
```

## Логирование и отладка

```python
# Настройка логирования
from neurograph.core.logging import setup_logging
setup_logging(level="DEBUG", log_file="integration.log")

# Отладочная информация
health_status = engine.get_health_status()
print(f"Статус: {health_status['overall_status']}")

# Детальные метрики
metrics = monitor.metrics.get_detailed_report()
print(metrics)
```

## Обработка ошибок

Модуль включает комплексную обработку ошибок:
- Автоматические fallback стратегии
- Circuit breaker для защиты от каскадных сбоев
- Детальное логирование ошибок
- Graceful degradation при проблемах в компонентах

## Тестирование

```bash
# Запуск базовых примеров
python -m neurograph.integration.examples.basic_usage

# Тесты производительности
python -m neurograph.integration.examples.performance_testing

# Продвинутые сценарии
python -m neurograph.integration.examples.advanced_integration
```

---

Модуль Integration является сердцем системы NeuroGraph, обеспечивая 
seamless интеграцию всех компонентов и предоставляя удобный 
интерфейс для разработчиков и конечных пользователей.