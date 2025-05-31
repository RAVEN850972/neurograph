# Руководство разработчика: Модуль Integration NeuroGraph

## Обзор модуля

Модуль Integration является центральным связующим звеном системы NeuroGraph, обеспечивающим унифицированный интерфейс для работы со всеми компонентами системы. Он реализует архитектуру, основанную на принципах инверсии зависимостей и слабой связанности компонентов.

### Основные задачи модуля

- **Оркестрация компонентов** - координация работы графа знаний, памяти, векторных представлений, NLP и логического вывода
- **Унификация интерфейсов** - предоставление единого API для различных типов обработки
- **Управление жизненным циклом** - инициализация, конфигурация и корректное завершение работы компонентов
- **Мониторинг и диагностика** - отслеживание состояния системы и производительности

## Архитектурные принципы

### Принцип единственной ответственности
Каждый класс имеет четко определенную задачу:
- `NeuroGraphEngine` - управление системой
- `ComponentProvider` - предоставление компонентов
- `Pipeline` классы - обработка определенных типов запросов
- `Adapter` классы - преобразование данных между компонентами

### Принцип инверсии зависимостей
Все компоненты зависят от абстракций (интерфейсов), а не от конкретных реализаций. Это обеспечивает гибкость и тестируемость системы.

### Ленивая инициализация
Компоненты создаются только при первом обращении к ним, что экономит ресурсы и ускоряет запуск системы.

## Основные компоненты

### NeuroGraphEngine
Центральный движок системы, который:
- Инициализирует и конфигурирует все компоненты
- Маршрутизирует запросы к соответствующим конвейерам обработки
- Собирает метрики производительности
- Обеспечивает корректное завершение работы

Движок поддерживает различные режимы работы через конфигурацию и может быть адаптирован под разные сценарии использования.

### ComponentProvider
Провайдер компонентов реализует паттерн "Сервис-локатор" и обеспечивает:
- Регистрацию компонентов с отложенной инициализацией
- Контроль состояния здоровья компонентов
- Изоляцию зависимостей между компонентами

### Конвейеры обработки (Pipelines)
Система конвейеров реализует паттерн "Pipeline" для обработки различных типов запросов:

#### TextProcessingPipeline
Предназначен для обработки произвольного текста:
- Анализ через NLP модуль
- Извлечение сущностей и отношений
- Добавление информации в граф знаний
- Создание векторных представлений
- Сохранение в память

#### QueryProcessingPipeline  
Обрабатывает запросы к системе знаний:
- Анализ запроса
- Поиск в графе знаний
- Семантический поиск в векторах
- Поиск в памяти
- Распространение активации
- Логический вывод
- Синтез итогового ответа

#### LearningPipeline
Специализирован на обучении системы:
- Базовая обработка через TextProcessingPipeline
- Создание правил логического вывода
- Обновление векторных индексов
- Консолидация памяти

#### InferencePipeline
Выполняет логический вывод:
- Анализ предпосылок
- Подготовка контекста для вывода
- Выполнение логических операций
- Распространение от выводов
- Генерация объяснений

### Адаптеры интеграции
Система адаптеров обеспечивает преобразование данных между различными компонентами:

#### GraphMemoryAdapter
Преобразует данные между графом знаний и системой памяти:
- Узлы графа → элементы памяти типа "концепт"
- Ребра графа → элементы памяти типа "отношение"
- Элементы памяти → структура графа

#### VectorProcessorAdapter  
Связывает векторные представления с процессором логического вывода:
- Векторные данные → контекст для процессора
- Правила процессора → векторные правила

#### NLPGraphAdapter
Интегрирует NLP модуль с графом знаний:
- Результаты NLP → обновления графа
- Данные графа → контекст для NLP

#### MemoryProcessorAdapter
Связывает память с процессором:
- Элементы памяти → контекст для вывода
- Результаты процессора → обновления памяти

### Система мониторинга
Комплексная система мониторинга включает:

#### IntegrationMetrics
Сбор и анализ метрик:
- Метрики запросов (количество, время, успешность)
- Метрики компонентов (операции, ошибки)
- Системные метрики (CPU, память)

#### HealthChecker
Проверка состояния компонентов:
- Базовые проверки доступности
- Специфичные проверки для каждого типа компонента
- Анализ времени ответа и частоты ошибок

#### ComponentMonitor
Мониторинг в реальном времени:
- Периодические проверки здоровья
- Система алертов с настраиваемыми порогами
- История алертов и трендов

## Конфигурационная система

### Типы конфигураций

#### Default (Стандартная)
Сбалансированная конфигурация для общего использования:
- Умеренное потребление ресурсов
- Включены все основные компоненты
- Настройки производительности по умолчанию

#### Lightweight (Облегченная)  
Минималистичная конфигурация для ограниченных ресурсов:
- Уменьшенные размеры памяти и кешей
- Отключены неосновные функции
- Снижены пороги качества для ускорения

#### Research (Исследовательская)
Максимальная функциональность для исследований:
- Увеличенные размеры памяти и глубина анализа
- Включены все продвинутые функции
- Детальное логирование и метрики

#### Production (Продакшн)
Оптимизированная конфигурация для продакшн-среды:
- Настройки производительности и надежности
- Включены мониторинг и алерты
- Настройки безопасности и валидации

### Управление конфигурациями
Система конфигураций поддерживает:
- Загрузку из JSON файлов
- Создание шаблонов конфигураций
- Валидацию параметров
- Значения по умолчанию

## Обработка данных

### Структура запросов
Система работает с унифицированными объектами `ProcessingRequest`, которые содержат:
- Контент для обработки
- Тип запроса (обработка текста, запрос, обучение, вывод)
- Параметры обработки (включение/отключение компонентов)
- Настройки качества и производительности

### Структура ответов  
Все ответы системы имеют единую структуру `ProcessingResponse`:
- Основной ответ в текстовом виде
- Структурированные данные от различных компонентов
- Метаинформация об обработке
- Объяснения и источники информации

### Потоки данных
Система реализует несколько основных потоков обработки данных:

1. **Поток обучения**: Текст → NLP → Граф → Векторы → Память
2. **Поток запроса**: Запрос → Поиск → Анализ → Вывод → Ответ  
3. **Поток вывода**: Предпосылки → Контекст → Правила → Выводы → Объяснения

## Расширение функциональности

### Добавление новых конвейеров
Для создания нового конвейера обработки:
1. Унаследуйтесь от `BasePipeline`
2. Реализуйте метод `process` с вашей логикой
3. Зарегистрируйте конвейер в движке

### Создание адаптеров
Для интеграции новых компонентов:
1. Унаследуйтесь от `BaseAdapter`  
2. Реализуйте метод `adapt` для преобразования данных
3. Определите поддерживаемые форматы

### Регистрация компонентов
Новые компоненты регистрируются через `ComponentProvider`:
- Для немедленной инициализации используйте `register_component`
- Для отложенной инициализации используйте `register_lazy_component`

## Обработка ошибок

### Стратегии обработки ошибок
Система реализует несколько уровней обработки ошибок:

#### Graceful Degradation
При сбое отдельных компонентов система продолжает работать с ограниченной функциональностью.

#### Circuit Breaker
Защита от каскадных сбоев через временное отключение проблемных компонентов.

#### Fallback Strategies  
Автоматическое переключение на альтернативные методы обработки при сбоях.

#### Retry Logic
Автоматические повторные попытки с экспоненциальной задержкой.

### Логирование ошибок
Комплексная система логирования включает:
- Структурированные логи с контекстом
- Различные уровни детализации
- Корреляция ошибок между компонентами

## Производительность и оптимизация

### Кеширование
Система включает многоуровневое кеширование:
- Кеширование результатов NLP обработки
- Кеширование векторных операций  
- Кеширование результатов поиска

### Параллелизация
Поддержка параллельной обработки:
- Многопоточная обработка запросов
- Асинхронные операции (в планах)
- Пакетная обработка для оптимизации

### Мониторинг производительности
Отслеживание ключевых метрик:
- Время ответа по компонентам
- Пропускная способность системы
- Использование ресурсов

## Тестирование

### Модульное тестирование
Каждый компонент имеет изолированные тесты:
- Тестирование интерфейсов через моки
- Проверка корректности обработки данных
- Валидация обработки ошибок

### Интеграционное тестирование  
Тестирование взаимодействия компонентов:
- Проверка корректности адаптеров
- Тестирование полных конвейеров обработки
- Валидация производительности

### Нагрузочное тестирование
Проверка системы под нагрузкой:
- Тестирование пропускной способности
- Проверка стабильности при пиковых нагрузках
- Валидация времени отклика

## Развертывание

### Режимы развертывания

#### Standalone
Автономное развертывание всех компонентов в одном процессе.

#### Distributed (планируется)
Распределенное развертывание с разделением компонентов по узлам.

### Конфигурация окружения
Настройка для различных сред:
- Development - отладочные настройки
- Testing - конфигурация для тестов  
- Production - оптимизированные настройки

### Мониторинг в продакшн
Система мониторинга для продакшн-среды:
- Health checks для load balancers
- Метрики для систем мониторинга
- Алерты для критических ситуаций

## Безопасность

### Валидация входных данных
Многоуровневая валидация:
- Проверка размера и формата входных данных
- Санитизация пользовательского ввода
- Защита от injection атак

### Управление доступом (планируется)
Система контроля доступа:
- Аутентификация пользователей
- Авторизация операций
- Аудит действий

### Rate Limiting
Защита от перегрузки:
- Ограничение частоты запросов
- Приоритизация запросов
- Защита от DDoS

## Лучшие практики

### Проектирование компонентов
- Следуйте принципам SOLID
- Используйте dependency injection
- Проектируйте для тестируемости

### Обработка данных
- Валидируйте все входные данные  
- Используйте типизированные интерфейсы
- Обеспечивайте идемпотентность операций

### Мониторинг и логирование
- Логируйте все критические операции
- Используйте структурированные логи
- Мониторьте ключевые метрики

### Обработка ошибок
- Обрабатывайте ошибки на всех уровнях
- Предоставляйте понятные сообщения об ошибках
- Реализуйте стратегии восстановления

## Roadmap развития

### Краткосрочные планы
- Реализация полной асинхронной обработки
- Улучшение системы кеширования
- Расширение мониторинга

### Долгосрочные планы  
- Распределенная архитектура
- Машинное обучение для оптимизации
- Продвинутая система безопасности

## Быстрый старт для разработчиков

### Установка и настройка

#### 1. Установка зависимостей
```bash
# Клонирование репозитория
git clone https://github.com/your-org/neurograph.git
cd neurograph

# Установка через pip
pip install -e .

# Или через poetry (рекомендуется)
poetry install
```

#### 2. Проверка установки
```python
# test_installation.py
from neurograph.integration import create_default_engine

try:
    engine = create_default_engine()
    print("✅ NeuroGraph Integration установлен корректно")
    engine.shutdown()
except Exception as e:
    print(f"❌ Ошибка установки: {e}")
```

### Первые шаги

#### Минимальный пример использования
```python
# app.py
from neurograph.integration import create_default_engine

def main():
    # Создание движка
    engine = create_default_engine()
    
    try:
        # Обучение системы
        engine.learn("Python - язык программирования для ИИ")
        
        # Запрос к системе
        response = engine.query("Что такое Python?")
        print(f"Ответ: {response.primary_response}")
        
    finally:
        # Корректное завершение
        engine.shutdown()

if __name__ == "__main__":
    main()
```

#### Интеграция в веб-приложение (Flask)
```python
# web_app.py
from flask import Flask, request, jsonify
from neurograph.integration import create_default_engine

app = Flask(__name__)
engine = create_default_engine()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    
    response = engine.query(question)
    
    return jsonify({
        'answer': response.primary_response,
        'confidence': response.confidence,
        'processing_time': response.processing_time
    })

@app.route('/learn', methods=['POST'])
def learn_content():
    data = request.get_json()
    content = data.get('content', '')
    
    response = engine.learn(content)
    
    return jsonify({
        'success': response.success,
        'message': response.primary_response
    })

# Graceful shutdown
import atexit
atexit.register(lambda: engine.shutdown())

if __name__ == '__main__':
    app.run(debug=True)
```

#### Интеграция в существующий проект Django
```python
# django_integration.py
from django.apps import AppConfig
from neurograph.integration import create_default_engine

class NeuroGraphConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'neurograph_app'
    
    def ready(self):
        # Инициализация при запуске Django
        from . import signals
        self.engine = create_default_engine()

# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.apps import apps
import json

@csrf_exempt
def neurograph_query(request):
    if request.method == 'POST':
        engine = apps.get_app_config('neurograph_app').engine
        data = json.loads(request.body)
        
        response = engine.query(data['question'])
        
        return JsonResponse({
            'answer': response.primary_response,
            'success': response.success
        })
```

### Интеграция с существующими системами

#### Интеграция с базой данных
```python
# database_integration.py
import sqlite3
from neurograph.integration import create_default_engine
from contextlib import contextmanager

class DatabaseKnowledgeSystem:
    def __init__(self, db_path='knowledge.db'):
        self.db_path = db_path
        self.engine = create_default_engine()
        self._init_database()
    
    def _init_database(self):
        with self.get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
    
    @contextmanager
    def get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def add_knowledge(self, content):
        with self.get_db_connection() as conn:
            conn.execute(
                'INSERT INTO knowledge_entries (content) VALUES (?)',
                (content,)
            )
    
    def process_pending_knowledge(self):
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                'SELECT id, content FROM knowledge_entries WHERE processed = FALSE'
            )
            
            for entry_id, content in cursor.fetchall():
                # Обучение NeuroGraph
                response = self.engine.learn(content)
                
                if response.success:
                    conn.execute(
                        'UPDATE knowledge_entries SET processed = TRUE WHERE id = ?',
                        (entry_id,)
                    )
    
    def query_knowledge(self, question):
        return self.engine.query(question)
    
    def shutdown(self):
        self.engine.shutdown()
```

#### Интеграция с API внешних сервисов
```python
# external_api_integration.py
import requests
from neurograph.integration import create_default_engine

class EnhancedKnowledgeSystem:
    def __init__(self):
        self.engine = create_default_engine()
        self.external_apis = {
            'wikipedia': 'https://en.wikipedia.org/api/rest_v1/page/summary/',
            'openai': 'https://api.openai.com/v1/completions'
        }
    
    def query_with_external_enrichment(self, question):
        # Сначала запрашиваем у NeuroGraph
        response = self.engine.query(question)
        
        # Если уверенность низкая, обращаемся к внешним источникам
        if response.confidence < 0.7:
            external_data = self._fetch_external_knowledge(question)
            if external_data:
                # Обучаем систему новыми данными
                self.engine.learn(external_data)
                # Повторный запрос
                response = self.engine.query(question)
        
        return response
    
    def _fetch_external_knowledge(self, query):
        # Простой пример обращения к Wikipedia
        try:
            # Извлекаем ключевые слова из запроса
            keywords = self._extract_keywords(query)
            
            for keyword in keywords:
                wiki_response = requests.get(
                    f"{self.external_apis['wikipedia']}{keyword}",
                    timeout=5
                )
                
                if wiki_response.status_code == 200:
                    data = wiki_response.json()
                    return data.get('extract', '')
                    
        except Exception as e:
            print(f"Ошибка получения внешних данных: {e}")
        
        return None
    
    def _extract_keywords(self, text):
        # Простое извлечение ключевых слов
        # В реальности здесь был бы более сложный NLP
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3]
        return keywords[:3]  # Первые 3 ключевых слова
```

### Кастомизация под ваши потребности

#### Создание пользовательского конвейера
```python
# custom_pipeline.py
from neurograph.integration.pipelines import BasePipeline
from neurograph.integration.base import ProcessingRequest, ProcessingResponse
import time

class CustomBusinessLogicPipeline(BasePipeline):
    """Пример пользовательского конвейера для бизнес-логики."""
    
    def __init__(self):
        super().__init__("custom_business")
        # Ваши настройки
        self.business_rules = self._load_business_rules()
    
    def process(self, request, provider):
        start_time = time.time()
        
        try:
            # Валидация запроса
            is_valid, error_msg = self.validate_request(request)
            if not is_valid:
                return self._create_response(request, False, error_message=error_msg)
            
            # Ваша бизнес-логика
            result = self._apply_business_logic(request.content, provider)
            
            processing_time = time.time() - start_time
            
            return self._create_response(
                request,
                success=True,
                primary_response=result['answer'],
                structured_data={'business_data': result}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка в бизнес-конвейере: {e}")
            return self._create_response(request, False, error_message=str(e))
    
    def _apply_business_logic(self, content, provider):
        # Пример: анализ тональности для бизнеса
        nlp = provider.get_component('nlp')
        nlp_result = nlp.process_text(content)
        
        # Ваша логика анализа
        sentiment = self._analyze_business_sentiment(nlp_result)
        recommendations = self._generate_recommendations(sentiment)
        
        return {
            'answer': f"Анализ завершен. Тональность: {sentiment}",
            'sentiment': sentiment,
            'recommendations': recommendations
        }
    
    def _load_business_rules(self):
        # Загрузка ваших бизнес-правил
        return {
            'positive_keywords': ['отлично', 'хорошо', 'успех'],
            'negative_keywords': ['плохо', 'ошибка', 'проблема']
        }
    
    def _analyze_business_sentiment(self, nlp_result):
        # Ваш анализ тональности
        positive_count = 0
        negative_count = 0
        
        for entity in nlp_result.entities:
            if entity.text.lower() in self.business_rules['positive_keywords']:
                positive_count += 1
            elif entity.text.lower() in self.business_rules['negative_keywords']:
                negative_count += 1
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _generate_recommendations(self, sentiment):
        recommendations = {
            'positive': ['Продолжайте в том же духе', 'Расширьте успешные практики'],
            'negative': ['Проанализируйте проблемы', 'Примите корректирующие меры'],
            'neutral': ['Мониторьте ситуацию', 'Ищите точки роста']
        }
        return recommendations.get(sentiment, [])

# Регистрация пользовательского конвейера
def register_custom_pipeline(engine):
    custom_pipeline = CustomBusinessLogicPipeline()
    engine.pipelines['custom_business'] = custom_pipeline
    return engine
```

#### Создание пользовательского провайдера компонентов
```python
# custom_provider.py
from neurograph.integration.engine import ComponentProvider

class CustomComponentProvider(ComponentProvider):
    """Расширенный провайдер с поддержкой внешних сервисов."""
    
    def __init__(self):
        super().__init__()
        self.external_services = {}
    
    def register_external_service(self, service_name, service_config):
        """Регистрация внешнего сервиса."""
        self.external_services[service_name] = service_config
        
        # Ленивая инициализация внешнего сервиса
        def init_external_service():
            return self._create_external_service(service_name, service_config)
        
        self.register_lazy_component(f"external_{service_name}", init_external_service)
    
    def _create_external_service(self, service_name, config):
        # Фабрика для создания различных внешних сервисов
        if config['type'] == 'rest_api':
            return RestAPIService(config)
        elif config['type'] == 'database':
            return DatabaseService(config)
        else:
            raise ValueError(f"Неизвестный тип сервиса: {config['type']}")

class RestAPIService:
    def __init__(self, config):
        self.base_url = config['base_url']
        self.headers = config.get('headers', {})
    
    def query(self, endpoint, params=None):
        import requests
        response = requests.get(f"{self.base_url}/{endpoint}", 
                              params=params, headers=self.headers)
        return response.json()

class DatabaseService:
    def __init__(self, config):
        self.connection_string = config['connection_string']
    
    def execute_query(self, query, params=None):
        # Выполнение SQL запроса
        pass
```

### Сценарии использования

#### Чат-бот для customer support
```python
# chatbot_integration.py
from neurograph.integration import create_default_engine, ProcessingRequest

class CustomerSupportBot:
    def __init__(self):
        self.engine = create_default_engine()
        self._load_knowledge_base()
        self.conversation_context = {}
    
    def _load_knowledge_base(self):
        """Загрузка базы знаний для поддержки."""
        knowledge_items = [
            "Для сброса пароля перейдите в настройки аккаунта",
            "Техподдержка работает с 9:00 до 18:00 в будние дни",
            "Возврат товара возможен в течение 14 дней с момента покупки",
            # ... другие элементы базы знаний
        ]
        
        for item in knowledge_items:
            self.engine.learn(item)
    
    def handle_user_message(self, user_id, message):
        # Получение контекста беседы
        context = self.conversation_context.get(user_id, {})
        
        # Создание запроса с контекстом
        request = ProcessingRequest(
            content=message,
            request_type="query",
            context=context,
            response_format="conversational"
        )
        
        response = self.engine.process_request(request)
        
        # Обновление контекста
        self._update_conversation_context(user_id, message, response.primary_response)
        
        return response.primary_response
    
    def _update_conversation_context(self, user_id, user_message, bot_response):
        if user_id not in self.conversation_context:
            self.conversation_context[user_id] = {'history': []}
        
        self.conversation_context[user_id]['history'].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': time.time()
        })
        
        # Ограничиваем историю
        if len(self.conversation_context[user_id]['history']) > 10:
            self.conversation_context[user_id]['history'] = \
                self.conversation_context[user_id]['history'][-5:]
```

#### Система анализа документов
```python
# document_analyzer.py
from neurograph.integration import create_research_engine
import os

class DocumentAnalyzer:
    def __init__(self, output_dir='analysis_results'):
        self.engine = create_research_engine()  # Максимальная функциональность
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_document(self, file_path, document_type='general'):
        """Анализ одного документа."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Обучение системы на документе
        learn_response = self.engine.learn(content)
        
        # Анализ документа
        analysis_queries = [
            "Какие основные темы обсуждаются в документе?",
            "Какие ключевые концепции упоминаются?",
            "Какие отношения между концептами можно выделить?"
        ]
        
        analysis_results = {}
        for query in analysis_queries:
            response = self.engine.query(query)
            analysis_results[query] = {
                'answer': response.primary_response,
                'confidence': response.confidence,
                'structured_data': response.structured_data
            }
        
        # Сохранение результатов
        self._save_analysis_results(file_path, analysis_results)
        
        return analysis_results
    
    def batch_analyze_documents(self, documents_dir):
        """Пакетный анализ документов."""
        results = {}
        
        for filename in os.listdir(documents_dir):
            if filename.endswith(('.txt', '.md', '.doc')):
                file_path = os.path.join(documents_dir, filename)
                try:
                    results[filename] = self.analyze_document(file_path)
                    print(f"✅ Проанализирован: {filename}")
                except Exception as e:
                    print(f"❌ Ошибка анализа {filename}: {e}")
                    results[filename] = {'error': str(e)}
        
        return results
    
    def _save_analysis_results(self, original_file, results):
        import json
        filename = os.path.basename(original_file)
        output_file = os.path.join(self.output_dir, f"{filename}_analysis.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
```

### Мониторинг и отладка

#### Настройка логирования
```python
# logging_setup.py
import logging
from neurograph.core.logging import setup_logging

def setup_custom_logging():
    """Настройка логирования для вашего приложения."""
    
    # Базовая настройка
    setup_logging(
        level="INFO",
        log_file="neurograph_app.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Дополнительные настройки для конкретных компонентов
    logging.getLogger("neurograph.integration").setLevel(logging.DEBUG)
    logging.getLogger("neurograph.nlp").setLevel(logging.WARNING)
    
    # Настройка для внешних библиотек
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

# Использование в приложении
setup_custom_logging()
```

#### Система мониторинга
```python
# monitoring_setup.py
from neurograph.integration import ComponentMonitor, HealthChecker
import time
import threading

class ApplicationMonitor:
    def __init__(self, engine):
        self.engine = engine
        self.monitor = ComponentMonitor(check_interval=30.0)
        self.health_checker = HealthChecker()
        self.monitoring_thread = None
        self.running = False
    
    def start_monitoring(self):
        """Запуск мониторинга в отдельном потоке."""
        self.running = True
        self.monitor.start_monitoring(self.engine.provider)
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        print("🔍 Мониторинг запущен")
    
    def stop_monitoring(self):
        """Остановка мониторинга."""
        self.running = False
        self.monitor.stop_monitoring()
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        print("🛑 Мониторинг остановлен")
    
    def _monitoring_loop(self):
        """Основной цикл мониторинга."""
        while self.running:
            try:
                # Проверка здоровья
                health = self.health_checker.get_overall_health()
                
                if health['status'] == 'critical':
                    self._handle_critical_status(health)
                
                # Получение метрик
                metrics = self.monitor.get_monitoring_report()
                self._log_metrics(metrics)
                
                time.sleep(30)  # Интервал проверки
                
            except Exception as e:
                print(f"❌ Ошибка мониторинга: {e}")
                time.sleep(60)  # Увеличенный интервал при ошибке
    
    def _handle_critical_status(self, health):
        """Обработка критического состояния системы."""
        print(f"🚨 КРИТИЧЕСКОЕ СОСТОЯНИЕ: {health}")
        
        # Здесь можно добавить:
        # - Отправку уведомлений
        # - Автоматическое восстановление
        # - Запись в лог критических событий
    
    def _log_metrics(self, metrics):
        """Логирование метрик."""
        summary = metrics['metrics_summary']
        print(f"📊 Метрики: Запросов/мин: {summary['requests']['requests_per_minute']:.1f}, "
              f"Успешность: {summary['requests']['success_rate']:.1%}")
    
    def get_status_dashboard(self):
        """Получение данных для дашборда."""
        return {
            'health': self.health_checker.get_overall_health(),
            'metrics': self.monitor.get_dashboard_data(),
            'alerts': self.monitor.get_monitoring_report()['recent_alerts'][-5:]
        }
```

### Производственное развертывание

#### Docker интеграция
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY . .

# Установка NeuroGraph
RUN pip install -e .

# Переменные окружения
ENV NEUROGRAPH_ENV=production
ENV NEUROGRAPH_LOG_LEVEL=INFO

# Экспорт порта
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from app import health_check; health_check()" || exit 1

# Запуск приложения
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  neurograph-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEUROGRAPH_ENV=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/neurograph
    volumes:
      - ./config:/app/config
      - neurograph_data:/app/data
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: neurograph
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  redis:
    image: redis:6-alpine
    restart: unless-stopped

volumes:
  neurograph_data:
  postgres_data:
```

#### Конфигурация для продакшна
```python
# production_config.py
import os
from neurograph.integration import IntegrationConfig

def create_production_config():
    """Создание конфигурации для продакшна."""
    
    return IntegrationConfig(
        engine_name="production_neurograph",
        components={
            "semgraph": {
                "type": "persistent",
                "params": {
                    "file_path": os.getenv("NEUROGRAPH_GRAPH_PATH", "/app/data/graph.json"),
                    "auto_save_interval": float(os.getenv("GRAPH_SAVE_INTERVAL", "600"))
                }
            },
            "memory": {
                "params": {
                    "stm_capacity": int(os.getenv("STM_CAPACITY", "200")),
                    "ltm_capacity": int(os.getenv("LTM_CAPACITY", "20000")),
                    "use_semantic_indexing": True,
                    "auto_consolidation": True
                }
            },
            "nlp": {
                "params": {
                    "language": os.getenv("DEFAULT_LANGUAGE", "ru"),
                    "confidence_threshold": float(os.getenv("NLP_CONFIDENCE", "0.6"))
                }
            }
        },
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "50")),
        default_timeout=float(os.getenv("DEFAULT_TIMEOUT", "45.0")),
        enable_caching=True,
        cache_ttl=int(os.getenv("CACHE_TTL", "900")),
        enable_metrics=True,
        enable_health_checks=True,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_input_length=int(os.getenv("MAX_INPUT_LENGTH", "20000"))
    )
```

### Решение типичных проблем

#### Обработка ошибок и восстановление
```python
# error_handling.py
from neurograph.integration import create_default_engine
import time
import logging

class ResilientNeuroGraphWrapper:
    """Обертка для устойчивой работы с NeuroGraph."""
    
    def __init__(self, max_retries=3, retry_delay=1.0):
        self.engine = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Инициализация движка с повторными попытками."""
        for attempt in range(self.max_retries):
            try:
                self.engine = create_default_engine()
                self.logger.info("✅ NeuroGraph движок инициализирован")
                return
            except Exception as e:
                self.logger.error(f"❌ Попытка {attempt + 1} неуспешна: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Экспоненциальная задержка
        
        raise RuntimeError("Не удалось инициализировать NeuroGraph")
    
    def safe_query(self, question, fallback_response="Извините, временные технические проблемы"):
        """Безопасный запрос с fallback."""
        for attempt in range(self.max_retries):
            try:
                response = self.engine.query(question)
                if response.success:
                    return response.primary_response
                else:
                    self.logger.warning(f"Неуспешный ответ: {response.error_message}")
            
            except Exception as e:
                self.logger.error(f"Ошибка запроса (попытка {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    # Попытка переинициализации движка
                    try:
                        self.engine.shutdown()
                        self._initialize_engine()
                    except:
                        pass
        
        return fallback_response
    
    def safe_learn(self, content):
        """Безопасное обучение."""
        try:
            response = self.engine.learn(content)
            return response.success
        except Exception as e:
            self.logger.error(f"Ошибка обучения: {e}")
            return False
    
    def health_check(self):
        """Проверка состояния системы."""
        try:
            health = self.engine.get_health_status()
            return health.get('overall_status') == 'healthy'
        except:
            return False
    
    def shutdown(self):
        """Корректное завершение работы."""
        if self.engine:
            try:
                self.engine.shutdown()
            except Exception as e:
                self.logger.error(f"Ошибка при завершении: {e}")
```

#### Оптимизация производительности
```python
# performance_optimization.py
from neurograph.integration import create_lightweight_engine, IntegrationConfig
import functools
import time

class PerformanceOptimizedNeuroGraph:
    """Оптимизированная версия для высокой производительности."""
    
    def __init__(self):
        # Конфигурация для производительности
        config = IntegrationConfig(
            engine_name="performance_optimized",
            components={
                "memory": {
                    "params": {
                        "stm_capacity": 50,  # Уменьшена для скорости
                        "ltm_capacity": 5000,
                        