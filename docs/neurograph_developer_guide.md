
## Практическое руководство разработчика

---

# 📋 Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Установка и настройка](#установка-и-настройка)
3. [Основные концепции](#основные-концепции)
4. [API Reference](#api-reference)
5. [Практические примеры](#практические-примеры)
6. [Конфигурация](#конфигурация)
7. [Интеграция](#интеграция)
8. [Расширение системы](#расширение-системы)
9. [Развертывание](#развертывание)
10. [Troubleshooting](#troubleshooting)

---

# 🚀 Быстрый старт

## 30-секундный пример

```python
from neurograph.integration import create_default_engine

# Создание движка NeuroGraph
engine = create_default_engine()

# Обучение системы
engine.learn("Python - это язык программирования для ИИ")

# Запрос к системе
response = engine.query("Что такое Python?")
print(response.primary_response)
# Output: "Python - это язык программирования, который широко используется для разработки искусственного интеллекта..."

# Корректное завершение
engine.shutdown()
```

## Что вы получите

✅ **Полнофункциональный ИИ-ассистент** за 4 строки кода  
✅ **Автоматическое извлечение знаний** из текста  
✅ **Интеллектуальные ответы** на вопросы  
✅ **Биоморфную память** с консолидацией знаний  
✅ **Семантический поиск** и ассоциативные связи  

---

# 📦 Установка и настройка

## Системные требования

### Минимальные требования
- **Python**: 3.8+
- **RAM**: 2GB
- **CPU**: 2 ядра
- **Диск**: 1GB свободного места

### Рекомендуемые требования
- **Python**: 3.11+
- **RAM**: 8GB
- **CPU**: 4+ ядра
- **Диск**: 5GB свободного места
- **GPU**: для больших языковых моделей (опционально)

## Установка

### Способ 1: pip (рекомендуется)
```bash
# Базовая установка
pip install neurograph

# С дополнительными зависимостями
pip install neurograph[full]

# Для разработки
pip install neurograph[dev]
```

### Способ 2: из исходников
```bash
git clone https://github.com/neurograph/neurograph.git
cd neurograph
pip install -e .
```

### Способ 3: Docker
```bash
docker pull neurograph/neurograph:latest
docker run -p 8080:8080 neurograph/neurograph:latest
```

## Проверка установки

```python
# test_installation.py
from neurograph.integration import create_default_engine
from neurograph.core import get_version

print(f"NeuroGraph версия: {get_version()}")

# Быстрый тест функциональности
engine = create_default_engine()
test_response = engine.query("Тест системы")
print(f"Система работает: {test_response.success}")
engine.shutdown()
```

## Начальная конфигурация

### Создание конфигурационного файла
```bash
# Генерация базовой конфигурации
neurograph init --config-type default

# Генерация для продакшена
neurograph init --config-type production

# Генерация для разработки
neurograph init --config-type development
```

### Структура проекта
```
my-neurograph-app/
├── config/
│   ├── default.json
│   ├── development.json
│   └── production.json
├── data/
│   ├── knowledge_graph.json
│   └── memory_storage/
├── logs/
└── app.py
```

---

# 🧠 Основные концепции

## Архитектура системы

### Уровни абстракции
```
┌─────────────────┐
│   Application   │  ← Ваше приложение
├─────────────────┤
│   Integration   │  ← Оркестрация модулей
├─────────────────┤
│   Processing    │  ← NLP, Processor, Propagation
├─────────────────┤
│   Knowledge     │  ← SemGraph, ContextVec, Memory
├─────────────────┤
│      Core       │  ← Инфраструктура
└─────────────────┘
```

### Ключевые компоненты

#### NeuroGraphEngine
Центральный движок системы:
- **Координирует** все модули
- **Маршрутизирует** запросы к нужным компонентам
- **Управляет** жизненным циклом системы
- **Предоставляет** единый API

#### Модули знаний
- **SemGraph**: структурированный граф связей
- **ContextVec**: векторные представления для семантического поиска
- **Memory**: биоморфная память с консолидацией

#### Модули обработки
- **NLP**: извлечение сущностей и отношений из текста
- **Processor**: логический вывод и рассуждения
- **Propagation**: распространение активации по графу

## Потоки данных

### Поток обучения
```
Текст → NLP → SemGraph → ContextVec → Memory
      ↓       ↓           ↓           ↓
   Сущности Узлы      Векторы    Элементы
            Связи                  памяти
```

### Поток запроса
```
Вопрос → Analysis → Multi-Search → Inference → Response
                    ↓            ↓           ↓
                  Graph        Memory    Processor
                 ContextVec   Propagation
```

---

# 📚 API Reference

## Основной API (NeuroGraphEngine)

### Создание движка

```python
from neurograph.integration import (
    create_default_engine,
    create_lightweight_engine,
    create_research_engine,
    create_production_engine
)

# Различные предустановленные конфигурации
engine = create_default_engine()           # Сбалансированная
engine = create_lightweight_engine()       # Минимальные ресурсы
engine = create_research_engine()          # Максимальная функциональность
engine = create_production_engine()        # Оптимизированная для продакшена

# Создание с пользовательской конфигурацией
from neurograph.integration import NeuroGraphEngine, IntegrationConfig

config = IntegrationConfig.load_from_file("my_config.json")
engine = NeuroGraphEngine(config)
```

### Базовые операции

```python
# Обучение системы
response = engine.learn(
    content="Искусственный интеллект изучает машинное обучение",
    source="user_input",           # Источник информации
    importance=1.0,                # Важность (0.0-1.0)
    tags=["AI", "ML"]              # Теги для категоризации
)

# Запрос к системе
response = engine.query(
    question="Что изучает ИИ?",
    context={"domain": "technology"},  # Контекст запроса
    max_results=5,                     # Максимум результатов
    include_explanations=True          # Включить объяснения
)

# Логический вывод
response = engine.infer(
    premises=["Все птицы летают", "Пингвин - птица"],
    max_depth=3                    # Глубина вывода
)

# Обработка произвольного текста
response = engine.process_text(
    text="Статья о нейронных сетях...",
    extract_entities=True,        # Извлечь сущности
    extract_relations=True,       # Извлечь отношения
    create_summary=True           # Создать резюме
)
```

### Структура ответов

```python
# ProcessingResponse - универсальная структура ответа
class ProcessingResponse:
    success: bool                          # Успешность операции
    primary_response: str                  # Основной ответ
    confidence: float                      # Уверенность (0.0-1.0)
    processing_time: float                 # Время обработки (сек)
    
    # Структурированные данные от модулей
    structured_data: Dict[str, Any] = {
        'nlp_result': {...},               # Результат NLP обработки
        'graph_data': {...},               # Данные из графа знаний
        'memory_matches': [...],           # Найденные в памяти элементы
        'inference_chain': [...],          # Цепочка логического вывода
        'propagation_result': {...}       # Результат распространения активации
    }
    
    # Метаинформация
    explanation: List[str]                 # Объяснения и источники
    sources: List[str]                     # Источники информации
    related_concepts: List[str]            # Связанные концепты
    error_message: Optional[str]           # Сообщение об ошибке
```

## Работа с конкретными модулями

### SemGraph API

```python
# Получение доступа к графу знаний
graph = engine.get_component('semgraph')

# Добавление узлов и связей
graph.add_node("Python", type="programming_language", popularity="high")
graph.add_edge("Python", "AI", "used_for", weight=0.9)

# Поиск в графе
neighbors = graph.get_neighbors("Python")
path = graph.find_shortest_path("Python", "Machine Learning")

# Анализ графа
central_nodes = graph.get_central_nodes(top_n=10)
communities = graph.find_communities()
```

### Memory API

```python
# Получение доступа к памяти
memory = engine.get_component('memory')

# Работа с элементами памяти
recent_items = memory.get_recent_items(hours=24)
important_memories = memory.get_most_accessed_items(limit=10)

# Поиск в памяти
results = memory.search_by_content("машинное обучение", limit=5)
semantic_results = memory.search_by_vector(query_vector, limit=5)

# Статистика памяти
stats = memory.get_memory_statistics()
```

### ContextVec API

```python
# Получение доступа к векторным представлениям
vectors = engine.get_component('contextvec')

# Работа с векторами
vector = vectors.get_vector("machine_learning")
similar = vectors.get_most_similar("artificial_intelligence", top_n=5)

# Добавление новых векторов
vectors.create_vector("new_concept", embedding_array)
```

### NLP API

```python
# Получение доступа к NLP процессору
nlp = engine.get_component('nlp')

# Прямая обработка текста
result = nlp.process_text(
    "Глубокое обучение использует нейронные сети",
    extract_entities=True,
    extract_relations=True
)

# Доступ к извлеченным данным
for entity in result.entities:
    print(f"Сущность: {entity.text}, Тип: {entity.entity_type}")

for relation in result.relations:
    print(f"Отношение: {relation.subject.text} -> {relation.predicate} -> {relation.object.text}")
```

---

# 💡 Практические примеры

## Пример 1: Персональный ассистент знаний

```python
from neurograph.integration import create_default_engine
import json

class PersonalKnowledgeAssistant:
    def __init__(self, user_name="User"):
        self.engine = create_default_engine()
        self.user_name = user_name
        self.context = {"user": user_name}
        
    def add_personal_fact(self, fact, category="general"):
        """Добавление личного факта"""
        response = self.engine.learn(
            content=f"{self.user_name}: {fact}",
            source="personal",
            tags=[category, "personal"],
            importance=0.8
        )
        return f"Запомнил: {fact}"
    
    def ask_question(self, question):
        """Задать вопрос ассистенту"""
        response = self.engine.query(
            question=question,
            context=self.context,
            include_explanations=True
        )
        
        return {
            "answer": response.primary_response,
            "confidence": response.confidence,
            "sources": response.sources,
            "related": response.related_concepts
        }
    
    def get_personal_insights(self):
        """Получить инсайты о личных данных"""
        # Поиск связей в личных данных
        memory = self.engine.get_component('memory')
        personal_items = memory.search_by_tags(["personal"], limit=20)
        
        # Анализ через граф
        graph = self.engine.get_component('semgraph')
        user_node = graph.get_node(self.user_name)
        
        if user_node:
            connections = graph.get_neighbors(self.user_name)
            return {
                "total_facts": len(personal_items),
                "key_interests": connections[:5],
                "knowledge_areas": self._extract_categories(personal_items)
            }
    
    def _extract_categories(self, items):
        categories = {}
        for item in items:
            tags = item.metadata.get('tags', [])
            for tag in tags:
                if tag != 'personal':
                    categories[tag] = categories.get(tag, 0) + 1
        return sorted(categories.items(), key=lambda x: x[1], reverse=True)

# Использование
assistant = PersonalKnowledgeAssistant("Алексей")

# Добавление личных фактов
assistant.add_personal_fact("Я работаю программистом в IT компании", "work")
assistant.add_personal_fact("Мне нравится изучать машинное обучение", "interests")
assistant.add_personal_fact("У меня есть кот по имени Мурзик", "personal")

# Запросы
result = assistant.ask_question("Что ты знаешь о моих интересах?")
print(json.dumps(result, indent=2, ensure_ascii=False))

# Инсайты
insights = assistant.get_personal_insights()
print("Ваш профиль знаний:", insights)
```

## Пример 2: Анализатор документов

```python
from neurograph.integration import create_research_engine
import os
from pathlib import Path

class DocumentAnalyzer:
    def __init__(self):
        # Используем исследовательскую конфигурацию для максимальной точности
        self.engine = create_research_engine()
        self.analysis_results = {}
    
    def analyze_document(self, file_path, document_type="general"):
        """Анализ одного документа"""
        
        # Чтение файла
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Обработка документа
        response = self.engine.process_text(
            text=content,
            extract_entities=True,
            extract_relations=True,
            create_summary=True
        )
        
        # Обучение системы на документе
        self.engine.learn(
            content=content,
            source=f"document:{os.path.basename(file_path)}",
            tags=[document_type, "document"],
            importance=0.9
        )
        
        # Анализ структуры
        analysis = self._analyze_structure(response, content)
        
        # Сохранение результатов
        doc_id = os.path.basename(file_path)
        self.analysis_results[doc_id] = analysis
        
        return analysis
    
    def _analyze_structure(self, response, content):
        """Глубокий анализ структуры документа"""
        
        nlp_data = response.structured_data.get('nlp_result', {})
        
        # Извлечение ключевой информации
        entities = nlp_data.get('entities', [])
        relations = nlp_data.get('relations', [])
        
        # Категоризация сущностей
        entity_types = {}
        for entity in entities:
            entity_type = entity.get('entity_type', 'UNKNOWN')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Анализ отношений
        relation_patterns = {}
        for relation in relations:
            rel_type = relation.get('predicate', 'unknown')
            relation_patterns[rel_type] = relation_patterns.get(rel_type, 0) + 1
        
        return {
            "summary": response.primary_response,
            "confidence": response.confidence,
            "statistics": {
                "total_entities": len(entities),
                "entity_distribution": entity_types,
                "total_relations": len(relations),
                "relation_patterns": relation_patterns,
                "text_length": len(content)
            },
            "key_entities": [e.get('text', '') for e in entities[:10]],
            "main_topics": self._extract_topics(entities),
            "processing_time": response.processing_time
        }
    
    def _extract_topics(self, entities):
        """Извлечение основных тем документа"""
        # Группировка сущностей по типам для выявления тем
        topics = {}
        for entity in entities:
            entity_type = entity.get('entity_type', 'UNKNOWN')
            if entity_type not in topics:
                topics[entity_type] = []
            topics[entity_type].append(entity.get('text', ''))
        
        return {k: v[:5] for k, v in topics.items()}  # Топ-5 по каждому типу
    
    def batch_analyze(self, directory_path, pattern="*.txt"):
        """Пакетный анализ документов"""
        
        directory = Path(directory_path)
        files = list(directory.glob(pattern))
        
        results = {}
        for file_path in files:
            try:
                print(f"Анализирую: {file_path.name}")
                analysis = self.analyze_document(str(file_path))
                results[file_path.name] = analysis
                print(f"✅ Готово: {file_path.name}")
            except Exception as e:
                print(f"❌ Ошибка при анализе {file_path.name}: {e}")
                results[file_path.name] = {"error": str(e)}
        
        return results
    
    def compare_documents(self, doc1_id, doc2_id):
        """Сравнение двух документов"""
        
        if doc1_id not in self.analysis_results or doc2_id not in self.analysis_results:
            return {"error": "Один или оба документа не найдены"}
        
        doc1 = self.analysis_results[doc1_id]
        doc2 = self.analysis_results[doc2_id]
        
        # Сравнение через векторные представления
        vectors = self.engine.get_component('contextvec')
        
        # Получение векторов документов (если они были векторизованы)
        similarity = self._calculate_similarity(doc1, doc2)
        
        return {
            "similarity_score": similarity,
            "common_entities": self._find_common_entities(doc1, doc2),
            "common_topics": self._find_common_topics(doc1, doc2),
            "differences": self._find_differences(doc1, doc2)
        }
    
    def _calculate_similarity(self, doc1, doc2):
        """Расчет семантической близости документов"""
        # Простое сравнение по пересечению ключевых сущностей
        entities1 = set(doc1.get('key_entities', []))
        entities2 = set(doc2.get('key_entities', []))
        
        if len(entities1) == 0 and len(entities2) == 0:
            return 0.0
        
        intersection = len(entities1.intersection(entities2))
        union = len(entities1.union(entities2))
        
        return intersection / union if union > 0 else 0.0
    
    def _find_common_entities(self, doc1, doc2):
        entities1 = set(doc1.get('key_entities', []))
        entities2 = set(doc2.get('key_entities', []))
        return list(entities1.intersection(entities2))
    
    def _find_common_topics(self, doc1, doc2):
        topics1 = set(doc1.get('main_topics', {}).keys())
        topics2 = set(doc2.get('main_topics', {}).keys())
        return list(topics1.intersection(topics2))
    
    def _find_differences(self, doc1, doc2):
        entities1 = set(doc1.get('key_entities', []))
        entities2 = set(doc2.get('key_entities', []))
        
        return {
            "unique_to_doc1": list(entities1 - entities2),
            "unique_to_doc2": list(entities2 - entities1)
        }
    
    def generate_report(self, output_file="analysis_report.json"):
        """Генерация отчета по всем проанализированным документам"""
        
        report = {
            "total_documents": len(self.analysis_results),
            "analysis_timestamp": str(datetime.now()),
            "summary_statistics": self._calculate_summary_stats(),
            "documents": self.analysis_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _calculate_summary_stats(self):
        """Расчет сводной статистики"""
        if not self.analysis_results:
            return {}
        
        total_entities = sum(
            doc.get('statistics', {}).get('total_entities', 0) 
            for doc in self.analysis_results.values() 
            if 'statistics' in doc
        )
        
        total_relations = sum(
            doc.get('statistics', {}).get('total_relations', 0) 
            for doc in self.analysis_results.values() 
            if 'statistics' in doc
        )
        
        avg_confidence = sum(
            doc.get('confidence', 0) 
            for doc in self.analysis_results.values() 
            if 'confidence' in doc
        ) / len(self.analysis_results)
        
        return {
            "total_entities_extracted": total_entities,
            "total_relations_found": total_relations,
            "average_confidence": round(avg_confidence, 3),
            "documents_with_errors": len([
                doc for doc in self.analysis_results.values() 
                if 'error' in doc
            ])
        }

# Использование
from datetime import datetime

analyzer = DocumentAnalyzer()

# Анализ одного документа
analysis = analyzer.analyze_document("research_paper.txt", "academic")
print("Анализ документа:")
print(f"Резюме: {analysis['summary'][:200]}...")
print(f"Найдено сущностей: {analysis['statistics']['total_entities']}")
print(f"Найдено отношений: {analysis['statistics']['total_relations']}")

# Пакетный анализ
results = analyzer.batch_analyze("./documents/", "*.txt")
print(f"Проанализировано документов: {len(results)}")

# Генерация отчета
report = analyzer.generate_report("document_analysis_report.json")
```

## Пример 3: Чат-бот с контекстом

```python
from neurograph.integration import create_default_engine
from datetime import datetime, timedelta
import uuid

class ContextualChatBot:
    def __init__(self, bot_name="NeuroBot"):
        self.engine = create_default_engine()
        self.bot_name = bot_name
        self.conversations = {}  # user_id -> conversation_history
        
        # Базовые знания бота
        self._initialize_bot_knowledge()
    
    def _initialize_bot_knowledge(self):
        """Инициализация базовых знаний бота"""
        base_knowledge = [
            f"Меня зовут {self.bot_name} и я ИИ-ассистент на основе NeuroGraph",
            "Я могу помочь с вопросами, обучаться на новой информации и запоминать контекст беседы",
            "Я использую нейросимволическую архитектуру для понимания и генерации ответов",
            "Моя память состоит из кратковременной и долговременной памяти, как у человека"
        ]
        
        for knowledge in base_knowledge:
            self.engine.learn(
                content=knowledge,
                source="bot_initialization",
                tags=["bot_info", "base_knowledge"],
                importance=0.9
            )
    
    def start_conversation(self, user_id, user_name=None):
        """Начало новой беседы с пользователем"""
        conversation_id = str(uuid.uuid4())
        
        self.conversations[user_id] = {
            "id": conversation_id,
            "user_name": user_name or f"User_{user_id}",
            "started_at": datetime.now(),
            "messages": [],
            "context": {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "user_preferences": {},
                "current_topic": None
            }
        }
        
        welcome_message = f"Привет! Я {self.bot_name}. Как дела? О чем хотите поговорить?"
        
        self._add_message(user_id, "bot", welcome_message)
        return welcome_message
    
    def send_message(self, user_id, message):
        """Отправка сообщения от пользователя"""
        
        if user_id not in self.conversations:
            self.start_conversation(user_id)
        
        # Добавляем сообщение пользователя в историю
        self._add_message(user_id, "user", message)
        
        # Обучаем систему на сообщении пользователя
        self._learn_from_user_message(user_id, message)
        
        # Генерируем ответ
        response = self._generate_response(user_id, message)
        
        # Добавляем ответ бота в историю
        self._add_message(user_id, "bot", response["text"])
        
        # Обновляем контекст
        self._update_context(user_id, message, response)
        
        return response
    
    def _add_message(self, user_id, sender, text):
        """Добавление сообщения в историю беседы"""
        message = {
            "sender": sender,
            "text": text,
            "timestamp": datetime.now(),
            "message_id": str(uuid.uuid4())
        }
        
        self.conversations[user_id]["messages"].append(message)
        
        # Ограничиваем историю последними 50 сообщениями
        if len(self.conversations[user_id]["messages"]) > 50:
            self.conversations[user_id]["messages"] = \
                self.conversations[user_id]["messages"][-50:]
    
    def _learn_from_user_message(self, user_id, message):
        """Обучение системы на сообщении пользователя"""
        conversation = self.conversations[user_id]
        user_name = conversation["user_name"]
        
        # Формируем контекстуализированное знание
        contextualized_message = f"{user_name} сказал: {message}"
        
        self.engine.learn(
            content=contextualized_message,
            source=f"conversation:{conversation['id']}",
            tags=["conversation", "user_input", user_id],
            importance=0.7
        )
    
    def _generate_response(self, user_id, message):
        """Генерация ответа на сообщение пользователя"""
        conversation = self.conversations[user_id]
        
        # Формируем контекст для запроса
        context = self._build_query_context(user_id)
        
        # Основной запрос к системе
        response = self.engine.query(
            question=message,
            context=context,
            include_explanations=True,
            max_results=3
        )
        
        # Если системный ответ слишком формальный, делаем его более разговорным
        bot_response = self._make_conversational(response.primary_response, conversation)
        
        return {
            "text": bot_response,
            "confidence": response.confidence,
            "context_used": len(context),
            "related_topics": response.related_concepts[:3],
            "processing_time": response.processing_time
        }
    
    def _build_query_context(self, user_id):
        """Построение контекста для запроса"""
        conversation = self.conversations[user_id]
        
        # Базовый контекст
        context = {
            "user_id": user_id,
            "user_name": conversation["user_name"],
            "conversation_id": conversation["id"],
            "bot_name": self.bot_name
        }
        
        # Добавляем последние сообщения как контекст
        recent_messages = conversation["messages"][-10:]  # Последние 10 сообщений
        if recent_messages:
            context["recent_conversation"] = [
                f"{msg['sender']}: {msg['text']}" 
                for msg in recent_messages
            ]
        
        # Добавляем текущую тему, если есть
        if conversation["context"]["current_topic"]:
            context["current_topic"] = conversation["context"]["current_topic"]
        
        return context
    
    def _make_conversational(self, system_response, conversation):
        """Делаем ответ более разговорным"""
        user_name = conversation["user_name"]
        
        # Простые правила для улучшения разговорности
        if len(system_response) > 200:
            # Длинный ответ - добавляем разбивку
            return f"{system_response}\n\nЧто еще вас интересует по этой теме?"
        elif system_response.lower().startswith("я не знаю"):
            # Незнание - более дружелюбная формулировка
            return f"Хм, я пока не уверен в ответе на этот вопрос. Может, расскажете мне больше об этом?"
        else:
            # Обычный ответ
            return system_response
    
    def _update_context(self, user_id, user_message, bot_response):
        """Обновление контекста беседы"""
        conversation = self.conversations[user_id]
        
        # Определяем тему сообщения через NLP
        nlp = self.engine.get_component('nlp')
        nlp_result = nlp.process_text(user_message, extract_entities=True)
        
        # Извлекаем основные сущности как потенциальные темы
        if hasattr(nlp_result, 'entities') and nlp_result.entities:
            entities = [entity.text for entity in nlp_result.entities[:3]]
            if entities:
                conversation["context"]["current_topic"] = entities[0]
        
        # Обновляем предпочтения пользователя на основе темы
        current_topic = conversation["context"]["current_topic"]
        if current_topic:
            preferences = conversation["context"]["user_preferences"]
            preferences[current_topic] = preferences.get(current_topic, 0) + 1
    
    def get_conversation_summary(self, user_id):
        """Получение резюме беседы"""
        if user_id not in self.conversations:
            return {"error": "Беседа не найдена"}
        
        conversation = self.conversations[user_id]
        messages = conversation["messages"]
        
        if not messages:
            return {"summary": "Беседа только началась"}
        
        # Статистика беседы
        total_messages = len(messages)
        user_messages = len([m for m in messages if m["sender"] == "user"])
        bot_messages = len([m for m in messages if m["sender"] == "bot"])
        
        # Основные темы
        topics = list(conversation["context"]["user_preferences"].keys())
        duration = datetime.now() - conversation["started_at"]
        
        return {
            "conversation_id": conversation["id"],
            "user_name": conversation["user_name"],
            "duration_minutes": int(duration.total_seconds() / 60),
            "total_messages": total_messages,
            "user_messages": user_messages,
            "bot_messages": bot_messages,
            "main_topics": topics[:5],
            "current_topic": conversation["context"]["current_topic"],
            "started_at": conversation["started_at"].isoformat()
        }
    
    def get_user_profile(self, user_id):
        """Получение профиля пользователя на основе истории бесед"""
        if user_id not in self.conversations:
            return {"error": "Пользователь не найден"}
        
        conversation = self.conversations[user_id]
        
        # Анализ интересов пользователя
        preferences = conversation["context"]["user_preferences"]
        sorted_interests = sorted(preferences.items(), key=lambda x: x[1], reverse=True)
        
        # Стиль общения (анализ длины сообщений)
        user_messages = [m for m in conversation["messages"] if m["sender"] == "user"]
        if user_messages:
            avg_message_length = sum(len(m["text"]) for m in user_messages) / len(user_messages)
            communication_style = "краткий" if avg_message_length < 50 else "подробный"
        else:
            communication_style = "неопределенный"
        
        return {
            "user_name": conversation["user_name"],
            "primary_interests": [interest for interest, count in sorted_interests[:5]],
            "communication_style": communication_style,
            "total_interactions": len(user_messages),
            "favorite_topics": dict(sorted_interests[:3]),
            "profile_confidence": min(len(user_messages) / 10, 1.0)  # Уверенность в профиле
        }

# Использование чат-бота
bot = ContextualChatBot("НейроБот")

# Моделирование беседы
user_id = "user_123"

# Начало беседы
welcome = bot.start_conversation(user_id, "Анна")
print(f"Бот: {welcome}")

# Диалог
messages = [
    "Привет! Меня интересует машинное обучение",
    "Что такое нейронные сети?",
    "А как они используются в компьютерном зрении?",
    "Спасибо за объяснения! А что ты знаешь о Python?",
    "Какие библиотеки лучше использовать для ML?"
]

for message in messages:
    print(f"Пользователь: {message}")
    response = bot.send_message(user_id, message)
    print(f"Бот: {response['text']}")
    print(f"Уверенность: {response['confidence']:.2f}")
    if response['related_topics']:
        print(f"Связанные темы: {', '.join(response['related_topics'])}")
    print("-" * 50)

# Анализ беседы
summary = bot.get_conversation_summary(user_id)
print("\nРезюме беседы:")
print(json.dumps(summary, indent=2, ensure_ascii=False))

profile = bot.get_user_profile(user_id)
print("\nПрофиль пользователя:")
print(json.dumps(profile, indent=2, ensure_ascii=False))
```

---

# ⚙️ Конфигурация

## Типы конфигураций

### Default Configuration
```json
{
  "engine_name": "default_neurograph",
  "components": {
    "semgraph": {
      "type": "memory_efficient",
      "params": {}
    },
    "contextvec": {
      "type": "dynamic",
      "params": {
        "vector_size": 384,
        "use_indexing": true
      }
    },
    "memory": {
      "params": {
        "stm_capacity": 100,
        "ltm_capacity": 10000,
        "use_semantic_indexing": true
      }
    },
    "nlp": {
      "params": {
        "language": "ru",
        "use_spacy": true
      }
    },
    "processor": {
      "params": {
        "confidence_threshold": 0.5,
        "max_depth": 5
      }
    },
    "propagation": {
      "params": {
        "max_iterations": 100,
        "activation_threshold": 0.1
      }
    }
  },
  "performance": {
    "max_concurrent_requests": 10,
    "default_timeout": 30.0,
    "enable_caching": true,
    "cache_ttl": 300
  }
}
```

### Lightweight Configuration
```json
{
  "engine_name": "lightweight_neurograph",
  "components": {
    "memory": {
      "params": {
        "stm_capacity": 25,
        "ltm_capacity": 500,
        "use_semantic_indexing": false
      }
    },
    "nlp": {
      "params": {
        "language": "ru",
        "use_spacy": false
      }
    },
    "processor": {
      "params": {
        "confidence_threshold": 0.3,
        "max_depth": 2
      }
    }
  },
  "performance": {
    "max_concurrent_requests": 3,
    "default_timeout": 10.0,
    "enable_caching": false
  }
}
```

### Production Configuration
```json
{
  "engine_name": "production_neurograph",
  "components": {
    "semgraph": {
      "type": "persistent",
      "params": {
        "file_path": "/data/knowledge_graph.json",
        "auto_save_interval": 300.0
      }
    },
    "memory": {
      "params": {
        "stm_capacity": 200,
        "ltm_capacity": 50000,
        "auto_consolidation": true,
        "consolidation_interval": 600.0
      }
    }
  },
  "performance": {
    "max_concurrent_requests": 100,
    "default_timeout": 60.0,
    "enable_caching": true,
    "cache_ttl": 900
  },
  "monitoring": {
    "enable_metrics": true,
    "enable_health_checks": true,
    "metrics_interval": 60
  },
  "logging": {
    "level": "INFO",
    "file": "/logs/neurograph.log",
    "rotation": "100MB",
    "retention": "30 days"
  }
}
```

## Создание пользовательской конфигурации

```python
from neurograph.integration import IntegrationConfig

# Создание конфигурации программно
config = IntegrationConfig(
    engine_name="my_custom_engine",
    components={
        "memory": {
            "params": {
                "stm_capacity": 150,
                "ltm_capacity": 20000
            }
        },
        "nlp": {
            "params": {
                "language": "en",
                "confidence_threshold": 0.8
            }
        }
    },
    performance={
        "max_concurrent_requests": 50,
        "enable_caching": True,
        "cache_ttl": 600
    }
)

# Сохранение в файл
config.save_to_file("my_config.json")

# Создание движка с пользовательской конфигурацией
from neurograph.integration import NeuroGraphEngine
engine = NeuroGraphEngine(config)
```

## Переменные окружения

```bash
# Основные настройки
export NEUROGRAPH_CONFIG_FILE="/path/to/config.json"
export NEUROGRAPH_LOG_LEVEL="INFO"
export NEUROGRAPH_LOG_FILE="/logs/neurograph.log"

# Пути к данным
export NEUROGRAPH_DATA_DIR="/data"
export NEUROGRAPH_GRAPH_FILE="/data/graph.json"
export NEUROGRAPH_MEMORY_DIR="/data/memory"

# Производительность
export NEUROGRAPH_MAX_WORKERS="4"
export NEUROGRAPH_CACHE_TTL="300"
export NEUROGRAPH_REQUEST_TIMEOUT="30"

# Интеграции
export NEUROGRAPH_SPACY_MODEL="ru_core_news_sm"
export NEUROGRAPH_VECTOR_SIZE="384"
```

---

# 🔌 Интеграция

## Web Framework Integration

### Flask Integration
```python
from flask import Flask, request, jsonify
from neurograph.integration import create_default_engine
import logging

app = Flask(__name__)
engine = create_default_engine()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/learn', methods=['POST'])
def learn():
    try:
        data = request.get_json()
        content = data.get('content', '')
        source = data.get('source', 'api')
        tags = data.get('tags', [])
        
        if not content:
            return jsonify({'error': 'Content is required'}), 400
        
        response = engine.learn(
            content=content,
            source=source,
            tags=tags,
            importance=data.get('importance', 0.7)
        )
        
        return jsonify({
            'success': response.success,
            'message': response.primary_response,
            'processing_time': response.processing_time
        })
        
    except Exception as e:
        logger.error(f"Error in /api/learn: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get('question', '')
        context = data.get('context', {})
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        response = engine.query(
            question=question,
            context=context,
            include_explanations=data.get('include_explanations', True),
            max_results=data.get('max_results', 5)
        )
        
        return jsonify({
            'success': response.success,
            'answer': response.primary_response,
            'confidence': response.confidence,
            'sources': response.sources,
            'related_concepts': response.related_concepts,
            'processing_time': response.processing_time
        })
        
    except Exception as e:
        logger.error(f"Error in /api/query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    try:
        health_status = engine.get_health_status()
        return jsonify(health_status)
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    try:
        stats = engine.get_system_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Graceful shutdown
import atexit
def shutdown_engine():
    logger.info("Shutting down NeuroGraph engine...")
    engine.shutdown()

atexit.register(shutdown_engine)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Django Integration
```python
# django_app/neurograph_service.py
from django.conf import settings
from neurograph.integration import create_default_engine
import logging

logger = logging.getLogger(__name__)

class NeuroGraphService:
    _instance = None
    _engine = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._engine is None:
            config_file = getattr(settings, 'NEUROGRAPH_CONFIG', None)
            if config_file:
                from neurograph.integration import NeuroGraphEngine, IntegrationConfig
                config = IntegrationConfig.load_from_file(config_file)
                self._engine = NeuroGraphEngine(config)
            else:
                self._engine = create_default_engine()
            
            logger.info("NeuroGraph engine initialized")
    
    @property
    def engine(self):
        return self._engine
    
    def shutdown(self):
        if self._engine:
            self._engine.shutdown()
            logger.info("NeuroGraph engine shutdown")

# Singleton instance
neurograph_service = NeuroGraphService()

# django_app/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .neurograph_service import neurograph_service
import json

@csrf_exempt
@require_http_methods(["POST"])
def neurograph_learn(request):
    try:
        data = json.loads(request.body)
        response = neurograph_service.engine.learn(
            content=data.get('content', ''),
            source=data.get('source', 'django_api'),
            tags=data.get('tags', [])
        )
        
        return JsonResponse({
            'success': response.success,
            'message': response.primary_response
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def neurograph_query(request):
    try:
        data = json.loads(request.body)
        response = neurograph_service.engine.query(
            question=data.get('question', ''),
            context=data.get('context', {})
        )
        
        return JsonResponse({
            'success': response.success,
            'answer': response.primary_response,
            'confidence': response.confidence
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# django_app/apps.py
from django.apps import AppConfig
from django.conf import settings

class NeuroGraphAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'django_app'
    
    def ready(self):
        # Инициализация NeuroGraph при запуске Django
        from .neurograph_service import neurograph_service
        
        # Регистрация сигнала для корректного завершения
        import signal
        import sys
        
        def signal_handler(sig, frame):
            neurograph_service.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
```

## Database Integration

### SQLAlchemy Integration
```python
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from neurograph.integration import create_default_engine
from datetime import datetime
import json

Base = declarative_base()

class KnowledgeEntry(Base):
    __tablename__ = 'knowledge_entries'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    source = Column(String(255), default='unknown')
    tags = Column(Text)  # JSON string
    importance = Column(Float, default=0.5)
    processed = Column(Integer, default=0)  # Boolean as int
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)

class QueryLog(Base):
    __tablename__ = 'query_log'
    
    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    confidence = Column(Float)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class NeuroGraphDatabase:
    def __init__(self, database_url="sqlite:///neurograph.db"):
        self.engine_db = create_engine(database_url)
        Base.metadata.create_all(self.engine_db)
        
        Session = sessionmaker(bind=self.engine_db)
        self.session = Session()
        
        self.neurograph = create_default_engine()
    
    def add_knowledge(self, content, source="user", tags=None, importance=0.7):
        """Добавление знания в БД и обучение NeuroGraph"""
        
        # Сохранение в БД
        entry = KnowledgeEntry(
            content=content,
            source=source,
            tags=json.dumps(tags or []),
            importance=importance
        )
        self.session.add(entry)
        self.session.commit()
        
        # Обучение NeuroGraph
        response = self.neurograph.learn(
            content=content,
            source=source,
            tags=tags or [],
            importance=importance
        )
        
        # Обновление статуса обработки
        if response.success:
            entry.processed = 1
            entry.processed_at = datetime.utcnow()
            self.session.commit()
        
        return entry.id, response.success
    
    def process_pending_knowledge(self):
        """Обработка необработанных записей"""
        
        pending_entries = self.session.query(KnowledgeEntry).filter(
            KnowledgeEntry.processed == 0
        ).all()
        
        processed_count = 0
        for entry in pending_entries:
            try:
                tags = json.loads(entry.tags) if entry.tags else []
                
                response = self.neurograph.learn(
                    content=entry.content,
                    source=entry.source,
                    tags=tags,
                    importance=entry.importance
                )
                
                if response.success:
                    entry.processed = 1
                    entry.processed_at = datetime.utcnow()
                    processed_count += 1
                
            except Exception as e:
                print(f"Error processing entry {entry.id}: {e}")
        
        self.session.commit()
        return processed_count
    
    def query_with_logging(self, question, context=None):
        """Запрос с логированием в БД"""
        
        start_time = datetime.utcnow()
        
        response = self.neurograph.query(
            question=question,
            context=context or {},
            include_explanations=True
        )
        
        # Логирование запроса
        query_log = QueryLog(
            question=question,
            answer=response.primary_response,
            confidence=response.confidence,
            processing_time=response.processing_time
        )
        self.session.add(query_log)
        self.session.commit()
        
        return response
    
    def get_knowledge_stats(self):
        """Статистика знаний в БД"""
        
        total_entries = self.session.query(KnowledgeEntry).count()
        processed_entries = self.session.query(KnowledgeEntry).filter(
            KnowledgeEntry.processed == 1
        ).count()
        
        recent_queries = self.session.query(QueryLog).filter(
            QueryLog.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
        ).count()
        
        return {
            "total_knowledge_entries": total_entries,
            "processed_entries": processed_entries,
            "pending_entries": total_entries - processed_entries,
            "queries_today": recent_queries
        }
    
    def cleanup_old_data(self, days=30):
        """Очистка старых данных"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Удаление старых логов запросов
        old_queries = self.session.query(QueryLog).filter(
            QueryLog.created_at < cutoff_date
        ).delete()
        
        self.session.commit()
        return old_queries

# Использование
db = NeuroGraphDatabase("postgresql://user:pass@localhost/neurograph")

# Добавление знаний
entry_id, success = db.add_knowledge(
    "Машинное обучение - это подраздел искусственного интеллекта",
    source="textbook",
    tags=["AI", "ML", "education"],
    importance=0.9
)

# Обработка необработанных записей
processed = db.process_pending_knowledge()
print(f"Обработано записей: {processed}")

# Запрос с логированием
response = db.query_with_logging("Что такое машинное обучение?")
print(f"Ответ: {response.primary_response}")

# Статистика
stats = db.get_knowledge_stats()
print(f"Статистика: {stats}")
```

## Cloud Services Integration

### AWS Integration
```python
import boto3
from neurograph.integration import create_default_engine
import json
from datetime import datetime

class AWSNeuroGraphIntegration:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.secrets_client = boto3.client('secretsmanager', region_name=region)
        
        self.neurograph = create_default_engine()
        
    def process_s3_documents(self, bucket_name, prefix="documents/"):
        """Обработка документов из S3"""
        
        response = self.s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return {"processed": 0, "errors": []}
        
        processed_count = 0
        errors = []
        
        for obj in response['Contents']:
            try:
                # Скачивание файла
                file_content = self.s3_client.get_object(
                    Bucket=bucket_name,
                    Key=obj['Key']
                )
                
                content = file_content['Body'].read().decode('utf-8')
                
                # Обработка через NeuroGraph
                result = self.neurograph.learn(
                    content=content,
                    source=f"s3://{bucket_name}/{obj['Key']}",
                    tags=["s3_document", "auto_processed"]
                )
                
                if result.success:
                    processed_count += 1
                    
                    # Сохранение результата обратно в S3
                    self._save_processing_result(bucket_name, obj['Key'], result)
                
            except Exception as e:
                errors.append(f"Error processing {obj['Key']}: {str(e)}")
        
        return {"processed": processed_count, "errors": errors}
    
    def _save_processing_result(self, bucket, key, result):
        """Сохранение результата обработки в S3"""
        
        result_data = {
            "processed_at": datetime.utcnow().isoformat(),
            "success": result.success,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "structured_data": result.structured_data
        }
        
        result_key = key.replace(".txt", "_processed.json")
        
        self.s3_client.put_object(
            Bucket=bucket,
            Key=f"processed/{result_key}",
            Body=json.dumps(result_data, indent=2),
            ContentType='application/json'
        )
    
    def deploy_as_lambda(self, function_name="neurograph-processor"):
        """Развертывание как Lambda функция"""
        
        # Код Lambda функции
        lambda_code = '''
import json
from neurograph.integration import create_lightweight_engine

engine = None

def lambda_handler(event, context):
    global engine
    
    if engine is None:
        engine = create_lightweight_engine()
    
    try:
        if event.get('action') == 'learn':
            response = engine.learn(
                content=event['content'],
                source=event.get('source', 'lambda'),
                tags=event.get('tags', [])
            )
        elif event.get('action') == 'query':
            response = engine.query(
                question=event['question'],
                context=event.get('context', {})
            )
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid action'})
            }
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': response.success,
                'result': response.primary_response,
                'confidence': response.confidence
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
        '''
        
        # Создание Lambda функции (псевдокод)
        print(f"Lambda функция {function_name} готова к развертыванию")
        print("Код функции сохранен для развертывания")
        
        return lambda_code

# Использование
aws_integration = AWSNeuroGraphIntegration(region='us-east-1')

# Обработка документов из S3
result = aws_integration.process_s3_documents('my-documents-bucket')
print(f"Обработано документов: {result['processed']}")

# Подготовка Lambda функции
lambda_code = aws_integration.deploy_as_lambda()
```

---

# 🔧 Расширение системы

## Создание пользовательских модулей

### Базовый модуль
```python
from neurograph.core import Component, Configurable
from neurograph.core.logging import get_logger
from abc import ABC, abstractmethod

# Интерфейс для пользовательского модуля
class ICustomProcessor(ABC):
    @abstractmethod
    def process_data(self, data: str) -> dict:
        pass
    
    @abstractmethod
    def get_capabilities(self) -> list:
        pass

# Реализация пользовательского модуля
class SentimentAnalysisModule(Component, Configurable, ICustomProcessor):
    def __init__(self, component_id: str = "sentiment_analyzer"):
        super().__init__(component_id)
        self.logger = get_logger(self.__class__.__name__)
        self.config = {}
        
        # Простая модель настроения
        self.positive_words = set([
            "хорошо", "отлично", "прекрасно", "замечательно", 
            "великолепно", "превосходно", "удивительно"
        ])
        self.negative_words = set([
            "плохо", "ужасно", "отвратительно", "кошмарно",
            "неприятно", "раздражает", "разочарование"
        ])
    
    def initialize(self) -> bool:
        self.logger.info("Инициализация модуля анализа настроений")
        return True
    
        def initialize(self) -> bool:
        self.logger.info("Инициализация модуля анализа настроений")
        return True
    
    def shutdown(self) -> bool:
        self.logger.info("Завершение работы модуля анализа настроений")
        return True
    
    def configure(self, config: dict) -> bool:
        self.config = config
        
        # Загрузка пользовательских словарей, если указаны
        if 'positive_words_file' in config:
            self._load_word_list(config['positive_words_file'], self.positive_words)
        
        if 'negative_words_file' in config:
            self._load_word_list(config['negative_words_file'], self.negative_words)
        
        return True
    
    def get_config(self) -> dict:
        return self.config.copy()
    
    def process_data(self, data: str) -> dict:
        """Анализ настроения текста"""
        
        words = data.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        if total_words == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / total_words
        
        # Классификация настроения
        if sentiment_score > 0.1:
            sentiment = "positive"
        elif sentiment_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "positive_words_found": positive_count,
            "negative_words_found": negative_count,
            "confidence": min(abs(sentiment_score) * 2, 1.0)
        }
    
    def get_capabilities(self) -> list:
        return [
            "sentiment_analysis",
            "emotion_detection",
            "text_polarity"
        ]
    
    def _load_word_list(self, file_path: str, word_set: set):
        """Загрузка списка слов из файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = [line.strip().lower() for line in f if line.strip()]
                word_set.update(words)
            self.logger.info(f"Загружено {len(words)} слов из {file_path}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки слов из {file_path}: {e}")

# Регистрация модуля в системе
from neurograph.integration import ProcessorFactory

# Регистрация фабрики для создания модуля
ProcessorFactory.register_processor("sentiment_analysis", SentimentAnalysisModule)

# Использование в составе NeuroGraph
from neurograph.integration import create_default_engine

engine = create_default_engine()

# Добавление пользовательского модуля
sentiment_module = SentimentAnalysisModule()
sentiment_module.initialize()

# Интеграция с движком (через провайдер компонентов)
provider = engine.provider
provider.register_component("sentiment_analyzer", sentiment_module)

# Использование модуля
result = sentiment_module.process_data("Это отличный день! Я очень рад!")
print(f"Настроение: {result['sentiment']}, Оценка: {result['sentiment_score']:.2f}")
```

### Пользовательский конвейер обработки

```python
from neurograph.integration.pipelines import BasePipeline
from neurograph.integration.base import ProcessingRequest, ProcessingResponse
import time

class SentimentAnalysisPipeline(BasePipeline):
    """Конвейер для анализа настроений"""
    
    def __init__(self):
        super().__init__("sentiment_analysis")
        self.sentiment_module = None
    
    def process(self, request: ProcessingRequest, provider) -> ProcessingResponse:
        start_time = time.time()
        
        try:
            # Получение модуля анализа настроений
            if self.sentiment_module is None:
                self.sentiment_module = provider.get_component('sentiment_analyzer')
            
            # Базовая обработка текста через NLP
            nlp = provider.get_component('nlp')
            nlp_result = nlp.process_text(request.content)
            
            # Анализ настроения
            sentiment_result = self.sentiment_module.process_data(request.content)
            
            # Интеграция с памятью
            memory = provider.get_component('memory')
            if sentiment_result['sentiment'] != 'neutral':
                # Сохраняем тексты с выраженным настроением
                from neurograph.memory.base import MemoryItem
                import numpy as np
                
                memory_item = MemoryItem(
                    content=request.content,
                    embedding=np.random.random(384),  # В реальности - через энкодер
                    content_type="sentiment_text",
                    metadata={
                        "sentiment": sentiment_result['sentiment'],
                        "sentiment_score": sentiment_result['sentiment_score'],
                        "confidence": sentiment_result['confidence']
                    }
                )
                memory.add(memory_item)
            
            # Формирование ответа
            processing_time = time.time() - start_time
            
            response_text = f"Анализ настроения завершен. Определено настроение: {sentiment_result['sentiment']} (оценка: {sentiment_result['sentiment_score']:.2f})"
            
            return ProcessingResponse(
                success=True,
                primary_response=response_text,
                confidence=sentiment_result['confidence'],
                processing_time=processing_time,
                structured_data={
                    'nlp_result': nlp_result.__dict__ if hasattr(nlp_result, '__dict__') else {},
                    'sentiment_analysis': sentiment_result
                },
                explanation=[
                    f"Найдено положительных слов: {sentiment_result['positive_words_found']}",
                    f"Найдено отрицательных слов: {sentiment_result['negative_words_found']}",
                    f"Итоговая оценка: {sentiment_result['sentiment_score']:.3f}"
                ]
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка в конвейере анализа настроений: {e}")
            
            return ProcessingResponse(
                success=False,
                primary_response="Ошибка при анализе настроения",
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )

# Регистрация конвейера
def register_sentiment_pipeline(engine):
    """Регистрация конвейера анализа настроений"""
    sentiment_pipeline = SentimentAnalysisPipeline()
    engine.pipelines['sentiment_analysis'] = sentiment_pipeline
    return engine

# Использование
engine = create_default_engine()

# Добавление модуля и конвейера
sentiment_module = SentimentAnalysisModule()
sentiment_module.initialize()
engine.provider.register_component("sentiment_analyzer", sentiment_module)

register_sentiment_pipeline(engine)

# Обработка через пользовательский конвейер
request = ProcessingRequest(
    content="Сегодня замечательный день! Я чувствую себя прекрасно!",
    request_type="sentiment_analysis"
)

response = engine.process_request(request)
print(f"Результат: {response.primary_response}")
print(f"Структурированные данные: {response.structured_data['sentiment_analysis']}")
```

## Создание пользовательских адаптеров

```python
from neurograph.integration.adapters import BaseAdapter
from typing import Any, Dict, List

class CustomDataAdapter(BaseAdapter):
    """Адаптер для интеграции с пользовательскими источниками данных"""
    
    def __init__(self, data_source_config: Dict[str, Any]):
        super().__init__("custom_data_adapter")
        self.config = data_source_config
        self.supported_formats = ["json", "xml", "csv"]
    
    def adapt(self, source_data: Any, target_format: str) -> Dict[str, Any]:
        """Адаптация данных из источника в формат NeuroGraph"""
        
        if target_format == "neurograph_entities":
            return self._convert_to_entities(source_data)
        elif target_format == "neurograph_relations":
            return self._convert_to_relations(source_data)
        elif target_format == "memory_items":
            return self._convert_to_memory_items(source_data)
        else:
            raise ValueError(f"Неподдерживаемый формат: {target_format}")
    
    def _convert_to_entities(self, data: Dict) -> List[Dict]:
        """Конвертация в формат сущностей NeuroGraph"""
        entities = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 2:
                    entities.append({
                        "text": value,
                        "entity_type": self._classify_entity_type(key, value),
                        "confidence": 0.8,
                        "source": "custom_adapter",
                        "metadata": {"original_key": key}
                    })
        
        return entities
    
    def _convert_to_relations(self, data: Dict) -> List[Dict]:
        """Конвертация в формат отношений NeuroGraph"""
        relations = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        relations.append({
                            "subject": key,
                            "predicate": self._normalize_predicate(sub_key),
                            "object": str(sub_value),
                            "confidence": 0.7,
                            "source": "custom_adapter"
                        })
                elif isinstance(value, list):
                    for item in value:
                        relations.append({
                            "subject": key,
                            "predicate": "contains",
                            "object": str(item),
                            "confidence": 0.6,
                            "source": "custom_adapter"
                        })
        
        return relations
    
    def _convert_to_memory_items(self, data: Any) -> List[Dict]:
        """Конвертация в формат элементов памяти"""
        items = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                content = f"{key}: {value}"
                items.append({
                    "content": content,
                    "content_type": "custom_data",
                    "metadata": {
                        "source": "custom_adapter",
                        "original_key": key,
                        "data_type": type(value).__name__
                    }
                })
        
        return items
    
    def _classify_entity_type(self, key: str, value: str) -> str:
        """Классификация типа сущности"""
        key_lower = key.lower()
        
        if any(word in key_lower for word in ['name', 'title', 'label']):
            return "ENTITY_NAME"
        elif any(word in key_lower for word in ['date', 'time', 'when']):
            return "DATE"
        elif any(word in key_lower for word in ['place', 'location', 'where']):
            return "LOCATION"
        elif any(word in key_lower for word in ['person', 'author', 'user']):
            return "PERSON"
        else:
            return "CONCEPT"
    
    def _normalize_predicate(self, key: str) -> str:
        """Нормализация предиката"""
        key_lower = key.lower()
        
        predicate_mapping = {
            'has': 'has',
            'is': 'is_a',
            'contains': 'contains',
            'belongs': 'belongs_to',
            'located': 'located_in',
            'created': 'created_by',
            'owns': 'owns'
        }
        
        for pattern, predicate in predicate_mapping.items():
            if pattern in key_lower:
                return predicate
        
        return key_lower.replace(' ', '_')

# Использование адаптера
custom_adapter = CustomDataAdapter({
    "source_type": "json",
    "encoding": "utf-8"
})

# Пример данных для конвертации
sample_data = {
    "product_name": "Смартфон iPhone 15",
    "manufacturer": "Apple",
    "release_date": "2023-09-15",
    "features": ["Face ID", "Wireless Charging", "5G"],
    "specifications": {
        "screen_size": "6.1 inch",
        "storage": "128GB",
        "ram": "6GB"
    }
}

# Конвертация в разные форматы
entities = custom_adapter.adapt(sample_data, "neurograph_entities")
relations = custom_adapter.adapt(sample_data, "neurograph_relations")
memory_items = custom_adapter.adapt(sample_data, "memory_items")

print("Сущности:")
for entity in entities:
    print(f"  - {entity['text']} ({entity['entity_type']})")

print("\nОтношения:")
for relation in relations:
    print(f"  - {relation['subject']} {relation['predicate']} {relation['object']}")

print("\nЭлементы памяти:")
for item in memory_items:
    print(f"  - {item['content']}")
```

---

# 🚀 Развертывание

## Docker Deployment

### Базовый Dockerfile
```dockerfile
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя приложения
RUN useradd --create-home --shell /bin/bash neurograph
USER neurograph
WORKDIR /home/neurograph

# Копирование и установка зависимостей
COPY --chown=neurograph:neurograph requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Установка NeuroGraph
COPY --chown=neurograph:neurograph . .
RUN pip install --user -e .

# Создание необходимых директорий
RUN mkdir -p data logs config

# Переменные окружения
ENV PYTHONPATH=/home/neurograph
ENV NEUROGRAPH_DATA_DIR=/home/neurograph/data
ENV NEUROGRAPH_LOG_DIR=/home/neurograph/logs
ENV NEUROGRAPH_CONFIG_DIR=/home/neurograph/config

# Копирование конфигурации
COPY --chown=neurograph:neurograph config/production.json config/

# Порт приложения
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from app import health_check; health_check()" || exit 1

# Команда запуска
CMD ["python", "app.py"]
```

### Docker Compose для продакшена
```yaml
version: '3.8'

services:
  neurograph-app:
    build: .
    container_name: neurograph-app
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - NEUROGRAPH_ENV=production
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./data:/home/neurograph/data
      - ./logs:/home/neurograph/logs
      - ./config:/home/neurograph/config
    depends_on:
      - postgres
      - redis
    networks:
      - neurograph-network
    
  postgres:
    image: postgres:15
    container_name: neurograph-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: neurograph
      POSTGRES_USER: neurograph
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - neurograph-network
    
  redis:
    image: redis:7-alpine
    container_name: neurograph-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - neurograph-network
    
  nginx:
    image: nginx:alpine
    container_name: neurograph-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - neurograph-app
    networks:
      - neurograph-network

volumes:
  postgres_data:
  redis_data:

networks:
  neurograph-network:
    driver: bridge
```

### Nginx конфигурация
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream neurograph_backend {
        server neurograph-app:8080;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://neurograph_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 60s;
        }
        
        # Health check
        location /health {
            proxy_pass http://neurograph_backend;
            access_log off;
        }
        
        # Static files (если есть)
        location /static/ {
            alias /var/www/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

## Kubernetes Deployment

### Deployment манифест
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neurograph-app
  labels:
    app: neurograph
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neurograph
  template:
    metadata:
      labels:
        app: neurograph
    spec:
      containers:
      - name: neurograph
        image: neurograph/neurograph:latest
        ports:
        - containerPort: 8080
        env:
        - name: NEUROGRAPH_ENV
          value: "production"
        - name: POSTGRES_HOST
          value: "postgres-service"
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /home/neurograph/config
        - name: data-volume
          mountPath: /home/neurograph/data
      volumes:
      - name: config-volume
        configMap:
          name: neurograph-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: neurograph-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: neurograph-service
spec:
  selector:
    app: neurograph
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: neurograph-config
data:
  production.json: |
    {
      "engine_name": "k8s_neurograph",
      "components": {
        "memory": {
          "params": {
            "stm_capacity": 200,
            "ltm_capacity": 50000
          }
        }
      },
      "performance": {
        "max_concurrent_requests": 50,
        "enable_caching": true
      }
    }
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: neurograph-data-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### HorizontalPodAutoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neurograph-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neurograph-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

# 🔍 Troubleshooting

## Распространенные проблемы и решения

### Проблемы с установкой

#### Ошибка: "No module named 'neurograph'"
```bash
# Решение 1: Установка в development режиме
pip install -e .

# Решение 2: Проверка PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/neurograph"

# Решение 3: Переустановка
pip uninstall neurograph
pip install neurograph
```

#### Ошибка: "Unable to build wheels"
```bash
# Установка build tools
pip install --upgrade pip setuptools wheel

# Для Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# Для CentOS/RHEL
sudo yum install gcc python3-devel

# Для macOS
xcode-select --install
```

### Проблемы с памятью

#### OutOfMemoryError при работе с большими данными
```python
# Решение: Использование lightweight конфигурации
from neurograph.integration import create_lightweight_engine

engine = create_lightweight_engine()

# Или настройка параметров памяти
from neurograph.integration import IntegrationConfig

config = IntegrationConfig(
    components={
        "memory": {
            "params": {
                "stm_capacity": 50,    # Уменьшено
                "ltm_capacity": 1000   # Уменьшено
            }
        }
    }
)
```

#### Медленная работа системы
```python
# Решение: Включение кеширования и оптимизация
config = IntegrationConfig(
    performance={
        "enable_caching": True,
        "cache_ttl": 600,
        "max_concurrent_requests": 5  # Ограничение нагрузки
    },
    components={
        "nlp": {
            "params": {
                "use_spacy": False  # Отключение spaCy для экономии ресурсов
            }
        }
    }
)
```

### Проблемы с компонентами

#### Ошибка: "Component not found"
```python
# Проверка доступности компонента
engine = create_default_engine()
available_components = engine.provider.get_available_components()
print("Доступные компоненты:", available_components)

# Принудительная инициализация компонента
try:
    nlp = engine.get_component('nlp')
except Exception as e:
    print(f"Ошибка получения NLP: {e}")
    # Попытка повторной инициализации
    engine.provider.initialize_component('nlp')
```

#### Ошибка: "Failed to initialize spaCy model"
```bash
# Установка языковой модели spaCy
python -m spacy download ru_core_news_sm

# Или использование fallback конфигурации
```

```python
config = IntegrationConfig(
    components={
        "nlp": {
            "params": {
                "use_spacy": False,  # Отключить spaCy
                "fallback_to_rules": True
            }
        }
    }
)
```

### Проблемы с производительностью

#### Долгое время отклика
```python
# Диагностика производительности
import time

start_time = time.time()
response = engine.query("test question")
end_time = time.time()

print(f"Время обработки: {end_time - start_time:.2f} сек")
print(f"Системная статистика: {response.structured_data}")

# Профилирование отдельных компонентов
components_time = {}
for component_name in ['nlp', 'memory', 'semgraph']:
    start = time.time()
    component = engine.get_component(component_name)
    # Выполнение операции компонента
    components_time[component_name] = time.time() - start

print("Время по компонентам:", components_time)
```

#### Высокое потребление CPU
```python
# Мониторинг ресурсов
from neurograph.core.resources import get_resource_usage

usage = get_resource_usage()
print(f"CPU: {usage['cpu_percent']}%")
print(f"Память: {usage['memory_rss'] / 1024 / 1024:.1f} MB")

# Оптимизация через конфигурацию
config = IntegrationConfig(
    components={
        "propagation": {
            "params": {
                "max_iterations": 50,  # Уменьшено
                "activation_threshold": 0.3  # Увеличено
            }
        }
    }
)
```

### Проблемы с данными

#### Ошибка: "Failed to save graph data"
```python
# Проверка прав доступа
import os

data_dir = "/path/to/data"
if not os.access(data_dir, os.W_OK):
    print(f"Нет прав записи в {data_dir}")

# Создание директории, если не существует
os.makedirs(data_dir, exist_ok=True)

# Использование временной директории
import tempfile
temp_dir = tempfile.mkdtemp()
config = IntegrationConfig(
    components={
        "semgraph": {
            "type": "persistent",
            "params": {
                "file_path": f"{temp_dir}/graph.json"
            }
        }
    }
)
```

#### Повреждение данных графа
```python
# Резервное копирование и восстановление
def backup_graph_data(engine, backup_path):
    """Создание резервной копии данных графа"""
    graph = engine.get_component('semgraph')
    
    if hasattr(graph, 'save'):
        graph.save(backup_path)
        print(f"Резервная копия сохранена: {backup_path}")

def restore_graph_data(engine, backup_path):
    """Восстановление данных графа из резервной копии"""
    try:
        # Создание нового графа из резервной копии
        from neurograph.semgraph.impl.memory_graph import MemoryEfficientSemGraph
        restored_graph = MemoryEfficientSemGraph.load(backup_path)
        
        # Замена компонента
        engine.provider.register_component('semgraph', restored_graph)
        print(f"Граф восстановлен из: {backup_path}")
        
    except Exception as e:
        print(f"Ошибка восстановления: {e}")

# Использование
backup_graph_data(engine, "backup_graph.json")
# При необходимости:
# restore_graph_data(engine, "backup_graph.json")
```

## Логи и диагностика

### Настройка детального логирования
```python
from neurograph.core.logging import setup_logging

# Максимально подробное логирование
setup_logging(
    level="DEBUG",
    log_file="neurograph_debug.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)

# Включение логирования для конкретных модулей
import logging
logging.getLogger("neurograph.integration").setLevel(logging.DEBUG)
logging.getLogger("neurograph.memory").setLevel(logging.DEBUG)
logging.getLogger("neurograph.semgraph").setLevel(logging.DEBUG)
```

### Система мониторинга
```python
from neurograph.integration import ComponentMonitor, HealthChecker

# Создание системы мониторинга
monitor = ComponentMonitor(check_interval=10.0)
health_checker = HealthChecker()

# Запуск мониторинга
monitor.start_monitoring(engine.provider)

# Проверка здоровья системы
health = health_checker.get_overall_health()
print(f"Общее состояние: {health['status']}")

if health['status'] != 'healthy':
    print("Проблемные компоненты:")
    for component, status in health['components'].items():
        if status['status'] != 'healthy':
            print(f"  - {component}: {status['error']}")

# Получение детального отчета
report = monitor.get_monitoring_report()
print(f"Метрики системы:")
print(f"  - Время работы: {report['uptime_seconds']} сек")
print(f"  - Всего алертов: {len(report['recent_alerts'])}")
```

### Инструменты отладки

```python
class NeuroGraphDebugger:
    """Утилита для отладки NeuroGraph"""
    
    def __init__(self, engine):
        self.engine = engine
    
    def diagnose_system(self):
        """Полная диагностика системы"""
        
        print("=== Диагностика NeuroGraph ===")
        
        # Проверка компонентов
        self._check_components()
        
        # Проверка конфигурации
        self._check_configuration()
        
        # Проверка ресурсов
        self._check_resources()
        
        # Тест базовой функциональности
        self._test_basic_functionality()
    
    def _check_components(self):
        """Проверка доступности компонентов"""
        
        print("\n--- Проверка компонентов ---")
        
        required_components = ['nlp', 'semgraph', 'memory', 'contextvec', 'processor']
        
        for component_name in required_components:
            try:
                component = self.engine.get_component(component_name)
                print(f"✅ {component_name}: OK")
            except Exception as e:
                print(f"❌ {component_name}: ОШИБКА - {e}")
    
    def _check_configuration(self):
        """Проверка конфигурации"""
        
        print("\n--- Проверка конфигурации ---")
        
        try:
            config = self.engine.config
            print(f"✅ Конфигурация загружена: {config.engine_name}")
            
            # Проверка критичных настроек
            memory_config = config.components.get('memory', {}).get('params', {})
            stm_capacity = memory_config.get('stm_capacity', 0)
            ltm_capacity = memory_config.get('ltm_capacity', 0)
            
            if stm_capacity > 0 and ltm_capacity > 0:
                print(f"✅ Память настроена: STM={stm_capacity}, LTM={ltm_capacity}")
            else:
                print("⚠️  Память не настроена или имеет нулевую емкость")
            
        except Exception as e:
            print(f"❌ Конфигурация: ОШИБКА - {e}")
    
    def _check_resources(self):
        """Проверка системных ресурсов"""
        
        print("\n--- Проверка ресурсов ---")
        
        try:
            from neurograph.core.resources import get_resource_usage
            usage = get_resource_usage()
            
            cpu_percent = usage.get('cpu_percent', 0)
            memory_mb = usage.get('memory_rss', 0) / 1024 / 1024
            
            print(f"CPU: {cpu_percent:.1f}%")
            print(f"Память: {memory_mb:.1f} MB")
            
            # Предупреждения
            if cpu_percent > 80:
                print("⚠️  Высокая загрузка CPU")
            if memory_mb > 1000:
                print("⚠️  Высокое потребление памяти")
            
        except Exception as e:
            print(f"❌ Ресурсы: ОШИБКА - {e}")
    
    def _test_basic_functionality(self):
        """Тест базовой функциональности"""
        
        print("\n--- Тест функциональности ---")
        
        try:
            # Тест обучения
            learn_response = self.engine.learn("Тест: NeuroGraph работает правильно")
            if learn_response.success:
                print("✅ Обучение: OK")
            else:
                print(f"❌ Обучение: ОШИБКА - {learn_response.error_message}")
            
            # Тест запроса
            query_response = self.engine.query("Что такое тест?")
            if query_response.success:
                print("✅ Запросы: OK")
                print(f"   Время ответа: {query_response.processing_time:.2f}с")
            else:
                print(f"❌ Запросы: ОШИБКА - {query_response.error_message}")
            
        except Exception as e:
            print(f"❌ Функциональность: ОШИБКА - {e}")
    
    def benchmark_performance(self, num_operations=10):
        """Бенчмарк производительности"""
        
        print(f"\n=== Бенчмарк производительности ({num_operations} операций) ===")
        
        import time
        
        # Бенчмарк обучения
        learn_times = []
        for i in range(num_operations):
            start = time.time()
            self.engine.learn(f"Тестовое знание номер {i}")
            learn_times.append(time.time() - start)
        
        # Бенчмарк запросов
        query_times = []
        for i in range(num_operations):
            start = time.time()
            self.engine.query(f"Вопрос номер {i}")
            query_times.append(time.time() - start)
        
        # Статистика
        avg_learn_time = sum(learn_times) / len(learn_times)
        avg_query_time = sum(query_times) / len(query_times)
        
        print(f"Среднее время обучения: {avg_learn_time:.3f}с")
        print(f"Среднее время запроса: {avg_query_time:.3f}с")
        print(f"Пропускная способность обучения: {1/avg_learn_time:.1f} оп/с")
        print(f"Пропускная способность запросов: {1/avg_query_time:.1f} оп/с")

# Использование отладчика
debugger = NeuroGraphDebugger(engine)
debugger.diagnose_system()
debugger.benchmark_performance(num_operations=5)
```

## Часто задаваемые вопросы (FAQ)

### Q: Как увеличить производительность системы?

**A:** Есть несколько способов:

1. **Оптимизация конфигурации:**
```python
config = IntegrationConfig(
    performance={
        "enable_caching": True,
        "cache_ttl": 900,  # Увеличить время кеширования
        "max_concurrent_requests": 20
    },
    components={
        "memory": {
            "params": {
                "stm_capacity": 50,  # Уменьшить для экономии ресурсов
                "ltm_capacity": 5000
            }
        }
    }
)
```

2. **Использование lightweight конфигурации** для небольших задач

3. **Мониторинг и профилирование** для выявления узких мест

### Q: Как сохранить данные между перезапусками?

**A:** Используйте persistent конфигурацию:

```python
config = IntegrationConfig(
    components={
        "semgraph": {
            "type": "persistent",
            "params": {
                "file_path": "./data/knowledge_graph.json",
                "auto_save_interval": 300.0
            }
        },
        "memory": {
            "params": {
                "use_persistent_storage": True,
                "storage_path": "./data/memory/"
            }
        }
    }
)
```

### Q: Как интегрировать собственные языковые модели?

**A:** Создайте адаптер для вашей модели:

```python
from neurograph.contextvec.adapters.base import BaseVectorAdapter

class CustomModelAdapter(BaseVectorAdapter):
    def __init__(self, model_path):
        self.model = self.load_custom_model(model_path)
    
    def encode(self, text):
        return self.model.encode(text)
    
    def encode_batch(self, texts):
        return self.model.encode_batch(texts)

# Регистрация адаптера
from neurograph.contextvec import ContextVectorsFactory
ContextVectorsFactory.register_adapter("custom_model", CustomModelAdapter)
```

### Q: Как масштабировать систему для больших нагрузок?

**A:** Используйте несколько подходов:

1. **Горизонтальное масштабирование через Kubernetes**
2. **Балансировка нагрузки** между несколькими экземплярами
3. **Кеширование на уровне инфраструктуры** (Redis/Memcached)
4. **Асинхронная обработка** тяжелых операций

### Q: Что делать, если система потребляет слишком много памяти?

**A:** Оптимизируйте использование памяти:

```python
# 1. Уменьшите размеры памяти
config.components["memory"]["params"]["stm_capacity"] = 25
config.components["memory"]["params"]["ltm_capacity"] = 1000

# 2. Отключите семантическое индексирование
config.components["memory"]["params"]["use_semantic_indexing"] = False

# 3. Используйте простые алгоритмы
config.components["nlp"]["params"]["use_spacy"] = False

# 4. Настройте более агрессивное забывание
config.components["memory"]["params"]["consolidation_interval"] = 60.0
```

---

# 📚 Дополнительные ресурсы

## Полезные ссылки

- **Официальная документация**: https://neurograph.readthedocs.io
- **GitHub репозиторий**: https://github.com/neurograph/neurograph
- **Примеры использования**: https://github.com/neurograph/examples
- **Сообщество**: https://discord.gg/neurograph
- **Блог разработчиков**: https://blog.neurograph.ai

## Обучающие материалы

### Видеоуроки
1. "Введение в NeuroGraph" - базовые концепции
2. "Создание первого ИИ-ассистента" - пошаговое руководство
3. "Продвинутая настройка и оптимизация" - лучшие практики
4. "Развертывание в продакшене" - DevOps аспекты

### Статьи и туториалы
1. "Нейросимволические системы: теория и практика"
2. "Биоморфная архитектура памяти в NeuroGraph"
3. "Интеграция с современными языковыми моделями"
4. "Масштабирование ИИ-систем в облаке"

## Поддержка сообщества

### Как получить помощь
1. **GitHub Issues** - для багов и запросов функций
2. **Discord сервер** - для вопросов сообщества
3. **Stack Overflow** - тег `neurograph`
4. **Документация** - подробные гайды и API reference

### Как внести вклад
1. **Fork** репозитория
2. **Создайте ветку** для новой функции
3. **Напишите тесты** для вашего кода
4. **Отправьте Pull Request** с описанием изменений

---

# 🎯 Заключение

NeuroGraph предоставляет мощную и гибкую платформу для создания интеллектуальных приложений. Эта документация охватывает основные аспекты работы с системой:

## ✅ Что вы изучили

- **Быстрый старт** - создание первого ИИ-ассистента за минуты
- **API Reference** - полное описание всех интерфейсов
- **Практические примеры** - реальные сценарии использования
- **Конфигурация** - настройка под различные задачи
- **Интеграция** - подключение к существующим системам
- **Расширение** - создание собственных модулей
- **Развертывание** - продакшн-готовые решения
- **Troubleshooting** - решение типичных проблем

## 🚀 Следующие шаги

1. **Экспериментируйте** с примерами кода
2. **Изучите** исходный код для глубокого понимания
3. **Создайте** свое первое приложение
4. **Поделитесь** опытом с сообществом
5. **Внесите вклад** в развитие проекта

## 💡 Помните

- NeuroGraph - это не просто библиотека, а целостная экосистема
- Система построена на принципах модульности и расширяемости
- Биоморфная архитектура делает ИИ более понятным и предсказуемым
- Сообщество всегда готово помочь в решении сложных задач

**Добро пожаловать в будущее нейросимволического ИИ!** 🧠✨

---

*Документация обновлена для версии NeuroGraph 1.0.0*# NeuroGraph Developer Guide