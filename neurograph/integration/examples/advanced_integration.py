"""
Продвинутые примеры интеграции между модулями.
"""

import time
import asyncio
from typing import List, Dict, Any
from neurograph.integration import (
    NeuroGraphEngine,
    ComponentProvider,
    ProcessingRequest,
    IntegrationConfig
)


class CustomKnowledgeSystem:
    """Пример пользовательской системы знаний на базе NeuroGraph."""
    
    def __init__(self, config_path: str = None):
        self.config = self._create_custom_config()
        self.provider = ComponentProvider()
        self.engine = NeuroGraphEngine(self.provider)
        
        if not self.engine.initialize(self.config):
            raise RuntimeError("Не удалось инициализировать систему")
        
        # Статистика работы
        self.stats = {
            "documents_processed": 0,
            "queries_answered": 0,
            "knowledge_items": 0,
            "start_time": time.time()
        }
    
    def _create_custom_config(self) -> IntegrationConfig:
        """Создание пользовательской конфигурации."""
        return IntegrationConfig(
            engine_name="custom_knowledge_system",
            components={
                "semgraph": {
                    "type": "persistent",
                    "params": {
                        "file_path": "knowledge_base.json",
                        "auto_save_interval": 180.0
                    }
                },
                "memory": {
                    "params": {
                        "stm_capacity": 150,
                        "ltm_capacity": 15000,
                        "use_semantic_indexing": True,
                        "auto_consolidation": True
                    }
                },
                "processor": {
                    "type": "pattern_matching",
                    "params": {
                        "confidence_threshold": 0.6,
                        "enable_explanations": True
                    }
                }
            },
            max_concurrent_requests=15,
            enable_caching=True,
            enable_metrics=True
        )
    
    def process_document(self, document_text: str, document_title: str = None) -> Dict[str, Any]:
        """Обработка и индексация документа."""
        request = ProcessingRequest(
            content=document_text,
            request_type="learning",
            metadata={
                "document_title": document_title,
                "processing_timestamp": time.time()
            }
        )
        
        response = self.engine.process_request(request)
        
        if response.success:
            self.stats["documents_processed"] += 1
            
            # Извлекаем ключевую информацию
            nlp_data = response.structured_data.get("nlp", {})
            graph_data = response.structured_data.get("graph", {})
            
            self.stats["knowledge_items"] += len(nlp_data.get("entities", []))
        
        return {
            "success": response.success,
            "entities_found": len(nlp_data.get("entities", [])),
            "relations_found": len(nlp_data.get("relations", [])),
            "processing_time": response.processing_time,
            "error": response.error_message
        }
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Ответ на вопрос пользователя."""
        request = ProcessingRequest(
            content=question,
            request_type="query",
            response_format="conversational",
            explanation_level="detailed"
        )
        
        response = self.engine.process_request(request)
        
        if response.success:
            self.stats["queries_answered"] += 1
        
        return {
            "answer": response.primary_response,
            "confidence": response.confidence,
            "sources": response.sources,
            "explanation": response.explanation,
            "processing_time": response.processing_time,
            "success": response.success
        }
    
    def get_knowledge_graph_summary(self) -> Dict[str, Any]:
        """Получение сводки по графу знаний."""
        try:
            graph = self.provider.get_component("semgraph")
            memory = self.provider.get_component("memory")
            
            return {
                "graph_nodes": len(graph.get_all_nodes()),
                "graph_edges": len(graph.get_all_edges()),
                "memory_size": memory.size(),
                "memory_stats": memory.get_memory_statistics()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Получение статистики системы."""
        uptime = time.time() - self.stats["start_time"]
        
        base_stats = self.stats.copy()
        base_stats["uptime_seconds"] = uptime
        base_stats["engine_health"] = self.engine.get_health_status()
        base_stats["knowledge_summary"] = self.get_knowledge_graph_summary()
        
        return base_stats
    
    def shutdown(self):
        """Корректное завершение работы."""
        return self.engine.shutdown()


def example_custom_knowledge_system():
    """Пример использования пользовательской системы знаний."""
    print("=== Пользовательская система знаний ===")
    
    # Создание системы
    knowledge_system = CustomKnowledgeSystem()
    
    # Обучение на документах
    documents = [
        ("Python", 
         "Python - высокоуровневый язык программирования общего назначения. "
         "Он был создан Гвидо ван Россумом в 1991 году. Python широко используется "
         "для веб-разработки, анализа данных и машинного обучения."),
        
        ("Django",
         "Django - это веб-фреймворк высокого уровня для Python, который способствует "
         "быстрой разработке и чистому, прагматичному дизайну. Он был создан в 2003 году "
         "и следует принципу DRY (Don't Repeat Yourself)."),
        
        ("Машинное обучение",
         "Машинное обучение - это область искусственного интеллекта, которая дает "
         "компьютерам способность обучаться без явного программирования. Оно использует "
         "алгоритмы для анализа данных, выявления закономерностей и принятия решений."),
        
        ("TensorFlow",
         "TensorFlow - это открытая платформа машинного обучения, разработанная Google. "
         "Она предоставляет всеобъемлющую экосистему инструментов, библиотек и ресурсов "
         "для создания и развертывания приложений машинного обучения.")
    ]
    
    print("Обучение системы на документах...")
    for title, content in documents:
        result = knowledge_system.process_document(content, title)
        print(f"📄 {title}: {'✓' if result['success'] else '✗'} "
              f"({result['entities_found']} сущностей, {result['processing_time']:.3f}с)")
    
    # Задаем вопросы
    questions = [
        "Что такое Python?",
        "Кто создал Python?",
        "Для чего используется Django?",
        "Что такое машинное обучение?",
        "Какая связь между Python и машинным обучением?",
        "Расскажи о TensorFlow"
    ]
    
    print(f"\nОтветы на вопросы:")
    for question in questions:
        result = knowledge_system.answer_question(question)
        print(f"\n❓ {question}")
        print(f"💬 {result['answer']}")
        if result['confidence'] > 0:
            print(f"🎯 Уверенность: {result['confidence']:.2f}")
        print(f"⏱️ Время: {result['processing_time']:.3f}с")
    
    # Статистика системы
    print(f"\n=== Статистика системы ===")
    stats = knowledge_system.get_system_statistics()
    print(f"📚 Документов обработано: {stats['documents_processed']}")
    print(f"❓ Вопросов отвечено: {stats['queries_answered']}")
    print(f"🧠 Элементов знаний: {stats['knowledge_items']}")
    print(f"⏰ Время работы: {stats['uptime_seconds']:.1f}с")
    
    knowledge_summary = stats['knowledge_summary']
    if 'graph_nodes' in knowledge_summary:
        print(f"🕸️ Узлов в графе: {knowledge_summary['graph_nodes']}")
        print(f"🔗 Связей в графе: {knowledge_summary['graph_edges']}")
        print(f"💾 Размер памяти: {knowledge_summary['memory_size']}")
    
    # Завершение
    knowledge_system.shutdown()


class ConversationalAgent:
    """Пример разговорного агента на базе NeuroGraph."""
    
    def __init__(self):
        # Специальная конфигурация для диалогов
        config = IntegrationConfig(
            engine_name="conversational_agent",
            components={
                "nlp": {
                    "params": {
                        "language": "ru",
                        "conversation_mode": True
                    }
                },
                "memory": {
                    "params": {
                        "stm_capacity": 50,  # Больше для контекста диалога
                        "ltm_capacity": 5000,
                        "conversation_memory": True
                    }
                },
                "processor": {
                    "type": "pattern_matching",
                    "params": {
                        "confidence_threshold": 0.4,  # Ниже для разговоров
                        "conversation_rules": True
                    }
                }
            },
            max_concurrent_requests=5,
            enable_caching=True
        )
        
        self.provider = ComponentProvider()
        self.engine = NeuroGraphEngine(self.provider)
        self.engine.initialize(config)
        
        # Контекст диалога
        self.conversation_context = {
            "user_name": None,
            "topics_discussed": [],
            "last_responses": [],
            "session_start": time.time()
        }
    
    def chat(self, user_input: str, user_name: str = None) -> str:
        """Диалог с пользователем."""
        if user_name:
            self.conversation_context["user_name"] = user_name
        
        # Создаем запрос с контекстом диалога
        request = ProcessingRequest(
            content=user_input,
            request_type="query",
            response_format="conversational",
            context=self.conversation_context.copy(),
            metadata={
                "conversation_turn": len(self.conversation_context["last_responses"]),
                "user_name": self.conversation_context["user_name"]
            }
        )
        
        response = self.engine.process_request(request)
        
        # Обновляем контекст диалога
        self._update_conversation_context(user_input, response.primary_response)
        
        return response.primary_response
    
    def _update_conversation_context(self, user_input: str, agent_response: str):
        """Обновление контекста диалога."""
        # Добавляем топики (простая эвристика)
        words = user_input.lower().split()
        important_words = [w for w in words if len(w) > 4 and w not in ['который', 'потому', 'сказать']]
        self.conversation_context["topics_discussed"].extend(important_words[:2])
        
        # Ограничиваем размер контекста
        if len(self.conversation_context["topics_discussed"]) > 20:
            self.conversation_context["topics_discussed"] = (
                self.conversation_context["topics_discussed"][-10:]
            )
        
        # Сохраняем последние реплики
        self.conversation_context["last_responses"].append({
            "user": user_input,
            "agent": agent_response,
            "timestamp": time.time()
        })
        
        # Ограничиваем историю
        if len(self.conversation_context["last_responses"]) > 5:
            self.conversation_context["last_responses"] = (
                self.conversation_context["last_responses"][-3:]
            )
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Получение сводки по диалогу."""
        duration = time.time() - self.conversation_context["session_start"]
        
        return {
            "user_name": self.conversation_context["user_name"],
            "duration_seconds": duration,
            "turns_count": len(self.conversation_context["last_responses"]),
            "topics_discussed": list(set(self.conversation_context["topics_discussed"])),
            "last_interaction": (
                self.conversation_context["last_responses"][-1]["timestamp"]
                if self.conversation_context["last_responses"] else None
            )
        }
    
    def shutdown(self):
        """Завершение работы агента."""
        return self.engine.shutdown()


def example_conversational_agent():
    """Пример разговорного агента."""
    print("\n=== Разговорный агент ===")
    
    agent = ConversationalAgent()
    
    # Сначала обучаем агента базовым знаниям
    agent.engine.learn("Меня зовут NeuroGraph. Я - интеллектуальный ассистент.")
    agent.engine.learn("Я помогаю отвечать на вопросы и обрабатывать информацию.")
    agent.engine.learn("Python - это язык программирования.")
    
    # Диалог
    conversation = [
        ("Привет! Как дела?", "Алиса"),
        ("Расскажи о себе", None),
        ("Что ты знаешь о программировании?", None),
        ("А что такое Python?", None),
        ("Спасибо за информацию!", None),
        ("До свидания!", None)
    ]
    
    print("💬 Начинаем диалог:")
    for user_message, user_name in conversation:
        print(f"\n👤 {user_name or 'Пользователь'}: {user_message}")
        
        response = agent.chat(user_message, user_name)
        print(f"🤖 NeuroGraph: {response}")
        
        time.sleep(0.5)  # Имитация паузы в диалоге
    
    # Сводка по диалогу
    summary = agent.get_conversation_summary()
    print(f"\n=== Сводка по диалогу ===")
    print(f"👤 Пользователь: {summary['user_name'] or 'Анонимный'}")
    print(f"⏰ Длительность: {summary['duration_seconds']:.1f}с")
    print(f"💬 Реплик: {summary['turns_count']}")
    print(f"📝 Обсуждавшиеся темы: {', '.join(summary['topics_discussed'][:5])}")
    
    agent.shutdown()


class BatchProcessor:
    """Пример пакетной обработки документов."""
    
    def __init__(self):
        # Конфигурация для пакетной обработки
        config = IntegrationConfig(
            engine_name="batch_processor",
            components={
                "nlp": {
                    "params": {
                        "language": "ru",
                        "batch_mode": True
                    }
                }
            },
            max_concurrent_requests=20,  # Больше для пакетной обработки
            enable_caching=True,
            enable_metrics=True
        )
        
        self.provider = ComponentProvider()
        self.engine = NeuroGraphEngine(self.provider)
        self.engine.initialize(config)
        
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0.0
        }
    
    def process_batch(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Пакетная обработка документов."""
        results = []
        start_time = time.time()
        
        print(f"📦 Начинаем пакетную обработку {len(documents)} документов...")
        
        for i, doc in enumerate(documents, 1):
            try:
                doc_start = time.time()
                
                request = ProcessingRequest(
                    content=doc["content"],
                    request_type="learning",
                    metadata={
                        "title": doc.get("title", f"Документ {i}"),
                        "batch_id": doc.get("id", str(i)),
                        "batch_position": i
                    }
                )
                
                response = self.engine.process_request(request)
                doc_time = time.time() - doc_start
                
                result = {
                    "id": doc.get("id", str(i)),
                    "title": doc.get("title", f"Документ {i}"),
                    "success": response.success,
                    "processing_time": doc_time,
                    "entities_count": 0,
                    "relations_count": 0
                }
                
                if response.success:
                    nlp_data = response.structured_data.get("nlp", {})
                    result["entities_count"] = len(nlp_data.get("entities", []))
                    result["relations_count"] = len(nlp_data.get("relations", []))
                    self.processing_stats["successful"] += 1
                else:
                    result["error"] = response.error_message
                    self.processing_stats["failed"] += 1
                
                results.append(result)
                self.processing_stats["total_processed"] += 1
                
                # Прогресс
                if i % 5 == 0 or i == len(documents):
                    progress = (i / len(documents)) * 100
                    print(f"📊 Прогресс: {progress:.1f}% ({i}/{len(documents)})")
                
            except Exception as e:
                results.append({
                    "id": doc.get("id", str(i)),
                    "title": doc.get("title", f"Документ {i}"),
                    "success": False,
                    "error": str(e),
                    "processing_time": 0.0
                })
                self.processing_stats["failed"] += 1
        
        self.processing_stats["total_time"] = time.time() - start_time
        return results
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Получение статистики пакетной обработки."""
        total = self.processing_stats["total_processed"]
        avg_time = (
            self.processing_stats["total_time"] / total
            if total > 0 else 0
        )
        
        return {
            "total_documents": total,
            "successful": self.processing_stats["successful"],
            "failed": self.processing_stats["failed"],
            "success_rate": (
                self.processing_stats["successful"] / total
                if total > 0 else 0
            ),
            "total_time": self.processing_stats["total_time"],
            "average_time_per_document": avg_time,
            "throughput_docs_per_second": (
                total / self.processing_stats["total_time"]
                if self.processing_stats["total_time"] > 0 else 0
            )
        }
    
    def shutdown(self):
        """Завершение работы процессора."""
        return self.engine.shutdown()


def example_batch_processing():
    """Пример пакетной обработки документов."""
    print("\n=== Пакетная обработка ===")
    
    processor = BatchProcessor()
    
    # Тестовые документы
    documents = [
        {
            "id": "doc_1",
            "title": "Основы Python",
            "content": "Python - интерпретируемый язык программирования высокого уровня. "
                      "Он поддерживает множественное наследование и динамическую типизацию."
        },
        {
            "id": "doc_2", 
            "title": "Веб-разработка",
            "content": "Django и Flask - популярные веб-фреймворки для Python. "
                      "Они позволяют быстро создавать веб-приложения."
        },
        {
            "id": "doc_3",
            "title": "Машинное обучение",
            "content": "TensorFlow и PyTorch - ведущие библиотеки для машинного обучения. "
                      "Они предоставляют инструменты для создания нейронных сетей."
        },
        {
            "id": "doc_4",
            "title": "Анализ данных", 
            "content": "Pandas и NumPy - основные библиотеки для анализа данных в Python. "
                      "Они обеспечивают эффективную работу с большими наборами данных."
        },
        {
            "id": "doc_5",
            "title": "Искусственный интеллект",
            "content": "ИИ включает машинное обучение, обработку естественного языка и компьютерное зрение. "
                      "Python широко используется в разработке ИИ-систем."
        }
    ]
    
    # Обработка
    results = processor.process_batch(documents)
    
    # Анализ результатов
    print(f"\n=== Результаты обработки ===")
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['title']}: {result['entities_count']} сущностей, "
              f"{result['relations_count']} отношений ({result['processing_time']:.3f}с)")
        
        if not result["success"] and "error" in result:
            print(f"   Ошибка: {result['error']}")
    
    # Статистика
    stats = processor.get_batch_statistics()
    print(f"\n=== Статистика пакетной обработки ===")
    print(f"📄 Всего документов: {stats['total_documents']}")
    print(f"✅ Успешно: {stats['successful']}")
    print(f"❌ Ошибок: {stats['failed']}")
    print(f"📊 Успешность: {stats['success_rate']:.1%}")
    print(f"⏱️ Общее время: {stats['total_time']:.2f}с")
    print(f"🚀 Среднее время на документ: {stats['average_time_per_document']:.3f}с")
    print(f"📈 Производительность: {stats['throughput_docs_per_second']:.1f} док/с")
    
    processor.shutdown()


def main():
    """Запуск продвинутых примеров."""
    print("🚀 Продвинутые примеры интеграции NeuroGraph\n")
    
    examples = [
        ("Пользовательская система знаний", example_custom_knowledge_system),
        ("Разговорный агент", example_conversational_agent),
        ("Пакетная обработка документов", example_batch_processing)
    ]
    
    for name, example_func in examples:
        print(f"📋 {name}")
        try:
            example_func()
        except Exception as e:
            print(f"❌ Ошибка в примере '{name}': {e}")
        
        print("\n" + "="*70)
    
    print("✅ Все продвинутые примеры завершены!")


if __name__ == "__main__":
    main()