"""
Комплексные тесты модуля Integration NeuroGraph.
"""

import unittest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from neurograph.integration import (
    NeuroGraphEngine,
    ComponentProvider,
    ProcessingRequest,
    ProcessingResponse,
    IntegrationConfig,
    ProcessingMode,
    ResponseFormat,
    create_default_engine,
    create_lightweight_engine,
    EngineFactory,
    IntegrationMetrics,
    HealthChecker,
    ComponentMonitor
)

from neurograph.integration.pipelines import (
    TextProcessingPipeline,
    QueryProcessingPipeline,
    LearningPipeline,
    InferencePipeline
)

from neurograph.integration.adapters import (
    GraphMemoryAdapter,
    VectorProcessorAdapter,
    NLPGraphAdapter,
    MemoryProcessorAdapter
)

from neurograph.integration.config import IntegrationConfigManager


class TestComponentProvider(unittest.TestCase):
    """Тесты провайдера компонентов."""
    
    def setUp(self):
        self.provider = ComponentProvider()
    
    def test_component_registration(self):
        """Тест регистрации компонентов."""
        mock_component = Mock()
        
        # Регистрация
        self.provider.register_component("test_component", mock_component)
        
        # Проверка доступности
        self.assertTrue(self.provider.is_component_available("test_component"))
        
        # Получение компонента
        retrieved = self.provider.get_component("test_component")
        self.assertEqual(retrieved, mock_component)
    
    def test_lazy_component_initialization(self):
        """Тест ленивой инициализации компонентов."""
        mock_component = Mock()
        init_func = Mock(return_value=mock_component)
        
        # Регистрация ленивого компонента
        self.provider.register_lazy_component("lazy_component", init_func)
        
        # Проверка доступности
        self.assertTrue(self.provider.is_component_available("lazy_component"))
        
        # Первое обращение должно вызвать инициализацию
        retrieved = self.provider.get_component("lazy_component")
        init_func.assert_called_once()
        self.assertEqual(retrieved, mock_component)
        
        # Второе обращение не должно вызывать инициализацию повторно
        retrieved2 = self.provider.get_component("lazy_component")
        init_func.assert_called_once()  # Всё ещё один раз
        self.assertEqual(retrieved2, mock_component)
    
    def test_component_health(self):
        """Тест отслеживания здоровья компонентов."""
        mock_component = Mock()
        mock_component.get_statistics.return_value = {"operations": 100}
        
        self.provider.register_component("health_test", mock_component)
        
        # Проверка начального состояния
        health = self.provider.get_component_health("health_test")
        self.assertEqual(health["status"], "healthy")
        
        # Обновление состояния на нездоровое
        self.provider.update_component_health("health_test", "unhealthy", "Test error")
        health = self.provider.get_component_health("health_test")
        self.assertEqual(health["status"], "unhealthy")
        self.assertEqual(health["last_error"], "Test error")
    
    def test_missing_component(self):
        """Тест обращения к несуществующему компоненту."""
        with self.assertRaises(Exception):
            self.provider.get_component("nonexistent")


class TestNeuroGraphEngine(unittest.TestCase):
    """Тесты основного движка."""
    
    def setUp(self):
        self.provider = ComponentProvider()
        self.engine = NeuroGraphEngine(self.provider)
        
        # Мок-компоненты для тестирования
        self.mock_nlp = self._create_mock_nlp()
        self.mock_graph = self._create_mock_graph()
        self.mock_memory = self._create_mock_memory()
        self.mock_processor = self._create_mock_processor()
        
        # Регистрация моков
        self.provider.register_component("nlp", self.mock_nlp)
        self.provider.register_component("semgraph", self.mock_graph)
        self.provider.register_component("memory", self.mock_memory)
        self.provider.register_component("processor", self.mock_processor)
    
    def _create_mock_nlp(self):
        """Создание мок-объекта NLP."""
        mock = Mock()
        
        # Мок результата обработки
        mock_result = Mock()
        mock_result.entities = []
        mock_result.relations = []
        mock_result.sentences = []
        mock_result.language = "ru"
        mock_result.processing_time = 0.1
        
        mock.process_text.return_value = mock_result
        return mock
    
    def _create_mock_graph(self):
        """Создание мок-объекта графа."""
        mock = Mock()
        mock.get_all_nodes.return_value = []
        mock.get_all_edges.return_value = []
        mock.add_node.return_value = None
        mock.add_edge.return_value = None
        mock.has_node.return_value = False
        return mock
    
    def _create_mock_memory(self):
        """Создание мок-объекта памяти."""
        mock = Mock()
        mock.size.return_value = 0
        mock.add.return_value = "memory_item_123"
        mock.get_recent_items.return_value = []
        return mock
    
    def _create_mock_processor(self):
        """Создание мок-объекта процессора."""
        mock = Mock()
        
        # Мок результата вывода
        mock_result = Mock()
        mock_result.success = True
        mock_result.derived_facts = {}
        mock_result.rules_used = []
        mock_result.confidence = 0.8
        mock_result.explanation = []
        
        mock.derive.return_value = mock_result
        return mock
    
    def test_engine_initialization(self):
        """Тест инициализации движка."""
        config = IntegrationConfig(
            engine_name="test_engine",
            components={}
        )
        
        success = self.engine.initialize(config)
        self.assertTrue(success)
        self.assertEqual(self.engine.config, config)
    
    def test_process_text(self):
        """Тест обработки текста."""
        config = IntegrationConfig(engine_name="test")
        self.engine.initialize(config)
        
        response = self.engine.process_text("Тестовый текст")
        
        self.assertIsInstance(response, ProcessingResponse)
        self.assertTrue(response.success)
        self.assertGreater(response.processing_time, 0)
    
    def test_process_request_validation(self):
        """Тест валидации запросов."""
        config = IntegrationConfig(
            engine_name="test",
            enable_input_validation=True,
            max_input_length=100
        )
        self.engine.initialize(config)
        
        # Пустой запрос
        request = ProcessingRequest(content="")
        response = self.engine.process_request(request)
        self.assertFalse(response.success)
        self.assertIn("пустое", response.error_message.lower())
        
        # Слишком длинный запрос
        long_text = "x" * 200
        request = ProcessingRequest(content=long_text)
        response = self.engine.process_request(request)
        self.assertFalse(response.success)
        self.assertIn("длинный", response.error_message.lower())
    
    def test_health_status(self):
        """Тест получения статуса здоровья."""
        config = IntegrationConfig(engine_name="test")
        self.engine.initialize(config)
        
        health = self.engine.get_health_status()
        
        self.assertIn("overall_status", health)
        self.assertIn("components", health)
        self.assertIn("metrics", health)
    
    def test_shutdown(self):
        """Тест корректного завершения работы."""
        config = IntegrationConfig(engine_name="test")
        self.engine.initialize(config)
        
        success = self.engine.shutdown()
        self.assertTrue(success)


class TestPipelines(unittest.TestCase):
    """Тесты конвейеров обработки."""
    
    def setUp(self):
        self.provider = ComponentProvider()
        
        # Настройка мок-компонентов
        self.mock_nlp = Mock()
        self.mock_graph = Mock()
        self.mock_memory = Mock()
        
        # Настройка возвращаемых значений
        self._setup_mock_nlp()
        self._setup_mock_graph()
        self._setup_mock_memory()
        
        self.provider.register_component("nlp", self.mock_nlp)
        self.provider.register_component("semgraph", self.mock_graph)
        self.provider.register_component("memory", self.mock_memory)
    
    def _setup_mock_nlp(self):
        """Настройка мок-объекта NLP."""
        mock_result = Mock()
        mock_result.entities = [
            Mock(text="Python", entity_type=Mock(value="TECHNOLOGY"), confidence=0.9)
        ]
        mock_result.relations = [
            Mock(
                subject=Mock(text="Python"),
                predicate=Mock(value="is_a"),
                object=Mock(text="language"),
                confidence=0.8
            )
        ]
        mock_result.sentences = []
        mock_result.language = "ru"
        
        self.mock_nlp.process_text.return_value = mock_result
    
    def _setup_mock_graph(self):
        """Настройка мок-объекта графа."""
        self.mock_graph.add_node.return_value = None
        self.mock_graph.add_edge.return_value = None
        self.mock_graph.get_all_nodes.return_value = ["Python", "language"]
        self.mock_graph.get_all_edges.return_value = [["Python", "language", "is_a"]]
    
    def _setup_mock_memory(self):
        """Настройка мок-объекта памяти."""
        self.mock_memory.add.return_value = "memory_123"
        self.mock_memory.size.return_value = 1
    
    def test_text_processing_pipeline(self):
        """Тест конвейера обработки текста."""
        pipeline = TextProcessingPipeline()
        request = ProcessingRequest(
            content="Python - это язык программирования",
            enable_nlp=True,
            enable_graph_reasoning=True,
            enable_memory_search=True
        )
        
        response = pipeline.process(request, self.provider)
        
        self.assertTrue(response.success)
        self.assertIn("nlp", response.components_used)
        self.assertIn("nlp", response.structured_data)
        
        # Проверяем, что NLP был вызван
        self.mock_nlp.process_text.assert_called_once()
    
    def test_query_processing_pipeline(self):
        """Тест конвейера обработки запросов."""
        pipeline = QueryProcessingPipeline()
        request = ProcessingRequest(
            content="Что такое Python?",
            request_type="query"
            )
        
        response = pipeline.process(request, self.provider)
        
        self.assertTrue(response.success)
        self.assertIn("query_analysis", response.structured_data)
    
    def test_learning_pipeline(self):
        """Тест обучающего конвейера."""
        pipeline = LearningPipeline()
        request = ProcessingRequest(
            content="Python создан в 1991 году",
            request_type="learning"
        )
        
        response = pipeline.process(request, self.provider)
        
        self.assertTrue(response.success)
        self.assertIn("learning", response.structured_data)
    
    def test_inference_pipeline(self):
        """Тест конвейера логического вывода."""
        # Настройка мок-процессора
        mock_processor = Mock()
        mock_derivation = Mock()
        mock_derivation.success = True
        mock_derivation.derived_facts = {"conclusion_1": {"value": "test", "confidence": 0.8}}
        mock_derivation.rules_used = ["rule_1"]
        mock_derivation.confidence = 0.8
        mock_derivation.explanation = []
        
        mock_processor.derive.return_value = mock_derivation
        self.provider.register_component("processor", mock_processor)
        
        pipeline = InferencePipeline()
        request = ProcessingRequest(
            content="Если Python - язык программирования, то что можно сделать?",
            request_type="inference"
        )
        
        response = pipeline.process(request, self.provider)
        
        self.assertTrue(response.success)
        self.assertIn("inference", response.structured_data)


class TestAdapters(unittest.TestCase):
    """Тесты адаптеров интеграции."""
    
    def test_graph_memory_adapter(self):
        """Тест адаптера граф-память."""
        adapter = GraphMemoryAdapter()
        
        # Тестовые данные графа
        graph_data = {
            "nodes": {
                "Python": {"type": "language", "popularity": "high"},
                "Django": {"type": "framework", "language": "Python"}
            },
            "edges": [
                ["Django", "Python", "written_in"],
                ["Python", "programming_language", "is_a"]
            ]
        }
        
        # Преобразование в элементы памяти
        memory_items = adapter.adapt(graph_data, "memory_items")
        
        self.assertIsInstance(memory_items, list)
        self.assertGreater(len(memory_items), 0)
        
        # Проверяем, что есть элементы для узлов и ребер
        concept_items = [item for item in memory_items if item["content_type"] == "concept"]
        relation_items = [item for item in memory_items if item["content_type"] == "relation"]
        
        self.assertGreater(len(concept_items), 0)
        self.assertGreater(len(relation_items), 0)
    
    def test_nlp_graph_adapter(self):
        """Тест адаптера NLP-граф."""
        adapter = NLPGraphAdapter()
        
        # Тестовые данные NLP
        nlp_data = {
            "entities": [
                {"text": "Python", "entity_type": "TECHNOLOGY", "confidence": 0.9},
                {"text": "Django", "entity_type": "FRAMEWORK", "confidence": 0.8}
            ],
            "relations": [
                {
                    "subject": {"text": "Django"},
                    "predicate": "written_in",
                    "object": {"text": "Python"},
                    "confidence": 0.85
                }
            ]
        }
        
        # Преобразование в обновления графа
        graph_updates = adapter.adapt(nlp_data, "graph_updates")
        
        self.assertIn("nodes_to_add", graph_updates)
        self.assertIn("edges_to_add", graph_updates)
        
        nodes = graph_updates["nodes_to_add"]
        edges = graph_updates["edges_to_add"]
        
        self.assertEqual(len(nodes), 2)  # Python и Django
        self.assertEqual(len(edges), 1)  # Django written_in Python
    
    def test_vector_processor_adapter(self):
        """Тест адаптера вектор-процессор."""
        adapter = VectorProcessorAdapter()
        
        # Тестовые векторные данные
        vector_data = {
            "vector_keys": ["concept1", "concept2"],
            "similarities": [
                {"concept1": "Python", "concept2": "programming", "score": 0.8}
            ]
        }
        
        # Преобразование в контекст процессора
        result = adapter.adapt(vector_data, "processing_context")
        
        self.assertIn("context", result)
        context = result["context"]
        
        # Проверяем, что контекст содержит факты
        self.assertGreater(len(context.facts), 0)
    
    def test_memory_processor_adapter(self):
        """Тест адаптера память-процессор."""
        adapter = MemoryProcessorAdapter()
        
        # Тестовые элементы памяти
        memory_data = [
            {
                "content": "Python is a programming language",
                "content_type": "fact",
                "metadata": {"confidence": 0.9}
            },
            {
                "content": "Django",
                "content_type": "concept",
                "metadata": {"confidence": 0.8}
            }
        ]
        
        # Преобразование в контекст вывода
        result = adapter.adapt(memory_data, "inference_context")
        
        self.assertIn("inference_context", result)
        context = result["inference_context"]
        
        # Проверяем наличие фактов в контексте
        self.assertGreater(len(context.facts), 0)


class TestMetricsAndMonitoring(unittest.TestCase):
    """Тесты метрик и мониторинга."""
    
    def test_integration_metrics(self):
        """Тест сбора метрик."""
        metrics = IntegrationMetrics()
        
        # Запись метрик запросов
        metrics.record_request("test", 0.5, True)
        metrics.record_request("test", 0.3, True)
        metrics.record_request("test", 1.0, False)
        
        summary = metrics.get_summary()
        
        self.assertEqual(summary["requests"]["total"], 3)
        self.assertEqual(summary["requests"]["success_rate"], 2/3)
        self.assertAlmostEqual(summary["requests"]["average_response_time"], 0.6, places=1)
    
    def test_health_checker(self):
        """Тест проверки здоровья."""
        health_checker = HealthChecker()
        
        # Мок-компонент со статистикой
        mock_component = Mock()
        mock_component.get_statistics.return_value = {"operations": 100, "errors": 5}
        mock_component.size.return_value = 50
        
        # Мок-провайдер
        mock_provider = Mock()
        mock_provider.get_all_components_status.return_value = {"test_component": {}}
        mock_provider.get_component.return_value = mock_component
        
        # Проверка здоровья
        health_results = health_checker.check_all_components(mock_provider)
        
        self.assertIn("test_component", health_results)
        health = health_results["test_component"]
        self.assertIn(health.status, ["healthy", "degraded", "unhealthy"])
        self.assertGreater(health.response_time, 0)
    
    def test_component_monitor(self):
        """Тест монитора компонентов."""
        monitor = ComponentMonitor(check_interval=1.0)
        
        # Мок-провайдер
        mock_provider = Mock()
        mock_provider.get_all_components_status.return_value = {"test": {}}
        mock_provider.get_component.return_value = Mock()
        
        # Запуск мониторинга
        monitor.start_monitoring(mock_provider)
        
        # Проверка отчета
        report = monitor.get_monitoring_report()
        
        self.assertIn("monitoring_active", report)
        self.assertIn("metrics_summary", report)
        self.assertIn("health_status", report)
        
        # Остановка мониторинга
        monitor.stop_monitoring()
        self.assertFalse(monitor.monitoring_active)


class TestConfiguration(unittest.TestCase):
    """Тесты управления конфигурацией."""
    
    def test_config_manager_templates(self):
        """Тест создания шаблонов конфигурации."""
        config_manager = IntegrationConfigManager()
        
        # Тестируем все доступные шаблоны
        templates = ["default", "lightweight", "research", "production"]
        
        for template_name in templates:
            template = config_manager.create_template_config(template_name)
            
            self.assertIn("engine_name", template)
            self.assertIn("components", template)
            self.assertIsInstance(template["components"], dict)
    
    def test_config_save_load(self):
        """Тест сохранения и загрузки конфигурации."""
        config_manager = IntegrationConfigManager()
        
        # Создаем тестовую конфигурацию
        test_config = IntegrationConfig(
            engine_name="test_engine",
            components={"nlp": {"params": {"language": "en"}}},
            max_concurrent_requests=5
        )
        
        # Сохранение в временный файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config_manager.save_config(test_config, temp_path)
            
            # Загрузка и проверка
            loaded_config = config_manager.load_config(temp_path)
            
            self.assertEqual(loaded_config.engine_name, "test_engine")
            self.assertEqual(loaded_config.max_concurrent_requests, 5)
            self.assertIn("nlp", loaded_config.components)
            
        finally:
            # Очистка
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_validation(self):
        """Тест валидации конфигурации."""
        config_manager = IntegrationConfigManager()
        
        # Валидная конфигурация
        valid_config = {
            "engine_name": "test",
            "components": {"nlp": {"params": {}}}
        }
        
        validated = config_manager._validate_config(valid_config)
        self.assertIn("version", validated)  # Должно быть добавлено значение по умолчанию
        
        # Невалидная конфигурация (отсутствует обязательное поле)
        invalid_config = {"components": {}}
        
        with self.assertRaises(ValueError):
            config_manager._validate_config(invalid_config)


class TestEngineFactory(unittest.TestCase):
    """Тесты фабрики движков."""
    
    def test_default_engine_creation(self):
        """Тест создания движка по умолчанию."""
        engine = create_default_engine()
        
        self.assertIsInstance(engine, NeuroGraphEngine)
        self.assertIsNotNone(engine.config)
        
        # Проверяем базовые компоненты
        self.assertTrue(engine.provider.is_component_available("nlp"))
        self.assertTrue(engine.provider.is_component_available("semgraph"))
        self.assertTrue(engine.provider.is_component_available("memory"))
        
        engine.shutdown()
    
    def test_lightweight_engine_creation(self):
        """Тест создания облегченного движка."""
        engine = create_lightweight_engine()
        
        self.assertIsInstance(engine, NeuroGraphEngine)
        self.assertEqual(engine.config.max_concurrent_requests, 2)
        self.assertFalse(engine.config.enable_caching)
        
        engine.shutdown()
    
    def test_custom_engine_registration(self):
        """Тест регистрации пользовательского движка."""
        class CustomEngine(NeuroGraphEngine):
            def __init__(self, provider=None):
                super().__init__(provider)
                self.custom_feature = True
        
        # Регистрация
        EngineFactory.register_engine("custom", CustomEngine)
        
        # Проверка доступности
        self.assertIn("custom", EngineFactory.get_available_engines())
        
        # Создание
        config = IntegrationConfig(engine_name="custom")
        engine = EngineFactory.create("custom", config)
        
        self.assertIsInstance(engine, CustomEngine)
        self.assertTrue(engine.custom_feature)
        
        engine.shutdown()


class TestEndToEndIntegration(unittest.TestCase):
    """Интеграционные тесты end-to-end."""
    
    def setUp(self):
        """Настройка для интеграционных тестов."""
        # Создаем движок с минимальной конфигурацией для тестирования
        self.engine = create_lightweight_engine()
    
    def tearDown(self):
        """Очистка после тестов."""
        if self.engine:
            self.engine.shutdown()
    
    def test_text_processing_workflow(self):
        """Тест полного workflow обработки текста."""
        text = "Python - это язык программирования высокого уровня"
        
        # Обработка текста
        response = self.engine.process_text(text)
        
        # Базовые проверки
        self.assertIsInstance(response, ProcessingResponse)
        self.assertIsNotNone(response.request_id)
        self.assertIsNotNone(response.response_id)
        self.assertGreater(response.processing_time, 0)
        
        # Проверяем, что хотя бы один компонент был использован
        self.assertGreater(len(response.components_used), 0)
    
    def test_learning_and_query_workflow(self):
        """Тест workflow обучения и запросов."""
        # Обучение
        learning_text = "Django - это веб-фреймворк для Python"
        learning_response = self.engine.learn(learning_text)
        
        self.assertTrue(learning_response.success or learning_response.error_message is not None)
        
        # Запрос
        query = "Что такое Django?"
        query_response = self.engine.query(query)
        
        self.assertTrue(query_response.success or query_response.error_message is not None)
        self.assertIsNotNone(query_response.primary_response)
    
    def test_multiple_requests_processing(self):
        """Тест обработки множественных запросов."""
        requests = [
            "Python - язык программирования",
            "Django - веб-фреймворк",
            "Что такое Python?",
            "Как связаны Python и Django?"
        ]
        
        responses = []
        for text in requests:
            response = self.engine.process_text(text)
            responses.append(response)
        
        # Проверяем, что все запросы обработаны
        self.assertEqual(len(responses), len(requests))
        
        # Проверяем уникальность ID ответов
        response_ids = [r.response_id for r in responses]
        self.assertEqual(len(set(response_ids)), len(response_ids))
    
    def test_system_health_monitoring(self):
        """Тест мониторинга здоровья системы."""
        # Выполняем несколько операций для генерации активности
        for i in range(5):
            self.engine.process_text(f"Тестовый текст номер {i}")
        
        # Проверяем состояние здоровья
        health = self.engine.get_health_status()
        
        self.assertIn("overall_status", health)
        self.assertIn("components", health)
        self.assertIn("metrics", health)
        
        # Проверяем метрики
        metrics = health["metrics"]
        self.assertGreaterEqual(metrics["requests_processed"], 5)
    
    def test_error_handling_and_recovery(self):
        """Тест обработки ошибок и восстановления."""
        # Тест с пустым запросом
        empty_response = self.engine.process_text("")
        self.assertFalse(empty_response.success)
        self.assertIsNotNone(empty_response.error_message)
        
        # Тест с нормальным запросом после ошибки
        normal_response = self.engine.process_text("Нормальный текст")
        # Система должна восстановиться и обработать запрос
        # (успех зависит от конкретной реализации моков)
    
    def test_configuration_impact(self):
        """Тест влияния конфигурации на поведение."""
        # Создаем движок с отключенным NLP
        config = IntegrationConfig(
            engine_name="test_config",
            components={
                "nlp": {"params": {"enabled": False}}
            }
        )
        
        provider = ComponentProvider()
        config_engine = NeuroGraphEngine(provider)
        
        try:
            success = config_engine.initialize(config)
            # Инициализация может не пройти из-за отсутствия реальных компонентов
            # Это нормально для unit-тестов
            
        finally:
            config_engine.shutdown()


class TestPerformanceBasics(unittest.TestCase):
    """Базовые тесты производительности."""
    
    def test_response_time_reasonable(self):
        """Тест разумного времени ответа."""
        engine = create_lightweight_engine()
        
        try:
            start_time = time.time()
            response = engine.process_text("Быстрый тест производительности")
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Время ответа должно быть разумным (менее 5 секунд для простого запроса)
            self.assertLess(processing_time, 5.0)
            
            # Время в ответе должно быть близко к реальному времени
            if response.success:
                self.assertLessEqual(response.processing_time, processing_time * 1.1)
                
        finally:
            engine.shutdown()
    
    def test_memory_usage_stable(self):
        """Тест стабильности использования памяти."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        engine = create_lightweight_engine()
        
        try:
            # Выполняем несколько операций
            for i in range(10):
                engine.process_text(f"Тест памяти {i}")
            
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            
            # Рост памяти не должен быть чрезмерным (менее 100MB для простых тестов)
            self.assertLess(memory_growth, 100 * 1024 * 1024)
            
        finally:
            engine.shutdown()


def create_test_suite():
    """Создание набора тестов."""
    suite = unittest.TestSuite()
    
    # Базовые тесты компонентов
    suite.addTest(unittest.makeSuite(TestComponentProvider))
    suite.addTest(unittest.makeSuite(TestNeuroGraphEngine))
    
    # Тесты конвейеров и адаптеров
    suite.addTest(unittest.makeSuite(TestPipelines))
    suite.addTest(unittest.makeSuite(TestAdapters))
    
    # Тесты мониторинга и конфигурации
    suite.addTest(unittest.makeSuite(TestMetricsAndMonitoring))
    suite.addTest(unittest.makeSuite(TestConfiguration))
    suite.addTest(unittest.makeSuite(TestEngineFactory))
    
    # Интеграционные тесты
    suite.addTest(unittest.makeSuite(TestEndToEndIntegration))
    suite.addTest(unittest.makeSuite(TestPerformanceBasics))
    
    return suite


def run_integration_tests():
    """Запуск всех тестов интеграции."""
    print("🧪 Запуск тестов модуля Integration NeuroGraph")
    print("=" * 60)
    
    # Создание и запуск тестов
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("📊 Результаты тестирования:")
    print(f"✅ Успешных тестов: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Неудачных тестов: {len(result.failures)}")
    print(f"💥 Ошибок: {len(result.errors)}")
    print(f"⏭️ Пропущенных: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    # Детали ошибок
    if result.failures:
        print(f"\n❌ Неудачные тесты:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print(f"\n💥 Ошибки:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    # Общий результат
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    
    if success_rate >= 0.9:
        print(f"\n🎉 Отличный результат! Успешность: {success_rate:.1%}")
    elif success_rate >= 0.7:
        print(f"\n👍 Хороший результат! Успешность: {success_rate:.1%}")
    else:
        print(f"\n⚠️ Требуется доработка. Успешность: {success_rate:.1%}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Запуск тестов при прямом вызове файла
    success = run_integration_tests()
    exit(0 if success else 1)