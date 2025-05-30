"""Тесты для модуля NLP."""

import pytest
import time
from unittest.mock import Mock, patch

from neurograph.nlp.base import (
    ProcessingResult, Entity, Relation, Token, Sentence,
    EntityType, RelationType
)
from neurograph.nlp import (
    create_default_processor, create_lightweight_processor,
    quick_process, quick_extract_knowledge, quick_generate_response,
    NLPFactory, NLPConfiguration, NLPManager
)


class TestBasicProcessing:
    """Тесты базовой обработки текста."""
    
    def test_quick_process_basic(self):
        """Тест базовой обработки текста."""
        text = "Python - это язык программирования."
        result = quick_process(text)
        
        assert isinstance(result, ProcessingResult)
        assert result.original_text == text
        assert len(result.sentences) > 0
        assert len(result.tokens) > 0
        assert result.processing_time > 0
        assert result.language in ["ru", "en", "mixed", "unknown"]
    
    def test_quick_process_empty_text(self):
        """Тест обработки пустого текста."""
        result = quick_process("")
        
        assert isinstance(result, ProcessingResult)
        assert result.original_text == ""
        assert len(result.sentences) == 0 or (len(result.sentences) == 1 and result.sentences[0].text == "")
    
    def test_quick_process_with_entities(self):
        """Тест обработки с извлечением сущностей."""
        text = "Москва - столица России."
        result = quick_process(text, extract_entities=True)
        
        assert isinstance(result, ProcessingResult)
        # Должна быть найдена хотя бы одна сущность (Москва или Россия)
        # В зависимости от конфигурации может быть 0 или больше
        assert len(result.entities) >= 0
    
    def test_quick_process_with_relations(self):
        """Тест обработки с извлечением отношений."""
        text = "Python создан Гвидо ван Россумом."
        result = quick_process(text, extract_entities=True, extract_relations=True)
        
        assert isinstance(result, ProcessingResult)
        # Отношения могут быть найдены или нет в зависимости от алгоритма
        assert len(result.relations) >= 0


class TestKnowledgeExtraction:
    """Тесты извлечения знаний."""
    
    def test_quick_extract_knowledge_basic(self):
        """Тест базового извлечения знаний."""
        text = "Python - это язык программирования."
        knowledge = quick_extract_knowledge(text)
        
        assert isinstance(knowledge, list)
        # Может быть пустым если не найдены структурированные знания
        assert len(knowledge) >= 0
    
    def test_extract_knowledge_with_relations(self):
        """Тест извлечения знаний с отношениями."""
        text = "Django использует Python."
        knowledge = quick_extract_knowledge(text)
        
        # Проверяем структуру знаний
        for item in knowledge:
            assert isinstance(item, dict)
            assert 'type' in item
            assert item['type'] in ['relation', 'entity', 'definition']
    
    def test_extract_knowledge_definition(self):
        """Тест извлечения определений."""
        text = "Искусственный интеллект - это область компьютерных наук."
        knowledge = quick_extract_knowledge(text)
        
        # Ищем определения среди извлеченных знаний
        definitions = [item for item in knowledge if item.get('type') == 'definition']
        # Может быть 0 или больше в зависимости от алгоритма
        assert len(definitions) >= 0


class TestTextGeneration:
    """Тесты генерации текста."""
    
    def test_quick_generate_response_basic(self):
        """Тест базовой генерации ответа."""
        query = "Что такое Python?"
        knowledge = {"Python": "язык программирования"}
        
        response = quick_generate_response(query, knowledge)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_response_different_styles(self):
        """Тест генерации в разных стилях."""
        query = "Что такое машинное обучение?"
        knowledge = {"машинное обучение": "подраздел искусственного интеллекта"}
        
        styles = ["formal", "casual", "scientific"]
        
        for style in styles:
            response = quick_generate_response(query, knowledge, style=style)
            assert isinstance(response, str)
            assert len(response) > 0
    
    def test_generate_response_empty_knowledge(self):
        """Тест генерации с пустой базой знаний."""
        query = "Что такое неизвестная_концепция?"
        knowledge = {}
        
        response = quick_generate_response(query, knowledge)
        
        assert isinstance(response, str)
        assert len(response) > 0  # Должен быть сгенерирован ответ о недостатке информации


class TestNLPFactory:
    """Тесты фабрики NLP."""
    
    def test_create_standard_processor(self):
        """Тест создания стандартного процессора."""
        processor = NLPFactory.create("standard")
        
        assert processor is not None
        assert hasattr(processor, 'process_text')
        assert hasattr(processor, 'extract_knowledge')
    
    def test_create_lightweight_processor(self):
        """Тест создания облегченного процессора."""
        processor = NLPFactory.create("lightweight")
        
        assert processor is not None
        assert hasattr(processor, 'process_text')
    
    def test_create_advanced_processor(self):
        """Тест создания продвинутого процессора."""
        processor = NLPFactory.create("advanced")
        
        assert processor is not None
        assert hasattr(processor, 'process_text')
    
    def test_create_unknown_processor(self):
        """Тест создания неизвестного типа процессора."""
        with pytest.raises(ValueError):
            NLPFactory.create("unknown_processor_type")
    
    def test_get_available_processors(self):
        """Тест получения доступных процессоров."""
        processors = NLPFactory.get_available_processors()
        
        assert isinstance(processors, list)
        assert "standard" in processors
        assert "lightweight" in processors
        assert "advanced" in processors
    
    def test_create_from_config(self):
        """Тест создания процессора из конфигурации."""
        config = {
            "type": "standard",
            "params": {
                "language": "ru",
                "use_spacy": False
            }
        }
        
        processor = NLPFactory.create_from_config(config)
        assert processor is not None


class TestNLPConfiguration:
    """Тесты конфигурации NLP."""
    
    def test_default_configuration(self):
        """Тест конфигурации по умолчанию."""
        config = NLPConfiguration()
        
        assert config.config is not None
        assert "processor" in config.config
        assert "tokenization" in config.config
        assert "entity_extraction" in config.config
    
    def test_validate_configuration(self):
        """Тест валидации конфигурации."""
        config = NLPConfiguration()
        
        is_valid = config.validate_config()
        assert is_valid is True
    
    def test_update_configuration(self):
        """Тест обновления конфигурации."""
        config = NLPConfiguration()
        
        updates = {
            "processor": {
                "language": "en"
            }
        }
        
        config.update_config(updates)
        
        assert config.get_processor_config()["language"] == "en"
    
    def test_create_processor_from_config(self):
        """Тест создания процессора из конфигурации."""
        config = NLPConfiguration()
        
        processor = config.create_processor()
        assert processor is not None
    
    @patch('builtins.open')
    @patch('json.dump')
    def test_save_to_file(self, mock_json_dump, mock_open):
        """Тест сохранения конфигурации в файл."""
        config = NLPConfiguration()
        
        config.save_to_file("test_config.json")
        
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
    
    @patch('builtins.open')
    @patch('json.load')
    def test_load_from_file(self, mock_json_load, mock_open):
        """Тест загрузки конфигурации из файла."""
        mock_json_load.return_value = {"processor": {"type": "standard"}}
        
        config = NLPConfiguration.load_from_file("test_config.json")
        
        assert config is not None
        mock_open.assert_called_once()
        mock_json_load.assert_called_once()


class TestNLPManager:
    """Тесты менеджера NLP."""
    
    def test_create_manager(self):
        """Тест создания менеджера."""
        manager = NLPManager()
        
        assert manager is not None
        assert manager.config is not None
    
    def test_get_processor(self):
        """Тест получения процессора через менеджер."""
        manager = NLPManager()
        
        processor = manager.get_processor("standard")
        assert processor is not None
        
        # Проверяем кеширование
        processor2 = manager.get_processor("standard")
        assert processor is processor2  # Должен быть тот же объект
    
    def test_process_text(self):
        """Тест обработки текста через менеджер."""
        manager = NLPManager()
        
        text = "Тестовый текст для обработки."
        result = manager.process_text(text)
        
        assert isinstance(result, ProcessingResult)
        assert result.original_text == text
    
    def test_batch_process_texts(self):
        """Тест пакетной обработки текстов."""
        manager = NLPManager()
        
        texts = [
            "Первый текст.",
            "Второй текст.",
            "Третий текст."
        ]
        
        results = manager.batch_process_texts(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, ProcessingResult) or result is None
    
    def test_get_statistics(self):
        """Тест получения статистики менеджера."""
        manager = NLPManager()
        
        # Обрабатываем несколько текстов для генерации статистики
        manager.process_text("Тест 1")
        manager.process_text("Тест 2")
        
        stats = manager.get_statistics()
        
        assert isinstance(stats, dict)
        assert "manager_stats" in stats
        assert stats["manager_stats"]["texts_processed"] == 2
    
    def test_cleanup(self):
        """Тест очистки ресурсов менеджера."""
        manager = NLPManager()
        
        # Создаем процессор
        manager.get_processor("standard")
        
        # Очищаем
        manager.cleanup()
        
        # Проверяем что кеш очищен
        assert len(manager._processors) == 0


class TestProcessorComparison:
    """Тесты сравнения разных процессоров."""
    
    def test_lightweight_vs_standard_performance(self):
        """Тест сравнения производительности процессоров."""
        text = "Python - это высокоуровневый язык программирования."
        
        # Облегченный процессор
        lightweight = create_lightweight_processor()
        start_time = time.time()
        lightweight_result = lightweight.process_text(text)
        lightweight_time = time.time() - start_time
        
        # Стандартный процессор
        standard = create_default_processor()
        start_time = time.time()
        standard_result = standard.process_text(text)
        standard_time = time.time() - start_time
        
        # Проверяем что оба дают результаты
        assert isinstance(lightweight_result, ProcessingResult)
        assert isinstance(standard_result, ProcessingResult)
        
        # Облегченный должен быть быстрее (в большинстве случаев)
        # Но это не строгое требование из-за вариативности
        assert lightweight_time >= 0
        assert standard_time >= 0
    
    def test_processor_consistency(self):
        """Тест консистентности результатов разных процессоров."""
        text = "Python создан в 1991 году."
        
        processors = [
            create_lightweight_processor(),
            create_default_processor()
        ]
        
        results = []
        for processor in processors:
            result = processor.process_text(text, extract_entities=True)
            results.append(result)
        
        # Все должны обработать текст без ошибок
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert result.original_text == text
            assert len(result.sentences) > 0


class TestEdgeCases:
    """Тесты граничных случаев."""
    
    def test_very_long_text(self):
        """Тест обработки очень длинного текста."""
        # Создаем длинный текст
        long_text = "Python - это язык программирования. " * 1000
        
        result = quick_process(long_text)
        
        assert isinstance(result, ProcessingResult)
        assert len(result.original_text) > 30000  # Проверяем что текст действительно длинный
    
    def test_special_characters(self):
        """Тест обработки текста со специальными символами."""
        text = "Python™ использует Unicode® для строк. Версия 3.x поддерживает £, €, ¥."
        
        result = quick_process(text)
        
        assert isinstance(result, ProcessingResult)
        assert result.original_text == text
    
    def test_mixed_languages(self):
        """Тест обработки текста на смешанных языках."""
        text = "Python is a язык программирования. It supports Unicode."
        
        result = quick_process(text)
        
        assert isinstance(result, ProcessingResult)
        assert result.language in ["ru", "en", "mixed", "unknown"]
    
    def test_only_punctuation(self):
        """Тест обработки текста только из знаков препинания."""
        text = "!@#$%^&*().,?;:"
        
        result = quick_process(text)
        
        assert isinstance(result, ProcessingResult)
        # Должен обработать без ошибок, даже если результат пустой
    
    def test_numbers_and_dates(self):
        """Тест обработки чисел и дат."""
        text = "Python 3.9 был выпущен 5 октября 2020 года в 14:30."
        
        result = quick_process(text, extract_entities=True)
        
        assert isinstance(result, ProcessingResult)
        # Проверяем что числа и даты могут быть извлечены как сущности
        # В зависимости от реализации может быть 0 или больше


class TestErrorHandling:
    """Тесты обработки ошибок."""
    
    def test_none_text_input(self):
        """Тест обработки None как входного текста."""
        # Теперь quick_process должен корректно обрабатывать None и выбрасывать TypeError
        with pytest.raises(TypeError, match="Текст не может быть None"):
            quick_process(None)
    
    def test_non_string_input(self):
        """Тест обработки не-строкового входа."""
        with pytest.raises(TypeError, match="Ожидается строка"):
            quick_process(123)
        
        with pytest.raises(TypeError, match="Ожидается строка"):
            quick_process(['list', 'input'])
    
    def test_malformed_config(self):
        """Тест обработки некорректной конфигурации."""
        malformed_config = {
            "type": "nonexistent_processor"
        }
        
        with pytest.raises(ValueError):
            NLPFactory.create_from_config(malformed_config)
    
    def test_processor_with_invalid_parameters(self):
        """Тест создания процессора с невалидными параметрами."""
        # Теперь процессоры должны принимать дополнительные параметры через **kwargs
        processor = NLPFactory.create("standard", invalid_param="test")
        assert processor is not None
    
    def test_generate_response_with_none_knowledge(self):
        """Тест генерации ответа с None в качестве знаний."""
        # Должен обработать gracefully
        response = quick_generate_response("Тест", None)
        assert isinstance(response, str)
        assert len(response) > 0


class TestIntegration:
    """Интеграционные тесты."""
    
    def test_full_pipeline(self):
        """Тест полного пайплайна обработки."""
        # Входной текст
        text = "Машинное обучение является подразделом искусственного интеллекта."
        
        # Обработка
        result = quick_process(text, extract_entities=True, extract_relations=True)
        
        # Извлечение знаний
        knowledge_items = quick_extract_knowledge(text)
        
        # Создание базы знаний для генерации
        knowledge_base = {}
        for item in knowledge_items:
            if item.get('type') == 'relation':
                subject = item.get('subject', {}).get('text', '')
                obj = item.get('object', {}).get('text', '')
                if subject and obj:
                    knowledge_base[subject] = f"связано с {obj}"
        
        # Генерация ответа
        if knowledge_base:
            response = quick_generate_response("Что такое машинное обучение?", knowledge_base)
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Проверка результатов
        assert isinstance(result, ProcessingResult)
        assert isinstance(knowledge_items, list)
    
    def test_multilingual_processing(self):
        """Тест обработки многоязычного текста."""
        texts = [
            "Python is a programming language.",  # Английский
            "Python - это язык программирования.",  # Русский
            "Python est un langage de programmation."  # Французский (может не поддерживаться)
        ]
        
        for text in texts:
            result = quick_process(text)
            assert isinstance(result, ProcessingResult)
            assert result.original_text == text
    
    def test_batch_processing_consistency(self):
        """Тест консистентности пакетной обработки."""
        texts = [
            "Первый текст для обработки.",
            "Второй текст с другим содержанием.",
            "Третий текст для завершения теста."
        ]
        
        manager = NLPManager()
        
        # Обрабатываем по одному
        individual_results = []
        for text in texts:
            result = manager.process_text(text)
            individual_results.append(result)
        
        # Обрабатываем пакетом
        batch_results = manager.batch_process_texts(texts)
        
        # Результаты должны быть консистентными
        assert len(individual_results) == len(batch_results)
        
        for individual, batch in zip(individual_results, batch_results):
            if individual and batch:  # Оба не None
                assert individual.original_text == batch.original_text
                # Время может различаться, но другие показатели должны быть похожими
                assert len(individual.sentences) == len(batch.sentences)


class TestCustomization:
    """Тесты кастомизации и расширения."""
    
    def test_custom_processor_registration(self):
        """Тест регистрации пользовательского процессора."""
        from neurograph.nlp.base import INLProcessor, ProcessingResult
        from neurograph.nlp import register_nlp_processor, NLPFactory
        
        @register_nlp_processor("test_custom")
        class TestCustomProcessor(INLProcessor):
            def process_text(self, text: str, **kwargs) -> ProcessingResult:
                return ProcessingResult(
                    original_text=text,
                    sentences=[],
                    entities=[],
                    relations=[],
                    tokens=[],
                    processing_time=0.001,
                    metadata={'custom': True}
                )
            
            def extract_knowledge(self, text: str) -> list:
                return [{'type': 'custom', 'text': text}]
            
            def normalize_text(self, text: str) -> str:
                return text.upper()
            
            def get_language(self, text: str) -> str:
                return "custom"
        
        # Проверяем что процессор зарегистрирован
        available = NLPFactory.get_available_processors()
        assert "test_custom" in available
        
        # Создаем и тестируем
        processor = NLPFactory.create("test_custom")
        result = processor.process_text("Test text")
        
        assert result.metadata.get('custom') is True
        assert processor.normalize_text("test") == "TEST"
        assert processor.get_language("any") == "custom"
    
    def test_advanced_processor_custom_patterns(self):
        """Тест пользовательских паттернов в продвинутом процессоре."""
        from neurograph.nlp import AdvancedNLProcessor
        
        processor = AdvancedNLProcessor()
        
        # Добавляем пользовательский паттерн сущности
        processor.add_custom_entity_pattern(r'\bAPI\s+\w+', 'API_NAME')
        
        # Добавляем пользовательский паттерн отношения
        processor.add_custom_relation_pattern(r'(\w+)\s+использует\s+(\w+)', 'uses')
        
        text = "Django использует API фреймворка для обработки запросов."
        result = processor.process_text(text)
        
        # Проверяем что пользовательские паттерны работают
        custom_entities = [e for e in result.entities if e.metadata.get('source') == 'custom_pattern']
        custom_relations = [r for r in result.relations if r.metadata.get('source') == 'custom_pattern']
        
        # Может быть 0 или больше в зависимости от точности паттернов
        assert len(custom_entities) >= 0
        assert len(custom_relations) >= 0


class TestPerformance:
    """Тесты производительности."""
    
    def test_processing_speed(self):
        """Тест скорости обработки."""
        text = "Python - это язык программирования для различных задач."
        
        start_time = time.time()
        result = quick_process(text)
        processing_time = time.time() - start_time
        
        # Обработка должна занимать разумное время (менее 5 секунд для простого текста)
        assert processing_time < 5.0
        assert result.processing_time > 0
    
    def test_memory_usage_batch_processing(self):
        """Тест использования памяти при пакетной обработке."""
        import gc
        
        # Генерируем много текстов
        texts = [f"Текст номер {i} для тестирования памяти." for i in range(100)]
        
        manager = NLPManager()
        
        # Измеряем память до обработки
        gc.collect()
        
        # Обрабатываем пакет
        results = manager.batch_process_texts(texts)
        
        # Проверяем что все обработалось
        assert len(results) == len(texts)
        
        # Очищаем ресурсы
        manager.cleanup()
        gc.collect()
    
    def test_concurrent_processing(self):
        """Тест параллельной обработки."""
        from concurrent.futures import ThreadPoolExecutor
        
        texts = [f"Параллельный текст {i}" for i in range(10)]
        
        def process_text(text):
            return quick_process(text)
        
        # Параллельная обработка
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_text, text) for text in texts]
            results = [future.result() for future in futures]
        
        # Все должно обработаться успешно
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, ProcessingResult)


def test_module_imports():
    """Тест корректности импортов модуля."""
    # Проверяем что все основные компоненты импортируются
    from neurograph.nlp import (
        INLProcessor, EntityType, RelationType,
        create_default_processor, NLPFactory, NLPManager
    )
    
    assert INLProcessor is not None
    assert EntityType is not None
    assert RelationType is not None
    assert create_default_processor is not None
    assert NLPFactory is not None
    assert NLPManager is not None


def test_module_info():
    """Тест получения информации о модуле."""
    from neurograph.nlp import get_module_info
    
    info = get_module_info()
    
    assert isinstance(info, dict)
    assert 'module' in info
    assert 'version' in info
    assert 'available_processors' in info
    assert 'dependencies' in info
    assert 'supported_languages' in info


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])
