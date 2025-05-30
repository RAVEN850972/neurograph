"""Примеры использования модуля NLP."""

from neurograph.nlp import (
    quick_process, quick_extract_knowledge, quick_generate_response,
    create_default_processor, create_lightweight_processor,
    NLPManager, get_production_config, get_development_config,
    EntityType, RelationType
)


def example_basic_usage():
    """Базовый пример использования NLP модуля."""
    
    print("=== Базовое использование NLP ===")
    
    # Простая обработка текста
    text = "Python - это высокоуровневый язык программирования, созданный Гвидо ван Россумом."
    
    result = quick_process(text)
    
    print(f"Исходный текст: {text}")
    print(f"Обработано за: {result.processing_time:.3f} сек")
    print(f"Язык: {result.language}")
    print(f"Предложений: {len(result.sentences)}")
    print(f"Токенов: {len(result.tokens)}")
    
    # Сущности
    print(f"\nНайдено сущностей: {len(result.entities)}")
    for entity in result.entities:
        print(f"  - {entity.text} ({entity.entity_type.value}, уверенность: {entity.confidence:.2f})")
    
    # Отношения
    print(f"\nНайдено отношений: {len(result.relations)}")
    for relation in result.relations:
        print(f"  - {relation.subject.text} --[{relation.predicate.value}]--> {relation.object.text}")
        print(f"    Уверенность: {relation.confidence:.2f}")


def example_knowledge_extraction():
    """Пример извлечения знаний из текста."""
    
    print("\n=== Извлечение знаний ===")
    
    text = """
    Искусственный интеллект - это область компьютерных наук, которая занимается созданием умных машин.
    Машинное обучение является подразделом ИИ. Python широко используется в машинном обучении.
    TensorFlow - это библиотека для машинного обучения, созданная Google.
    """
    
    knowledge_items = quick_extract_knowledge(text)
    
    print(f"Извлечено {len(knowledge_items)} элементов знаний:")
    
    for i, item in enumerate(knowledge_items, 1):
        print(f"\n{i}. Тип: {item['type']}")
        
        if item['type'] == 'relation':
            print(f"   Субъект: {item['subject']['text']} ({item['subject']['type']})")
            print(f"   Предикат: {item['predicate']}")
            print(f"   Объект: {item['object']['text']} ({item['object']['type']})")
            print(f"   Уверенность: {item['confidence']:.2f}")
        
        elif item['type'] == 'entity':
            print(f"   Текст: {item['text']}")
            print(f"   Тип: {item['entity_type']}")
            print(f"   Уверенность: {item['confidence']:.2f}")
        
        elif item['type'] == 'definition':
            print(f"   Определение: {item['text']}")
            print(f"   Сущности: {[e['text'] for e in item.get('entities', [])]}")


def example_text_generation():
    """Пример генерации текста."""
    
    print("\n=== Генерация текста ===")
    
    # База знаний для генерации ответов
    knowledge = {
        "Python": "высокоуровневый язык программирования общего назначения",
        "машинное обучение": "подраздел искусственного интеллекта, изучающий алгоритмы обучения",
        "TensorFlow": "библиотека для машинного обучения от Google",
        "нейронные сети": "вычислительные системы, вдохновленные биологическими нейронными сетями"
    }
    
    # Различные типы вопросов
    questions = [
        "Что такое Python?",
        "Как используется машинное обучение?", 
        "Почему TensorFlow популярен?",
        "Сравни Python и машинное обучение"
    ]
    
    styles = ["formal", "casual", "scientific"]
    
    for question in questions:
        print(f"\nВопрос: {question}")
        
        for style in styles:
            response = quick_generate_response(question, knowledge, style=style)
            print(f"  {style.capitalize()}: {response}")


def example_advanced_processor():
    """Пример использования продвинутого процессора с пользовательскими паттернами."""
    
    print("\n=== Продвинутый процессор ===")
    
    from neurograph.nlp import AdvancedNLProcessor
    
    # Создаем продвинутый процессор
    processor = AdvancedNLProcessor(language="ru", use_spacy=True)
    
    # Добавляем пользовательские паттерны
    processor.add_custom_entity_pattern(r'\b[А-ЯЁ]\w+\s+\d+\.\d+', 'VERSION')
    processor.add_custom_relation_pattern(r'(\w+)\s+версии\s+(\d+\.\d+)', 'has_version')
    
    text = "Python версии 3.9 поддерживает новые возможности. Django версии 4.0 включает улучшения."
    
    result = processor.process_text(text)
    
    print(f"Текст: {text}")
    print(f"\nСущности:")
    for entity in result.entities:
        source = entity.metadata.get('source', 'built-in')
        print(f"  - {entity.text} ({entity.entity_type.value}) [источник: {source}]")
    
    print(f"\nОтношения:")
    for relation in result.relations:
        source = relation.metadata.get('source', 'built-in')
        print(f"  - {relation.subject.text} --[{relation.predicate.value}]--> {relation.object.text} [источник: {source}]")
    
    # Статистика процессора
    stats = processor.get_statistics()
    print(f"\nСтатистика процессора:")
    print(f"  Текстов обработано: {stats['processing_stats']['texts_processed']}")
    print(f"  Пользовательских паттернов: {stats['custom_patterns']}")


def example_nlp_manager():
    """Пример использования NLP менеджера для управления процессорами."""
    
    print("\n=== NLP Менеджер ===")
    
    # Создаем менеджер с продакшн конфигурацией
    config = get_production_config()
    manager = NLPManager(config)
    
    # Тексты для обработки
    texts = [
        "Машинное обучение революционизирует технологии.",
        "Python является популярным языком для анализа данных.",
        "TensorFlow упрощает создание нейронных сетей.",
        "Искусственный интеллект меняет мир."
    ]
    
    print(f"Обработка {len(texts)} текстов...")
    
    # Пакетная обработка
    results = manager.batch_process_texts(texts, extract_entities=True, extract_relations=True)
    
    print(f"\nРезультаты обработки:")
    for i, result in enumerate(results, 1):
        if result:
            print(f"{i}. Сущности: {len(result.entities)}, Отношения: {len(result.relations)}")
            print(f"   Время: {result.processing_time:.3f}с")
        else:
            print(f"{i}. Ошибка обработки")
    
    # Статистика менеджера
    stats = manager.get_statistics()
    print(f"\nСтатистика менеджера:")
    print(f"  Создано процессоров: {stats['manager_stats']['processors_created']}")
    print(f"  Обработано текстов: {stats['manager_stats']['texts_processed']}")
    print(f"  Среднее время обработки: {stats['manager_stats']['avg_processing_time']:.3f}с")
    
    # Очистка ресурсов
    manager.cleanup()


def example_lightweight_vs_standard():
    """Сравнение облегченного и стандартного процессоров."""
    
    print("\n=== Сравнение процессоров ===")
    
    text = "Компания Google разработала TensorFlow для машинного обучения."
    
    # Облегченный процессор
    lightweight_processor = create_lightweight_processor()
    lightweight_result = lightweight_processor.process_text(text)
    
    # Стандартный процессор  
    standard_processor = create_default_processor()
    standard_result = standard_processor.process_text(text)
    
    print(f"Текст: {text}")
    
    print(f"\nОблегченный процессор:")
    print(f"  Время: {lightweight_result.processing_time:.3f}с")
    print(f"  Сущности: {len(lightweight_result.entities)}")
    print(f"  Отношения: {len(lightweight_result.relations)}")
    
    print(f"\nСтандартный процессор:")
    print(f"  Время: {standard_result.processing_time:.3f}с")
    print(f"  Сущности: {len(standard_result.entities)}")
    print(f"  Отношения: {len(standard_result.relations)}")
    
    # Детальное сравнение сущностей
    print(f"\nСущности (облегченный):")
    for entity in lightweight_result.entities:
        print(f"  - {entity.text} ({entity.entity_type.value})")
    
    print(f"\nСущности (стандартный):")
    for entity in standard_result.entities:
        print(f"  - {entity.text} ({entity.entity_type.value})")


def example_configuration_management():
    """Пример управления конфигурациями."""
    
    print("\n=== Управление конфигурациями ===")
    
    from neurograph.nlp import NLPConfiguration
    
    # Создаем пользовательскую конфигурацию
    custom_config = NLPConfiguration()
    
    # Изменяем настройки
    custom_config.update_config({
        "processor": {
            "type": "advanced",
            "language": "ru"
        },
        "entity_extraction": {
            "confidence_threshold": 0.8
        },
        "performance": {
            "cache_enabled": True,
            "parallel_processing": True
        }
    })
    
    # Валидация конфигурации
    is_valid = custom_config.validate_config()
    print(f"Конфигурация валидна: {is_valid}")
    
    # Сохранение конфигурации
    custom_config.save_to_file("custom_nlp_config.json")
    print("Конфигурация сохранена в custom_nlp_config.json")
    
    # Загрузка конфигурации
    loaded_config = NLPConfiguration.load_from_file("custom_nlp_config.json")
    
    # Создание процессора из конфигурации
    processor = loaded_config.create_processor()
    
    print(f"Создан процессор типа: {type(processor).__name__}")


def example_custom_processor():
    """Пример создания пользовательского процессора."""
    
    print("\n=== Пользовательский процессор ===")
    
    from neurograph.nlp.base import INLProcessor, ProcessingResult
    from neurograph.nlp import register_nlp_processor, NLPFactory
    
    @register_nlp_processor("demo")
    class DemoNLProcessor(INLProcessor):
        """Демонстрационный процессор с минимальной функциональностью."""
        
        def __init__(self, custom_param: str = "default"):
            self.custom_param = custom_param
        
        def process_text(self, text: str, **kwargs) -> ProcessingResult:
            import time
            start_time = time.time()
            
            # Простая обработка - подсчет слов
            words = text.split()
            
            # Создаем минимальный результат
            result = ProcessingResult(
                original_text=text,
                sentences=[],
                entities=[],
                relations=[],
                tokens=[],
                processing_time=time.time() - start_time,
                metadata={
                    'processor': 'demo',
                    'custom_param': self.custom_param,
                    'word_count': len(words)
                }
            )
            
            return result
        
        def extract_knowledge(self, text: str) -> list:
            return [{'type': 'demo', 'text': text, 'word_count': len(text.split())}]
        
        def normalize_text(self, text: str) -> str:
            return text.strip()
        
        def get_language(self, text: str) -> str:
            return "unknown"
    
    # Использование пользовательского процессора
    demo_processor = NLPFactory.create("demo", custom_param="test_value")
    
    text = "Это тестовый текст для демонстрации."
    result = demo_processor.process_text(text)
    
    print(f"Пользовательский процессор:")
    print(f"  Обработал: {result.original_text}")
    print(f"  Время: {result.processing_time:.3f}с")
    print(f"  Количество слов: {result.metadata['word_count']}")
    print(f"  Параметр: {result.metadata['custom_param']}")


def example_integration_with_other_modules():
    """Пример интеграции NLP с другими модулями NeuroGraph."""
    
    print("\n=== Интеграция с другими модулями ===")
    
    # Симулируем интеграцию с SemGraph и Memory
    text = "Python создан Гвидо ван Россумом в 1991 году для простого программирования."
    
    # Обработка текста
    result = quick_process(text)
    
    print(f"Исходный текст: {text}")
    print(f"Извлечено отношений: {len(result.relations)}")
    
    # Преобразование в формат для SemGraph
    graph_data = []
    for relation in result.relations:
        graph_data.append({
            'source': relation.subject.text,
            'target': relation.object.text,
            'relation': relation.predicate.value,
            'confidence': relation.confidence,
            'source_type': relation.subject.entity_type.value,
            'target_type': relation.object.entity_type.value
        })
    
    print(f"\nДанные для SemGraph:")
    for item in graph_data:
        print(f"  {item['source']} --[{item['relation']}]--> {item['target']}")
    
    # Преобразование в формат для Memory
    memory_items = []
    for sentence in result.sentences:
        if sentence.entities:
            memory_items.append({
                'content': sentence.text,
                'entities': [e.text for e in sentence.entities],
                'entity_types': [e.entity_type.value for e in sentence.entities],
                'importance': len(sentence.entities) * 0.1,  # Простая эвристика важности
                'content_type': 'processed_sentence'
            })
    
    print(f"\nДанные для Memory:")
    for item in memory_items:
        print(f"  Содержание: {item['content'][:50]}...")
        print(f"  Сущности: {item['entities']}")
        print(f"  Важность: {item['importance']:.2f}")


def run_all_examples():
    """Запуск всех примеров."""
    
    print("Запуск всех примеров использования NLP модуля...")
    
    try:
        example_basic_usage()
        example_knowledge_extraction()
        example_text_generation()
        example_advanced_processor()
        example_nlp_manager()
        example_lightweight_vs_standard()
        example_configuration_management()
        example_custom_processor()
        example_integration_with_other_modules()
        
        print("\n=== Все примеры выполнены успешно! ===")
        
    except Exception as e:
        print(f"\nОшибка при выполнении примеров: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()