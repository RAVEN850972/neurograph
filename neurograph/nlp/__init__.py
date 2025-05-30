"""Модуль обработки естественного языка и интеграции с другими компонентами."""

__version__ = "0.1.0"

# Импорт основных интерфейсов и классов
from neurograph.nlp.base import (
    # Интерфейсы
    INLProcessor,
    ITokenizer, 
    IEntityExtractor,
    IRelationExtractor,
    ITextGenerator,
    
    # Классы данных
    ProcessingResult,
    Entity,
    Relation, 
    Token,
    Sentence,
    
    # Перечисления
    EntityType,
    RelationType
)

# Импорт основных реализаций
from neurograph.nlp.processor import (
    StandardNLProcessor,
    LightweightNLProcessor,
    AdvancedNLProcessor
)

# Импорт фабрики и конфигурации  
from neurograph.nlp.factory import (
    NLPFactory,
    NLPConfiguration,
    NLPManager,
    register_nlp_processor,
    
    # Предустановленные фабричные функции
    create_default_processor,
    create_lightweight_processor, 
    create_high_performance_processor,
    
    # Предустановленные конфигурации
    get_development_config,
    get_production_config,
    get_minimal_config
)

# Импорт компонентов для расширенного использования
from neurograph.nlp.tokenization import (
    SimpleTokenizer,
    SpacyTokenizer, 
    SubwordTokenizer
)

from neurograph.nlp.entity_extraction import (
    RuleBasedEntityExtractor,
    SpacyEntityExtractor,
    HybridEntityExtractor
)

from neurograph.nlp.relation_extraction import (
    PatternBasedRelationExtractor,
    SpacyRelationExtractor,
    RuleBasedRelationExtractor,
    HybridRelationExtractor
)

from neurograph.nlp.text_generation import (
    TemplateTextGenerator,
    MarkovTextGenerator,
    RuleBasedTextGenerator,
    HybridTextGenerator
)

# Все публичные символы
__all__ = [
    # Версия
    "__version__",
    
    # Основные интерфейсы
    "INLProcessor",
    "ITokenizer", 
    "IEntityExtractor",
    "IRelationExtractor", 
    "ITextGenerator",
    
    # Классы данных
    "ProcessingResult",
    "Entity",
    "Relation",
    "Token", 
    "Sentence",
    
    # Перечисления
    "EntityType",
    "RelationType",
    
    # Основные процессоры
    "StandardNLProcessor",
    "LightweightNLProcessor",
    "AdvancedNLProcessor",
    
    # Фабрика и управление
    "NLPFactory",
    "NLPConfiguration", 
    "NLPManager",
    "register_nlp_processor",
    
    # Фабричные функции
    "create_default_processor",
    "create_lightweight_processor",
    "create_high_performance_processor",
    
    # Конфигурации
    "get_development_config",
    "get_production_config", 
    "get_minimal_config",
    
    # Компоненты токенизации
    "SimpleTokenizer",
    "SpacyTokenizer",
    "SubwordTokenizer",
    
    # Компоненты извлечения сущностей
    "RuleBasedEntityExtractor", 
    "SpacyEntityExtractor",
    "HybridEntityExtractor",
    
    # Компоненты извлечения отношений
    "PatternBasedRelationExtractor",
    "SpacyRelationExtractor", 
    "RuleBasedRelationExtractor",
    "HybridRelationExtractor",
    
    # Компоненты генерации текста
    "TemplateTextGenerator",
    "MarkovTextGenerator",
    "RuleBasedTextGenerator", 
    "HybridTextGenerator"
]


# Функции удобства для быстрого начала работы

def quick_process(text: str, language: str = "ru", **kwargs) -> ProcessingResult:
    """Быстрая обработка текста с настройками по умолчанию.
    
    Args:
        text: Текст для обработки
        language: Язык текста ("ru", "en")
        **kwargs: Дополнительные параметры для process_text
        
    Returns:
        Результат обработки текста
        
    Example:
        >>> result = quick_process("Python - это язык программирования")
        >>> print(f"Найдено сущностей: {len(result.entities)}")
        >>> print(f"Найдено отношений: {len(result.relations)}")
    """
    
    processor = create_default_processor(language=language)
    return processor.process_text(text, **kwargs)


def quick_extract_knowledge(text: str, language: str = "ru") -> list:
    """Быстрое извлечение знаний из текста.
    
    Args:
        text: Текст для анализа
        language: Язык текста ("ru", "en")
        
    Returns:
        Список извлеченных элементов знаний
        
    Example:
        >>> knowledge = quick_extract_knowledge("Python создан Гвидо ван Россумом")
        >>> for item in knowledge:
        ...     if item['type'] == 'relation':
        ...         print(f"{item['subject']['text']} -> {item['predicate']} -> {item['object']['text']}")
    """
    
    processor = create_default_processor(language=language)
    return processor.extract_knowledge(text)


def quick_generate_response(query: str, knowledge: dict, language: str = "ru", style: str = "formal") -> str:
    """Быстрая генерация ответа на запрос.
    
    Args:
        query: Запрос пользователя
        knowledge: База знаний для ответа
        language: Язык ответа ("ru", "en") 
        style: Стиль ответа ("formal", "casual", "scientific")
        
    Returns:
        Сгенерированный ответ
        
    Example:
        >>> knowledge = {"Python": "язык программирования высокого уровня"}
        >>> response = quick_generate_response("Что такое Python?", knowledge)
        >>> print(response)
    """
    
    processor = create_default_processor(language=language)
    return processor.generate_response(query, knowledge, style)


# Информация о модуле
def get_module_info() -> dict:
    """Получение информации о модуле NLP.
    
    Returns:
        Словарь с информацией о модуле
    """
    
    try:
        import spacy
        spacy_available = True
        spacy_version = spacy.__version__
    except ImportError:
        spacy_available = False
        spacy_version = None
    
    try:
        import jinja2
        jinja2_available = True 
        jinja2_version = jinja2.__version__
    except ImportError:
        jinja2_available = False
        jinja2_version = None
    
    return {
        "module": "neurograph.nlp", 
        "version": __version__,
        "available_processors": NLPFactory.get_available_processors(),
        "dependencies": {
            "spacy": {
                "available": spacy_available,
                "version": spacy_version
            },
            "jinja2": {
                "available": jinja2_available, 
                "version": jinja2_version
            }
        },
        "supported_languages": ["ru", "en", "mixed"],
        "entity_types": [e.value for e in EntityType],
        "relation_types": [r.value for r in RelationType]
    }


# Проверка зависимостей при импорте
def _check_dependencies():
    """Проверка доступности опциональных зависимостей."""
    
    from neurograph.core.logging import get_logger
    logger = get_logger("nlp_init")
    
    # Проверяем spaCy
    try:
        import spacy
        logger.info(f"spaCy доступен (версия: {spacy.__version__})")
    except ImportError:
        logger.warning("spaCy не установлен. Некоторые функции будут недоступны.")
    
    # Проверяем Jinja2
    try:
        import jinja2
        logger.info(f"Jinja2 доступен (версия: {jinja2.__version__})")
    except ImportError:
        logger.warning("Jinja2 не установлен. Шаблонная генерация текста будет ограничена.")
    
    logger.info(f"Модуль NLP инициализирован (версия: {__version__})")


# Выполняем проверку при импорте модуля
_check_dependencies()