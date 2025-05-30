"""Фабрика и конфигурация для модуля NLP."""

from typing import Dict, Any, Optional, Type
from neurograph.nlp.base import INLProcessor
from neurograph.nlp.processor import StandardNLProcessor, LightweightNLProcessor, AdvancedNLProcessor
from neurograph.core.logging import get_logger
from neurograph.core.utils.registry import Registry


# Реестр NLP процессоров
nlp_registry = Registry[INLProcessor]("nlp_processors")


class NLPFactory:
    """Фабрика для создания NLP процессоров."""
    
    @staticmethod
    def create(processor_type: str = "standard", **kwargs) -> INLProcessor:
        """Создание NLP процессора указанного типа.
        
        Args:
            processor_type: Тип процессора ("standard", "lightweight", "advanced")
            **kwargs: Дополнительные параметры для конструктора
            
        Returns:
            Экземпляр INLProcessor
            
        Raises:
            ValueError: Если указан неизвестный тип процессора
        """
        
        logger = get_logger("nlp_factory")
        
        # Встроенные типы процессоров
        builtin_processors = {
            "standard": StandardNLProcessor,
            "lightweight": LightweightNLProcessor,
            "advanced": AdvancedNLProcessor
        }
        
        # Проверяем встроенные процессоры
        if processor_type in builtin_processors:
            processor_class = builtin_processors[processor_type]
            logger.info(f"Создание встроенного процессора: {processor_type}")
            return processor_class(**kwargs)
        
        # Проверяем зарегистрированные процессоры
        try:
            processor = nlp_registry.create(processor_type, **kwargs)
            logger.info(f"Создание зарегистрированного процессора: {processor_type}")
            return processor
        except KeyError:
            pass
        
        # Если ничего не найдено
        available_types = list(builtin_processors.keys()) + nlp_registry.get_names()
        raise ValueError(
            f"Неизвестный тип процессора: {processor_type}. "
            f"Доступные типы: {available_types}"
        )
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> INLProcessor:
        """Создание процессора из конфигурации.
        
        Args:
            config: Словарь конфигурации с ключами 'type' и 'params'
            
        Returns:
            Экземпляр INLProcessor
        """
        
        processor_type = config.get("type", "standard")
        params = config.get("params", {})
        
        return NLPFactory.create(processor_type, **params)
    
    @staticmethod
    def register_processor(name: str, processor_class: Type[INLProcessor]):
        """Регистрация пользовательского процессора.
        
        Args:
            name: Имя процессора для использования в create()
            processor_class: Класс процессора, реализующий INLProcessor
        """
        
        nlp_registry.register(name, processor_class)
    
    @staticmethod
    def get_available_processors() -> list:
        """Получение списка доступных типов процессоров.
        
        Returns:
            Список имен доступных процессоров
        """
        
        builtin = ["standard", "lightweight", "advanced"]
        registered = nlp_registry.get_names()
        
        return builtin + registered


def create_default_processor(language: str = "ru", use_spacy: bool = True) -> INLProcessor:
    """Создание процессора с настройками по умолчанию.
    
    Args:
        language: Язык для обработки ("ru", "en")
        use_spacy: Использовать ли spaCy (если доступен)
        
    Returns:
        Стандартный NLP процессор
    """
    
    return NLPFactory.create(
        "standard",
        language=language,
        use_spacy=use_spacy,
        tokenizer_type="spacy" if use_spacy else "simple",
        cache_results=True
    )


def create_lightweight_processor(language: str = "ru") -> INLProcessor:
    """Создание облегченного процессора для ограниченных ресурсов.
    
    Args:
        language: Язык для обработки ("ru", "en")
        
    Returns:
        Облегченный NLP процессор
    """
    
    return NLPFactory.create("lightweight", language=language)


def create_high_performance_processor(language: str = "ru") -> INLProcessor:
    """Создание высокопроизводительного процессора.
    
    Args:
        language: Язык для обработки ("ru", "en")
        
    Returns:
        Продвинутый NLP процессор с оптимизациями
    """
    
    return NLPFactory.create(
        "advanced",
        language=language,
        use_spacy=True,
        tokenizer_type="spacy",
        cache_results=True
    )


class NLPConfiguration:
    """Класс для управления конфигурацией NLP модуля."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Инициализация конфигурации.
        
        Args:
            config_dict: Словарь конфигурации или None для значений по умолчанию
        """
        
        self.config = config_dict or self._get_default_config()
        self.logger = get_logger("nlp_configuration")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Получение конфигурации по умолчанию."""
        
        return {
            "processor": {
                "type": "standard",
                "language": "ru",
                "use_spacy": True,
                "cache_results": True
            },
            "tokenization": {
                "tokenizer_type": "simple",
                "language": "ru"
            },
            "entity_extraction": {
                "use_spacy": True,
                "use_rules": True,
                "confidence_threshold": 0.5
            },
            "relation_extraction": {
                "use_spacy": True,
                "use_patterns": True,
                "use_rules": True,
                "confidence_threshold": 0.5
            },
            "text_generation": {
                "use_templates": True,
                "use_markov": False,
                "use_rules": True,
                "default_style": "formal"
            },
            "advanced_features": {
                "custom_patterns": True,
                "sentiment_analysis": False,
                "language_detection": True
            },
            "performance": {
                "cache_enabled": True,
                "cache_ttl": 300,
                "parallel_processing": False,
                "batch_size": 10
            }
        }
    
    def get_processor_config(self) -> Dict[str, Any]:
        """Получение конфигурации процессора."""
        return self.config.get("processor", {})
    
    def get_tokenization_config(self) -> Dict[str, Any]:
        """Получение конфигурации токенизации."""
        return self.config.get("tokenization", {})
    
    def get_entity_extraction_config(self) -> Dict[str, Any]:
        """Получение конфигурации извлечения сущностей."""
        return self.config.get("entity_extraction", {})
    
    def get_relation_extraction_config(self) -> Dict[str, Any]:
        """Получение конфигурации извлечения отношений."""
        return self.config.get("relation_extraction", {})
    
    def get_text_generation_config(self) -> Dict[str, Any]:
        """Получение конфигурации генерации текста."""
        return self.config.get("text_generation", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Получение конфигурации производительности."""
        return self.config.get("performance", {})
    
    def update_config(self, updates: Dict[str, Any]):
        """Обновление конфигурации.
        
        Args:
            updates: Словарь с обновлениями для конфигурации
        """
        
        def deep_update(base_dict, update_dict):
            """Рекурсивное обновление словаря."""
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        self.logger.info("Конфигурация обновлена")
    
    def create_processor(self) -> INLProcessor:
        """Создание процессора на основе текущей конфигурации."""
        
        processor_config = self.get_processor_config()
        
        # Убираем 'type' из параметров, так как он нужен только для выбора класса
        params = processor_config.copy()
        processor_type = params.pop('type', 'standard')  # Удаляем type из params
        
        return NLPFactory.create(processor_type, **params)
    
    def validate_config(self) -> bool:
        """Валидация конфигурации.
        
        Returns:
            True если конфигурация валидна, иначе False
        """
        
        try:
            # Проверяем обязательные секции
            required_sections = ["processor", "tokenization", "entity_extraction", "relation_extraction"]
            
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Отсутствует обязательная секция: {section}")
                    return False
            
            # Проверяем типы процессоров
            processor_type = self.config["processor"].get("type")
            available_types = NLPFactory.get_available_processors()
            
            if processor_type not in available_types:
                self.logger.error(f"Неизвестный тип процессора: {processor_type}")
                return False
            
            # Проверяем языки
            language = self.config["processor"].get("language", "ru")
            supported_languages = ["ru", "en", "mixed"]
            
            if language not in supported_languages:
                self.logger.warning(f"Язык {language} может не поддерживаться полностью")
            
            self.logger.info("Конфигурация прошла валидацию")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации конфигурации: {e}")
            return False
    
    def save_to_file(self, filepath: str):
        """Сохранение конфигурации в файл.
        
        Args:
            filepath: Путь к файлу для сохранения
        """
        
        import json
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Конфигурация сохранена в {filepath}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'NLPConfiguration':
        """Загрузка конфигурации из файла.
        
        Args:
            filepath: Путь к файлу конфигурации
            
        Returns:
            Экземпляр NLPConfiguration
        """
        
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            return cls(config_dict)
            
        except Exception as e:
            logger = get_logger("nlp_configuration")
            logger.error(f"Ошибка загрузки конфигурации из {filepath}: {e}")
            return cls()  # Возвращаем конфигурацию по умолчанию


# Предустановленные конфигурации

def get_development_config() -> NLPConfiguration:
    """Конфигурация для разработки (быстрая обработка, подробное логирование)."""
    
    config = {
        "processor": {
            "type": "lightweight",
            "language": "ru",
            "use_spacy": False,
            "cache_results": False
        },
        "tokenization": {
            "tokenizer_type": "simple"
        },
        "entity_extraction": {
            "use_spacy": False,
            "use_rules": True,
            "confidence_threshold": 0.3
        },
        "relation_extraction": {
            "use_spacy": False,
            "use_patterns": True,
            "use_rules": True,
            "confidence_threshold": 0.3
        },
        "text_generation": {
            "use_templates": True,
            "use_markov": False,
            "use_rules": True
        },
        "performance": {
            "cache_enabled": False,
            "parallel_processing": False
        }
    }
    
    return NLPConfiguration(config)


def get_production_config() -> NLPConfiguration:
    """Конфигурация для продакшена (оптимальное качество и производительность)."""
    
    config = {
        "processor": {
            "type": "advanced",
            "language": "ru",
            "use_spacy": True,
            "cache_results": True
        },
        "tokenization": {
            "tokenizer_type": "spacy"
        },
        "entity_extraction": {
            "use_spacy": True,
            "use_rules": True,
            "confidence_threshold": 0.7
        },
        "relation_extraction": {
            "use_spacy": True,
            "use_patterns": True,
            "use_rules": True,
            "confidence_threshold": 0.6
        },
        "text_generation": {
            "use_templates": True,
            "use_markov": True,
            "use_rules": True,
            "default_style": "formal"
        },
        "advanced_features": {
            "custom_patterns": True,
            "sentiment_analysis": True,
            "language_detection": True
        },
        "performance": {
            "cache_enabled": True,
            "cache_ttl": 600,
            "parallel_processing": True,
            "batch_size": 20
        }
    }
    
    return NLPConfiguration(config)


def get_minimal_config() -> NLPConfiguration:
    """Минимальная конфигурация для ограниченных ресурсов."""
    
    config = {
        "processor": {
            "type": "lightweight",
            "language": "ru",
            "use_spacy": False,
            "cache_results": False
        },
        "tokenization": {
            "tokenizer_type": "simple"
        },
        "entity_extraction": {
            "use_spacy": False,
            "use_rules": True,
            "confidence_threshold": 0.5
        },
        "relation_extraction": {
            "use_spacy": False,
            "use_patterns": False,
            "use_rules": True,
            "confidence_threshold": 0.5
        },
        "text_generation": {
            "use_templates": False,
            "use_markov": False,
            "use_rules": True
        },
        "performance": {
            "cache_enabled": False,
            "parallel_processing": False,
            "batch_size": 1
        }
    }
    
    return NLPConfiguration(config)


class NLPManager:
    """Менеджер для управления NLP процессорами и их конфигурациями."""
    
    def __init__(self, config: Optional[NLPConfiguration] = None):
        """Инициализация менеджера.
        
        Args:
            config: Конфигурация NLP или None для значений по умолчанию
        """
        
        self.config = config or NLPConfiguration()
        self.logger = get_logger("nlp_manager")
        
        # Кеш процессоров
        self._processors = {}
        
        # Статистика использования
        self.usage_stats = {
            'processors_created': 0,
            'texts_processed': 0,
            'total_processing_time': 0.0
        }
    
    def get_processor(self, processor_type: Optional[str] = None, **kwargs) -> INLProcessor:
        """Получение процессора (с кешированием).
        
        Args:
            processor_type: Тип процессора или None для использования из конфигурации
            **kwargs: Дополнительные параметры для процессора
            
        Returns:
            Экземпляр INLProcessor
        """
        
        if processor_type is None:
            processor_type = self.config.get_processor_config().get("type", "standard")
        
        # Создаем ключ для кеша
        cache_key = f"{processor_type}_{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._processors:
            # Объединяем параметры из конфигурации и переданные
            config_params = self.config.get_processor_config().copy()
            config_params.update(kwargs)
            
            # Создаем процессор
            processor = NLPFactory.create(processor_type, **config_params)
            self._processors[cache_key] = processor
            
            self.usage_stats['processors_created'] += 1
            self.logger.info(f"Создан процессор: {processor_type}")
        
        return self._processors[cache_key]
    
    def process_text(self, text: str, processor_type: Optional[str] = None, **kwargs) -> Any:
        """Обработка текста с использованием подходящего процессора.
        
        Args:
            text: Текст для обработки
            processor_type: Тип процессора или None для автоматического выбора
            **kwargs: Дополнительные параметры для обработки
            
        Returns:
            Результат обработки текста
        """
        
        import time
        
        start_time = time.time()
        
        # Получаем процессор
        processor = self.get_processor(processor_type)
        
        # Обрабатываем текст
        result = processor.process_text(text, **kwargs)
        
        # Обновляем статистику
        processing_time = time.time() - start_time
        self.usage_stats['texts_processed'] += 1
        self.usage_stats['total_processing_time'] += processing_time
        
        return result
    
    def batch_process_texts(self, texts: list, processor_type: Optional[str] = None, **kwargs) -> list:
        """Пакетная обработка текстов.
        
        Args:
            texts: Список текстов для обработки
            processor_type: Тип процессора
            **kwargs: Дополнительные параметры для обработки
            
        Returns:
            Список результатов обработки
        """
        
        # Проверяем возможность параллельной обработки
        parallel_processing = self.config.get_performance_config().get("parallel_processing", False)
        batch_size = self.config.get_performance_config().get("batch_size", 10)
        
        if parallel_processing and len(texts) > batch_size:
            return self._parallel_process_texts(texts, processor_type, **kwargs)
        else:
            return self._sequential_process_texts(texts, processor_type, **kwargs)
    
    def _sequential_process_texts(self, texts: list, processor_type: Optional[str] = None, **kwargs) -> list:
        """Последовательная обработка текстов."""
        
        results = []
        processor = self.get_processor(processor_type)
        
        for text in texts:
            try:
                result = processor.process_text(text, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Ошибка обработки текста: {e}")
                results.append(None)
        
        return results
    
    def _parallel_process_texts(self, texts: list, processor_type: Optional[str] = None, **kwargs) -> list:
        """Параллельная обработка текстов."""
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        batch_size = self.config.get_performance_config().get("batch_size", 10)
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Создаем задачи для обработки
            future_to_index = {}
            
            for i, text in enumerate(texts):
                future = executor.submit(self.process_text, text, processor_type, **kwargs)
                future_to_index[future] = i
            
            # Собираем результаты
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Ошибка параллельной обработки текста {index}: {e}")
                    results[index] = None
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики использования менеджера."""
        
        # Базовая статистика менеджера
        base_stats = self.usage_stats.copy()
        
        # Добавляем статистику процессоров
        processor_stats = {}
        for cache_key, processor in self._processors.items():
            if hasattr(processor, 'get_statistics'):
                processor_stats[cache_key] = processor.get_statistics()
        
        # Средние показатели
        if base_stats['texts_processed'] > 0:
            base_stats['avg_processing_time'] = (
                base_stats['total_processing_time'] / base_stats['texts_processed']
            )
        else:
            base_stats['avg_processing_time'] = 0.0
        
        return {
            'manager_stats': base_stats,
            'processor_stats': processor_stats,
            'config': self.config.config
        }
    
    def cleanup(self):
        """Очистка ресурсов менеджера."""
        
        # Очищаем кеш процессоров
        self._processors.clear()
        
        self.logger.info("Менеджер NLP очищен")


# Регистрация декораторов для удобства

def register_nlp_processor(name: str):
    """Декоратор для регистрации пользовательского NLP процессора.
    
    Args:
        name: Имя процессора для регистрации
    """
    
    def decorator(cls):
        NLPFactory.register_processor(name, cls)
        return cls
    
    return decorator


# Пример использования декоратора:
# @register_nlp_processor("custom")
# class CustomNLProcessor(INLProcessor):
#     pass