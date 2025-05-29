"""Базовые интерфейсы и классы для модуля NLP."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import uuid


class EntityType(Enum):
    """Типы сущностей для извлечения."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    CONCEPT = "CONCEPT"
    FACT = "FACT"
    UNKNOWN = "UNKNOWN"


class RelationType(Enum):
    """Типы отношений между сущностями."""
    IS_A = "is_a"
    HAS = "has"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    CREATED_BY = "created_by"
    USES = "uses"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    PART_OF = "part_of"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Извлеченная сущность из текста."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Relation:
    """Отношение между сущностями."""
    subject: Entity
    predicate: RelationType
    object: Entity
    confidence: float = 1.0
    text_span: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Token:
    """Токен с лингвистическими атрибутами."""
    text: str
    lemma: str
    pos: str  # Part-of-speech tag
    tag: str  # Detailed POS tag
    is_alpha: bool
    is_stop: bool
    is_punct: bool
    start: int
    end: int
    dependency: Optional[str] = None
    head: Optional[int] = None  # Index of head token


@dataclass
class Sentence:
    """Предложение с токенами и метаданными."""
    text: str
    tokens: List[Token]
    start: int
    end: int
    entities: Optional[List[Entity]] = None
    relations: Optional[List[Relation]] = None
    sentiment: Optional[float] = None  # -1 to 1
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.relations is None:
            self.relations = []


@dataclass
class ProcessingResult:
    """Результат обработки текста."""
    original_text: str
    sentences: List[Sentence]
    entities: List[Entity]
    relations: List[Relation]
    tokens: List[Token]
    processing_time: float
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ITokenizer(ABC):
    """Интерфейс для токенизации текста."""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[Token]:
        """Токенизация текста с лингвистическими атрибутами.
        
        Args:
            text: Входной текст для токенизации
            
        Returns:
            Список токенов с атрибутами
        """
        pass
    
    @abstractmethod
    def sent_tokenize(self, text: str) -> List[str]:
        """Разбиение текста на предложения.
        
        Args:
            text: Входной текст
            
        Returns:
            Список предложений
        """
        pass


class IEntityExtractor(ABC):
    """Интерфейс для извлечения сущностей."""
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        """Извлечение именованных сущностей из текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Список извлеченных сущностей
        """
        pass
    
    @abstractmethod
    def extract_entities_from_tokens(self, tokens: List[Token]) -> List[Entity]:
        """Извлечение сущностей из токенов.
        
        Args:
            tokens: Список токенов
            
        Returns:
            Список извлеченных сущностей
        """
        pass


class IRelationExtractor(ABC):
    """Интерфейс для извлечения отношений."""
    
    @abstractmethod
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Извлечение отношений между сущностями.
        
        Args:
            text: Исходный текст
            entities: Список сущностей в тексте
            
        Returns:
            Список извлеченных отношений
        """
        pass
    
    @abstractmethod
    def extract_relations_from_sentence(self, sentence: Sentence) -> List[Relation]:
        """Извлечение отношений из предложения.
        
        Args:
            sentence: Предложение с токенами и сущностями
            
        Returns:
            Список отношений в предложении
        """
        pass


class ITextGenerator(ABC):
    """Интерфейс для генерации текста."""
    
    @abstractmethod
    def generate_text(self, template: str, params: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Генерация текста по шаблону.
        
        Args:
            template: Шаблон для генерации
            params: Параметры для подстановки
            context: Дополнительный контекст
            
        Returns:
            Сгенерированный текст
        """
        pass
    
    @abstractmethod
    def generate_response(self, query: str, knowledge: Dict[str, Any],
                         style: Optional[str] = None) -> str:
        """Генерация ответа на запрос.
        
        Args:
            query: Запрос пользователя
            knowledge: База знаний для ответа
            style: Стиль ответа (formal, casual, etc.)
            
        Returns:
            Сгенерированный ответ
        """
        pass


class INLProcessor(ABC):
    """Основной интерфейс для обработки естественного языка."""
    
    @abstractmethod
    def process_text(self, text: str, extract_entities: bool = True,
                    extract_relations: bool = True,
                    analyze_sentiment: bool = False) -> ProcessingResult:
        """Полная обработка текста.
        
        Args:
            text: Текст для обработки
            extract_entities: Извлекать ли сущности
            extract_relations: Извлекать ли отношения
            analyze_sentiment: Анализировать ли тональность
            
        Returns:
            Результат обработки текста
        """
        pass
    
    @abstractmethod
    def extract_knowledge(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение структурированных знаний из текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Список извлеченных знаний
        """
        pass
    
    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """Нормализация текста.
        
        Args:
            text: Исходный текст
            
        Returns:
            Нормализованный текст
        """
        pass
    
    @abstractmethod
    def get_language(self, text: str) -> str:
        """Определение языка текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Код языка (iso 639-1)
        """
        pass