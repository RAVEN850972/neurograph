"""
Базовые интерфейсы и классы данных для модуля интеграции.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime
from enum import Enum
import uuid

T = TypeVar('T')


class ProcessingMode(Enum):
    """Режимы обработки запросов."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"


class ResponseFormat(Enum):
    """Форматы ответов системы."""
    TEXT = "text"
    STRUCTURED = "structured"
    JSON = "json"
    CONVERSATIONAL = "conversational"


@dataclass
class ProcessingRequest:
    """Запрос на обработку в системе NeuroGraph."""
    
    # Основные данные
    content: str                                    # Текст или запрос
    request_type: str = "query"                     # Тип запроса
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None                # ID сессии пользователя
    
    # Параметры обработки
    mode: ProcessingMode = ProcessingMode.SYNCHRONOUS
    response_format: ResponseFormat = ResponseFormat.CONVERSATIONAL
    max_processing_time: float = 30.0               # Максимальное время обработки
    
    # Контекст и настройки
    context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Компоненты для использования
    enable_nlp: bool = True
    enable_memory_search: bool = True
    enable_graph_reasoning: bool = True
    enable_vector_search: bool = True
    enable_logical_inference: bool = True
    
    # Параметры качества
    confidence_threshold: float = 0.5
    max_results: int = 10
    explanation_level: str = "basic"                # basic, detailed, debug
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingResponse:
    """Ответ системы NeuroGraph."""
    
    # Идентификация
    request_id: str
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Результаты
    success: bool = True
    primary_response: str = ""                      # Основной ответ
    structured_data: Dict[str, Any] = field(default_factory=dict)
    
    # Детали обработки
    processing_time: float = 0.0
    components_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # Пояснения и контекст
    explanation: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    # Мета-информация
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class IntegrationConfig:
    """Конфигурация интеграционного слоя."""
    
    # Основные настройки
    engine_name: str = "default"
    version: str = "1.0.0"
    
    # Компоненты
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Производительность
    max_concurrent_requests: int = 10
    default_timeout: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 300
    
    # Обработка ошибок
    enable_fallbacks: bool = True
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    
    # Мониторинг
    enable_metrics: bool = True
    enable_health_checks: bool = True
    log_level: str = "INFO"
    
    # Безопасность
    enable_input_validation: bool = True
    max_input_length: int = 10000
    rate_limiting: bool = False


class IComponentProvider(ABC):
    """Интерфейс провайдера компонентов системы."""
    
    @abstractmethod
    def get_component(self, component_type: str, **kwargs) -> Any:
        """Получение компонента по типу."""
        pass
    
    @abstractmethod
    def register_component(self, component_type: str, component: Any) -> None:
        """Регистрация компонента."""
        pass
    
    @abstractmethod
    def is_component_available(self, component_type: str) -> bool:
        """Проверка доступности компонента."""
        pass
    
    @abstractmethod
    def get_component_health(self, component_type: str) -> Dict[str, Any]:
        """Получение состояния здоровья компонента."""
        pass


class IPipeline(ABC):
    """Интерфейс конвейера обработки."""
    
    @abstractmethod
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Обработка запроса через конвейер."""
        pass
    
    @abstractmethod
    def validate_request(self, request: ProcessingRequest) -> tuple[bool, Optional[str]]:
        """Валидация запроса."""
        pass
    
    @abstractmethod
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Информация о конвейере."""
        pass


class IComponentAdapter(ABC):
    """Интерфейс адаптера между компонентами."""
    
    @abstractmethod
    def adapt(self, source_data: Any, target_format: str) -> Any:
        """Адаптация данных из одного формата в другой."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Получение поддерживаемых форматов."""
        pass


class INeuroGraphEngine(ABC):
    """Главный интерфейс движка NeuroGraph."""
    
    @abstractmethod
    def initialize(self, config: IntegrationConfig) -> bool:
        """Инициализация движка."""
        pass
    
    @abstractmethod
    def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Обработка запроса."""
        pass
    
    @abstractmethod
    def process_text(self, text: str, **kwargs) -> ProcessingResponse:
        """Упрощенная обработка текста."""
        pass
    
    @abstractmethod
    def query(self, query: str, **kwargs) -> ProcessingResponse:
        """Выполнение запроса к системе знаний."""
        pass
    
    @abstractmethod
    def learn(self, content: str, **kwargs) -> ProcessingResponse:
        """Обучение системы на новом контенте."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Получение состояния здоровья системы."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Корректное завершение работы."""
        pass