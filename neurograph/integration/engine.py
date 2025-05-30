"""
Основная реализация движка NeuroGraph и провайдера компонентов.
"""

import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Callable

from neurograph.core import Component, Configuration
from neurograph.core.events import publish, subscribe
from neurograph.core.logging import get_logger
from neurograph.core.cache import global_cache
from neurograph.core.errors import NeuroGraphError

from .base import (
    INeuroGraphEngine, IComponentProvider, IPipeline,
    ProcessingRequest, ProcessingResponse, IntegrationConfig,
    ProcessingMode, ResponseFormat
)


class IntegrationError(NeuroGraphError):
    """Ошибки интеграционного слоя."""
    pass


class ComponentProvider(IComponentProvider):
    """Провайдер компонентов системы NeuroGraph."""
    
    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.component_configs: Dict[str, Dict[str, Any]] = {}
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger("component_provider")
        
        # Автоматическая инициализация компонентов при первом доступе
        self._lazy_init_funcs: Dict[str, Callable] = {}
    
    def register_component(self, component_type: str, component: Any) -> None:
        """Регистрация компонента."""
        self.components[component_type] = component
        self.component_health[component_type] = {
            "status": "healthy",
            "last_check": time.time(),
            "error_count": 0
        }
        self.logger.info(f"Зарегистрирован компонент: {component_type}")
    
    def register_lazy_component(self, component_type: str, init_func: Callable) -> None:
        """Регистрация компонента с отложенной инициализацией."""
        self._lazy_init_funcs[component_type] = init_func
        self.logger.info(f"Зарегистрирован lazy компонент: {component_type}")
    
    def get_component(self, component_type: str, **kwargs) -> Any:
        """Получение компонента по типу."""
        # Ленивая инициализация
        if (component_type not in self.components and 
            component_type in self._lazy_init_funcs):
            try:
                self.logger.info(f"Ленивая инициализация компонента: {component_type}")
                component = self._lazy_init_funcs[component_type](**kwargs)
                self.register_component(component_type, component)
            except Exception as e:
                self.logger.error(f"Ошибка инициализации {component_type}: {e}")
                raise IntegrationError(f"Не удалось инициализировать {component_type}")
        
        if component_type not in self.components:
            raise IntegrationError(f"Компонент {component_type} не найден")
        
        # Проверка здоровья компонента
        health = self.component_health.get(component_type, {})
        if health.get("status") == "unhealthy":
            self.logger.warning(f"Компонент {component_type} нездоров")
        
        return self.components[component_type]
    
    def is_component_available(self, component_type: str) -> bool:
        """Проверка доступности компонента."""
        return (component_type in self.components or 
                component_type in self._lazy_init_funcs)
    
    def get_component_health(self, component_type: str) -> Dict[str, Any]:
        """Получение состояния здоровья компонента."""
        if component_type not in self.component_health:
            return {"status": "unknown", "message": "Компонент не найден"}
        
        health = self.component_health[component_type].copy()
        
        # Добавляем дополнительную информацию
        if component_type in self.components:
            component = self.components[component_type]
            if hasattr(component, 'get_statistics'):
                try:
                    health["statistics"] = component.get_statistics()
                except Exception as e:
                    health["statistics_error"] = str(e)
        
        return health
    
    def update_component_health(self, component_type: str, status: str, 
                              error_message: Optional[str] = None) -> None:
        """Обновление состояния здоровья компонента."""
        if component_type not in self.component_health:
            self.component_health[component_type] = {}
        
        health = self.component_health[component_type]
        health["status"] = status
        health["last_check"] = time.time()
        
        if status == "unhealthy":
            health["error_count"] = health.get("error_count", 0) + 1
            if error_message:
                health["last_error"] = error_message
        else:
            health["error_count"] = 0
    
    def get_all_components_status(self) -> Dict[str, Dict[str, Any]]:
        """Получение статуса всех компонентов."""
        status = {}
        
        for component_type in self.components:
            status[component_type] = self.get_component_health(component_type)
        
        # Добавляем информацию о lazy компонентах
        for component_type in self._lazy_init_funcs:
            if component_type not in status:
                status[component_type] = {
                    "status": "not_initialized", 
                    "lazy": True
                }
        
        return status


class NeuroGraphEngine(INeuroGraphEngine, Component):
    """Основной движок системы NeuroGraph."""
    
    def __init__(self, provider: Optional[IComponentProvider] = None):
        super().__init__("neurograph_engine")
        self.provider = provider or ComponentProvider()
        self.config: Optional[IntegrationConfig] = None
        self.pipelines: Dict[str, IPipeline] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.request_history: List[Dict[str, Any]] = []
        self._shutdown_requested = False
        
        # Метрики
        self.metrics = {
            "requests_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        
        # Подписка на события системы
        subscribe("system.shutdown", self._handle_shutdown)
    
    def initialize(self, config: IntegrationConfig) -> bool:
        """Инициализация движка."""
        try:
            self.config = config
            self.logger.info(f"Инициализация движка {config.engine_name}")
            
            # Настройка executor
            max_workers = min(config.max_concurrent_requests, 16)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            
            # Инициализация компонентов
            self._initialize_components(config)
            
            # Регистрация конвейеров
            self._register_pipelines()
            
            # Публикация события инициализации
            publish("integration.engine_initialized", {
                "engine_name": config.engine_name,
                "components_count": len(self.provider.get_all_components_status())
            })
            
            self.logger.info("Движок успешно инициализирован")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации движка: {e}")
            return False
    
    def _initialize_components(self, config: IntegrationConfig) -> None:
        """Инициализация компонентов системы."""
        # Регистрация ленивых инициализаторов для основных компонентов
        
        def init_graph():
            from neurograph.semgraph import SemGraphFactory
            graph_config = config.components.get("semgraph", {})
            return SemGraphFactory.create(
                graph_config.get("type", "memory_efficient"),
                **graph_config.get("params", {})
            )
        
        def init_vectors():
            from neurograph.contextvec import ContextVectorsFactory  
            vector_config = config.components.get("contextvec", {})
            return ContextVectorsFactory.create(
                vector_config.get("type", "dynamic"),
                **vector_config.get("params", {})
            )
        
        def init_memory():
            from neurograph.memory import create_default_biomorphic_memory
            memory_config = config.components.get("memory", {})
            return create_default_biomorphic_memory(**memory_config.get("params", {}))
        
        def init_processor():
            from neurograph.processor import ProcessorFactory
            processor_config = config.components.get("processor", {})
            return ProcessorFactory.create(
                processor_config.get("type", "pattern_matching"),
                **processor_config.get("params", {})
            )
        
        def init_propagation():
            from neurograph.propagation import create_default_engine
            # Propagation нуждается в графе
            graph = self.provider.get_component("semgraph")
            return create_default_engine(graph)
        
        def init_nlp():
            from neurograph.nlp import create_default_processor
            nlp_config = config.components.get("nlp", {})
            return create_default_processor(**nlp_config.get("params", {}))
        
        # Регистрация компонентов с отложенной инициализацией
        self.provider.register_lazy_component("semgraph", init_graph)
        self.provider.register_lazy_component("contextvec", init_vectors)
        self.provider.register_lazy_component("memory", init_memory)
        self.provider.register_lazy_component("processor", init_processor)
        self.provider.register_lazy_component("propagation", init_propagation)
        self.provider.register_lazy_component("nlp", init_nlp)
    
    def _register_pipelines(self) -> None:
        """Регистрация конвейеров обработки."""
        from .pipelines import (
            TextProcessingPipeline, QueryProcessingPipeline,
            LearningPipeline, InferencePipeline
        )
        
        self.pipelines["text"] = TextProcessingPipeline()
        self.pipelines["query"] = QueryProcessingPipeline()
        self.pipelines["learning"] = LearningPipeline()
        self.pipelines["inference"] = InferencePipeline()
        
        self.logger.info(f"Зарегистрировано {len(self.pipelines)} конвейеров")
    
    def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Обработка запроса."""
        start_time = time.time()
        self.metrics["requests_processed"] += 1
        
        try:
            # Валидация запроса
            is_valid, error_msg = self._validate_request(request)
            if not is_valid:
                return self._create_error_response(request, error_msg)
            
            # Выбор конвейера обработки
            pipeline = self._select_pipeline(request)
            if not pipeline:
                return self._create_error_response(request, "Не найден подходящий конвейер")
            
            # Обработка в зависимости от режима
            if request.mode == ProcessingMode.ASYNCHRONOUS:
                return self._process_async(request, pipeline)
            elif request.mode == ProcessingMode.BATCH:
                return self._process_batch(request, pipeline)
            else:
                # Синхронная обработка
                response = pipeline.process(request, self.provider)
            
            # Обновление метрик
            processing_time = time.time() - start_time
            response.processing_time = processing_time
            
            self._update_metrics(True, processing_time)
            self._log_request(request, response)
            
            # Публикация события
            publish("integration.request_processed", {
                "request_id": request.request_id,
                "success": response.success,
                "processing_time": processing_time,
                "components_used": response.components_used
            })
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка обработки запроса {request.request_id}: {e}")
            
            self._update_metrics(False, processing_time)
            
            return self._create_error_response(request, str(e))
    
    def process_text(self, text: str, **kwargs) -> ProcessingResponse:
        """Упрощенная обработка текста."""
        request = ProcessingRequest(
            content=text,
            request_type="text_processing",
            **kwargs
        )
        return self.process_request(request)
    
    def query(self, query: str, **kwargs) -> ProcessingResponse:
        """Выполнение запроса к системе знаний."""
        request = ProcessingRequest(
            content=query,
            request_type="query",
            **kwargs
        )
        return self.process_request(request)
    
    def learn(self, content: str, **kwargs) -> ProcessingResponse:
        """Обучение системы на новом контенте."""
        request = ProcessingRequest(
            content=content,
            request_type="learning",
            **kwargs
        )
        return self.process_request(request)
    
    def _validate_request(self, request: ProcessingRequest) -> tuple[bool, Optional[str]]:
        """Валидация запроса."""
        if not self.config:
            return False, "Движок не инициализирован"
        
        if not request.content or not request.content.strip():
            return False, "Пустое содержимое запроса"
        
        if self.config.enable_input_validation:
            if len(request.content) > self.config.max_input_length:
                return False, f"Слишком длинный запрос: {len(request.content)} символов"
        
        return True, None
    
    def _select_pipeline(self, request: ProcessingRequest) -> Optional[IPipeline]:
        """Выбор конвейера для обработки запроса."""
        pipeline_map = {
            "text_processing": "text",
            "query": "query", 
            "learning": "learning",
            "inference": "inference"
        }
        
        pipeline_name = pipeline_map.get(request.request_type)
        return self.pipelines.get(pipeline_name) if pipeline_name else None
    
    def _process_async(self, request: ProcessingRequest, pipeline: IPipeline) -> ProcessingResponse:
        """Асинхронная обработка запроса."""
        # Для простоты пока возвращаем синхронный результат
        # В реальной реализации здесь была бы работа с asyncio
        return pipeline.process(request, self.provider)
    
    def _process_batch(self, request: ProcessingRequest, pipeline: IPipeline) -> ProcessingResponse:
        """Пакетная обработка запроса."""
        # Для простоты пока возвращаем синхронный результат
        return pipeline.process(request, self.provider)
    
    def _create_error_response(self, request: ProcessingRequest, error_message: str) -> ProcessingResponse:
        """Создание ответа с ошибкой."""
        return ProcessingResponse(
            request_id=request.request_id,
            success=False,
            error_message=error_message,
            processing_time=0.0
        )
    
    def _update_metrics(self, success: bool, processing_time: float) -> None:
        """Обновление метрик движка."""
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        self.metrics["total_processing_time"] += processing_time
        
        if self.metrics["requests_processed"] > 0:
            self.metrics["average_processing_time"] = (
                self.metrics["total_processing_time"] / self.metrics["requests_processed"]
            )
    
    def _log_request(self, request: ProcessingRequest, response: ProcessingResponse) -> None:
        """Логирование запроса и ответа."""
        log_entry = {
            "request_id": request.request_id,
            "request_type": request.request_type,
            "content_length": len(request.content),
            "success": response.success,
            "processing_time": response.processing_time,
            "components_used": response.components_used,
            "timestamp": time.time()
        }
        
        # Ограничиваем историю
        self.request_history.append(log_entry)
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-500:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Получение состояния здоровья системы."""
        components_status = self.provider.get_all_components_status()
        
        # Общий статус системы
        overall_healthy = all(
            status.get("status") != "unhealthy" 
            for status in components_status.values()
        )
        
        return {
            "overall_status": "healthy" if overall_healthy else "degraded",
            "engine_status": "running" if not self._shutdown_requested else "shutting_down",
            "components": components_status,
            "metrics": self.metrics.copy(),
            "uptime": time.time() - getattr(self, '_start_time', time.time()),
            "request_history_size": len(self.request_history)
        }
    
    def _handle_shutdown(self, data: Dict[str, Any]) -> None:
        """Обработка сигнала завершения работы."""
        self.logger.info("Получен сигнал завершения работы")
        self._shutdown_requested = True
    
    def shutdown(self) -> bool:
        """Корректное завершение работы."""
        try:
            self.logger.info("Завершение работы движка NeuroGraph")
            
            # Завершение executor
            self.executor.shutdown(wait=True, timeout=30)
            
            # Завершение компонентов
            for component_type, component in self.provider.components.items():
                if hasattr(component, 'shutdown'):
                    try:
                        component.shutdown()
                        self.logger.info(f"Компонент {component_type} завершен")
                    except Exception as e:
                        self.logger.error(f"Ошибка завершения {component_type}: {e}")
            
            # Публикация события
            publish("integration.engine_shutdown", {
                "final_metrics": self.metrics.copy()
            })
            
            self.logger.info("Движок успешно завершен")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при завершении работы: {e}")
            return False


class IntegrationManager:
    """Менеджер интеграции для управления жизненным циклом системы."""
    
    def __init__(self):
        self.engine: Optional[NeuroGraphEngine] = None
        self.config: Optional[IntegrationConfig] = None
        self.logger = get_logger("integration_manager")
    
    def create_engine(self, config: IntegrationConfig, 
                     provider: Optional[IComponentProvider] = None) -> NeuroGraphEngine:
        """Создание и настройка движка."""
        self.config = config
        self.engine = NeuroGraphEngine(provider)
        
        if not self.engine.initialize(config):
            raise IntegrationError("Не удалось инициализировать движок")
        
        return self.engine
    
    def get_engine(self) -> NeuroGraphEngine:
        """Получение текущего движка."""
        if not self.engine:
            raise IntegrationError("Движок не создан")
        return self.engine
    
    def shutdown(self) -> bool:
        """Завершение работы менеджера."""
        if self.engine:
            return self.engine.shutdown()
        return True