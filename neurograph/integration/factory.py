"""
Фабрики для создания компонентов интеграционного слоя.
"""

from typing import Dict, Any, Optional
from neurograph.core import Configuration
from neurograph.core.logging import get_logger

from .base import IntegrationConfig
from .engine import NeuroGraphEngine, ComponentProvider, IntegrationManager


class EngineFactory:
    """Фабрика для создания движков NeuroGraph."""
    
    _engines_registry: Dict[str, type] = {}
    
    @classmethod
    def register_engine(cls, engine_type: str, engine_class: type) -> None:
        """Регистрация типа движка."""
        cls._engines_registry[engine_type] = engine_class
    
    @classmethod
    def create(cls, engine_type: str = "default", 
              config: Optional[IntegrationConfig] = None,
              provider: Optional[ComponentProvider] = None) -> NeuroGraphEngine:
        """Создание движка указанного типа."""
        
        if engine_type not in cls._engines_registry:
            # Используем стандартный движок для неизвестных типов
            engine_class = NeuroGraphEngine
        else:
            engine_class = cls._engines_registry[engine_type]
        
        engine = engine_class(provider)
        
        if config:
            if not engine.initialize(config):
                raise RuntimeError(f"Не удалось инициализировать движок {engine_type}")
        
        return engine
    
    @classmethod
    def create_from_config(cls, config_dict: Dict[str, Any]) -> NeuroGraphEngine:
        """Создание движка из словаря конфигурации."""
        
        # Преобразуем словарь в IntegrationConfig
        config = IntegrationConfig(**config_dict)
        
        engine_type = config_dict.get("engine_type", "default")
        return cls.create(engine_type, config)
    
    @classmethod
    def get_available_engines(cls) -> list[str]:
        """Получение списка доступных типов движков."""
        return list(cls._engines_registry.keys()) + ["default"]


def create_default_engine(provider: Optional[ComponentProvider] = None) -> NeuroGraphEngine:
    """Создание движка с настройками по умолчанию."""
    
    config = IntegrationConfig(
        engine_name="default_neurograph",
        components={
            "semgraph": {
                "type": "memory_efficient",
                "params": {}
            },
            "contextvec": {
                "type": "dynamic",
                "params": {"vector_size": 384, "use_indexing": True}
            },
            "memory": {
                "params": {
                    "stm_capacity": 100,
                    "ltm_capacity": 10000,
                    "use_semantic_indexing": True
                }
            },
            "processor": {
                "type": "pattern_matching",
                "params": {"confidence_threshold": 0.5}
            },
            "nlp": {
                "params": {"language": "ru"}
            }
        },
        max_concurrent_requests=10,
        enable_caching=True,
        enable_metrics=True
    )
    
    return EngineFactory.create("default", config, provider)


def create_lightweight_engine(provider: Optional[ComponentProvider] = None) -> NeuroGraphEngine:
    """Создание облегченного движка для ограниченных ресурсов."""
    
    config = IntegrationConfig(
        engine_name="lightweight_neurograph",
        components={
            "semgraph": {
                "type": "memory_efficient",
                "params": {}
            },
            "contextvec": {
                "type": "static",
                "params": {"vector_size": 100}
            },
            "memory": {
                "params": {
                    "stm_capacity": 25,
                    "ltm_capacity": 500,
                    "use_semantic_indexing": False
                }
            },
            "processor": {
                "type": "pattern_matching",
                "params": {"confidence_threshold": 0.7, "max_depth": 2}
            },
            "nlp": {
                "params": {"language": "ru", "use_spacy": False}
            }
        },
        max_concurrent_requests=2,
        enable_caching=False,
        enable_metrics=False,
        default_timeout=10.0
    )
    
    return EngineFactory.create("default", config, provider)


def create_research_engine(provider: Optional[ComponentProvider] = None) -> NeuroGraphEngine:
    """Создание движка для исследований с расширенными возможностями."""
    
    config = IntegrationConfig(
        engine_name="research_neurograph",
        components={
            "semgraph": {
                "type": "persistent",
                "params": {
                    "file_path": "research_graph.json",
                    "auto_save_interval": 300.0
                }
            },
            "contextvec": {
                "type": "dynamic",
                "params": {"vector_size": 768, "use_indexing": True}
            },
            "memory": {
                "params": {
                    "stm_capacity": 300,
                    "ltm_capacity": 50000,
                    "use_semantic_indexing": True,
                    "auto_consolidation": True
                }
            },
            "processor": {
                "type": "graph_based",
                "params": {
                    "confidence_threshold": 0.3,
                    "max_depth": 10,
                    "enable_explanations": True
                }
            },
            "nlp": {
                "params": {
                    "language": "ru",
                    "use_spacy": True,
                    "extract_entities": True,
                    "extract_relations": True
                }
            }
        },
        max_concurrent_requests=20,
        enable_caching=True,
        cache_ttl=600,
        enable_metrics=True,
        enable_health_checks=True,
        default_timeout=120.0
    )
    
    return EngineFactory.create("default", config, provider)


# Регистрируем стандартный движок
EngineFactory.register_engine("default", NeuroGraphEngine)