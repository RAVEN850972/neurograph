"""
NeuroGraph Integration Module

Модуль интеграции обеспечивает связующий слой между всеми компонентами системы NeuroGraph.
Предоставляет единый интерфейс для работы с графом знаний, памятью, векторными представлениями,
NLP обработкой и логическим выводом.
"""

from .base import (
    INeuroGraphEngine,
    IComponentProvider,
    IPipeline,
    IComponentAdapter,
    ProcessingRequest,
    ProcessingResponse,
    ProcessingMode,
    ResponseFormat,
    IntegrationConfig
)

from .engine import (
    NeuroGraphEngine,
    ComponentProvider,
    IntegrationManager
)

from .pipelines import (
    TextProcessingPipeline,
    QueryProcessingPipeline,
    LearningPipeline,
    InferencePipeline
)

from .adapters import (
    GraphMemoryAdapter,
    VectorProcessorAdapter,
    NLPGraphAdapter,
    MemoryProcessorAdapter
)

from .factory import (
    EngineFactory,
    create_default_engine,
    create_lightweight_engine,
    create_research_engine
)

from .utils import (
    IntegrationMetrics,
    HealthChecker,
    ComponentMonitor
)

__version__ = "1.0.0"
__all__ = [
    # Base interfaces
    "INeuroGraphEngine",
    "IComponentProvider", 
    "IPipeline",
    "IComponentAdapter",
    "ProcessingRequest",
    "ProcessingResponse",
    "IntegrationConfig",
    
    # Main implementations
    "NeuroGraphEngine",
    "ComponentProvider",
    "IntegrationManager",
    
    # Pipelines
    "TextProcessingPipeline",
    "QueryProcessingPipeline", 
    "LearningPipeline",
    "InferencePipeline",
    
    # Adapters
    "GraphMemoryAdapter",
    "VectorProcessorAdapter",
    "NLPGraphAdapter",
    "MemoryProcessorAdapter",
    
    # Factory functions
    "EngineFactory",
    "create_default_engine",
    "create_lightweight_engine", 
    "create_research_engine",
    
    # Utilities
    "IntegrationMetrics",
    "HealthChecker",
    "ComponentMonitor"
]
