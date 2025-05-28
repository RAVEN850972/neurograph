"""Модуль для работы с многоуровневой биоморфной памятью."""

from neurograph.memory.base import IMemory, MemoryItem
from neurograph.memory.factory import (
    MemoryFactory, 
    create_default_biomorphic_memory,
    create_lightweight_memory,
    create_high_performance_memory
)
from neurograph.memory.impl.biomorphic import BiomorphicMemory
from neurograph.memory.strategies import (
    ConsolidationStrategy,
    ForgettingStrategy,
    TimeBasedConsolidation,
    ImportanceBasedConsolidation,
    EbbinghausBasedForgetting,
    LeastRecentlyUsedForgetting,
    SemanticClusteringConsolidation,
    AdaptiveConsolidation,
    MemoryPressureMonitor
)
from neurograph.memory.consolidation import (
    ConsolidationManager,
    ConsolidationOrchestrator,
    TransitionLogger,
    MemoryTransition
)

__all__ = [
    # Базовые интерфейсы и классы
    "IMemory",
    "MemoryItem",
    
    # Фабрика и удобные функции создания
    "MemoryFactory",
    "create_default_biomorphic_memory",
    "create_lightweight_memory", 
    "create_high_performance_memory",
    
    # Реализации памяти
    "BiomorphicMemory",
    
    # Стратегии консолидации и забывания
    "ConsolidationStrategy",
    "ForgettingStrategy",
    "TimeBasedConsolidation",
    "ImportanceBasedConsolidation",
    "EbbinghausBasedForgetting",
    "LeastRecentlyUsedForgetting",
    "SemanticClusteringConsolidation",
    "AdaptiveConsolidation",
    "MemoryPressureMonitor",
    
    # Система консолидации
    "ConsolidationManager",
    "ConsolidationOrchestrator",
    "TransitionLogger",
    "MemoryTransition"
]

__version__ = "0.1.0"