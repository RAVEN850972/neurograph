"""Модуль нейросимволического процессора для логического вывода и рассуждений."""

from .base import (
    INeuroSymbolicProcessor,
    SymbolicRule,
    ProcessingContext,
    DerivationResult,
    ExplanationStep
)
from .factory import ProcessorFactory
from .impl.pattern_matching import PatternMatchingProcessor
from .impl.graph_based import GraphBasedProcessor

__version__ = "0.1.0"

__all__ = [
    "INeuroSymbolicProcessor",
    "SymbolicRule", 
    "ProcessingContext",
    "DerivationResult",
    "ExplanationStep",
    "ProcessorFactory",
    "PatternMatchingProcessor",
    "GraphBasedProcessor"
]