"""Модуль для работы с векторными представлениями (эмбеддингами)."""

from neurograph.contextvec.base import IContextVectors
from neurograph.contextvec.factory import ContextVectorsFactory

__all__ = [
    "IContextVectors",
    "ContextVectorsFactory"
]