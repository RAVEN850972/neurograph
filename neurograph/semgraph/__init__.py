"""Модуль реализации семантического графа для хранения знаний."""

from neurograph.semgraph.base import ISemGraph, Node, Edge
from neurograph.semgraph.factory import SemGraphFactory

__all__ = [
    "ISemGraph",
    "Node",
    "Edge",
    "SemGraphFactory"
]