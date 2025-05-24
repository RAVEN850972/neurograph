"""Базовая часть фреймворка NeuroGraph, определяющая основные интерфейсы и утилиты."""

__version__ = "0.1.0"

from neurograph.core.component import Component, Configurable
from neurograph.core.config import Configuration
from neurograph.core.errors import NeuroGraphError
from neurograph.core.utils.registry import Registry
from neurograph.core.events import (
    EventBus, subscribe, unsubscribe, publish, has_subscribers, global_event_bus
)

__all__ = [
    "Component",
    "Configurable",
    "Configuration",
    "NeuroGraphError",
    "Registry",
    "EventBus",
    "subscribe",
    "unsubscribe",
    "publish",
    "has_subscribers",
    "global_event_bus",
]