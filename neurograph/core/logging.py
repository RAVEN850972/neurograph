"""Система логирования для NeuroGraph."""

import os
import sys
from typing import Optional, Dict, Any, Union
from pathlib import Path

from loguru import logger

# Базовая настройка логгера
logger.remove()  # Удаляем стандартный обработчик


class Logger:
    """Класс для настройки и управления логированием."""
    
    DEFAULT_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    @staticmethod
    def setup(level: str = "INFO", 
              format_string: Optional[str] = None, 
              log_file: Optional[Union[str, Path]] = None,
              rotation: Optional[str] = "10 MB",
              retention: Optional[str] = "1 week") -> None:
        """Настраивает логирование.
        
        Args:
            level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            format_string: Строка формата сообщения.
            log_file: Путь к файлу для записи логов.
            rotation: Настройка ротации файлов логов.
            retention: Настройка хранения файлов логов.
        """
        # Добавляем обработчик для консоли
        logger.add(
            sys.stderr, 
            format=format_string or Logger.DEFAULT_FORMAT, 
            level=level.upper(),
            colorize=True
        )
        
        # Добавляем обработчик для файла, если указан
        if log_file:
            logger.add(
                log_file,
                format=format_string or Logger.DEFAULT_FORMAT,
                level=level.upper(),
                rotation=rotation,
                retention=retention,
                encoding="utf-8"
            )
    
    @staticmethod
    def get_logger(name: str, **context) -> "logger":
        """Возвращает логгер с указанным контекстом.
        
        Args:
            name: Имя логгера.
            **context: Дополнительный контекст для логирования.
            
        Returns:
            Настроенный логгер.
        """
        return logger.bind(name=name, **context)
    
    @staticmethod
    def set_level(level: str) -> None:
        """Устанавливает уровень логирования.
        
        Args:
            level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        for handler_id in logger._core.handlers:
            logger.level(handler_id, level.upper())


# Экспортируем глобальный логгер
get_logger = Logger.get_logger
setup_logging = Logger.setup
set_log_level = Logger.set_level