"""Система логирования для NeuroGraph."""

import os
import sys
from typing import Optional, Dict, Any, Union
from pathlib import Path

from loguru import logger

# Базовая настройка логгера
logger.remove()  # Удаляем стандартный обработчик

# Глобальные переменные для сохранения настроек логирования
_current_format = None
_current_log_file = None
_current_rotation = None
_current_retention = None


class Logger:
    """Класс для настройки и управления логированием."""
    
    # Формат включает поле с именем логгера из контекста
    DEFAULT_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
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
        # Сохраняем настройки для возможного вызова set_level
        global _current_format, _current_log_file, _current_rotation, _current_retention
        _current_format = format_string or Logger.DEFAULT_FORMAT
        _current_log_file = log_file
        _current_rotation = rotation
        _current_retention = retention
        
        # Удаляем существующие обработчики
        for handler_id in list(logger._core.handlers.keys()):
            logger.remove(handler_id)
        
        # Добавляем обработчик для консоли
        logger.add(
            sys.stderr, 
            format=_current_format, 
            level=level.upper(),
            colorize=True
        )
        
        # Добавляем обработчик для файла, если указан
        if log_file:
            logger.add(
                log_file,
                format=_current_format,
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
        # Добавляем имя логгера в контекст
        context["name"] = name
        return logger.bind(**context)
    
    @staticmethod
    def set_level(level: str) -> None:
        """Устанавливает уровень логирования.
        
        Args:
            level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        # Просто пересоздаем логгеры с новым уровнем
        Logger.setup(
            level=level,
            format_string=_current_format,
            log_file=_current_log_file,
            rotation=_current_rotation,
            retention=_current_retention
        )


# Экспортируем глобальный логгер
get_logger = Logger.get_logger
setup_logging = Logger.setup
set_log_level = Logger.set_level
