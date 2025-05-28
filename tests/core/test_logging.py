"""Тесты для системы логирования."""

import os
import tempfile
import pytest
from pathlib import Path

from neurograph.core.logging import setup_logging, get_logger, set_log_level


def test_get_logger():
    """Проверка получения логгера и логирования."""
    # Создаем временный файл для логов
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Настраиваем логирование в файл
        setup_logging(level="DEBUG", log_file=temp_path)
        
        # Получаем логгер и логируем сообщение
        logger = get_logger("test_module", test_param="value")
        logger.debug("Test debug message")
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Проверяем, что файл лога содержит сообщения
        with open(temp_path, "r") as f:
            log_content = f.read()
            
        assert "DEBUG" in log_content
        assert "INFO" in log_content
        assert "WARNING" in log_content
        assert "ERROR" in log_content
        assert "test_module" in log_content
        assert "Test debug message" in log_content
    finally:
        # Удаляем временный файл
        os.unlink(temp_path)


def test_set_log_level():
    """Проверка изменения уровня логирования."""
    # Создаем временный файл для логов
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Настраиваем логирование в файл с уровнем INFO
        setup_logging(level="INFO", log_file=temp_path)
        
        # Получаем логгер и логируем сообщения
        logger = get_logger("test_level")
        logger.debug("Debug message 1")  # Не должно быть записано
        logger.info("Info message 1")    # Должно быть записано
        
        # Изменяем уровень логирования на DEBUG
        set_log_level("DEBUG")
        
        # Логируем еще сообщения
        logger.debug("Debug message 2")  # Должно быть записано
        logger.info("Info message 2")    # Должно быть записано
        
        # Проверяем, что файл лога содержит нужные сообщения
        with open(temp_path, "r") as f:
            log_content = f.read()
            
        assert "Debug message 1" not in log_content
        assert "Info message 1" in log_content
        assert "Debug message 2" in log_content
        assert "Info message 2" in log_content
    finally:
        # Удаляем временный файл
        os.unlink(temp_path)