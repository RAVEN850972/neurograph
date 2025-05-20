"""Тесты для модуля конфигурации."""

import sys
import os
import tempfile
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from neurograph.core.config import Configuration


def test_configuration_init():
    """Тест инициализации конфигурации."""
    config = Configuration()
    assert config.to_dict() == {}
    
    config = Configuration({"a": 1, "b": 2})
    assert config.to_dict() == {"a": 1, "b": 2}


def test_configuration_get():
    """Тест получения значений из конфигурации."""
    config = Configuration({
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3
            }
        }
    })
    
    assert config.get("a") == 1
    assert config.get("b.c") == 2
    assert config.get("b.d.e") == 3
    assert config.get("x", "default") == "default"
    assert config.get("b.x", "default") == "default"
    assert config.get("b.d.x", "default") == "default"


def test_configuration_set():
    """Тест установки значений в конфигурацию."""
    config = Configuration()
    
    config.set("a", 1)
    assert config.to_dict() == {"a": 1}
    
    config.set("b.c", 2)
    assert config.to_dict() == {"a": 1, "b": {"c": 2}}
    
    config.set("b.d.e", 3)
    assert config.to_dict() == {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    
    # Перезапись существующего значения
    config.set("a", 10)
    assert config.get("a") == 10


def test_configuration_update():
    """Тест обновления конфигурации."""
    config = Configuration({"a": 1, "b": {"c": 2}})
    
    config.update({"a": 10, "b": {"d": 3}, "e": 4})
    
    assert config.get("a") == 10
    assert config.get("b.c") == 2  # Существующее значение не должно быть удалено
    assert config.get("b.d") == 3  # Новое значение должно быть добавлено
    assert config.get("e") == 4  # Новый ключ должен быть добавлен


def test_configuration_save_load():
    """Тест сохранения и загрузки конфигурации из файла."""
    config = Configuration({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Сохранение конфигурации в файл
        config.save_to_file(temp_path)
        
        # Проверка, что файл был создан и содержит правильные данные
        with open(temp_path, "r") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == config.to_dict()
        
        # Загрузка конфигурации из файла
        loaded_config = Configuration.load_from_file(temp_path)
        
        assert loaded_config.to_dict() == config.to_dict()
    finally:
        # Удаление временного файла
        os.unlink(temp_path)