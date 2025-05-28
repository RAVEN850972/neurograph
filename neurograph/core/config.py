"""Система конфигурации для компонентов."""

import json
from typing import Dict, Any, Optional, Union, List


class Configuration:
    """Класс для хранения и управления настройками компонентов."""
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """Инициализирует конфигурацию.
        
        Args:
            config_data: Начальные данные конфигурации (опционально).
        """
        self._config = config_data or {}
        
    def get(self, path: str, default: Any = None) -> Any:
        """Получает значение из конфигурации по указанному пути.
        
        Args:
            path: Путь к значению, используя точечную нотацию (например, "database.host").
            default: Значение по умолчанию, если путь не найден.
            
        Returns:
            Значение из конфигурации или значение по умолчанию.
        """
        keys = path.split(".")
        config = self._config
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
                
        return config
        
    def set(self, path: str, value: Any) -> None:
        """Устанавливает значение в конфигурации по указанному пути.
        
        Args:
            path: Путь к значению, используя точечную нотацию (например, "database.host").
            value: Значение для установки.
        """
        keys = path.split(".")
        config = self._config
        
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            config = config[key]
                
        config[keys[-1]] = value
        
    def update(self, config_data: Dict[str, Any]) -> None:
        """Обновляет конфигурацию данными из словаря.
        
        Args:
            config_data: Словарь с данными для обновления конфигурации.
        """
        self._deep_update(self._config, config_data)
        
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Рекурсивно обновляет словарь target значениями из словаря source.
        
        Args:
            target: Целевой словарь для обновления.
            source: Исходный словарь с новыми значениями.
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
                
    def to_dict(self) -> Dict[str, Any]:
        """Возвращает конфигурацию в виде словаря.
        
        Returns:
            Словарь с настройками конфигурации.
        """
        return self._config.copy()
        
    def save_to_file(self, file_path: str) -> None:
        """Сохраняет конфигурацию в JSON-файл.
        
        Args:
            file_path: Путь к файлу для сохранения.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
            
    @classmethod
    def load_from_file(cls, file_path: str) -> "Configuration":
        """Загружает конфигурацию из JSON-файла.
        
        Args:
            file_path: Путь к файлу с конфигурацией.
            
        Returns:
            Экземпляр Configuration с загруженными настройками.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        return cls(config_data)
    
    # neurograph/core/config.py

def merge_configs(config1: Configuration, config2: Configuration) -> Configuration:
    """Объединяет две конфигурации в одну.
    
    Значения из config2 имеют приоритет при конфликтах.
    
    Args:
        config1: Первая конфигурация.
        config2: Вторая конфигурация.
        
    Returns:
        Объединенная конфигурация.
    """
    result = Configuration(config1.to_dict())
    result.update(config2.to_dict())
    return result

def from_env(prefix: str = "NEUROGRAPH_") -> Configuration:
    """Создает конфигурацию из переменных окружения.
    
    Args:
        prefix: Префикс для переменных окружения.
        
    Returns:
        Конфигурация, созданная из переменных окружения.
    """
    import os
    
    config_data = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Преобразуем переменную окружения в ключ конфигурации
            config_key = key[len(prefix):].lower().replace("__", ".")
            
            # Преобразуем значение в соответствующий тип
            if value.lower() == "true":
                config_value = True
            elif value.lower() == "false":
                config_value = False
            elif value.isdigit():
                config_value = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                config_value = float(value)
            else:
                config_value = value
                
            # Добавляем в конфигурацию
            config = Configuration()
            config.set(config_key, config_value)
            config_data.update(config.to_dict())
            
    return Configuration(config_data)