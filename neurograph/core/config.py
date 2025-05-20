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