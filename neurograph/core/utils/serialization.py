"""Утилиты для сериализации и десериализации данных."""

import json
import pickle
from typing import Any, Dict, List, Union, Optional, Type, TypeVar, Generic

import numpy as np

T = TypeVar('T')


class Serializable:
    """Интерфейс для объектов, поддерживающих сериализацию."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализует объект в словарь.
        
        Returns:
            Словарь с данными объекта.
        """
        raise NotImplementedError("Метод to_dict должен быть реализован в подклассе")
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Создает объект из словаря.
        
        Args:
            data: Словарь с данными объекта.
            
        Returns:
            Созданный объект.
        """
        raise NotImplementedError("Метод from_dict должен быть реализован в подклассе")


class JSONSerializer:
    """Сериализатор для формата JSON."""
    
    @staticmethod
    def _prepare_for_json(obj: Any) -> Any:
        """Подготавливает объект для сериализации в JSON.
        
        Args:
            obj: Объект для подготовки.
            
        Returns:
            Подготовленный объект.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: JSONSerializer._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [JSONSerializer._prepare_for_json(i) for i in obj]
        else:
            return obj
    
    @staticmethod
    def serialize(obj: Any) -> str:
        """Сериализует объект в строку JSON.
        
        Args:
            obj: Объект для сериализации.
            
        Returns:
            Строка JSON.
        """
        prepared_obj = JSONSerializer._prepare_for_json(obj)
        return json.dumps(prepared_obj, ensure_ascii=False, indent=2)
    
    @staticmethod
    def deserialize(data: str) -> Any:
        """Десериализует объект из строки JSON.
        
        Args:
            data: Строка JSON.
            
        Returns:
            Десериализованный объект.
        """
        return json.loads(data)
    
    @staticmethod
    def save_to_file(obj: Any, file_path: str) -> None:
        """Сохраняет объект в файл JSON.
        
        Args:
            obj: Объект для сохранения.
            file_path: Путь к файлу.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(JSONSerializer.serialize(obj))
    
    @staticmethod
    def load_from_file(file_path: str) -> Any:
        """Загружает объект из файла JSON.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Загруженный объект.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return JSONSerializer.deserialize(f.read())


class PickleSerializer:
    """Сериализатор для формата Pickle."""
    
    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Сериализует объект в байты.
        
        Args:
            obj: Объект для сериализации.
            
        Returns:
            Сериализованный объект в виде байтов.
        """
        return pickle.dumps(obj)
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Десериализует объект из байтов.
        
        Args:
            data: Сериализованный объект в виде байтов.
            
        Returns:
            Десериализованный объект.
        """
        return pickle.loads(data)
    
    @staticmethod
    def save_to_file(obj: Any, file_path: str) -> None:
        """Сохраняет объект в файл.
        
        Args:
            obj: Объект для сохранения.
            file_path: Путь к файлу.
        """
        with open(file_path, 'wb') as f:
            f.write(PickleSerializer.serialize(obj))
    
    @staticmethod
    def load_from_file(file_path: str) -> Any:
        """Загружает объект из файла.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Загруженный объект.
        """
        with open(file_path, 'rb') as f:
            return PickleSerializer.deserialize(f.read())


class NumPySerializer:
    """Сериализатор для работы с массивами NumPy."""
    
    @staticmethod
    def save_array(arr: np.ndarray, file_path: str) -> None:
        """Сохраняет массив NumPy в файл.
        
        Args:
            arr: Массив для сохранения.
            file_path: Путь к файлу.
        """
        np.save(file_path, arr)
    
    @staticmethod
    def load_array(file_path: str) -> np.ndarray:
        """Загружает массив NumPy из файла.
        
        Args:
            file_path: Путь к файлу.
            
        Returns:
            Загруженный массив.
        """
        return np.load(file_path)
