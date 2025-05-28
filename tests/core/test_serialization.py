"""Тесты для утилит сериализации."""

import os
import tempfile
import numpy as np
import pytest

from neurograph.core.utils.serialization import (
    Serializable, JSONSerializer, PickleSerializer, NumPySerializer
)


class TestSerializable:
    """Тесты для интерфейса Serializable."""
    
    class TestObject(Serializable):
        """Тестовый класс, реализующий интерфейс Serializable."""
        
        def __init__(self, name, value):
            self.name = name
            self.value = value
        
        def to_dict(self):
            return {"name": self.name, "value": self.value}
        
        @classmethod
        def from_dict(cls, data):
            return cls(data["name"], data["value"])
    
    def test_to_dict(self):
        """Проверка сериализации объекта в словарь."""
        obj = self.TestObject("test", 42)
        data = obj.to_dict()
        
        assert data == {"name": "test", "value": 42}
    
    def test_from_dict(self):
        """Проверка десериализации объекта из словаря."""
        data = {"name": "test", "value": 42}
        obj = self.TestObject.from_dict(data)
        
        assert obj.name == "test"
        assert obj.value == 42


class TestJSONSerializer:
    """Тесты для JSONSerializer."""
    
    def test_serialize_deserialize(self):
        """Проверка сериализации и десериализации простых объектов."""
        data = {"name": "test", "value": 42, "list": [1, 2, 3]}
        
        # Сериализация
        json_str = JSONSerializer.serialize(data)
        
        # Десериализация
        restored = JSONSerializer.deserialize(json_str)
        
        assert restored == data
    
    def test_numpy_array_handling(self):
        """Проверка обработки массивов NumPy."""
        data = {"array": np.array([1, 2, 3])}
        
        # Сериализация
        json_str = JSONSerializer.serialize(data)
        
        # Десериализация
        restored = JSONSerializer.deserialize(json_str)
        
        assert restored["array"] == [1, 2, 3]
    
    def test_save_load_file(self):
        """Проверка сохранения и загрузки из файла."""
        data = {"name": "test", "value": 42}
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Сохраняем в файл
            JSONSerializer.save_to_file(data, temp_path)
            
            # Загружаем из файла
            restored = JSONSerializer.load_from_file(temp_path)
            
            assert restored == data
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)


class TestPickleSerializer:
    """Тесты для PickleSerializer."""
    
    def test_serialize_deserialize(self):
        """Проверка сериализации и десериализации объектов."""
        data = {"name": "test", "value": 42, "array": np.array([1, 2, 3])}
        
        # Сериализация
        pickled = PickleSerializer.serialize(data)
        
        # Десериализация
        restored = PickleSerializer.deserialize(pickled)
        
        assert restored["name"] == data["name"]
        assert restored["value"] == data["value"]
        assert np.array_equal(restored["array"], data["array"])
    
    def test_save_load_file(self):
        """Проверка сохранения и загрузки из файла."""
        data = {"name": "test", "value": 42, "array": np.array([1, 2, 3])}
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Сохраняем в файл
            PickleSerializer.save_to_file(data, temp_path)
            
            # Загружаем из файла
            restored = PickleSerializer.load_from_file(temp_path)
            
            assert restored["name"] == data["name"]
            assert restored["value"] == data["value"]
            assert np.array_equal(restored["array"], data["array"])
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)


class TestNumPySerializer:
    """Тесты для NumPySerializer."""
    
    def test_save_load_array(self):
        """Проверка сохранения и загрузки массива."""
        array = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Сохраняем массив
            NumPySerializer.save_array(array, temp_path)
            
            # Загружаем массив
            restored = NumPySerializer.load_array(temp_path)
            
            assert np.array_equal(restored, array)
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)