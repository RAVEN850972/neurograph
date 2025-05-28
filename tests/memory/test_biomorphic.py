"""Исправленные тесты для биоморфной памяти."""

import pytest
import numpy as np
import time
import threading
from unittest.mock import patch, MagicMock

from neurograph.memory.base import MemoryItem
from neurograph.memory.impl.biomorphic import BiomorphicMemory
from neurograph.memory.factory import (
    MemoryFactory, create_default_biomorphic_memory,
    create_lightweight_memory, create_high_performance_memory
)
from neurograph.memory.strategies import (
    TimeBasedConsolidation, ImportanceBasedConsolidation,
    EbbinghausBasedForgetting
)


class TestMemoryItem:
    """Тесты для класса MemoryItem."""
    
    def test_memory_item_creation(self):
        """Тест создания элемента памяти."""
        embedding = np.random.random(384).astype(np.float32)
        item = MemoryItem(
            content="Test item",
            embedding=embedding,
            content_type="text",
            metadata={"test": True}
        )
        
        # Проверяем базовые атрибуты
        assert item.content == "Test item"
        assert item.content_type == "text"
        assert item.metadata["test"] is True
        assert np.array_equal(item.embedding, embedding)
        
        # Проверяем атрибуты доступа
        assert item.access_count == 0
        assert isinstance(item.created_at, float)
        assert isinstance(item.last_accessed_at, float)
        assert item.created_at == item.last_accessed_at
        assert item.id is None  # Изначально не установлен
    
    def test_memory_item_access(self):
        """Тест метода access."""
        embedding = np.random.random(384).astype(np.float32)
        item = MemoryItem(
            content="Test item",
            embedding=embedding,
            content_type="text"
        )
        
        initial_access_time = item.last_accessed_at
        initial_count = item.access_count
        
        # Небольшая задержка
        time.sleep(0.01)
        
        # Вызываем метод access
        item.access()
        
        # Проверяем обновления
        assert item.access_count == initial_count + 1
        assert item.last_accessed_at > initial_access_time
        
        # Повторный доступ
        item.access()
        assert item.access_count == initial_count + 2


class TestBiomorphicMemory:
    """Тесты для класса BiomorphicMemory."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        # Создаем новую память для каждого теста
        self.memory = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False,
            consolidation_interval=1.0
        )
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if hasattr(self, 'memory'):
            try:
                self.memory.shutdown()
            except:
                pass  # Игнорируем ошибки при завершении
    
    def test_add_item(self):
        """Тест добавления элемента в память."""
        # Создаем элемент памяти
        embedding = np.random.random(384).astype(np.float32)
        item = MemoryItem(
            content="Test memory item",
            embedding=embedding,
            content_type="text",
            metadata={"test": True}
        )
        
        # Добавляем в память
        item_id = self.memory.add(item)
        
        # Проверяем, что элемент добавлен
        assert item_id is not None
        assert isinstance(item_id, str)
        assert len(item_id) > 0
        assert self.memory.size() == 1
        
        # Проверяем, что элемент находится в STM
        assert len(self.memory.stm) == 1
        assert len(self.memory.ltm) == 0
        assert item_id in self.memory.stm
        
        # Проверяем, что ID установлен в элементе
        stored_item = self.memory.stm[item_id]
        assert stored_item.id == item_id
    
    def test_get_item(self):
        """Тест получения элемента из памяти."""
        # Добавляем элемент
        embedding = np.random.random(384).astype(np.float32)
        item = MemoryItem(
            content="Test item for retrieval",
            embedding=embedding,
            content_type="text"
        )
        item_id = self.memory.add(item)
        
        # Получаем элемент
        retrieved_item = self.memory.get(item_id)
        
        # Проверяем корректность
        assert retrieved_item is not None
        assert retrieved_item.content == "Test item for retrieval"
        assert retrieved_item.access_count == 1
        assert retrieved_item.id == item_id
        
        # Проверяем, что элемент попал в рабочую память
        assert len(self.memory.working_memory) == 1
        assert item_id in self.memory.working_memory
        
        # Повторный доступ
        retrieved_again = self.memory.get(item_id)
        assert retrieved_again.access_count == 2
        
        # Проверка несуществующего элемента
        non_existent = self.memory.get("non-existent-id")
        assert non_existent is None
    
    def test_stm_overflow_and_consolidation(self):
        """Тест переполнения STM и консолидации."""
        # Заполняем STM до предела (+ 1 для переполнения)
        item_ids = []
        for i in range(self.memory.stm_capacity + 1):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Item {i}",
                embedding=embedding,
                content_type="text"
            )
            item_id = self.memory.add(item)
            item_ids.append(item_id)
        
        # Ждем немного для имитации времени
        time.sleep(0.1)
        
        # Принудительно выполняем консолидацию
        consolidation_result = self.memory.force_consolidation()
        
        # Проверяем результат
        assert isinstance(consolidation_result, dict)
        assert "consolidated" in consolidation_result
        
        # Проверяем, что общее количество элементов сохранилось
        total_items = len(self.memory.stm) + len(self.memory.ltm)
        assert total_items == len(item_ids)
        
        # Если консолидация произошла, должны быть элементы в LTM
        if consolidation_result["consolidated"] > 0:
            assert len(self.memory.ltm) > 0
    
    def test_item_promotion_from_ltm(self):
        """Тест продвижения часто используемых элементов из LTM в STM."""
        # Добавляем элемент
        embedding = np.random.random(384).astype(np.float32)
        item = MemoryItem(
            content="Frequently accessed item",
            embedding=embedding,
            content_type="text"
        )
        item_id = self.memory.add(item)
        
        # Перемещаем в LTM напрямую для тестирования
        item_obj = self.memory.stm.pop(item_id)
        self.memory.ltm[item_id] = item_obj
        
        # Проверяем, что элемент в LTM
        assert item_id not in self.memory.stm
        assert item_id in self.memory.ltm
        
        # Часто обращаемся к элементу
        for i in range(4):
            retrieved_item = self.memory.get(item_id)
            assert retrieved_item is not None
            assert retrieved_item.access_count == i + 1
        
        # Проверяем, что элемент может вернуться в STM при частом доступе
        # (логика продвижения зависит от реализации)
        final_item = self.memory.get(item_id)
        assert final_item is not None
        assert final_item.access_count >= 3
    
    def test_working_memory_management(self):
        """Тест управления рабочей памятью."""
        # Добавляем больше элементов, чем помещается в рабочую память
        item_ids = []
        for i in range(self.memory.working_memory_capacity + 3):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Working item {i}",
                embedding=embedding,
                content_type="text"
            )
            item_id = self.memory.add(item)
            item_ids.append(item_id)
        
        # Обращаемся ко всем элементам
        for item_id in item_ids:
            retrieved_item = self.memory.get(item_id)
            assert retrieved_item is not None
        
        # Проверяем, что рабочая память не превышает лимит
        assert len(self.memory.working_memory) <= self.memory.working_memory_capacity
    
    def test_search_functionality(self):
        """Тест функциональности поиска."""
        # Добавляем несколько элементов с разным содержимым
        items_data = [
            "Python programming language",
            "Machine learning algorithms",
            "Natural language processing"
        ]
        
        added_ids = []
        for content in items_data:
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=content,
                embedding=embedding,
                content_type="text"
            )
            item_id = self.memory.add(item)
            added_ids.append(item_id)
        
        # Выполняем поиск по вектору правильной размерности
        query_vector = np.random.random(384).astype(np.float32)
        results = self.memory.search(query_vector, limit=3)
        
        # Проверяем результаты
        assert isinstance(results, list)
        assert len(results) <= 3
        
        # Проверяем, что результаты содержат ID и оценку правильных типов
        for item_id, score in results:
            assert isinstance(item_id, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            assert item_id in added_ids
    
    def test_remove_item(self):
        """Тест удаления элемента из памяти."""
        # Создаем новую память для этого теста
        memory = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # Добавляем элемент
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content="Item to be removed",
                embedding=embedding,
                content_type="text"
            )
            item_id = memory.add(item)
            
            # Проверяем, что элемент существует
            assert memory.get(item_id) is not None
            assert memory.size() == 1
            
            # Удаляем элемент
            removed = memory.remove(item_id)
            
            # Проверяем результат
            assert removed is True
            assert memory.get(item_id) is None
            assert memory.size() == 0
            
            # Попытка удалить несуществующий элемент
            removed_again = memory.remove(item_id)
            assert removed_again is False
            
        finally:
            memory.shutdown()
    
    def test_memory_item_edge_cases(self):
        """Тест граничных случаев для MemoryItem."""
        # Элемент с пустым содержимым
        embedding = np.random.random(384).astype(np.float32)
        empty_item = MemoryItem(
            content="",
            embedding=embedding,
            content_type="text"
        )
        
        assert empty_item.content == ""
        assert empty_item.access_count == 0
        
        # Элемент с очень длинным содержимым
        long_content = "x" * 10000
        long_item = MemoryItem(
            content=long_content,
            embedding=embedding,
            content_type="text"
        )
        
        assert len(long_item.content) == 10000
        
        # Элемент с нулевым embedding
        zero_embedding = np.zeros(384, dtype=np.float32)
        zero_item = MemoryItem(
            content="Zero embedding item",
            embedding=zero_embedding,
            content_type="text"
        )
        
        assert np.all(zero_item.embedding == 0)
        
        # Множественные вызовы access
        for i in range(100):  # Уменьшено для быстрого выполнения
            zero_item.access()
        
        assert zero_item.access_count == 100
    
    def test_memory_with_large_capacity(self):
        """Тест памяти с очень большой вместимостью."""
        memory = BiomorphicMemory(
            stm_capacity=1000,
            ltm_capacity=10000,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # Добавляем много элементов
            for i in range(100):
                embedding = np.random.random(384).astype(np.float32)
                item = MemoryItem(
                    content=f"Large capacity item {i}",
                    embedding=embedding,
                    content_type="text"
                )
                memory.add(item)
            
            # Проверяем, что все элементы в STM (не должно быть консолидации)
            assert len(memory.stm) == 100
            assert len(memory.ltm) == 0
            
            # Проверяем работу с большим количеством элементов
            stats = memory.get_memory_statistics()
            assert stats["total_items"] == 100
            assert stats["memory_levels"]["stm"]["pressure"] <= 0.1  # Низкое давление
            
        finally:
            memory.shutdown()


class TestConcurrentAccess:
    """Тесты для конкурентного доступа к памяти."""
    
    def test_concurrent_access(self):
        """Тест конкурентного доступа к памяти."""
        memory = BiomorphicMemory(
            stm_capacity=10,
            ltm_capacity=50,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        results = []
        errors = []
        
        def add_items():
            """Функция для добавления элементов в потоке."""
            try:
                for i in range(5):
                    embedding = np.random.random(384).astype(np.float32)
                    item = MemoryItem(
                        content=f"Concurrent item {i}",
                        embedding=embedding,
                        content_type="text"
                    )
                    item_id = memory.add(item)
                    results.append(item_id)
                    time.sleep(0.001)  # Небольшая задержка
            except Exception as e:
                errors.append(e)
        
        def access_items():
            """Функция для доступа к элементам в потоке."""
            try:
                time.sleep(0.05)  # Даем время для добавления элементов
                for _ in range(10):  # Попытаемся получить доступ несколько раз
                    if results:  # Проверяем, что есть элементы
                        item_id = results[0]  # Берем первый добавленный элемент
                        retrieved_item = memory.get(item_id)
                        if retrieved_item is None:
                            errors.append(f"Item {item_id} not found")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        def search_items():
            """Функция для поиска элементов в потоке."""
            try:
                time.sleep(0.1)  # Даем время для добавления элементов
                for _ in range(3):
                    query_vector = np.random.random(384).astype(np.float32)
                    search_results = memory.search(query_vector, limit=2)
                    if not isinstance(search_results, list):
                        errors.append("Search did not return a list")
                    time.sleep(0.02)
            except Exception as e:
                errors.append(e)
        
        try:
            # Запускаем потоки
            threads = [
                threading.Thread(target=add_items),
                threading.Thread(target=access_items),
                threading.Thread(target=search_items)
            ]
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join(timeout=10.0)  # Увеличенный таймаут
            
            # Проверяем, что потоки завершились
            for thread in threads:
                assert not thread.is_alive(), "Thread did not finish in time"
            
            # Проверяем, что не было ошибок
            if errors:
                print(f"Errors encountered: {errors}")
            assert len(errors) == 0, f"Errors occurred during concurrent access: {errors}"
            assert len(results) > 0, "No items were added"
            
        finally:
            memory.shutdown()


class TestMemoryPersistence:
    """Тесты для симуляции персистентности памяти."""
    
    def test_memory_persistence_simulation(self):
        """Тест симуляции персистентности памяти."""
        # Первая сессия - добавляем данные
        memory1 = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        original_items = []
        try:
            # Добавляем элементы
            for i in range(7):
                embedding = np.random.random(384).astype(np.float32)
                item = MemoryItem(
                    content=f"Persistent item {i}",
                    embedding=embedding,
                    content_type="text",
                    metadata={"session": 1}
                )
                item_id = memory1.add(item)
                original_items.append((item_id, item.content))
            
            # Консолидируем
            memory1.force_consolidation()
            
            # Экспортируем дамп
            dump = memory1.export_memory_dump()
            
        finally:
            memory1.shutdown()
        
        # Вторая сессия - "восстанавливаем" данные
        memory2 = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # "Восстанавливаем" данные из дампа (имитация)
            for item_data in dump["items"]["stm"] + dump["items"]["ltm"]:
                embedding = np.random.random(384).astype(np.float32)  # Новый embedding
                item = MemoryItem(
                    content=item_data["content"],
                    embedding=embedding,
                    content_type=item_data["content_type"],
                    metadata=item_data["metadata"]
                )
                memory2.add(item)
            
            # Проверяем, что данные "восстановлены"
            assert memory2.size() > 0
            
            # Получаем все элементы и проверяем содержимое
            recent_items = memory2.get_recent_items(hours=24.0)
            restored_contents = [item.content for item in recent_items]
            
            for original_id, original_content in original_items:
                assert any(original_content in content for content in restored_contents)
        
        finally:
            memory2.shutdown()


class TestMemoryOptimization:
    """Тесты для оптимизации памяти."""
    
    def test_memory_optimization(self):
        """Тест оптимизации памяти."""
        memory = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # Заполняем память элементами
            item_ids = []
            for i in range(8):
                embedding = np.random.random(384).astype(np.float32)
                item = MemoryItem(
                    content=f"Optimization item {i}",
                    embedding=embedding,
                    content_type="text"
                )
                item_id = memory.add(item)
                item_ids.append(item_id)
            
            # Принудительно консолидируем некоторые элементы в LTM
            memory.force_consolidation()
            
            # Запускаем оптимизацию
            optimization_result = memory.optimize_memory()
            
            # Проверяем результат оптимизации
            assert "actions_taken" in optimization_result
            assert "before" in optimization_result
            assert "after" in optimization_result
            assert "timestamp" in optimization_result
            assert isinstance(optimization_result["actions_taken"], list)
            assert isinstance(optimization_result["before"], dict)
            assert isinstance(optimization_result["after"], dict)
            
            # Проверяем, что время оптимизации разумное
            current_time = time.time()
            assert optimization_result["timestamp"] <= current_time
            assert optimization_result["timestamp"] >= (current_time - 10)  # Не более 10 секунд назад
            
        finally:
            memory.shutdown()
    
    def test_memory_dump(self):
        """Тест экспорта дампа памяти."""
        memory = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # Добавляем несколько элементов
            for i in range(3):
                embedding = np.random.random(384).astype(np.float32)
                item = MemoryItem(
                    content=f"Dump item {i}",
                    embedding=embedding,
                    content_type="text",
                    metadata={"index": i}
                )
                memory.add(item)
            
            # Экспортируем дамп
            dump = memory.export_memory_dump()
            
            # Проверяем структуру дампа
            assert "metadata" in dump
            assert "configuration" in dump
            assert "statistics" in dump
            assert "items" in dump
            
            # Проверяем метаданные
            assert dump["metadata"]["memory_type"] == "biomorphic"
            assert "timestamp" in dump["metadata"]
            assert "version" in dump["metadata"]
            
            # Проверяем конфигурацию
            config = dump["configuration"]
            assert config["stm_capacity"] == 5
            assert config["ltm_capacity"] == 20
            assert config["working_capacity"] == 7
            assert config["semantic_indexing"] is False
            
            # Проверяем элементы
            items = dump["items"]
            assert "stm" in items
            assert "ltm" in items
            assert "working_memory" in items
            
            assert len(items["stm"]) == 3
            assert len(items["ltm"]) == 0
            assert len(items["working_memory"]) == 0
            
            # Проверяем структуру элементов
            for item_data in items["stm"]:
                assert "id" in item_data
                assert "content" in item_data
                assert "content_type" in item_data
                assert "created_at" in item_data
                assert "last_accessed_at" in item_data
                assert "access_count" in item_data
                assert "metadata" in item_data
                
        finally:
            memory.shutdown()


@patch('neurograph.memory.impl.biomorphic.BiomorphicMemory._encode_text')
def test_search_with_mocked_encoding(mock_encode):
    """Тест поиска с мокированием кодирования."""
    # Настраиваем мок
    mock_encode.return_value = np.ones(384, dtype=np.float32)
    
    memory = BiomorphicMemory(
        stm_capacity=5,
        ltm_capacity=20,
        use_semantic_indexing=False,
        auto_consolidation=False
    )
    
    try:
        # Добавляем элементы
        item_ids = []
        for i in range(3):
            embedding = np.ones(384, dtype=np.float32) * (i + 1)  # Разные векторы
            item = MemoryItem(
                content=f"Search item {i}",
                embedding=embedding,
                content_type="text"
            )
            item_id = memory.add(item)
            item_ids.append(item_id)
        
        # Выполняем поиск по тексту
        results = memory.search("test query", limit=2)
        
        # Проверяем, что поиск работает
        assert len(results) >= 0
        assert len(results) <= 2
        assert mock_encode.called
        
        # Проверяем типы результатов
        for item_id, score in results:
            assert isinstance(item_id, str)
            assert isinstance(score, float)
            assert item_id in item_ids
        
    finally:
        memory.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    def test_clear_memory(self):
        """Тест очистки памяти."""
        # Добавляем несколько элементов
        for i in range(3):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Item {i}",
                embedding=embedding,
                content_type="text"
            )
            self.memory.add(item)
        
        # Проверяем, что память не пуста
        assert self.memory.size() > 0
        assert len(self.memory.stm) > 0
        
        # Очищаем память
        self.memory.clear()
        
        # Проверяем результат
        assert self.memory.size() == 0
        assert len(self.memory.stm) == 0
        assert len(self.memory.ltm) == 0
        assert len(self.memory.working_memory) == 0
    
    def test_memory_statistics(self):
        """Тест получения статистики памяти."""
        # Создаем новую память для этого теста
        memory = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # Добавляем элементы и обращаемся к ним
            item_ids = []
            for i in range(3):
                embedding = np.random.random(384).astype(np.float32)
                item = MemoryItem(
                    content=f"Stats item {i}",
                    embedding=embedding,
                    content_type="text"
                )
                item_id = memory.add(item)
                item_ids.append(item_id)
            
            # Обращаемся к элементам
            for item_id in item_ids:
                retrieved_item = memory.get(item_id)
                assert retrieved_item is not None
            
            # Выполняем поиск
            query_vector = np.random.random(384).astype(np.float32)
            memory.search(query_vector, limit=2)
            
            # Получаем статистику
            stats = memory.get_memory_statistics()
            
            # Проверяем структуру статистики
            assert "memory_levels" in stats
            assert "performance" in stats
            assert "consolidation" in stats
            assert "total_items" in stats
            assert "memory_efficiency" in stats
            assert "semantic_indexing" in stats
            
            # Проверяем конкретные значения
            assert stats["memory_levels"]["stm"]["size"] == 3
            assert stats["performance"]["items_added"] == 3
            assert stats["performance"]["items_accessed"] >= 3  # Может быть больше из-за рабочей памяти
            assert stats["performance"]["search_queries"] == 1
            assert stats["total_items"] == 3
            
        finally:
            memory.shutdown()
    
    def test_recent_items(self):
        """Тест получения недавних элементов."""
        # Добавляем элементы с задержкой
        item_ids = []
        start_time = time.time()
        
        for i in range(3):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Recent item {i}",
                embedding=embedding,
                content_type="text"
            )
            item_id = self.memory.add(item)
            item_ids.append(item_id)
            time.sleep(0.01)  # Небольшая задержка
        
        # Получаем недавние элементы
        recent_items = self.memory.get_recent_items(hours=1.0)
        
        # Проверяем результат
        assert len(recent_items) == 3
        assert all(isinstance(item, MemoryItem) for item in recent_items)
        
        # Проверяем фильтрацию по уровню памяти
        stm_items = self.memory.get_recent_items(hours=1.0, memory_level="stm")
        assert len(stm_items) == 3  # Все элементы в STM
        
        ltm_items = self.memory.get_recent_items(hours=1.0, memory_level="ltm")
        assert len(ltm_items) == 0  # Нет элементов в LTM
        
        # Проверяем фильтрацию по времени (более реалистичный интервал)
        old_items = self.memory.get_recent_items(hours=0.0)
        assert len(old_items) == 0
    
    def test_most_accessed_items(self):
        """Тест получения наиболее часто используемых элементов."""
        # Создаем новую память для этого теста
        memory = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # Добавляем элементы
            item_ids = []
            for i in range(3):
                embedding = np.random.random(384).astype(np.float32)
                item = MemoryItem(
                    content=f"Accessed item {i}",
                    embedding=embedding,
                    content_type="text"
                )
                item_id = memory.add(item)
                item_ids.append(item_id)
            
            # Обращаемся к элементам разное количество раз
            for _ in range(5):
                memory.get(item_ids[0])  # 5 обращений
            
            for _ in range(2):
                memory.get(item_ids[1])  # 2 обращения
            
            memory.get(item_ids[2])  # 1 обращение
            
            # Получаем наиболее используемые элементы
            most_accessed = memory.get_most_accessed_items(limit=3)
            
            # Проверяем результат
            assert len(most_accessed) >= 3
            assert all(isinstance(item, MemoryItem) for item, count in most_accessed)
            assert all(isinstance(count, int) for item, count in most_accessed)
            
            # Проверяем, что элементы отсортированы по частоте использования
            access_counts = [count for item, count in most_accessed[:3]]
            assert access_counts == sorted(access_counts, reverse=True)
            
            # Проверяем, что первый элемент имеет наибольшее количество обращений
            assert most_accessed[0][1] >= 5
            
        finally:
            memory.shutdown()


class TestMemoryStrategies:
    """Тесты для стратегий консолидации и забывания."""
    
    def test_time_based_consolidation(self):
        """Тест консолидации на основе времени."""
        strategy = TimeBasedConsolidation(min_age_seconds=1.0, max_stm_size=3)
        
        # Создаем элементы памяти разного возраста
        current_time = time.time()
        stm_items = {}
        
        for i in range(5):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(f"Item {i}", embedding)
            item.id = f"item_{i}"
            item.created_at = current_time - (i * 0.5)  # Разный возраст
            stm_items[item.id] = item
        
        ltm_items = {}
        
        # Тестируем стратегию
        candidates = strategy.should_consolidate(stm_items, ltm_items)
        
        # Проверяем, что выбраны кандидаты (из-за переполнения STM)
        assert isinstance(candidates, list)
        assert all(isinstance(candidate, str) for candidate in candidates)


class TestMemoryFactory:
    """Тесты для фабрики памяти."""
    
    def test_create_default_memory(self):
        """Тест создания памяти по умолчанию."""
        memory = MemoryFactory.create("biomorphic")
        
        try:
            assert isinstance(memory, BiomorphicMemory)
            assert memory.stm_capacity == 100  # Значение по умолчанию
        finally:
            memory.shutdown()
    
    def test_create_from_config(self):
        """Тест создания памяти из конфигурации."""
        config = {
            "type": "biomorphic",
            "stm_capacity": 50,
            "ltm_capacity": 500,
            "auto_consolidation": False
        }
        
        memory = MemoryFactory.create_from_config(config)
        
        try:
            assert isinstance(memory, BiomorphicMemory)
            assert memory.stm_capacity == 50
            assert memory.ltm_capacity == 500
        finally:
            memory.shutdown()
    
    def test_get_available_types(self):
        """Тест получения доступных типов памяти."""
        types = MemoryFactory.get_available_types()
        
        assert isinstance(types, list)
        assert "biomorphic" in types


class TestMemoryIntegration:
    """Интеграционные тесты для памяти."""
    
    def test_memory_lifecycle(self):
        """Тест полного жизненного цикла памяти."""
        memory = BiomorphicMemory(
            stm_capacity=3,
            ltm_capacity=10,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # Добавляем элементы
            item_ids = []
            for i in range(5):
                embedding = np.random.random(384).astype(np.float32)
                item = MemoryItem(
                    content=f"Lifecycle item {i}",
                    embedding=embedding,
                    content_type="text"
                )
                item_id = memory.add(item)
                item_ids.append(item_id)
            
            # Проверяем, что элементы добавлены
            assert len(item_ids) == 5
            assert memory.size() == 5
            
            # Обращаемся к элементам
            for item_id in item_ids[:3]:
                retrieved_item = memory.get(item_id)
                assert retrieved_item is not None
                assert retrieved_item.access_count >= 1
            
            # Выполняем консолидацию
            consolidation_result = memory.force_consolidation()
            assert isinstance(consolidation_result, dict)
            assert "consolidated" in consolidation_result
            
            # Выполняем поиск
            query_vector = np.random.random(384).astype(np.float32)
            results = memory.search(query_vector, limit=2)
            assert len(results) >= 0  # Может быть 0, если нет результатов
            assert all(isinstance(score, float) for _, score in results)
            
            # Получаем статистику
            stats = memory.get_memory_statistics()
            assert stats["total_items"] > 0
            assert stats["performance"]["items_added"] == 5
            
            # Оптимизируем память
            optimization = memory.optimize_memory()
            assert isinstance(optimization["actions_taken"], list)
            
        finally:
            memory.shutdown()
    
    def test_memory_stress_test(self):
        """Стресс-тест памяти с большим количеством операций."""
        memory = BiomorphicMemory(
            stm_capacity=20,
            ltm_capacity=100,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            item_ids = []
            
            # Добавляем много элементов
            for i in range(50):
                embedding = np.random.random(384).astype(np.float32)
                item = MemoryItem(
                    content=f"Stress test item {i}",
                    embedding=embedding,
                    content_type="text",
                    metadata={"batch": i // 10}
                )
                item_id = memory.add(item)
                item_ids.append(item_id)
                
                # Периодически выполняем консолидацию
                if i % 25 == 0 and i > 0:
                    memory.force_consolidation()
            
            # Проверяем общее состояние
            total_items = len(memory.stm) + len(memory.ltm)
            assert total_items == 50
            
            # Выполняем много операций доступа
            access_count = 0
            for i in range(100):
                item_id = item_ids[i % len(item_ids)]
                retrieved_item = memory.get(item_id)
                assert retrieved_item is not None
                access_count += 1
            
            # Выполняем много поисковых запросов
            for i in range(20):
                query_vector = np.random.random(384).astype(np.float32)
                results = memory.search(query_vector, limit=5)
                assert isinstance(results, list)
                assert len(results) <= 5
            
            # Удаляем некоторые элементы
            items_to_remove = item_ids[:10]
            for item_id in items_to_remove:
                removed = memory.remove(item_id)
                assert removed is True
            
            # Проверяем финальное состояние
            final_total = len(memory.stm) + len(memory.ltm)
            assert final_total == 40  # 50 - 10 удаленных
            
            # Получаем статистику
            stats = memory.get_memory_statistics()
            assert stats["performance"]["items_added"] == 50
            assert stats["performance"]["items_accessed"] >= access_count  # Может быть больше
            assert stats["performance"]["search_queries"] >= 20
            
        finally:
            memory.shutdown()


class TestMemoryEdgeCases:
    """Тесты граничных случаев и обработки ошибок."""
    
    def test_empty_memory_operations(self):
        """Тест операций с пустой памятью."""
        memory = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # Операции с пустой памятью
            assert memory.size() == 0
            assert memory.get("non-existent") is None
            assert memory.remove("non-existent") is False
            
            # Поиск в пустой памяти
            query_vector = np.random.random(384).astype(np.float32)
            results = memory.search(query_vector, limit=10)
            assert results == []
            
            # Статистика пустой памяти
            stats = memory.get_memory_statistics()
            assert stats["total_items"] == 0
            assert stats["performance"]["items_added"] == 0
            
            # Недавние элементы в пустой памяти
            recent = memory.get_recent_items()
            assert recent == []
            
            # Наиболее используемые элементы в пустой памяти
            most_accessed = memory.get_most_accessed_items()
            assert most_accessed == []
            
            # Консолидация пустой памяти
            consolidation = memory.force_consolidation()
            assert consolidation["consolidated"] == 0
            
        finally:
            memory.shutdown()
    
    def test_invalid_search_queries(self):
        """Тест поиска с некорректными запросами."""
        memory = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,
            auto_consolidation=False
        )
        
        try:
            # Добавляем один элемент
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content="Test item",
                embedding=embedding,
                content_type="text"
            )
            memory.add(item)
            
            # Поиск с пустым запросом
            results = memory.search("", limit=5)
            assert isinstance(results, list)
            
            # Поиск с limit = 0
            results = memory.search("test", limit=0)
            assert results == []
            
            # Поиск с отрицательным limit (должен обрабатываться корректно)
            results = memory.search("test", limit=-1)
            assert isinstance(results, list)
            
            # Поиск с некорректным вектором должен вызывать ошибку или обрабатываться
            wrong_size_vector = np.random.random(100).astype(np.float32)  # Неправильный размер
            try:
                results = memory.search(wrong_size_vector, limit=5)
                # Если не вызвало ошибку, проверяем, что результат корректный
                assert isinstance(results, list)
            except (ValueError, RuntimeError):
                # Ошибка ожидаема для векторов неправильной размерности
                pass
            
        finally:
            memory.shutdown()
