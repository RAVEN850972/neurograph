"""Тесты для биоморфной памяти."""

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


class TestBiomorphicMemory:
    """Тесты для класса BiomorphicMemory."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        # Создаем память с небольшими размерами для тестирования
        self.memory = BiomorphicMemory(
            stm_capacity=5,
            ltm_capacity=20,
            use_semantic_indexing=False,  # Отключаем для простоты тестов
            auto_consolidation=False,  # Контролируем вручную
            consolidation_interval=1.0
        )
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if hasattr(self, 'memory'):
            self.memory.shutdown()
    
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
        assert self.memory.size() == 1
        
        # Проверяем, что элемент находится в STM
        assert len(self.memory.stm) == 1
        assert len(self.memory.ltm) == 0
        
        # Проверяем статистику
        stats = self.memory.get_memory_statistics()
        assert stats["performance"]["items_added"] == 1
    
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
        
        # Проверяем, что элемент попал в рабочую память
        assert len(self.memory.working_memory) == 1
        
        # Повторный доступ
        retrieved_again = self.memory.get(item_id)
        assert retrieved_again.access_count == 2
    
    def test_stm_overflow_and_consolidation(self):
        """Тест переполнения STM и консолидации."""
        # Заполняем STM до предела
        item_ids = []
        for i in range(6):  # Больше чем stm_capacity=5
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Item {i}",
                embedding=embedding,
                content_type="text"
            )
            item_id = self.memory.add(item)
            item_ids.append(item_id)
        
        # Принудительно выполняем консолидацию
        consolidation_result = self.memory.force_consolidation()
        
        # Проверяем, что произошла консолидация
        assert consolidation_result["consolidated"] > 0
        assert len(self.memory.stm) < 6
        assert len(self.memory.ltm) > 0
    
    def test_item_promotion_from_ltm(self):
        """Тест продвижения часто используемых элементов из LTM в STM."""
        # Добавляем элемент и принудительно консолидируем его в LTM
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
        
        # Часто обращаемся к элементу
        for _ in range(4):
            self.memory.get(item_id)
        
        # Проверяем, что элемент вернулся в STM
        assert item_id in self.memory.stm
        assert item_id not in self.memory.ltm
    
    def test_working_memory_management(self):
        """Тест управления рабочей памятью."""
        # Добавляем больше элементов, чем помещается в рабочую память
        item_ids = []
        for i in range(10):  # Больше чем working_memory_capacity=7
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
            self.memory.get(item_id)
        
        # Проверяем, что рабочая память не превышает лимит
        assert len(self.memory.working_memory) <= self.memory.working_memory_capacity
    
    def test_search_functionality(self):
        """Тест функциональности поиска."""
        # Добавляем несколько элементов с разным содержимым
        items_data = [
            "Python programming language",
            "Machine learning algorithms",
            "Natural language processing",
            "Computer vision techniques",
            "Data science methods"
        ]
        
        for content in items_data:
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=content,
                embedding=embedding,
                content_type="text"
            )
            self.memory.add(item)
        
        # Выполняем поиск
        query_vector = np.random.random(384).astype(np.float32)
        results = self.memory.search(query_vector, limit=3)
        
        # Проверяем результаты
        assert len(results) <= 3
        assert len(results) > 0
        
        # Проверяем, что результаты содержат ID и оценку
        for item_id, score in results:
            assert isinstance(item_id, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_remove_item(self):
        """Тест удаления элемента из памяти."""
        # Добавляем элемент
        embedding = np.random.random(384).astype(np.float32)
        item = MemoryItem(
            content="Item to be removed",
            embedding=embedding,
            content_type="text"
        )
        item_id = self.memory.add(item)
        
        # Проверяем, что элемент существует
        assert self.memory.get(item_id) is not None
        
        # Удаляем элемент
        removed = self.memory.remove(item_id)
        
        # Проверяем результат
        assert removed is True
        assert self.memory.get(item_id) is None
        
        # Попытка удалить несуществующий элемент
        removed_again = self.memory.remove(item_id)
        assert removed_again is False
    
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
        
        # Очищаем память
        self.memory.clear()
        
        # Проверяем результат
        assert self.memory.size() == 0
        assert len(self.memory.stm) == 0
        assert len(self.memory.ltm) == 0
        assert len(self.memory.working_memory) == 0
    
    def test_memory_statistics(self):
        """Тест получения статистики памяти."""
        # Добавляем элементы и обращаемся к ним
        item_ids = []
        for i in range(3):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Stats item {i}",
                embedding=embedding,
                content_type="text"
            )
            item_id = self.memory.add(item)
            item_ids.append(item_id)
        
        # Обращаемся к элементам
        for item_id in item_ids:
            self.memory.get(item_id)
        
        # Получаем статистику
        stats = self.memory.get_memory_statistics()
        
        # Проверяем структуру статистики
        assert "memory_levels" in stats
        assert "performance" in stats
        assert "consolidation" in stats
        assert "total_items" in stats
        assert "memory_efficiency" in stats
        
        # Проверяем конкретные значения
        assert stats["memory_levels"]["stm"]["size"] == 3
        assert stats["performance"]["items_added"] == 3
        assert stats["performance"]["items_accessed"] == 3
        assert stats["total_items"] == 3
    
    def test_recent_items(self):
        """Тест получения недавних элементов."""
        # Добавляем элементы с задержкой
        item_ids = []
        for i in range(3):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Recent item {i}",
                embedding=embedding,
                content_type="text"
            )
            item_id = self.memory.add(item)
            item_ids.append(item_id)
            time.sleep(0.1)  # Небольшая задержка
        
        # Получаем недавние элементы
        recent_items = self.memory.get_recent_items(hours=1.0)
        
        # Проверяем результат
        assert len(recent_items) == 3
        
        # Проверяем, что элементы отсортированы по времени создания
        for i in range(len(recent_items) - 1):
            assert recent_items[i].created_at >= recent_items[i + 1].created_at
    
    def test_most_accessed_items(self):
        """Тест получения наиболее часто используемых элементов."""
        # Добавляем элементы
        item_ids = []
        for i in range(3):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Accessed item {i}",
                embedding=embedding,
                content_type="text"
            )
            item_id = self.memory.add(item)
            item_ids.append(item_id)
        
        # Обращаемся к элементам разное количество раз
        for _ in range(5):
            self.memory.get(item_ids[0])  # 5 обращений
        
        for _ in range(2):
            self.memory.get(item_ids[1])  # 2 обращения
        
        self.memory.get(item_ids[2])  # 1 обращение
        
        # Получаем наиболее используемые элементы
        most_accessed = self.memory.get_most_accessed_items(limit=3)
        
        # Проверяем результат
        assert len(most_accessed) == 3
        
        # Проверяем, что элементы отсортированы по частоте использования
        assert most_accessed[0][1] >= most_accessed[1][1] >= most_accessed[2][1]
        assert most_accessed[0][0].id == item_ids[0]  # Самый часто используемый
    
    def test_memory_optimization(self):
        """Тест оптимизации памяти."""
        # Заполняем память элементами
        for i in range(8):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Optimization item {i}",
                embedding=embedding,
                content_type="text"
            )
            self.memory.add(item)
        
        # Запускаем оптимизацию
        optimization_result = self.memory.optimize_memory()
        
        # Проверяем результат оптимизации
        assert "actions_taken" in optimization_result
        assert "before" in optimization_result
        assert "after" in optimization_result
        assert isinstance(optimization_result["actions_taken"], list)
    
    def test_memory_dump(self):
        """Тест экспорта дампа памяти."""
        # Добавляем несколько элементов
        for i in range(3):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(
                content=f"Dump item {i}",
                embedding=embedding,
                content_type="text",
                metadata={"index": i}
            )
            self.memory.add(item)
        
        # Экспортируем дамп
        dump = self.memory.export_memory_dump()
        
        # Проверяем структуру дампа
        assert "metadata" in dump
        assert "configuration" in dump
        assert "statistics" in dump
        assert "items" in dump
        
        # Проверяем содержимое
        assert dump["metadata"]["memory_type"] == "biomorphic"
        assert len(dump["items"]["stm"]) == 3
        assert len(dump["items"]["ltm"]) == 0
        assert len(dump["items"]["working_memory"]) == 0


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
        assert len(candidates) > 0
    
    def test_importance_based_consolidation(self):
        """Тест консолидации на основе важности."""
        strategy = ImportanceBasedConsolidation(importance_threshold=0.5)
        
        current_time = time.time()
        stm_items = {}
        
        # Создаем элементы с разным уровнем доступа
        for i in range(3):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(f"Important item {i}", embedding)
            item.id = f"important_{i}"
            item.access_count = 10  # Много обращений
            item.created_at = current_time - 100
            stm_items[item.id] = item
        
        ltm_items = {}
        
        # Тестируем стратегию
        candidates = strategy.should_consolidate(stm_items, ltm_items)
        
        # Важные элементы должны быть выбраны для консолидации
        assert len(candidates) > 0
    
    def test_ebbinghaus_forgetting(self):
        """Тест забывания по кривой Эббингауза."""
        strategy = EbbinghausBasedForgetting(base_retention=0.1, decay_rate=0.693)
        
        current_time = time.time()
        items = {}
        
        # Создаем старые элементы, которые должны быть забыты
        for i in range(3):
            embedding = np.random.random(384).astype(np.float32)
            item = MemoryItem(f"Old item {i}", embedding)
            item.id = f"old_{i}"
            item.created_at = current_time - 86400  # Сутки назад
            item.last_accessed_at = current_time - 86400
            item.access_count = 1
            items[item.id] = item
        
        # Тестируем стратегию (может быть стохастической)
        candidates = strategy.should_forget(items)
        
        # Старые элементы могут быть выбраны для забывания
        assert isinstance(candidates, list)


class TestMemoryFactory:
    """Тесты для фабрики памяти."""
    
    def test_create_default_memory(self):
        """Тест создания памяти по умолчанию."""
        memory = MemoryFactory.create("biomorphic")
        
        assert isinstance(memory, BiomorphicMemory)
        assert memory.stm_capacity == 100  # Значение по умолчанию
        
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
        
        assert isinstance(memory, BiomorphicMemory)
        assert memory.stm_capacity == 50
        assert memory.ltm_capacity == 500
        
        memory.shutdown()
    
    def test_create_lightweight_memory(self):
        """Тест создания облегченной памяти."""
        memory = create_lightweight_memory()
        
        assert isinstance(memory, BiomorphicMemory)
        assert memory.stm_capacity == 50
        assert memory.ltm_capacity == 1000
        assert memory.use_semantic_indexing is False
        
        memory.shutdown()
    
    def test_create_high_performance_memory(self):
        """Тест создания высокопроизводительной памяти."""
        memory = create_high_performance_memory()
        
        assert isinstance(memory, BiomorphicMemory)
        assert memory.stm_capacity == 200
        assert memory.ltm_capacity == 50000
        assert memory.use_semantic_indexing is True
        
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
            
            # Обращаемся к элементам
            for item_id in item_ids[:3]:
                memory.get(item_id)
            
            # Выполняем консолидацию
            memory.force_consolidation()
            
            # Проверяем, что консолидация произошла
            assert len(memory.ltm) > 0
            
            # Выполняем поиск
            query_vector = np.random.random(384).astype(np.float32)
            results = memory.search(query_vector, limit=2)
            assert len(results) > 0
            
            # Получаем статистику
            stats = memory.get_memory_statistics()
            assert stats["total_items"] > 0
            
            # Оптимизируем память
            optimization = memory.optimize_memory()
            assert isinstance(optimization["actions_taken"], list)
            
        finally:
            memory.shutdown()
    
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
            except Exception as e:
                errors.append(e)
        
        def access_items():
            """Функция для доступа к элементам в потоке."""
            try:
                time.sleep(0.1)  # Даем время для добавления элементов
                for item_id in results[:3]:
                    if item_id:
                        memory.get(item_id)
            except Exception as e:
                errors.append(e)
        
        try:
            # Запускаем потоки
            threads = [
                threading.Thread(target=add_items),
                threading.Thread(target=access_items)
            ]
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join(timeout=5.0)
            
            # Проверяем, что не было ошибок
            assert len(errors) == 0
            assert len(results) > 0
            
        finally:
            memory.shutdown()
    
    @patch('neurograph.memory.impl.biomorphic.BiomorphicMemory._encode_text')
    def test_search_with_mocked_encoding(self, mock_encode):
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
            for i in range(3):
                embedding = np.ones(384, dtype=np.float32) * (i + 1)  # Разные векторы
                item = MemoryItem(
                    content=f"Search item {i}",
                    embedding=embedding,
                    content_type="text"
                )
                memory.add(item)
            
            # Выполняем поиск по тексту
            results = memory.search("test query", limit=2)
            
            # Проверяем, что поиск работает
            assert len(results) > 0
            assert mock_encode.called
            
        finally:
            memory.shutdown()