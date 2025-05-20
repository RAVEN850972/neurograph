"""Тесты для поиска по шаблонам в графе."""

import pytest
from unittest.mock import MagicMock

from neurograph.semgraph.query.pattern import Pattern, PatternMatcher


class TestPattern:
    """Тесты для класса Pattern."""
    
    def test_init(self):
        """Проверка инициализации шаблона."""
        pattern = Pattern(subject="A", predicate="P", object="B")
        
        assert pattern.subject == "A"
        assert pattern.predicate == "P"
        assert pattern.object == "B"
    
    def test_str(self):
        """Проверка строкового представления шаблона."""
        pattern = Pattern(subject="A", predicate="P", object="B")
        
        assert str(pattern) == "(A, P, B)"
        
        # Шаблон с None
        pattern = Pattern(subject=None, predicate="P", object=None)
        
        assert str(pattern) == "(?, P, ?)"


class TestPatternMatcher:
    """Тесты для класса PatternMatcher."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        # Создаем мок графа
        self.graph = MagicMock()
        
        # Настраиваем поведение графа
        def get_all_edges():
            return [
                ("A", "B", "is_a"),
                ("B", "C", "has_property"),
                ("A", "D", "has_property"),
                ("D", "E", "is_a"),
                ("C", "E", "is_related_to")
            ]
        
        def get_node(node_id):
            nodes = {
                "A": {"type": "entity", "name": "Apple"},
                "B": {"type": "category", "name": "Fruit"},
                "C": {"type": "property", "name": "Sweet"},
                "D": {"type": "property", "name": "Red"},
                "E": {"type": "category", "name": "Color"}
            }
            return nodes.get(node_id, {})
        
        # Настраиваем методы мока
        self.graph.get_all_edges.side_effect = get_all_edges
        self.graph.get_node.side_effect = get_node
        
        # Создаем PatternMatcher
        self.matcher = PatternMatcher(self.graph)
    
    def test_match(self):
        """Проверка поиска по шаблону."""
        # Поиск всех треплетов с предикатом "is_a"
        pattern = Pattern(predicate="is_a")
        results = self.matcher.match(pattern)
        
        # Проверяем результаты
        assert len(results) == 2
        assert ("A", "is_a", "B") in results
        assert ("D", "is_a", "E") in results
        
        # Поиск треплетов с конкретным субъектом
        pattern = Pattern(subject="A")
        results = self.matcher.match(pattern)
        
        assert len(results) == 2
        assert ("A", "is_a", "B") in results
        assert ("A", "has_property", "D") in results
        
        # Поиск конкретного треплета
        pattern = Pattern(subject="A", predicate="is_a", object="B")
        results = self.matcher.match(pattern)
        
        assert len(results) == 1
        assert results[0] == ("A", "is_a", "B")
    
    def test_match_with_regex(self):
        """Проверка поиска с использованием регулярных выражений."""
        # Поиск всех треплетов, где предикат начинается с "is_"
        results = self.matcher.match_with_regex(predicate_pattern="^is_")
        
        # Проверяем результаты
        assert len(results) == 3
        assert ("A", "is_a", "B") in results
        assert ("D", "is_a", "E") in results
        assert ("C", "is_related_to", "E") in results
    
    def test_match_with_attributes(self):
        """Проверка поиска по атрибутам узлов."""
        # Поиск всех треплетов, где субъект имеет тип "entity"
        results = self.matcher.match_with_attributes(subject_attrs={"type": "entity"})
        
        # Проверяем результаты
        assert len(results) == 2
        assert ("A", "is_a", "B") in results
        assert ("A", "has_property", "D") in results
        
        # Поиск треплетов, где объект имеет тип "category" и предикат "is_a"
        results = self.matcher.match_with_attributes(
            object_attrs={"type": "category"},
            edge_type="is_a"
        )
        
        # Проверяем результаты
        assert len(results) == 2
        assert ("A", "is_a", "B") in results
        assert ("D", "is_a", "E") in results