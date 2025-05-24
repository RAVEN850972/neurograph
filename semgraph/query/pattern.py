"""Механизмы поиска по шаблонам в графе."""

from typing import Dict, List, Set, Tuple, Any, Optional, Iterator, Union
import re

from neurograph.semgraph.base import ISemGraph
from neurograph.core.logging import get_logger


logger = get_logger("semgraph.query.pattern")


class Pattern:
    """Класс для представления шаблона запроса к графу."""
    
    def __init__(self, subject: Optional[str] = None, predicate: Optional[str] = None, 
                object: Optional[str] = None):
        """Инициализирует шаблон.
        
        Args:
            subject: Шаблон для субъекта (None для любого).
            predicate: Шаблон для предиката/типа ребра (None для любого).
            object: Шаблон для объекта (None для любого).
        """
        self.subject = subject
        self.predicate = predicate
        self.object = object
    
    def __str__(self) -> str:
        """Возвращает строковое представление шаблона.
        
        Returns:
            Строковое представление.
        """
        subj = self.subject if self.subject is not None else "?"
        pred = self.predicate if self.predicate is not None else "?"
        obj = self.object if self.object is not None else "?"
        return f"({subj}, {pred}, {obj})"


class PatternMatcher:
    """Класс для поиска по шаблонам в графе."""
    
    def __init__(self, graph: ISemGraph):
        """Инициализирует поиск по шаблонам.
        
        Args:
            graph: Граф для поиска.
        """
        self.graph = graph
    
    def match(self, pattern: Pattern) -> List[Tuple[str, str, str]]:
        """Ищет триплеты, соответствующие шаблону.
        
        Args:
            pattern: Шаблон для поиска.
            
        Returns:
            Список триплетов (subject, predicate, object).
        """
        matches = []
        
        # Получаем все ребра в графе
        all_edges = self.graph.get_all_edges()
        
        for source, target, edge_type in all_edges:
            # Проверяем соответствие шаблону
            if self._match_pattern(source, edge_type, target, pattern):
                matches.append((source, edge_type, target))
        
        return matches
    
    def match_with_regex(self, subject_pattern: Optional[str] = None, 
                       predicate_pattern: Optional[str] = None,
                       object_pattern: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Ищет триплеты, соответствующие регулярным выражениям.
        
        Args:
            subject_pattern: Регулярное выражение для субъекта (None для любого).
            predicate_pattern: Регулярное выражение для предиката (None для любого).
            object_pattern: Регулярное выражение для объекта (None для любого).
            
        Returns:
            Список триплетов (subject, predicate, object).
        """
        matches = []
        
        # Компилируем регулярные выражения
        subj_regex = re.compile(subject_pattern) if subject_pattern else None
        pred_regex = re.compile(predicate_pattern) if predicate_pattern else None
        obj_regex = re.compile(object_pattern) if object_pattern else None
        
        # Получаем все ребра в графе
        all_edges = self.graph.get_all_edges()
        
        for source, target, edge_type in all_edges:
            # Проверяем соответствие регулярным выражениям
            if self._match_regex(source, edge_type, target, subj_regex, pred_regex, obj_regex):
                matches.append((source, edge_type, target))
        
        return matches
    
    def match_with_attributes(self, subject_attrs: Optional[Dict[str, Any]] = None,
                            object_attrs: Optional[Dict[str, Any]] = None,
                            edge_type: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Ищет триплеты с соответствующими атрибутами узлов.
        
        Args:
            subject_attrs: Атрибуты субъекта для сопоставления (None для любого).
            object_attrs: Атрибуты объекта для сопоставления (None для любого).
            edge_type: Тип ребра (None для любого).
            
        Returns:
            Список триплетов (subject, predicate, object).
        """
        matches = []
        
        # Получаем все ребра в графе
        all_edges = self.graph.get_all_edges()
        
        for source, target, pred in all_edges:
            # Если указан тип ребра и он не соответствует, пропускаем
            if edge_type is not None and pred != edge_type:
                continue
            
            # Проверяем атрибуты субъекта
            if subject_attrs is not None:
                source_attrs = self.graph.get_node(source)
                if not self._match_attributes(source_attrs, subject_attrs):
                    continue
            
            # Проверяем атрибуты объекта
            if object_attrs is not None:
                target_attrs = self.graph.get_node(target)
                if not self._match_attributes(target_attrs, object_attrs):
                    continue
            
            matches.append((source, pred, target))
        
        return matches
    
    def _match_pattern(self, subject: str, predicate: str, object: str, pattern: Pattern) -> bool:
        """Проверяет соответствие триплета шаблону.
        
        Args:
            subject: Субъект.
            predicate: Предикат.
            object: Объект.
            pattern: Шаблон для сопоставления.
            
        Returns:
            True, если триплет соответствует шаблону, иначе False.
        """
        # Проверяем соответствие каждой части шаблона
        if pattern.subject is not None and subject != pattern.subject:
            return False
        
        if pattern.predicate is not None and predicate != pattern.predicate:
            return False
        
        if pattern.object is not None and object != pattern.object:
            return False
        
        return True
    
    def _match_regex(self, subject: str, predicate: str, object: str,
                   subj_regex: Optional[re.Pattern], pred_regex: Optional[re.Pattern],
                   obj_regex: Optional[re.Pattern]) -> bool:
        """Проверяет соответствие триплета регулярным выражениям.
        
        Args:
            subject: Субъект.
            predicate: Предикат.
            object: Объект.
            subj_regex: Регулярное выражение для субъекта.
            pred_regex: Регулярное выражение для предиката.
            obj_regex: Регулярное выражение для объекта.
            
        Returns:
            True, если триплет соответствует регулярным выражениям, иначе False.
        """
        # Проверяем соответствие каждой части регулярным выражениям
        if subj_regex is not None and not subj_regex.search(subject):
            return False
        
        if pred_regex is not None and not pred_regex.search(predicate):
            return False
        
        if obj_regex is not None and not obj_regex.search(object):
            return False
        
        return True
    
    def _match_attributes(self, attrs: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Проверяет соответствие атрибутов узла шаблону.
        
        Args:
            attrs: Атрибуты узла.
            pattern: Шаблон атрибутов для сопоставления.
            
        Returns:
            True, если атрибуты соответствуют шаблону, иначе False.
        """
        for key, value in pattern.items():
            # Если ключ отсутствует в атрибутах или значение не соответствует
            if key not in attrs or attrs[key] != value:
                return False
        
        return True