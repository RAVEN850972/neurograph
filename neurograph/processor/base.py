"""Базовые интерфейсы и классы для нейросимволического процессора."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import re
from enum import Enum


class RuleType(Enum):
    """Типы правил."""
    SYMBOLIC = "symbolic"
    PATTERN = "pattern"
    CONDITIONAL = "conditional"
    INFERENCE = "inference"


class ActionType(Enum):
    """Типы действий."""
    ASSERT = "assert"
    RETRACT = "retract"
    DERIVE = "derive"
    QUERY = "query"
    EXECUTE = "execute"


@dataclass
class SymbolicRule:
    """Символическое правило для логического вывода."""
    
    condition: str
    action: str
    rule_type: RuleType = RuleType.SYMBOLIC
    action_type: ActionType = ActionType.DERIVE
    weight: float = 1.0
    confidence: float = 1.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Служебные поля
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    def __post_init__(self):
        """Инициализация после создания."""
        if isinstance(self.rule_type, str):
            self.rule_type = RuleType(self.rule_type)
        if isinstance(self.action_type, str):
            self.action_type = ActionType(self.action_type)
    
    def mark_used(self):
        """Отметка использования правила."""
        self.usage_count += 1
        self.last_used = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "id": self.id,
            "condition": self.condition,
            "action": self.action,
            "rule_type": self.rule_type.value,
            "action_type": self.action_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolicRule":
        """Десериализация из словаря."""
        rule = cls(
            condition=data["condition"],
            action=data["action"],
            rule_type=RuleType(data["rule_type"]),
            action_type=ActionType(data["action_type"]),
            weight=data["weight"],
            confidence=data["confidence"],
            priority=data["priority"],
            metadata=data["metadata"]
        )
        rule.id = data["id"]
        rule.created_at = datetime.fromisoformat(data["created_at"])
        rule.usage_count = data["usage_count"]
        if data["last_used"]:
            rule.last_used = datetime.fromisoformat(data["last_used"])
        return rule


@dataclass
class ProcessingContext:
    """Контекст обработки для логического вывода."""
    
    facts: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_depth: int = 5
    confidence_threshold: float = 0.5
    
    def add_fact(self, key: str, value: Any, confidence: float = 1.0):
        """Добавление факта в контекст."""
        self.facts[key] = {
            "value": value,
            "confidence": confidence,
            "added_at": datetime.now()
        }
    
    def get_fact(self, key: str) -> Optional[Any]:
        """Получение факта из контекста."""
        if key in self.facts:
            return self.facts[key]["value"]
        return None
    
    def has_fact(self, key: str) -> bool:
        """Проверка наличия факта."""
        return key in self.facts
    
    def set_variable(self, name: str, value: Any):
        """Установка переменной."""
        self.variables[name] = value
    
    def get_variable(self, name: str) -> Optional[Any]:
        """Получение переменной."""
        return self.variables.get(name)
    
    def clear_variables(self):
        """Очистка переменных."""
        self.variables.clear()
    
    def copy(self) -> "ProcessingContext":
        """Создание копии контекста."""
        new_context = ProcessingContext(
            facts=self.facts.copy(),
            variables=self.variables.copy(),
            query_params=self.query_params.copy(),
            session_id=self.session_id,
            max_depth=self.max_depth,
            confidence_threshold=self.confidence_threshold
        )
        return new_context


@dataclass
class ExplanationStep:
    """Шаг объяснения логического вывода."""
    
    step_number: int
    rule_id: str
    rule_description: str
    input_facts: Dict[str, Any]
    output_facts: Dict[str, Any]
    confidence: float
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "step_number": self.step_number,
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
            "input_facts": self.input_facts,
            "output_facts": self.output_facts,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


@dataclass
class DerivationResult:
    """Результат логического вывода."""
    
    success: bool
    derived_facts: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    explanation: List[ExplanationStep] = field(default_factory=list)
    rules_used: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    error_message: Optional[str] = None
    
    def add_derived_fact(self, key: str, value: Any, confidence: float = 1.0):
        """Добавление выведенного факта."""
        self.derived_facts[key] = {
            "value": value,
            "confidence": confidence
        }
    
    def add_explanation_step(self, step: ExplanationStep):
        """Добавление шага объяснения."""
        self.explanation.append(step)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "success": self.success,
            "derived_facts": self.derived_facts,
            "confidence": self.confidence,
            "explanation": [step.to_dict() for step in self.explanation],
            "rules_used": self.rules_used,
            "processing_time": self.processing_time,
            "error_message": self.error_message
        }


class INeuroSymbolicProcessor(ABC):
    """Интерфейс для нейросимволического процессора."""
    
    @abstractmethod
    def add_rule(self, rule: SymbolicRule) -> str:
        """Добавляет правило в базу знаний и возвращает его ID.
        
        Args:
            rule: Правило для добавления.
            
        Returns:
            Идентификатор добавленного правила.
            
        Raises:
            ProcessorError: Если правило невалидно или не может быть добавлено.
        """
        pass
    
    @abstractmethod
    def remove_rule(self, rule_id: str) -> bool:
        """Удаляет правило из базы знаний.
        
        Args:
            rule_id: Идентификатор правила.
            
        Returns:
            True, если правило было удалено, иначе False.
        """
        pass
    
    @abstractmethod
    def get_rule(self, rule_id: str) -> Optional[SymbolicRule]:
        """Возвращает правило по ID.
        
        Args:
            rule_id: Идентификатор правила.
            
        Returns:
            Правило или None, если правило не найдено.
        """
        pass
    
    @abstractmethod
    def execute_rule(self, rule_id: str, context: ProcessingContext) -> DerivationResult:
        """Выполняет указанное правило в заданном контексте.
        
        Args:
            rule_id: Идентификатор правила.
            context: Контекст выполнения.
            
        Returns:
            Результат выполнения правила.
            
        Raises:
            ProcessorError: Если правило не найдено или не может быть выполнено.
        """
        pass
    
    @abstractmethod
    def derive(self, context: ProcessingContext, depth: int = 1) -> DerivationResult:
        """Производит логический вывод на основе правил и контекста.
        
        Args:
            context: Контекст для вывода.
            depth: Глубина вывода.
            
        Returns:
            Результат логического вывода с объяснением.
            
        Raises:
            ProcessorError: Если вывод не может быть выполнен.
        """
        pass
    
    @abstractmethod
    def find_relevant_rules(self, context: ProcessingContext) -> List[str]:
        """Находит правила, релевантные заданному контексту.
        
        Args:
            context: Контекст для поиска правил.
            
        Returns:
            Список идентификаторов релевантных правил.
        """
        pass
    
    @abstractmethod
    def update_rule(self, rule_id: str, **attributes) -> bool:
        """Обновляет атрибуты правила.
        
        Args:
            rule_id: Идентификатор правила.
            **attributes: Атрибуты для обновления.
            
        Returns:
            True, если правило было обновлено, иначе False.
        """
        pass
    
    @abstractmethod
    def get_all_rules(self) -> List[SymbolicRule]:
        """Возвращает все правила в базе знаний.
        
        Returns:
            Список всех правил.
        """
        pass
    
    @abstractmethod
    def validate_rule(self, rule: SymbolicRule) -> Tuple[bool, Optional[str]]:
        """Проверяет валидность правила.
        
        Args:
            rule: Правило для проверки.
            
        Returns:
            Кортеж (валидность, сообщение об ошибке).
        """
        pass
    
    @abstractmethod
    def clear_rules(self) -> None:
        """Очищает все правила."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику процессора.
        
        Returns:
            Словарь со статистикой.
        """
        pass


class ProcessorError(Exception):
    """Базовое исключение для ошибок процессора."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PROCESSOR_ERROR"
        self.details = details or {}


class RuleValidationError(ProcessorError):
    """Исключение для ошибок валидации правил."""
    
    def __init__(self, message: str, rule_id: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "RULE_VALIDATION_ERROR", details)
        self.rule_id = rule_id


class RuleExecutionError(ProcessorError):
    """Исключение для ошибок выполнения правил."""
    
    def __init__(self, message: str, rule_id: str = None, context: ProcessingContext = None):
        super().__init__(message, "RULE_EXECUTION_ERROR")
        self.rule_id = rule_id
        self.context = context


class DerivationError(ProcessorError):
    """Исключение для ошибок логического вывода."""
    
    def __init__(self, message: str, context: ProcessingContext = None, partial_result: DerivationResult = None):
        super().__init__(message, "DERIVATION_ERROR")
        self.context = context
        self.partial_result = partial_result