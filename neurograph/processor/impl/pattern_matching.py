"""Процессор на основе сопоставления шаблонов."""

import re
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from neurograph.core.logging import get_logger
from neurograph.core.cache import cached

from ..base import (
    INeuroSymbolicProcessor,
    SymbolicRule,
    ProcessingContext,
    DerivationResult,
    ExplanationStep,
    ProcessorError,
    RuleValidationError,
    RuleExecutionError,
    DerivationError,
    RuleType,
    ActionType
)


class PatternMatcher:
    """Класс для сопоставления шаблонов."""
    
    @staticmethod
    def parse_condition(condition: str) -> Dict[str, Any]:
        """Парсит условие правила.
        
        Args:
            condition: Строка условия.
            
        Returns:
            Словарь с распарсенным условием.
        """
        # Простой парсер для условий вида:
        # "X является Y", "X имеет свойство Y", "X связан с Y через Z"
        
        patterns = {
            # Базовые шаблоны
            r"^(\w+)\s+является\s+(\w+)$": {"type": "is_a", "subject": 1, "object": 2},
            r"^(\w+)\s+имеет\s+свойство\s+(\w+)$": {"type": "has_property", "subject": 1, "property": 2},
            r"^(\w+)\s+связан\s+с\s+(\w+)$": {"type": "related_to", "subject": 1, "object": 2},
            r"^(\w+)\s+связан\s+с\s+(\w+)\s+через\s+(\w+)$": {"type": "related_via", "subject": 1, "object": 2, "relation": 3},
            
            # Логические операторы
            r"^(\w+)\s+И\s+(\w+)$": {"type": "and", "left": 1, "right": 2},
            r"^(\w+)\s+ИЛИ\s+(\w+)$": {"type": "or", "left": 1, "right": 2},
            r"^НЕ\s+(\w+)$": {"type": "not", "operand": 1},
            
            # Переменные и функции
            r"^(\?\w+)\s+является\s+(\w+)$": {"type": "is_a_var", "variable": 1, "object": 2},
            r"^exists\((\?\w+)\)$": {"type": "exists", "variable": 1},
            r"^count\((\w+)\)\s*>\s*(\d+)$": {"type": "count_gt", "entity": 1, "threshold": 2},
        }
        
        for pattern, template in patterns.items():
            match = re.match(pattern, condition, re.IGNORECASE)
            if match:
                result = {"type": template["type"]}
                for key, group_num in template.items():
                    if key != "type" and isinstance(group_num, int):
                        value = match.group(group_num)
                        # Конвертация чисел
                        if value.isdigit():
                            value = int(value)
                        result[key] = value
                return result
        
        # Если не найден шаблон, возвращаем как свободный текст
        return {"type": "free_text", "text": condition}
    
    @staticmethod
    def parse_action(action: str) -> Dict[str, Any]:
        """Парсит действие правила.
        
        Args:
            action: Строка действия.
            
        Returns:
            Словарь с распарсенным действием.
        """
        patterns = {
            # Утверждения
            r"^assert\s+(\w+)\s+является\s+(\w+)$": {"type": "assert_is_a", "subject": 1, "object": 2},
            r"^assert\s+(\w+)\s+имеет\s+свойство\s+(\w+)$": {"type": "assert_property", "subject": 1, "property": 2},
            r"^assert\s+(\w+)$": {"type": "assert_fact", "fact": 1},
            
            # Выводы
            r"^derive\s+(\w+)\s+является\s+(\w+)$": {"type": "derive_is_a", "subject": 1, "object": 2},
            r"^derive\s+(\w+)$": {"type": "derive_fact", "fact": 1},
            
            # Запросы
            r"^query\s+(\w+)$": {"type": "query", "target": 1},
            
            # Выполнение функций
            r"^execute\s+(\w+)\(([^)]*)\)$": {"type": "execute_function", "function": 1, "args": 2},
        }
        
        for pattern, template in patterns.items():
            match = re.match(pattern, action, re.IGNORECASE)
            if match:
                result = {"type": template["type"]}
                for key, group_num in template.items():
                    if key != "type" and isinstance(group_num, int):
                        value = match.group(group_num)
                        if key == "args":
                            # Парсинг аргументов функции
                            value = [arg.strip() for arg in value.split(",") if arg.strip()]
                        result[key] = value
                return result
        
        # Если не найден шаблон, возвращаем как свободный текст
        return {"type": "free_text", "text": action}
    
    @staticmethod
    def match_condition(parsed_condition: Dict[str, Any], context: ProcessingContext) -> Tuple[bool, float, Dict[str, Any]]:
        """Проверяет выполнение условия в данном контексте.
        
        Args:
            parsed_condition: Распарсенное условие.
            context: Контекст обработки.
            
        Returns:
            Кортеж (выполнено, уверенность, переменные).
        """
        condition_type = parsed_condition["type"]
        
        if condition_type == "is_a":
            subject = parsed_condition["subject"]
            obj = parsed_condition["object"]
            fact_key = f"{subject}_is_a_{obj}"
            if context.has_fact(fact_key):
                fact_data = context.facts[fact_key]
                return True, fact_data.get("confidence", 1.0), {}
            return False, 0.0, {}
        
        elif condition_type == "has_property":
            subject = parsed_condition["subject"]
            prop = parsed_condition["property"]
            fact_key = f"{subject}_has_{prop}"
            if context.has_fact(fact_key):
                fact_data = context.facts[fact_key]
                return True, fact_data.get("confidence", 1.0), {}
            return False, 0.0, {}
        
        elif condition_type == "related_to":
            subject = parsed_condition["subject"]
            obj = parsed_condition["object"]
            fact_key = f"{subject}_related_to_{obj}"
            if context.has_fact(fact_key):
                fact_data = context.facts[fact_key]
                return True, fact_data.get("confidence", 1.0), {}
            return False, 0.0, {}
        
        elif condition_type == "and":
            left_result = PatternMatcher.match_condition({"type": "free_text", "text": parsed_condition["left"]}, context)
            right_result = PatternMatcher.match_condition({"type": "free_text", "text": parsed_condition["right"]}, context)
            
            if left_result[0] and right_result[0]:
                combined_confidence = min(left_result[1], right_result[1])
                combined_vars = {**left_result[2], **right_result[2]}
                return True, combined_confidence, combined_vars
            return False, 0.0, {}
        
        elif condition_type == "or":
            left_result = PatternMatcher.match_condition({"type": "free_text", "text": parsed_condition["left"]}, context)
            right_result = PatternMatcher.match_condition({"type": "free_text", "text": parsed_condition["right"]}, context)
            
            if left_result[0] or right_result[0]:
                best_confidence = max(left_result[1], right_result[1])
                best_vars = left_result[2] if left_result[1] >= right_result[1] else right_result[2]
                return True, best_confidence, best_vars
            return False, 0.0, {}
        
        elif condition_type == "exists":
            variable = parsed_condition["variable"]
            if context.get_variable(variable.lstrip("?")):
                return True, 1.0, {variable: context.get_variable(variable.lstrip("?"))}
            return False, 0.0, {}
        
        elif condition_type == "free_text":
            # Простой поиск в фактах по тексту
            text = parsed_condition["text"].lower()
            
            # Извлекаем ключевые слова из условия
            condition_words = text.split()
            
            for fact_key, fact_data in context.facts.items():
                fact_key_lower = fact_key.lower()
                fact_value_lower = str(fact_data.get("value", "")).lower()
                
                # Проверяем пересечение ключевых слов
                fact_words = fact_key_lower.replace("_", " ").split()
                
                # Ищем совпадения слов с учетом морфологии
                matches = 0
                for cond_word in condition_words:
                    for fact_word in fact_words:
                        # Простая проверка на совпадение корня (первые 4 символа)
                        if len(cond_word) >= 4 and len(fact_word) >= 4:
                            if cond_word[:4] == fact_word[:4]:
                                matches += 1
                                break
                        elif cond_word == fact_word:
                            matches += 1
                            break
                
                # Если найдено достаточно совпадений
                if matches >= min(2, len(condition_words) // 2):
                    return True, fact_data.get("confidence", 1.0), {}
                
                # Простая проверка вхождения
                if any(word in fact_key_lower for word in condition_words if len(word) > 3):
                    return True, fact_data.get("confidence", 1.0), {}
            
            return False, 0.0, {}
        
        return False, 0.0, {}
    
    @staticmethod
    def execute_action(parsed_action: Dict[str, Any], context: ProcessingContext, confidence: float = 1.0) -> Dict[str, Any]:
        """Выполняет действие в данном контексте.
        
        Args:
            parsed_action: Распарсенное действие.
            context: Контекст обработки.
            confidence: Уверенность в действии.
            
        Returns:
            Словарь с результатами выполнения действия.
        """
        action_type = parsed_action["type"]
        results = {}
        
        if action_type == "assert_is_a":
            subject = parsed_action["subject"]
            obj = parsed_action["object"]
            fact_key = f"{subject}_is_a_{obj}"
            context.add_fact(fact_key, True, confidence)
            results[fact_key] = {"value": True, "confidence": confidence, "action": "asserted"}
        
        elif action_type == "assert_property":
            subject = parsed_action["subject"]
            prop = parsed_action["property"]
            fact_key = f"{subject}_has_{prop}"
            context.add_fact(fact_key, True, confidence)
            results[fact_key] = {"value": True, "confidence": confidence, "action": "asserted"}
        
        elif action_type == "assert_fact":
            fact = parsed_action["fact"]
            context.add_fact(fact, True, confidence)
            results[fact] = {"value": True, "confidence": confidence, "action": "asserted"}
        
        elif action_type == "derive_is_a":
            subject = parsed_action["subject"]
            obj = parsed_action["object"]
            fact_key = f"{subject}_is_a_{obj}"
            context.add_fact(fact_key, True, confidence)
            results[fact_key] = {"value": True, "confidence": confidence, "action": "derived"}
        
        elif action_type == "derive_fact":
            fact = parsed_action["fact"]
            context.add_fact(fact, True, confidence)
            results[fact] = {"value": True, "confidence": confidence, "action": "derived"}
        
        elif action_type == "query":
            target = parsed_action["target"]
            # Поиск фактов, связанных с целью
            matching_facts = {}
            for fact_key, fact_data in context.facts.items():
                if target.lower() in fact_key.lower():
                    matching_facts[fact_key] = fact_data
            results["query_results"] = matching_facts
        
        elif action_type == "execute_function":
            function_name = parsed_action["function"]
            args = parsed_action.get("args", [])
            # Простые встроенные функции
            if function_name == "print":
                print(" ".join(args))
                results["print_output"] = " ".join(args)
            elif function_name == "count":
                if args:
                    entity = args[0]
                    count = sum(1 for key in context.facts.keys() if entity.lower() in key.lower())
                    results["count_result"] = count
        
        elif action_type == "free_text":
            # Попытка интерпретировать как утверждение факта
            text = parsed_action["text"]
            
            # Убираем "derive " если есть
            if text.startswith("derive "):
                text = text[7:]
            
            # Интерпретация различных типов действий
            if 'является' in text:
                parts = text.split(' является ')
                if len(parts) == 2:
                    subject = parts[0].strip()
                    obj = parts[1].strip()
                    fact_key = f"{subject}_is_a_{obj}"
                    context.add_fact(fact_key, True, confidence)
                    results[fact_key] = {"value": True, "confidence": confidence, "action": "derived"}
                else:
                    fact_key = f"interpreted_{hash(text) % 10000}"
                    context.add_fact(fact_key, text, confidence)
                    results[fact_key] = {"value": text, "confidence": confidence, "action": "interpreted"}
            
            elif 'имеет' in text:
                if 'свойство' in text:
                    parts = text.split(' имеет свойство ')
                    if len(parts) == 2:
                        subject = parts[0].strip()
                        prop = parts[1].strip()
                        fact_key = f"{subject}_has_{prop}"
                        context.add_fact(fact_key, True, confidence)
                        results[fact_key] = {"value": True, "confidence": confidence, "action": "derived"}
                    else:
                        fact_key = f"interpreted_{hash(text) % 10000}"
                        context.add_fact(fact_key, text, confidence)
                        results[fact_key] = {"value": text, "confidence": confidence, "action": "interpreted"}
                else:
                    parts = text.split(' имеет ')
                    if len(parts) == 2:
                        subject = parts[0].strip()
                        obj = parts[1].strip()
                        fact_key = f"{subject}_has_{obj}"
                        context.add_fact(fact_key, True, confidence)
                        results[fact_key] = {"value": True, "confidence": confidence, "action": "derived"}
                    else:
                        fact_key = f"interpreted_{hash(text) % 10000}"
                        context.add_fact(fact_key, text, confidence)
                        results[fact_key] = {"value": text, "confidence": confidence, "action": "interpreted"}
            
            elif 'нуждается' in text:
                parts = text.split(' нуждается в ')
                if len(parts) == 2:
                    subject = parts[0].strip()
                    obj = parts[1].strip()
                    fact_key = f"{subject}_needs_{obj}"
                    context.add_fact(fact_key, True, confidence)
                    results[fact_key] = {"value": True, "confidence": confidence, "action": "derived"}
                else:
                    fact_key = f"interpreted_{hash(text) % 10000}"
                    context.add_fact(fact_key, text, confidence)
                    results[fact_key] = {"value": text, "confidence": confidence, "action": "interpreted"}
            
            elif 'ищет' in text:
                parts = text.split(' ищет ')
                if len(parts) == 2:
                    subject = parts[0].strip()
                    obj = parts[1].strip()
                    fact_key = f"{subject}_seeks_{obj}"
                    context.add_fact(fact_key, True, confidence)
                    results[fact_key] = {"value": True, "confidence": confidence, "action": "derived"}
                else:
                    fact_key = f"interpreted_{hash(text) % 10000}"
                    context.add_fact(fact_key, text, confidence)
                    results[fact_key] = {"value": text, "confidence": confidence, "action": "interpreted"}
            
            elif 'живая' in text or 'живой' in text or 'живое' in text:
                # собака живая
                words = text.split()
                if len(words) >= 2:
                    subject = words[0]
                    fact_key = f"{subject}_is_a_живая"
                    context.add_fact(fact_key, True, confidence)
                    results[fact_key] = {"value": True, "confidence": confidence, "action": "derived"}
                else:
                    fact_key = f"interpreted_{hash(text) % 10000}"
                    context.add_fact(fact_key, text, confidence)
                    results[fact_key] = {"value": text, "confidence": confidence, "action": "interpreted"}
            
            elif 'заботе' in text or 'заботу' in text:
                # собака нуждается в заботе
                words = text.split()
                if len(words) >= 3:
                    subject = words[0]
                    fact_key = f"{subject}_needs_заботе"
                    context.add_fact(fact_key, True, confidence)
                    results[fact_key] = {"value": True, "confidence": confidence, "action": "derived"}
                else:
                    fact_key = f"interpreted_{hash(text) % 10000}"
                    context.add_fact(fact_key, text, confidence)
                    results[fact_key] = {"value": text, "confidence": confidence, "action": "interpreted"}
            
            else:
                # Общий случай
                fact_key = f"interpreted_{hash(text) % 10000}"
                context.add_fact(fact_key, text, confidence)
                results[fact_key] = {"value": text, "confidence": confidence, "action": "interpreted"}
        
        return results


class PatternMatchingProcessor(INeuroSymbolicProcessor):
    """Процессор на основе сопоставления шаблонов."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 max_depth: int = 5,
                 enable_explanations: bool = True,
                 cache_rules: bool = True,
                 parallel_processing: bool = False,
                 rule_indexing: bool = True):
        """Инициализация процессора.
        
        Args:
            confidence_threshold: Минимальная уверенность для применения правил.
            max_depth: Максимальная глубина вывода.
            enable_explanations: Включить генерацию объяснений.
            cache_rules: Включить кеширование правил.
            parallel_processing: Включить параллельную обработку.
            rule_indexing: Включить индексацию правил.
        """
        self.confidence_threshold = confidence_threshold
        self.max_depth = max_depth
        self.enable_explanations = enable_explanations
        self.cache_rules = cache_rules
        self.parallel_processing = parallel_processing
        self.rule_indexing = rule_indexing
        
        # Хранилище правил
        self._rules: Dict[str, SymbolicRule] = {}
        
        # Индексы для быстрого поиска
        self._condition_index: Dict[str, Set[str]] = {}  # тип условия -> множество rule_id
        self._fact_index: Dict[str, Set[str]] = {}       # факт -> множество rule_id
        
        # Статистика
        self._stats = {
            "rules_added": 0,
            "rules_executed": 0,
            "derivations_performed": 0,
            "total_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.logger = get_logger("pattern_matching_processor")
        self.logger.info("Инициализирован PatternMatchingProcessor")
    
    def add_rule(self, rule: SymbolicRule) -> str:
        """Добавляет правило в базу знаний."""
        # Валидация правила
        is_valid, error_msg = self.validate_rule(rule)
        if not is_valid:
            raise RuleValidationError(f"Невалидное правило: {error_msg}", rule.id)
        
        # Добавление в хранилище
        self._rules[rule.id] = rule
        
        # Обновление индексов
        if self.rule_indexing:
            self._update_indexes(rule)
        
        self._stats["rules_added"] += 1
        self.logger.debug(f"Добавлено правило {rule.id}: {rule.condition} -> {rule.action}")
        
        return rule.id
    
    def remove_rule(self, rule_id: str) -> bool:
        """Удаляет правило из базы знаний."""
        if rule_id in self._rules:
            rule = self._rules[rule_id]
            
            # Удаление из индексов
            if self.rule_indexing:
                self._remove_from_indexes(rule)
            
            del self._rules[rule_id]
            self.logger.debug(f"Удалено правило {rule_id}")
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[SymbolicRule]:
        """Возвращает правило по ID."""
        return self._rules.get(rule_id)
    
    def execute_rule(self, rule_id: str, context: ProcessingContext) -> DerivationResult:
        """Выполняет указанное правило в заданном контексте."""
        start_time = time.time()
        
        try:
            rule = self._rules.get(rule_id)
            if not rule:
                raise RuleExecutionError(f"Правило {rule_id} не найдено", rule_id, context)
            
            # Парсинг условия и действия
            parsed_condition = PatternMatcher.parse_condition(rule.condition)
            parsed_action = PatternMatcher.parse_action(rule.action)
            
            # Проверка условия
            condition_met, condition_confidence, variables = PatternMatcher.match_condition(parsed_condition, context)
            
            result = DerivationResult(success=False)
            
            if condition_met and condition_confidence >= self.confidence_threshold:
                # Выполнение действия
                action_results = PatternMatcher.execute_action(parsed_action, context, condition_confidence * rule.confidence)
                
                # Обновление контекста переменными
                for var_name, var_value in variables.items():
                    context.set_variable(var_name, var_value)
                
                # Формирование результата
                result.success = True
                result.confidence = condition_confidence * rule.confidence
                result.rules_used = [rule_id]
                
                for fact_key, fact_data in action_results.items():
                    result.add_derived_fact(fact_key, fact_data["value"], fact_data["confidence"])
                
                # Добавление объяснения
                if self.enable_explanations:
                    explanation_step = ExplanationStep(
                        step_number=1,
                        rule_id=rule_id,
                        rule_description=f"{rule.condition} -> {rule.action}",
                        input_facts={k: v for k, v in context.facts.items() if condition_met},
                        output_facts=action_results,
                        confidence=result.confidence,
                        reasoning=f"Условие '{rule.condition}' выполнено с уверенностью {condition_confidence:.2f}"
                    )
                    result.add_explanation_step(explanation_step)
                
                # Отметка использования правила
                rule.mark_used()
                self._stats["rules_executed"] += 1
            
            execution_time = time.time() - start_time
            result.processing_time = execution_time
            self._stats["total_execution_time"] += execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = DerivationResult(
                success=False,
                processing_time=execution_time,
                error_message=str(e)
            )
            raise RuleExecutionError(f"Ошибка выполнения правила {rule_id}: {e}", rule_id, context)
    
    def derive(self, context: ProcessingContext, depth: int = 1) -> DerivationResult:
        """Производит логический вывод на основе правил и контекста."""
        start_time = time.time()
        
        try:
            # Ограничение глубины
            actual_depth = min(depth, self.max_depth)
            
            # Результат вывода
            final_result = DerivationResult(success=True)
            
            # Копирование контекста для безопасности
            working_context = context.copy()
            
            for current_depth in range(actual_depth):
                depth_result = self._perform_single_derivation_step(working_context, current_depth + 1)
                
                if not depth_result.success:
                    break
                
                # Объединение результатов
                final_result.derived_facts.update(depth_result.derived_facts)
                final_result.explanation.extend(depth_result.explanation)
                final_result.rules_used.extend(depth_result.rules_used)
                
                # Обновление уверенности (минимум из всех шагов)
                if final_result.confidence == 0.0:
                    final_result.confidence = depth_result.confidence
                else:
                    final_result.confidence = min(final_result.confidence, depth_result.confidence)
                
                # Если ничего нового не выведено, прекращаем
                if not depth_result.derived_facts:
                    break
            
            execution_time = time.time() - start_time
            final_result.processing_time = execution_time
            self._stats["total_execution_time"] += execution_time
            self._stats["derivations_performed"] += 1
            
            return final_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = DerivationResult(
                success=False,
                processing_time=execution_time,
                error_message=str(e)
            )
            raise DerivationError(f"Ошибка логического вывода: {e}", context, error_result)
    
    def _perform_single_derivation_step(self, context: ProcessingContext, step_number: int) -> DerivationResult:
        """Выполняет один шаг логического вывода."""
        step_result = DerivationResult(success=True)
        
        # Поиск применимых правил
        relevant_rules = self.find_relevant_rules(context)
        
        # Сортировка правил по приоритету и уверенности
        sorted_rules = sorted(
            [(rule_id, self._rules[rule_id]) for rule_id in relevant_rules],
            key=lambda x: (x[1].priority, x[1].confidence),
            reverse=True
        )
        
        applied_any_rule = False
        
        for rule_id, rule in sorted_rules:
            try:
                rule_result = self.execute_rule(rule_id, context)
                
                if rule_result.success:
                    applied_any_rule = True
                    
                    # Объединение результатов
                    step_result.derived_facts.update(rule_result.derived_facts)
                    step_result.rules_used.extend(rule_result.rules_used)
                    
                    # Обновление объяснений с корректными номерами шагов
                    for explanation in rule_result.explanation:
                        explanation.step_number = step_number
                        step_result.add_explanation_step(explanation)
                    
                    # Обновление уверенности
                    if step_result.confidence == 0.0:
                        step_result.confidence = rule_result.confidence
                    else:
                        step_result.confidence = min(step_result.confidence, rule_result.confidence)
            
            except RuleExecutionError as e:
                self.logger.warning(f"Не удалось выполнить правило {rule_id}: {e}")
                continue
        
        if not applied_any_rule:
            step_result.success = False
        
        return step_result
    
    def find_relevant_rules(self, context: ProcessingContext) -> List[str]:
        """Находит правила, релевантные заданному контексту."""
        # Создаем ключ кеша на основе фактов контекста
        cache_key = hash(tuple(sorted(context.facts.keys())))
        
        relevant_rules = set()
        
        if self.rule_indexing and self._condition_index:
            # Быстрый поиск через индекс
            for fact_key in context.facts.keys():
                if fact_key in self._fact_index:
                    relevant_rules.update(self._fact_index[fact_key])
                    if self.cache_rules:
                        self._stats["cache_hits"] += 1
                else:
                    if self.cache_rules:
                        self._stats["cache_misses"] += 1
        else:
            # Полный перебор
            for rule_id, rule in self._rules.items():
                if self._is_rule_relevant(rule, context):
                    relevant_rules.add(rule_id)
            
            if self.cache_rules:
                self._stats["cache_misses"] += 1
        
        return list(relevant_rules)
    
    def _is_rule_relevant(self, rule: SymbolicRule, context: ProcessingContext) -> bool:
        """Проверяет релевантность правила контексту."""
        try:
            parsed_condition = PatternMatcher.parse_condition(rule.condition)
            condition_met, confidence, _ = PatternMatcher.match_condition(parsed_condition, context)
            return confidence >= self.confidence_threshold
        except:
            return False
    
    def update_rule(self, rule_id: str, **attributes) -> bool:
        """Обновляет атрибуты правила."""
        if rule_id not in self._rules:
            return False
        
        rule = self._rules[rule_id]
        
        # Удаление из индексов
        if self.rule_indexing:
            self._remove_from_indexes(rule)
        
        # Обновление атрибутов
        for attr, value in attributes.items():
            if hasattr(rule, attr):
                setattr(rule, attr, value)
        
        # Обновление индексов
        if self.rule_indexing:
            self._update_indexes(rule)
        
        return True
    
    def get_all_rules(self) -> List[SymbolicRule]:
        """Возвращает все правила в базе знаний."""
        return list(self._rules.values())
    
    def validate_rule(self, rule: SymbolicRule) -> Tuple[bool, Optional[str]]:
        """Проверяет валидность правила."""
        if not rule.condition or not rule.condition.strip():
            return False, "Условие правила не может быть пустым"
        
        if not rule.action or not rule.action.strip():
            return False, "Действие правила не может быть пустым"
        
        if not 0 <= rule.confidence <= 1:
            return False, "Уверенность должна быть между 0 и 1"
        
        if not 0 <= rule.weight <= 10:
            return False, "Вес должен быть между 0 и 10"
        
        try:
            # Проверка парсинга условия и действия
            PatternMatcher.parse_condition(rule.condition)
            PatternMatcher.parse_action(rule.action)
        except Exception as e:
            return False, f"Ошибка парсинга: {e}"
        
        return True, None
    
    def clear_rules(self) -> None:
        """Очищает все правила."""
        self._rules.clear()
        self._condition_index.clear()
        self._fact_index.clear()
        self.logger.info("Все правила очищены")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику процессора."""
        return {
            "rules_count": len(self._rules),
            "rules_added": self._stats["rules_added"],
            "rules_executed": self._stats["rules_executed"],
            "derivations_performed": self._stats["derivations_performed"],
            "total_execution_time": self._stats["total_execution_time"],
            "average_execution_time": self._stats["total_execution_time"] / max(1, self._stats["rules_executed"]),
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_rate": self._stats["cache_hits"] / max(1, self._stats["cache_hits"] + self._stats["cache_misses"]),
            "configuration": {
                "confidence_threshold": self.confidence_threshold,
                "max_depth": self.max_depth,
                "enable_explanations": self.enable_explanations,
                "cache_rules": self.cache_rules,
                "parallel_processing": self.parallel_processing,
                "rule_indexing": self.rule_indexing
            }
        }
    
    def _update_indexes(self, rule: SymbolicRule):
        """Обновляет индексы для правила."""
        try:
            parsed_condition = PatternMatcher.parse_condition(rule.condition)
            condition_type = parsed_condition["type"]
            
            # Индекс по типу условия
            if condition_type not in self._condition_index:
                self._condition_index[condition_type] = set()
            self._condition_index[condition_type].add(rule.id)
            
            # Индекс по фактам
            if condition_type == "is_a":
                fact_key = f"{parsed_condition['subject']}_is_a_{parsed_condition['object']}"
                if fact_key not in self._fact_index:
                    self._fact_index[fact_key] = set()
                self._fact_index[fact_key].add(rule.id)
            
            elif condition_type == "has_property":
                fact_key = f"{parsed_condition['subject']}_has_{parsed_condition['property']}"
                if fact_key not in self._fact_index:
                    self._fact_index[fact_key] = set()
                self._fact_index[fact_key].add(rule.id)
                
        except Exception as e:
            self.logger.warning(f"Не удалось обновить индексы для правила {rule.id}: {e}")
    
    def _remove_from_indexes(self, rule: SymbolicRule):
        """Удаляет правило из индексов."""
        # Удаление из всех индексов
        for condition_type, rule_set in self._condition_index.items():
            rule_set.discard(rule.id)
        
        for fact_key, rule_set in self._fact_index.items():
            rule_set.discard(rule.id)
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """Экспортирует все правила в формате словарей."""
        return [rule.to_dict() for rule in self._rules.values()]
    
    def import_rules(self, rules_data: List[Dict[str, Any]]) -> int:
        """Импортирует правила из списка словарей.
        
        Returns:
            Количество успешно импортированных правил.
        """
        imported_count = 0
        
        for rule_data in rules_data:
            try:
                rule = SymbolicRule.from_dict(rule_data)
                self.add_rule(rule)
                imported_count += 1
            except Exception as e:
                self.logger.warning(f"Не удалось импортировать правило: {e}")
        
        return imported_count