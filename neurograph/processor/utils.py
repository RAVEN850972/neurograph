"""Утилиты для модуля Processor."""

import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path

from .base import SymbolicRule, ProcessingContext, DerivationResult, RuleType, ActionType


class RuleManager:
    """Менеджер для управления коллекциями правил."""
    
    def __init__(self):
        self.rule_collections: Dict[str, List[SymbolicRule]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def create_collection(self, name: str, description: str = "", domain: str = "general"):
        """Создает новую коллекцию правил."""
        if name not in self.rule_collections:
            self.rule_collections[name] = []
            self.metadata[name] = {
                "description": description,
                "domain": domain,
                "created_at": datetime.now().isoformat(),
                "rule_count": 0
            }
    
    def add_rule_to_collection(self, collection_name: str, rule: SymbolicRule):
        """Добавляет правило в коллекцию."""
        if collection_name not in self.rule_collections:
            self.create_collection(collection_name)
        
        self.rule_collections[collection_name].append(rule)
        self.metadata[collection_name]["rule_count"] += 1
    
    def get_collection(self, name: str) -> List[SymbolicRule]:
        """Возвращает коллекцию правил."""
        return self.rule_collections.get(name, [])
    
    def merge_collections(self, target: str, *sources: str) -> int:
        """Объединяет несколько коллекций в одну."""
        if target not in self.rule_collections:
            self.create_collection(target, "Merged collection")
        
        merged_count = 0
        for source in sources:
            if source in self.rule_collections:
                self.rule_collections[target].extend(self.rule_collections[source])
                merged_count += len(self.rule_collections[source])
        
        self.metadata[target]["rule_count"] = len(self.rule_collections[target])
        return merged_count
    
    def filter_rules(self, collection_name: str, 
                    min_confidence: float = 0.0,
                    rule_types: List[RuleType] = None,
                    keywords: List[str] = None) -> List[SymbolicRule]:
        """Фильтрует правила по критериям."""
        if collection_name not in self.rule_collections:
            return []
        
        filtered_rules = []
        for rule in self.rule_collections[collection_name]:
            # Фильтр по уверенности
            if rule.confidence < min_confidence:
                continue
            
            # Фильтр по типу правила
            if rule_types and rule.rule_type not in rule_types:
                continue
            
            # Фильтр по ключевым словам
            if keywords:
                rule_text = f"{rule.condition} {rule.action}".lower()
                if not any(keyword.lower() in rule_text for keyword in keywords):
                    continue
            
            filtered_rules.append(rule)
        
        return filtered_rules
    
    def export_collection(self, collection_name: str, file_path: str):
        """Экспортирует коллекцию в файл."""
        if collection_name not in self.rule_collections:
            raise ValueError(f"Коллекция '{collection_name}' не найдена")
        
        data = {
            "collection_name": collection_name,
            "metadata": self.metadata[collection_name],
            "rules": [rule.to_dict() for rule in self.rule_collections[collection_name]]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def import_collection(self, file_path: str) -> str:
        """Импортирует коллекцию из файла."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        collection_name = data["collection_name"]
        
        # Создание коллекции
        self.rule_collections[collection_name] = []
        self.metadata[collection_name] = data["metadata"]
        
        # Импорт правил
        for rule_data in data["rules"]:
            rule = SymbolicRule.from_dict(rule_data)
            self.rule_collections[collection_name].append(rule)
        
        return collection_name
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по коллекциям."""
        total_rules = sum(len(rules) for rules in self.rule_collections.values())
        
        stats = {
            "total_collections": len(self.rule_collections),
            "total_rules": total_rules,
            "collections": {}
        }
        
        for name, rules in self.rule_collections.items():
            collection_stats = {
                "rule_count": len(rules),
                "avg_confidence": sum(r.confidence for r in rules) / len(rules) if rules else 0,
                "rule_types": list(set(r.rule_type.value for r in rules)),
                "metadata": self.metadata.get(name, {})
            }
            stats["collections"][name] = collection_stats
        
        return stats


class RuleTemplateEngine:
    """Движок для создания правил по шаблонам."""
    
    def __init__(self):
        self.templates: Dict[str, str] = {}
        self.variables: Dict[str, List[str]] = {}
    
    def add_template(self, name: str, condition_template: str, action_template: str):
        """Добавляет шаблон правила."""
        self.templates[name] = {
            "condition": condition_template,
            "action": action_template
        }
    
    def add_variable_values(self, variable_name: str, values: List[str]):
        """Добавляет возможные значения для переменной."""
        self.variables[variable_name] = values
    
    def generate_rules_from_template(self, template_name: str, 
                                   variables: Dict[str, str] = None,
                                   confidence: float = 0.8) -> SymbolicRule:
        """Генерирует правило из шаблона."""
        if template_name not in self.templates:
            raise ValueError(f"Шаблон '{template_name}' не найден")
        
        template = self.templates[template_name]
        variables = variables or {}
        
        # Замена переменных в шаблоне
        condition = template["condition"]
        action = template["action"]
        
        for var_name, var_value in variables.items():
            condition = condition.replace(f"{{{var_name}}}", var_value)
            action = action.replace(f"{{{var_name}}}", var_value)
        
        return SymbolicRule(
            condition=condition,
            action=action,
            confidence=confidence,
            metadata={"template": template_name, "variables": variables}
        )
    
    def generate_all_combinations(self, template_name: str, 
                                confidence: float = 0.8) -> List[SymbolicRule]:
        """Генерирует все возможные комбинации правил из шаблона."""
        if template_name not in self.templates:
            raise ValueError(f"Шаблон '{template_name}' не найден")
        
        template = self.templates[template_name]
        
        # Найти все переменные в шаблоне
        variable_pattern = r'\{(\w+)\}'
        condition_vars = set(re.findall(variable_pattern, template["condition"]))
        action_vars = set(re.findall(variable_pattern, template["action"]))
        all_vars = condition_vars.union(action_vars)
        
        # Проверить наличие значений для всех переменных
        missing_vars = [var for var in all_vars if var not in self.variables]
        if missing_vars:
            raise ValueError(f"Отсутствуют значения для переменных: {missing_vars}")
        
        # Генерация всех комбинаций
        from itertools import product
        
        var_names = list(all_vars)
        var_value_lists = [self.variables[var] for var in var_names]
        
        rules = []
        for combination in product(*var_value_lists):
            variables = dict(zip(var_names, combination))
            rule = self.generate_rules_from_template(template_name, variables, confidence)
            rules.append(rule)
        
        return rules


class PerformanceProfiler:
    """Профилировщик производительности для процессора."""
    
    def __init__(self):
        self.execution_times: List[float] = []
        self.rule_usage: Dict[str, int] = {}
        self.fact_access: Dict[str, int] = {}
        self.start_time: Optional[float] = None
        
    def start_profiling(self):
        """Начинает профилирование."""
        self.start_time = time.time()
    
    def stop_profiling(self) -> float:
        """Останавливает профилирование и возвращает время выполнения."""
        if self.start_time is None:
            return 0.0
        
        execution_time = time.time() - self.start_time
        self.execution_times.append(execution_time)
        self.start_time = None
        return execution_time
    
    def record_rule_usage(self, rule_id: str):
        """Записывает использование правила."""
        self.rule_usage[rule_id] = self.rule_usage.get(rule_id, 0) + 1
    
    def record_fact_access(self, fact_key: str):
        """Записывает обращение к факту."""
        self.fact_access[fact_key] = self.fact_access.get(fact_key, 0) + 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Возвращает отчет о производительности."""
        if not self.execution_times:
            return {"error": "Нет данных для анализа"}
        
        sorted_rules = sorted(self.rule_usage.items(), key=lambda x: x[1], reverse=True)
        sorted_facts = sorted(self.fact_access.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "execution_statistics": {
                "total_executions": len(self.execution_times),
                "total_time": sum(self.execution_times),
                "avg_time": sum(self.execution_times) / len(self.execution_times),
                "min_time": min(self.execution_times),
                "max_time": max(self.execution_times)
            },
            "rule_usage": {
                "total_unique_rules": len(self.rule_usage),
                "most_used_rules": sorted_rules[:10],
                "total_rule_executions": sum(self.rule_usage.values())
            },
            "fact_access": {
                "total_unique_facts": len(self.fact_access),
                "most_accessed_facts": sorted_facts[:10],
                "total_fact_accesses": sum(self.fact_access.values())
            }
        }
    
    def reset(self):
        """Сбрасывает статистику профилирования."""
        self.execution_times.clear()
        self.rule_usage.clear()
        self.fact_access.clear()
        self.start_time = None


class RuleValidator:
    """Валидатор правил с расширенными проверками."""
    
    def __init__(self):
        self.validation_rules = []
        self._setup_default_validations()
    
    def _setup_default_validations(self):
        """Настройка валидаций по умолчанию."""
        self.validation_rules = [
            self._validate_not_empty,
            self._validate_confidence_range,
            self._validate_weight_range,
            self._validate_condition_syntax,
            self._validate_action_syntax,
            self._validate_circular_dependencies
        ]
    
    def validate_rule(self, rule: SymbolicRule, 
                     existing_rules: List[SymbolicRule] = None) -> Tuple[bool, List[str]]:
        """Выполняет полную валидацию правила."""
        errors = []
        existing_rules = existing_rules or []
        
        for validation_func in self.validation_rules:
            try:
                is_valid, error_message = validation_func(rule, existing_rules)
                if not is_valid:
                    errors.append(error_message)
            except Exception as e:
                errors.append(f"Ошибка валидации: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _validate_not_empty(self, rule: SymbolicRule, existing_rules: List[SymbolicRule]) -> Tuple[bool, str]:
        """Проверяет, что условие и действие не пустые."""
        if not rule.condition or not rule.condition.strip():
            return False, "Условие правила не может быть пустым"
        
        if not rule.action or not rule.action.strip():
            return False, "Действие правила не может быть пустым"
        
        return True, ""
    
    def _validate_confidence_range(self, rule: SymbolicRule, existing_rules: List[SymbolicRule]) -> Tuple[bool, str]:
        """Проверяет диапазон уверенности."""
        if not 0 <= rule.confidence <= 1:
            return False, f"Уверенность должна быть между 0 и 1, получено: {rule.confidence}"
        
        return True, ""
    
    def _validate_weight_range(self, rule: SymbolicRule, existing_rules: List[SymbolicRule]) -> Tuple[bool, str]:
        """Проверяет диапазон веса."""
        if not 0 <= rule.weight <= 10:
            return False, f"Вес должен быть между 0 и 10, получено: {rule.weight}"
        
        return True, ""
    
    def _validate_condition_syntax(self, rule: SymbolicRule, existing_rules: List[SymbolicRule]) -> Tuple[bool, str]:
        """Проверяет синтаксис условия."""
        condition = rule.condition.lower()
        
        # Проверка базовых шаблонов
        valid_patterns = [
            r'\w+\s+является\s+\w+',
            r'\w+\s+имеет\s+свойство\s+\w+',
            r'\w+\s+связан\s+с\s+\w+',
            r'\w+\s+И\s+\w+',
            r'\w+\s+ИЛИ\s+\w+',
            r'НЕ\s+\w+'
        ]
        
        for pattern in valid_patterns:
            if re.search(pattern, condition):
                return True, ""
        
        # Если ни один шаблон не подошел, это может быть свободный текст
        if len(condition.split()) >= 2:
            return True, ""
        
        return False, f"Нераспознанный синтаксис условия: {rule.condition}"
    
    def _validate_action_syntax(self, rule: SymbolicRule, existing_rules: List[SymbolicRule]) -> Tuple[bool, str]:
        """Проверяет синтаксис действия."""
        action = rule.action.lower()
        
        # Проверка базовых действий
        valid_action_patterns = [
            r'assert\s+\w+',
            r'derive\s+\w+',
            r'query\s+\w+',
            r'execute\s+\w+'
        ]
        
        for pattern in valid_action_patterns:
            if re.search(pattern, action):
                return True, ""
        
        # Свободный текст также допустим
        if len(action.split()) >= 1:
            return True, ""
        
        return False, f"Нераспознанный синтаксис действия: {rule.action}"
    
    def _validate_circular_dependencies(self, rule: SymbolicRule, existing_rules: List[SymbolicRule]) -> Tuple[bool, str]:
        """Проверяет циклические зависимости."""
        # Упрощенная проверка циклов
        condition_entities = self._extract_entities(rule.condition)
        action_entities = self._extract_entities(rule.action)
        
        for existing_rule in existing_rules:
            existing_condition_entities = self._extract_entities(existing_rule.condition)
            existing_action_entities = self._extract_entities(existing_rule.action)
            
            # Проверка прямого цикла: A -> B и B -> A
            if (set(condition_entities).intersection(set(existing_action_entities)) and
                set(action_entities).intersection(set(existing_condition_entities))):
                return False, f"Обнаружена циклическая зависимость с правилом {existing_rule.id}"
        
        return True, ""
    
    def _extract_entities(self, text: str) -> List[str]:
        """Извлекает сущности из текста."""
        # Простое извлечение слов
        words = re.findall(r'\b[А-Яа-я][А-Яа-я]*\b|\b[A-Za-z][A-Za-z]*\b', text)
        stop_words = {'и', 'или', 'не', 'является', 'имеет', 'свойство', 'связан', 'через',
                      'assert', 'derive', 'query', 'execute',
                      'and', 'or', 'not', 'is', 'has', 'property', 'related', 'via'}
        
        return [word for word in words if word.lower() not in stop_words]


class RuleOptimizer:
    """Оптимизатор правил для улучшения производительности."""
    
    def __init__(self):
        self.optimization_strategies = [
            self._remove_duplicate_rules,
            self._merge_similar_rules,
            self._reorder_by_frequency,
            self._optimize_condition_complexity
        ]
    
    def optimize_rules(self, rules: List[SymbolicRule], 
                      usage_stats: Dict[str, int] = None) -> List[SymbolicRule]:
        """Оптимизирует список правил."""
        optimized_rules = rules.copy()
        usage_stats = usage_stats or {}
        
        for strategy in self.optimization_strategies:
            optimized_rules = strategy(optimized_rules, usage_stats)
        
        return optimized_rules
    
    def _remove_duplicate_rules(self, rules: List[SymbolicRule], 
                               usage_stats: Dict[str, int]) -> List[SymbolicRule]:
        """Удаляет дублирующиеся правила."""
        seen_rules = {}
        unique_rules = []
        
        for rule in rules:
            rule_signature = (rule.condition, rule.action)
            
            if rule_signature in seen_rules:
                # Оставляем правило с большей уверенностью или чаще используемое
                existing_rule = seen_rules[rule_signature]
                existing_usage = usage_stats.get(existing_rule.id, 0)
                current_usage = usage_stats.get(rule.id, 0)
                
                if (rule.confidence > existing_rule.confidence or 
                    current_usage > existing_usage):
                    # Заменяем существующее правило
                    unique_rules = [r for r in unique_rules if r.id != existing_rule.id]
                    unique_rules.append(rule)
                    seen_rules[rule_signature] = rule
            else:
                seen_rules[rule_signature] = rule
                unique_rules.append(rule)
        
        return unique_rules
    
    def _merge_similar_rules(self, rules: List[SymbolicRule], 
                            usage_stats: Dict[str, int]) -> List[SymbolicRule]:
        """Объединяет похожие правила."""
        # Пока простая реализация - в будущем можно добавить машинное обучение
        return rules
    
    def _reorder_by_frequency(self, rules: List[SymbolicRule], 
                             usage_stats: Dict[str, int]) -> List[SymbolicRule]:
        """Переупорядочивает правила по частоте использования."""
        def get_usage(rule):
            return usage_stats.get(rule.id, 0)
        
        return sorted(rules, key=get_usage, reverse=True)
    
    def _optimize_condition_complexity(self, rules: List[SymbolicRule], 
                                      usage_stats: Dict[str, int]) -> List[SymbolicRule]:
        """Оптимизирует сложность условий."""
        # Простые условия должны проверяться первыми
        def condition_complexity(rule):
            condition = rule.condition.lower()
            complexity = 0
            
            if ' и ' in condition or ' or ' in condition:
                complexity += 2
            if ' или ' in condition or ' or ' in condition:
                complexity += 1
            if 'не ' in condition or ' not ' in condition:
                complexity += 1
            
            return complexity
        
        return sorted(rules, key=condition_complexity)


def create_standard_rule_collections() -> RuleManager:
    """Создает стандартные коллекции правил."""
    manager = RuleManager()
    
    # Коллекция животных
    manager.create_collection("animals", "Правила для классификации животных", "biology")
    
    animal_rules = [
        SymbolicRule(
            condition="собака является млекопитающим",
            action="derive собака является животным",
            confidence=1.0
        ),
        SymbolicRule(
            condition="кот является млекопитающим",
            action="derive кот является животным", 
            confidence=1.0
        ),
        SymbolicRule(
            condition="животное является живым",
            action="derive животное дышит",
            confidence=0.95
        )
    ]
    
    for rule in animal_rules:
        manager.add_rule_to_collection("animals", rule)
    
    # Коллекция медицинских правил
    manager.create_collection("medical", "Медицинские диагностические правила", "healthcare")
    
    medical_rules = [
        SymbolicRule(
            condition="пациент имеет свойство температура",
            action="derive пациент имеет свойство лихорадка",
            confidence=0.8
        ),
        SymbolicRule(
            condition="пациент имеет свойство кашель",
            action="derive пациент имеет свойство респираторные_симптомы",
            confidence=0.7
        )
    ]
    
    for rule in medical_rules:
        manager.add_rule_to_collection("medical", rule)
    
    return manager


def benchmark_processor_performance(processor, test_cases: List[Tuple[ProcessingContext, int]]) -> Dict[str, Any]:
    """Бенчмарк производительности процессора."""
    profiler = PerformanceProfiler()
    results = []
    
    for context, depth in test_cases:
        profiler.start_profiling()
        
        try:
            result = processor.derive(context, depth=depth)
            execution_time = profiler.stop_profiling()
            
            results.append({
                "success": result.success,
                "execution_time": execution_time,
                "derived_facts_count": len(result.derived_facts),
                "rules_used_count": len(result.rules_used),
                "confidence": result.confidence
            })
            
        except Exception as e:
            profiler.stop_profiling()
            results.append({
                "success": False,
                "execution_time": 0.0,
                "error": str(e)
            })
    
    # Анализ результатов
    successful_results = [r for r in results if r["success"]]
    
    if not successful_results:
        return {"error": "Все тесты завершились неудачей"}
    
    total_time = sum(r["execution_time"] for r in successful_results)
    avg_time = total_time / len(successful_results)
    
    return {
        "test_count": len(test_cases),
        "successful_tests": len(successful_results),
        "success_rate": len(successful_results) / len(test_cases),
        "total_execution_time": total_time,
        "average_execution_time": avg_time,
        "min_execution_time": min(r["execution_time"] for r in successful_results),
        "max_execution_time": max(r["execution_time"] for r in successful_results),
        "average_derived_facts": sum(r["derived_facts_count"] for r in successful_results) / len(successful_results),
        "average_confidence": sum(r["confidence"] for r in successful_results) / len(successful_results),
        "detailed_results": results
    }