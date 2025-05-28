"""Процессор на основе графа знаний."""

import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from neurograph.core.logging import get_logger

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


class GraphBasedProcessor(INeuroSymbolicProcessor):
    """Процессор на основе графа знаний."""
    
    def __init__(self,
                 graph_provider=None,
                 use_graph_structure: bool = True,
                 confidence_threshold: float = 0.4,
                 max_depth: int = 7,
                 enable_explanations: bool = True,
                 path_search_limit: int = 100):
        """Инициализация процессора.
        
        Args:
            graph_provider: Провайдер графа знаний.
            use_graph_structure: Использовать структуру графа для вывода.
            confidence_threshold: Минимальная уверенность для применения правил.
            max_depth: Максимальная глубина вывода.
            enable_explanations: Включить генерацию объяснений.
            path_search_limit: Лимит поиска путей в графе.
        """
        self.graph_provider = graph_provider
        self.use_graph_structure = use_graph_structure
        self.confidence_threshold = confidence_threshold
        self.max_depth = max_depth
        self.enable_explanations = enable_explanations
        self.path_search_limit = path_search_limit
        
        # Хранилище правил
        self._rules: Dict[str, SymbolicRule] = {}
        
        # Кеш путей графа
        self._path_cache: Dict[Tuple[str, str], List[List[str]]] = {}
        
        # Статистика
        self._stats = {
            "rules_added": 0,
            "rules_executed": 0,
            "derivations_performed": 0,
            "graph_queries": 0,
            "path_searches": 0,
            "total_execution_time": 0.0
        }
        
        self.logger = get_logger("graph_based_processor")
        self.logger.info("Инициализирован GraphBasedProcessor")
    
    def add_rule(self, rule: SymbolicRule) -> str:
        """Добавляет правило в базу знаний."""
        is_valid, error_msg = self.validate_rule(rule)
        if not is_valid:
            raise RuleValidationError(f"Невалидное правило: {error_msg}", rule.id)
        
        self._rules[rule.id] = rule
        
        # Если есть граф, добавляем правило как узел
        if self.graph_provider and hasattr(self.graph_provider, 'add_node'):
            try:
                self.graph_provider.add_node(
                    f"rule_{rule.id}",
                    type="rule",
                    condition=rule.condition,
                    action=rule.action,
                    confidence=rule.confidence,
                    rule_type=rule.rule_type.value
                )
            except Exception as e:
                self.logger.warning(f"Не удалось добавить правило в граф: {e}")
        
        self._stats["rules_added"] += 1
        self.logger.debug(f"Добавлено правило {rule.id}")
        
        return rule.id
    
    def remove_rule(self, rule_id: str) -> bool:
        """Удаляет правило из базы знаний."""
        if rule_id in self._rules:
            # Удаление из графа
            if self.graph_provider and hasattr(self.graph_provider, 'has_node'):
                try:
                    if self.graph_provider.has_node(f"rule_{rule_id}"):
                        # Здесь должен быть метод удаления узла
                        pass
                except Exception as e:
                    self.logger.warning(f"Не удалось удалить правило из графа: {e}")
            
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
            
            # Проверка применимости правила с использованием графа
            is_applicable, confidence, derived_facts = self._check_rule_applicability(rule, context)
            
            result = DerivationResult(success=False)
            
            if is_applicable and confidence >= self.confidence_threshold:
                result.success = True
                result.confidence = confidence
                result.rules_used = [rule_id]
                
                # Добавление выведенных фактов
                for fact_key, fact_value in derived_facts.items():
                    result.add_derived_fact(fact_key, fact_value, confidence)
                    # Обновление контекста
                    context.add_fact(fact_key, fact_value, confidence)
                
                # Добавление объяснения
                if self.enable_explanations:
                    explanation_step = ExplanationStep(
                        step_number=1,
                        rule_id=rule_id,
                        rule_description=f"{rule.condition} -> {rule.action}",
                        input_facts=self._extract_input_facts(rule, context),
                        output_facts=derived_facts,
                        confidence=confidence,
                        reasoning=self._generate_reasoning(rule, context, confidence)
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
        """Производит логический вывод на основе правил и графа."""
        start_time = time.time()
        
        try:
            actual_depth = min(depth, self.max_depth)
            final_result = DerivationResult(success=True)
            working_context = context.copy()
            
            for current_depth in range(actual_depth):
                # Вывод на основе правил
                rule_result = self._perform_rule_based_derivation(working_context, current_depth + 1)
                
                # Вывод на основе структуры графа
                graph_result = self._perform_graph_based_derivation(working_context, current_depth + 1)
                
                # Объединение результатов
                combined_result = self._combine_derivation_results(rule_result, graph_result)
                
                if not combined_result.success:
                    break
                
                # Обновление финального результата
                final_result.derived_facts.update(combined_result.derived_facts)
                final_result.explanation.extend(combined_result.explanation)
                final_result.rules_used.extend(combined_result.rules_used)
                
                if final_result.confidence == 0.0:
                    final_result.confidence = combined_result.confidence
                else:
                    final_result.confidence = min(final_result.confidence, combined_result.confidence)
                
                # Если ничего нового не выведено, прекращаем
                if not combined_result.derived_facts:
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
    
    def _check_rule_applicability(self, rule: SymbolicRule, context: ProcessingContext) -> Tuple[bool, float, Dict[str, Any]]:
        """Проверяет применимость правила с использованием графа."""
        # Парсинг условия и действия
        condition_entities = self._extract_entities_from_text(rule.condition)
        action_entities = self._extract_entities_from_text(rule.action)
        
        # Проверка наличия сущностей в контексте или графе
        entities_found = {}
        total_confidence = 0.0
        found_count = 0
        
        for entity in condition_entities:
            confidence = self._find_entity_confidence(entity, context)
            if confidence > 0:
                entities_found[entity] = confidence
                total_confidence += confidence
                found_count += 1
        
        if found_count == 0:
            return False, 0.0, {}
        
        # Вычисление средней уверенности
        avg_confidence = total_confidence / found_count
        
        # Если граф доступен, проверяем связи между сущностями
        if self.graph_provider and self.use_graph_structure:
            graph_confidence = self._check_graph_relationships(condition_entities, context)
            avg_confidence = (avg_confidence + graph_confidence) / 2
        
        # Применение правила с учетом его собственной уверенности
        final_confidence = avg_confidence * rule.confidence
        
        # Генерация выводимых фактов
        derived_facts = {}
        if final_confidence >= self.confidence_threshold:
            derived_facts = self._generate_derived_facts(rule, context, final_confidence)
        
        return final_confidence >= self.confidence_threshold, final_confidence, derived_facts
    
    def _perform_rule_based_derivation(self, context: ProcessingContext, step_number: int) -> DerivationResult:
        """Выполняет вывод на основе правил."""
        step_result = DerivationResult(success=True)
        
        relevant_rules = self.find_relevant_rules(context)
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
                    step_result.derived_facts.update(rule_result.derived_facts)
                    step_result.rules_used.extend(rule_result.rules_used)
                    
                    for explanation in rule_result.explanation:
                        explanation.step_number = step_number
                        step_result.add_explanation_step(explanation)
                    
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
    
    def _perform_graph_based_derivation(self, context: ProcessingContext, step_number: int) -> DerivationResult:
        """Выполняет вывод на основе структуры графа."""
        if not self.graph_provider or not self.use_graph_structure:
            return DerivationResult(success=False)
        
        step_result = DerivationResult(success=True)
        self._stats["graph_queries"] += 1
        
        try:
            # Извлечение сущностей из контекста
            context_entities = self._extract_context_entities(context)
            
            # Поиск новых связей в графе
            new_facts = {}
            
            for entity in context_entities:
                if hasattr(self.graph_provider, 'get_neighbors'):
                    try:
                        neighbors = self.graph_provider.get_neighbors(entity)
                        
                        for neighbor in neighbors:
                            # Получение информации о связи
                            if hasattr(self.graph_provider, 'get_edge'):
                                try:
                                    edge_data = self.graph_provider.get_edge(entity, neighbor)
                                    if edge_data:
                                        relation_type = edge_data.get('type', 'related_to')
                                        confidence = edge_data.get('weight', 0.5)
                                        
                                        # Создание нового факта
                                        fact_key = f"{entity}_{relation_type}_{neighbor}"
                                        if not context.has_fact(fact_key):
                                            new_facts[fact_key] = {
                                                'value': True,
                                                'confidence': confidence,
                                                'source': 'graph_structure'
                                            }
                                except:
                                    continue
                    except:
                        continue
            
            # Транзитивные отношения
            transitive_facts = self._find_transitive_relationships(context_entities, context)
            new_facts.update(transitive_facts)
            
            # Добавление найденных фактов в результат
            for fact_key, fact_data in new_facts.items():
                step_result.add_derived_fact(fact_key, fact_data['value'], fact_data['confidence'])
                context.add_fact(fact_key, fact_data['value'], fact_data['confidence'])
            
            if new_facts and self.enable_explanations:
                explanation_step = ExplanationStep(
                    step_number=step_number,
                    rule_id="graph_structure",
                    rule_description="Вывод на основе структуры графа знаний",
                    input_facts={entity: True for entity in context_entities},
                    output_facts=new_facts,
                    confidence=sum(f['confidence'] for f in new_facts.values()) / len(new_facts) if new_facts else 0.0,
                    reasoning="Найдены новые связи в графе знаний"
                )
                step_result.add_explanation_step(explanation_step)
                step_result.confidence = explanation_step.confidence
            
            step_result.success = len(new_facts) > 0
            
        except Exception as e:
            self.logger.warning(f"Ошибка вывода на основе графа: {e}")
            step_result.success = False
        
        return step_result
    
    def _find_transitive_relationships(self, entities: List[str], context: ProcessingContext) -> Dict[str, Dict[str, Any]]:
        """Поиск транзитивных отношений в графе."""
        if not self.graph_provider:
            return {}
        
        transitive_facts = {}
        self._stats["path_searches"] += 1
        
        # Поиск путей между сущностями
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                cache_key = (entity1, entity2)
                
                # Проверка кеша
                if cache_key in self._path_cache:
                    paths = self._path_cache[cache_key]
                else:
                    paths = self._find_paths_between_entities(entity1, entity2)
                    self._path_cache[cache_key] = paths
                
                # Анализ найденных путей
                for path in paths[:5]:  # Ограничиваем количество путей
                    if len(path) <= 4:  # Короткие пути более надежны
                        confidence = 1.0 / len(path)  # Чем короче путь, тем выше уверенность
                        
                        # Создание транзитивного факта
                        fact_key = f"{entity1}_transitively_related_to_{entity2}"
                        if not context.has_fact(fact_key):
                            transitive_facts[fact_key] = {
                                'value': True,
                                'confidence': confidence * 0.7,  # Снижаем уверенность для транзитивных связей
                                'source': 'transitive_inference',
                                'path': path
                            }
        
        return transitive_facts
    
    def _find_paths_between_entities(self, start: str, end: str) -> List[List[str]]:
        """Поиск путей между двумя сущностями в графе."""
        if not self.graph_provider:
            return []
        
        try:
            # Простой BFS для поиска путей
            from collections import deque
            
            queue = deque([(start, [start])])
            paths = []
            visited = set()
            
            while queue and len(paths) < self.path_search_limit:
                current, path = queue.popleft()
                
                if len(path) > 4:  # Ограничение глубины
                    continue
                
                if current == end and len(path) > 1:
                    paths.append(path)
                    continue
                
                if current in visited:
                    continue
                visited.add(current)
                
                if hasattr(self.graph_provider, 'get_neighbors'):
                    try:
                        neighbors = self.graph_provider.get_neighbors(current)
                        for neighbor in neighbors:
                            if neighbor not in path:  # Избегаем циклов
                                queue.append((neighbor, path + [neighbor]))
                    except:
                        continue
            
            return paths
            
        except Exception as e:
            self.logger.warning(f"Ошибка поиска путей: {e}")
            return []
    
    def _combine_derivation_results(self, rule_result: DerivationResult, graph_result: DerivationResult) -> DerivationResult:
        """Объединяет результаты вывода на основе правил и графа."""
        combined = DerivationResult(success=rule_result.success or graph_result.success)
        
        # Объединение фактов
        combined.derived_facts.update(rule_result.derived_facts)
        combined.derived_facts.update(graph_result.derived_facts)
        
        # Объединение объяснений
        combined.explanation.extend(rule_result.explanation)
        combined.explanation.extend(graph_result.explanation)
        
        # Объединение использованных правил
        combined.rules_used.extend(rule_result.rules_used)
        combined.rules_used.extend(graph_result.rules_used)
        
        # Вычисление общей уверенности
        confidences = [r.confidence for r in [rule_result, graph_result] if r.confidence > 0]
        if confidences:
            combined.confidence = sum(confidences) / len(confidences)
        
        return combined
    
    def find_relevant_rules(self, context: ProcessingContext) -> List[str]:
        """Находит правила, релевантные заданному контексту."""
        relevant_rules = []
        
        for rule_id, rule in self._rules.items():
            try:
                is_applicable, confidence, _ = self._check_rule_applicability(rule, context)
                if is_applicable:
                    relevant_rules.append(rule_id)
            except:
                continue
        
        return relevant_rules
    
    def update_rule(self, rule_id: str, **attributes) -> bool:
        """Обновляет атрибуты правила."""
        if rule_id not in self._rules:
            return False
        
        rule = self._rules[rule_id]
        
        for attr, value in attributes.items():
            if hasattr(rule, attr):
                setattr(rule, attr, value)
        
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
        
        return True, None
    
    def clear_rules(self) -> None:
        """Очищает все правила."""
        self._rules.clear()
        self._path_cache.clear()
        self.logger.info("Все правила очищены")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику процессора."""
        return {
            "rules_count": len(self._rules),
            "rules_added": self._stats["rules_added"],
            "rules_executed": self._stats["rules_executed"],
            "derivations_performed": self._stats["derivations_performed"],
            "graph_queries": self._stats["graph_queries"],
            "path_searches": self._stats["path_searches"],
            "total_execution_time": self._stats["total_execution_time"],
            "average_execution_time": self._stats["total_execution_time"] / max(1, self._stats["rules_executed"]),
            "path_cache_size": len(self._path_cache),
            "configuration": {
                "confidence_threshold": self.confidence_threshold,
                "max_depth": self.max_depth,
                "enable_explanations": self.enable_explanations,
                "use_graph_structure": self.use_graph_structure,
                "path_search_limit": self.path_search_limit
            }
        }
    
    # Вспомогательные методы
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Извлекает сущности из текста."""
        import re
        
        # Простое извлечение слов, которые могут быть сущностями
        words = re.findall(r'\b[А-Яа-я][А-Яа-я]*\b|\b[A-Za-z][A-Za-z]*\b', text)
        
        # Фильтрация служебных слов
        stop_words = {'и', 'или', 'не', 'является', 'имеет', 'свойство', 'связан', 'через',
                      'and', 'or', 'not', 'is', 'has', 'property', 'related', 'via'}
        
        entities = [word for word in words if word.lower() not in stop_words]
        return entities
    
    def _find_entity_confidence(self, entity: str, context: ProcessingContext) -> float:
        """Находит уверенность для сущности в контексте."""
        # Поиск в фактах контекста
        max_confidence = 0.0
        
        for fact_key, fact_data in context.facts.items():
            if entity.lower() in fact_key.lower():
                confidence = fact_data.get('confidence', 1.0)
                max_confidence = max(max_confidence, confidence)
        
        # Поиск в графе
        if self.graph_provider and hasattr(self.graph_provider, 'has_node'):
            try:
                if self.graph_provider.has_node(entity):
                    max_confidence = max(max_confidence, 0.8)  # Базовая уверенность для узлов графа
            except:
                pass
        
        return max_confidence
    
    def _check_graph_relationships(self, entities: List[str], context: ProcessingContext) -> float:
        """Проверяет связи между сущностями в графе."""
        if not self.graph_provider or len(entities) < 2:
            return 0.5  # Нейтральная уверенность
        
        total_confidence = 0.0
        relationship_count = 0
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                try:
                    if hasattr(self.graph_provider, 'has_edge'):
                        if self.graph_provider.has_edge(entity1, entity2):
                            total_confidence += 0.9
                            relationship_count += 1
                        elif hasattr(self.graph_provider, 'get_neighbors'):
                            # Проверка косвенных связей
                            neighbors1 = set(self.graph_provider.get_neighbors(entity1))
                            neighbors2 = set(self.graph_provider.get_neighbors(entity2))
                            
                            if neighbors1.intersection(neighbors2):
                                total_confidence += 0.6  # Косвенная связь
                                relationship_count += 1
                except:
                    continue
        
        if relationship_count > 0:
            return total_confidence / relationship_count
        
        return 0.3  # Низкая уверенность при отсутствии связей
    
    def _generate_derived_facts(self, rule: SymbolicRule, context: ProcessingContext, confidence: float) -> Dict[str, Any]:
        """Генерирует выводимые факты на основе правила."""
        derived_facts = {}
        
        # Простая интерпретация действия
        action_entities = self._extract_entities_from_text(rule.action)
        
        if 'является' in rule.action or 'is' in rule.action:
            if len(action_entities) >= 2:
                subject, obj = action_entities[0], action_entities[1]
                fact_key = f"{subject}_is_a_{obj}"
                derived_facts[fact_key] = True
        
        elif 'имеет' in rule.action or 'has' in rule.action:
            if len(action_entities) >= 2:
                subject, prop = action_entities[0], action_entities[1]
                fact_key = f"{subject}_has_{prop}"
                derived_facts[fact_key] = True
        
        else:
            # Общий случай - создаем факт из текста действия
            fact_key = f"derived_{hash(rule.action) % 10000}"
            derived_facts[fact_key] = rule.action
        
        return derived_facts
    
    def _extract_input_facts(self, rule: SymbolicRule, context: ProcessingContext) -> Dict[str, Any]:
        """Извлекает входные факты для правила."""
        condition_entities = self._extract_entities_from_text(rule.condition)
        input_facts = {}
        
        for entity in condition_entities:
            for fact_key, fact_data in context.facts.items():
                if entity.lower() in fact_key.lower():
                    input_facts[fact_key] = fact_data.get('value')
        
        return input_facts
    
    def _generate_reasoning(self, rule: SymbolicRule, context: ProcessingContext, confidence: float) -> str:
        """Генерирует объяснение рассуждения."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Правило '{rule.condition} -> {rule.action}' применено")
        reasoning_parts.append(f"с уверенностью {confidence:.2f}")
        
        if self.use_graph_structure:
            reasoning_parts.append("с учетом структуры графа знаний")
        
        return ". ".join(reasoning_parts)
    
    def _extract_context_entities(self, context: ProcessingContext) -> List[str]:
        """Извлекает сущности из контекста."""
        entities = set()
        
        for fact_key in context.facts.keys():
            # Простое извлечение сущностей из ключей фактов
            parts = fact_key.split('_')
            for part in parts:
                if len(part) > 2 and part.isalpha():
                    entities.add(part)
        
        return list(entities)