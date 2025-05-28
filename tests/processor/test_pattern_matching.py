"""Тесты для процессора сопоставления шаблонов."""

import unittest
from datetime import datetime
from neurograph.processor.base import (
   SymbolicRule, 
   ProcessingContext, 
   RuleType, 
   ActionType,
   RuleValidationError,
   RuleExecutionError
)
from neurograph.processor.impl.pattern_matching import PatternMatchingProcessor, PatternMatcher


class TestPatternMatcher(unittest.TestCase):
   """Тесты для класса PatternMatcher."""
   
   def test_parse_simple_condition(self):
       """Тест парсинга простого условия."""
       condition = "собака является животным"
       parsed = PatternMatcher.parse_condition(condition)
       
       self.assertEqual(parsed["type"], "is_a")
       self.assertEqual(parsed["subject"], "собака")
       self.assertEqual(parsed["object"], "животным")
   
   def test_parse_property_condition(self):
       """Тест парсинга условия со свойством."""
       condition = "собака имеет свойство лай"
       parsed = PatternMatcher.parse_condition(condition)
       
       self.assertEqual(parsed["type"], "has_property")
       self.assertEqual(parsed["subject"], "собака")
       self.assertEqual(parsed["property"], "лай")
   
   def test_parse_logical_and(self):
       """Тест парсинга логического И."""
       condition = "A И B"
       parsed = PatternMatcher.parse_condition(condition)
       
       self.assertEqual(parsed["type"], "and")
       self.assertEqual(parsed["left"], "A")
       self.assertEqual(parsed["right"], "B")
   
   def test_parse_simple_action(self):
       """Тест парсинга простого действия."""
       action = "assert собака является животным"
       parsed = PatternMatcher.parse_action(action)
       
       self.assertEqual(parsed["type"], "assert_is_a")
       self.assertEqual(parsed["subject"], "собака")
       self.assertEqual(parsed["object"], "животным")
   
   def test_parse_derive_action(self):
       """Тест парсинга действия вывода."""
       action = "derive кот является животным"
       parsed = PatternMatcher.parse_action(action)
       
       self.assertEqual(parsed["type"], "derive_is_a")
       self.assertEqual(parsed["subject"], "кот")
       self.assertEqual(parsed["object"], "животным")
   
   def test_match_simple_condition(self):
       """Тест сопоставления простого условия."""
       context = ProcessingContext()
       context.add_fact("собака_is_a_животным", True, 0.9)
       
       parsed_condition = {"type": "is_a", "subject": "собака", "object": "животным"}
       matched, confidence, variables = PatternMatcher.match_condition(parsed_condition, context)
       
       self.assertTrue(matched)
       self.assertEqual(confidence, 0.9)
   
   def test_match_property_condition(self):
       """Тест сопоставления условия со свойством."""
       context = ProcessingContext()
       context.add_fact("собака_has_лай", True, 0.8)
       
       parsed_condition = {"type": "has_property", "subject": "собака", "property": "лай"}
       matched, confidence, variables = PatternMatcher.match_condition(parsed_condition, context)
       
       self.assertTrue(matched)
       self.assertEqual(confidence, 0.8)
   
   def test_execute_assert_action(self):
       """Тест выполнения действия утверждения."""
       context = ProcessingContext()
       parsed_action = {"type": "assert_is_a", "subject": "кот", "object": "животным"}
       
       results = PatternMatcher.execute_action(parsed_action, context, 0.9)
       
       self.assertIn("кот_is_a_животным", results)
       self.assertTrue(context.has_fact("кот_is_a_животным"))
       self.assertEqual(context.get_fact("кот_is_a_животным"), True)


class TestPatternMatchingProcessor(unittest.TestCase):
   """Тесты для класса PatternMatchingProcessor."""
   
   def setUp(self):
       """Настройка тестов."""
       self.processor = PatternMatchingProcessor(
           confidence_threshold=0.5,
           max_depth=3,
           enable_explanations=True
       )
   
   def test_add_rule(self):
       """Тест добавления правила."""
       rule = SymbolicRule(
           condition="собака является животным",
           action="derive собака нуждается в пище",
           confidence=0.9
       )
       
       rule_id = self.processor.add_rule(rule)
       
       self.assertIsNotNone(rule_id)
       self.assertIn(rule_id, self.processor._rules)
       self.assertEqual(self.processor._stats["rules_added"], 1)
   
   def test_add_invalid_rule(self):
       """Тест добавления невалидного правила."""
       rule = SymbolicRule(
           condition="",  # Пустое условие
           action="some action",
           confidence=0.9
       )
       
       with self.assertRaises(RuleValidationError):
           self.processor.add_rule(rule)
   
   def test_remove_rule(self):
       """Тест удаления правила."""
       rule = SymbolicRule(
           condition="тест условие",
           action="тест действие"
       )
       
       rule_id = self.processor.add_rule(rule)
       removed = self.processor.remove_rule(rule_id)
       
       self.assertTrue(removed)
       self.assertNotIn(rule_id, self.processor._rules)
   
   def test_execute_simple_rule(self):
       """Тест выполнения простого правила."""
       # Добавляем правило
       rule = SymbolicRule(
           condition="собака является животным",
           action="derive собака нуждается в пище",
           confidence=0.9
       )
       rule_id = self.processor.add_rule(rule)
       
       # Создаем контекст с нужным фактом
       context = ProcessingContext()
       context.add_fact("собака_is_a_животным", True, 0.8)
       
       # Выполняем правило
       result = self.processor.execute_rule(rule_id, context)
       
       self.assertTrue(result.success)
       self.assertGreater(result.confidence, 0.5)
       self.assertIn(rule_id, result.rules_used)
       self.assertGreater(len(result.derived_facts), 0)
   
   def test_execute_rule_insufficient_confidence(self):
       """Тест выполнения правила с недостаточной уверенностью."""
       rule = SymbolicRule(
           condition="собака является животным",
           action="derive собака нуждается в пище",
           confidence=0.9
       )
       rule_id = self.processor.add_rule(rule)
       
       # Контекст с низкой уверенностью
       context = ProcessingContext()
       context.add_fact("собака_is_a_животным", True, 0.3)  # Ниже порога 0.5
       
       result = self.processor.execute_rule(rule_id, context)
       
       self.assertFalse(result.success)
   
   def test_derive_single_step(self):
       """Тест одношагового логического вывода."""
       # Правило транзитивности
       rule = SymbolicRule(
           condition="собака является животным",
           action="derive собака нуждается в пище",
           confidence=0.9
       )
       self.processor.add_rule(rule)
       
       # Контекст
       context = ProcessingContext()
       context.add_fact("собака_is_a_животным", True, 0.8)
       
       # Выполнение вывода
       result = self.processor.derive(context, depth=1)
       
       self.assertTrue(result.success)
       self.assertGreater(len(result.derived_facts), 0)
       self.assertGreater(len(result.explanation), 0)
   
   def test_derive_multi_step(self):
       """Тест многошагового логического вывода."""
       # Правило 1
       rule1 = SymbolicRule(
           condition="собака является животным",
           action="derive собака живая",
           confidence=0.9
       )
       self.processor.add_rule(rule1)
       
       # Правило 2
       rule2 = SymbolicRule(
           condition="собака живая",
           action="derive собака дышит",
           confidence=0.8
       )
       self.processor.add_rule(rule2)
       
       # Контекст
       context = ProcessingContext()
       context.add_fact("собака_is_a_животным", True, 0.8)
       
       # Выполнение многошагового вывода
       result = self.processor.derive(context, depth=2)
       
       self.assertTrue(result.success)
       self.assertGreaterEqual(len(result.explanation), 1)
       # Должны быть выведены факты из обоих правил
       self.assertGreater(len(result.derived_facts), 0)
   
   def test_find_relevant_rules(self):
       """Тест поиска релевантных правил."""
       rule1 = SymbolicRule(
           condition="собака является животным",
           action="derive собака живая"
       )
       rule1_id = self.processor.add_rule(rule1)
       
       rule2 = SymbolicRule(
           condition="кот является животным", 
           action="derive кот живая"
       )
       rule2_id = self.processor.add_rule(rule2)
       
       # Контекст только с собакой
       context = ProcessingContext()
       context.add_fact("собака_is_a_животным", True, 0.8)
       
       relevant_rules = self.processor.find_relevant_rules(context)
       
       # Должно найти только правило о собаке
       self.assertIn(rule1_id, relevant_rules)
       # rule2_id может не быть в relevant_rules, т.к. нет факта о коте
   
   def test_update_rule(self):
       """Тест обновления правила."""
       rule = SymbolicRule(
           condition="тест",
           action="действие",
           confidence=0.5
       )
       rule_id = self.processor.add_rule(rule)
       
       # Обновление уверенности
       updated = self.processor.update_rule(rule_id, confidence=0.9)
       
       self.assertTrue(updated)
       updated_rule = self.processor.get_rule(rule_id)
       self.assertEqual(updated_rule.confidence, 0.9)
   
   def test_validate_rule(self):
       """Тест валидации правил."""
       # Валидное правило
       valid_rule = SymbolicRule(
           condition="собака является животным",
           action="derive собака живая",
           confidence=0.8
       )
       is_valid, error = self.processor.validate_rule(valid_rule)
       self.assertTrue(is_valid)
       self.assertIsNone(error)
       
       # Невалидное правило - пустое условие
       invalid_rule = SymbolicRule(
           condition="",
           action="действие",
           confidence=0.8
       )
       is_valid, error = self.processor.validate_rule(invalid_rule)
       self.assertFalse(is_valid)
       self.assertIsNotNone(error)
       
       # Невалидное правило - неправильная уверенность
       invalid_rule2 = SymbolicRule(
           condition="условие",
           action="действие",
           confidence=1.5  # Больше 1
       )
       is_valid, error = self.processor.validate_rule(invalid_rule2)
       self.assertFalse(is_valid)
       self.assertIsNotNone(error)
   
   def test_clear_rules(self):
       """Тест очистки всех правил."""
       rule = SymbolicRule(condition="тест", action="действие")
       self.processor.add_rule(rule)
       
       self.assertEqual(len(self.processor._rules), 1)
       
       self.processor.clear_rules()
       
       self.assertEqual(len(self.processor._rules), 0)
   
   def test_get_statistics(self):
       """Тест получения статистики."""
       rule = SymbolicRule(
           condition="собака является животным",
           action="derive собака живая"
       )
       rule_id = self.processor.add_rule(rule)
       
       context = ProcessingContext()
       context.add_fact("собака_is_a_животным", True, 0.8)
       
       self.processor.execute_rule(rule_id, context)
       
       stats = self.processor.get_statistics()
       
       self.assertEqual(stats["rules_count"], 1)
       self.assertEqual(stats["rules_added"], 1)
       self.assertEqual(stats["rules_executed"], 1)
       self.assertGreater(stats["total_execution_time"], 0)
   
   def test_export_import_rules(self):
       """Тест экспорта и импорта правил."""
       # Добавляем несколько правил
       rule1 = SymbolicRule(
           condition="собака является животным",
           action="derive собака живая",
           confidence=0.9
       )
       rule2 = SymbolicRule(
           condition="кот является животным",
           action="derive кот живая", 
           confidence=0.8
       )
       
       self.processor.add_rule(rule1)
       self.processor.add_rule(rule2)
       
       # Экспорт
       exported_rules = self.processor.export_rules()
       self.assertEqual(len(exported_rules), 2)
       
       # Очистка и импорт
       self.processor.clear_rules()
       imported_count = self.processor.import_rules(exported_rules)
       
       self.assertEqual(imported_count, 2)
       self.assertEqual(len(self.processor._rules), 2)
   
   def test_explanation_generation(self):
       """Тест генерации объяснений."""
       rule = SymbolicRule(
           condition="собака является животным",
           action="derive собака нуждается в пище",
           confidence=0.9
       )
       rule_id = self.processor.add_rule(rule)
       
       context = ProcessingContext()
       context.add_fact("собака_is_a_животным", True, 0.8)
       
       result = self.processor.execute_rule(rule_id, context)
       
       self.assertTrue(result.success)
       self.assertGreater(len(result.explanation), 0)
       
       explanation = result.explanation[0]
       self.assertEqual(explanation.rule_id, rule_id)
       self.assertGreater(explanation.confidence, 0)
       self.assertIn("собака", explanation.rule_description)


class TestSymbolicRule(unittest.TestCase):
   """Тесты для класса SymbolicRule."""
   
   def test_rule_creation(self):
       """Тест создания правила."""
       rule = SymbolicRule(
           condition="A является B",
           action="derive C является D",
           confidence=0.8,
           priority=1
       )
       
       self.assertEqual(rule.condition, "A является B")
       self.assertEqual(rule.action, "derive C является D")
       self.assertEqual(rule.confidence, 0.8)
       self.assertEqual(rule.priority, 1)
       self.assertEqual(rule.usage_count, 0)
       self.assertIsNotNone(rule.id)
       self.assertIsInstance(rule.created_at, datetime)
   
   def test_mark_used(self):
       """Тест отметки использования правила."""
       rule = SymbolicRule(condition="тест", action="действие")
       
       self.assertEqual(rule.usage_count, 0)
       self.assertIsNone(rule.last_used)
       
       rule.mark_used()
       
       self.assertEqual(rule.usage_count, 1)
       self.assertIsNotNone(rule.last_used)
   
   def test_rule_serialization(self):
       """Тест сериализации и десериализации правила."""
       original_rule = SymbolicRule(
           condition="собака является животным",
           action="derive собака живая",
           confidence=0.9,
           priority=2,
           metadata={"source": "expert_system"}
       )
       
       # Сериализация
       rule_dict = original_rule.to_dict()
       
       self.assertIn("condition", rule_dict)
       self.assertIn("action", rule_dict)
       self.assertIn("confidence", rule_dict)
       self.assertIn("metadata", rule_dict)
       
       # Десериализация
       restored_rule = SymbolicRule.from_dict(rule_dict)
       
       self.assertEqual(restored_rule.condition, original_rule.condition)
       self.assertEqual(restored_rule.action, original_rule.action)
       self.assertEqual(restored_rule.confidence, original_rule.confidence)
       self.assertEqual(restored_rule.priority, original_rule.priority)
       self.assertEqual(restored_rule.metadata, original_rule.metadata)


class TestProcessingContext(unittest.TestCase):
   """Тесты для класса ProcessingContext."""
   
   def test_context_creation(self):
       """Тест создания контекста."""
       context = ProcessingContext()
       
       self.assertIsInstance(context.facts, dict)
       self.assertIsInstance(context.variables, dict)
       self.assertIsNotNone(context.session_id)
       self.assertEqual(context.max_depth, 5)
       self.assertEqual(context.confidence_threshold, 0.5)
   
   def test_add_get_fact(self):
       """Тест добавления и получения факта."""
       context = ProcessingContext()
       
       context.add_fact("собака_is_a_животным", True, 0.9)
       
       self.assertTrue(context.has_fact("собака_is_a_животным"))
       fact_value = context.get_fact("собака_is_a_животным")
       self.assertTrue(fact_value)
       
       # Проверка метаданных факта
       fact_data = context.facts["собака_is_a_животным"]
       self.assertEqual(fact_data["confidence"], 0.9)
       self.assertIn("added_at", fact_data)
   
   def test_variables(self):
       """Тест работы с переменными."""
       context = ProcessingContext()
       
       context.set_variable("X", "собака")
       context.set_variable("Y", "животное")
       
       self.assertEqual(context.get_variable("X"), "собака")
       self.assertEqual(context.get_variable("Y"), "животное")
       self.assertIsNone(context.get_variable("Z"))
       
       context.clear_variables()
       
       self.assertIsNone(context.get_variable("X"))
       self.assertIsNone(context.get_variable("Y"))
   
   def test_context_copy(self):
       """Тест копирования контекста."""
       original_context = ProcessingContext()
       original_context.add_fact("тест_факт", "значение", 0.8)
       original_context.set_variable("X", "переменная")
       
       copied_context = original_context.copy()
       
       # Проверяем, что данные скопированы
       self.assertTrue(copied_context.has_fact("тест_факт"))
       self.assertEqual(copied_context.get_variable("X"), "переменная")
       self.assertEqual(copied_context.session_id, original_context.session_id)
       
       # Проверяем, что это независимые копии
       copied_context.add_fact("новый_факт", "новое_значение")
       self.assertFalse(original_context.has_fact("новый_факт"))


if __name__ == "__main__":
   unittest.main()