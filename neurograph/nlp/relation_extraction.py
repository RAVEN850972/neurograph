"""Реализации извлечения отношений для модуля NLP."""

import re
from typing import List, Dict, Set, Optional, Tuple
from neurograph.nlp.base import (
    IRelationExtractor, Relation, Entity, EntityType, RelationType, 
    Token, Sentence
)
from neurograph.core.logging import get_logger


class PatternBasedRelationExtractor(IRelationExtractor):
    """Извлечение отношений на основе лингвистических паттернов."""
    
    def __init__(self, language: str = "ru"):
        self.language = language
        self.logger = get_logger("pattern_based_relation_extractor")
        
        # Инициализация паттернов отношений
        self._init_relation_patterns()
        
        # Стоп-слова для фильтрации
        self.stop_words = {
            "и", "в", "на", "с", "по", "для", "от", "до", "за", "под", 
            "над", "при", "через", "между", "среди", "внутри", "вне"
        }
    
    def _init_relation_patterns(self):
        """Инициализация паттернов для различных типов отношений."""
        
        # Паттерны для отношения "является" (is_a)
        self.is_a_patterns = [
            (r'(.+?)\s+(?:является|есть|это)\s+(.+)', RelationType.IS_A),
            (r'(.+?)\s*-\s*(?:это|есть)\s+(.+)', RelationType.IS_A),
            (r'(.+?)\s+представляет\s+собой\s+(.+)', RelationType.IS_A),
            (r'(.+?)\s+относится\s+к\s+(.+)', RelationType.IS_A),
            (r'(.+?)\s+входит\s+в\s+(?:состав|группу|категорию)\s+(.+)', RelationType.IS_A),
        ]
        
        # Паттерны для отношения "имеет" (has)
        self.has_patterns = [
            (r'(.+?)\s+(?:имеет|обладает|содержит)\s+(.+)', RelationType.HAS),
            (r'(.+?)\s+включает\s+(?:в\s+себя\s+)?(.+)', RelationType.HAS),
            (r'у\s+(.+?)\s+(?:есть|имеется)\s+(.+)', RelationType.HAS),
            (r'(.+?)\s+состоит\s+из\s+(.+)', RelationType.HAS),
            (r'(.+?)\s+оснащен\s+(.+)', RelationType.HAS),
        ]
        
        # Паттерны для локации (located_in)
        self.location_patterns = [
            (r'(.+?)\s+(?:находится|расположен|размещен)\s+в\s+(.+)', RelationType.LOCATED_IN),
            (r'(.+?)\s+живет\s+в\s+(.+)', RelationType.LOCATED_IN),
            (r'(.+?)\s+из\s+(.+)', RelationType.LOCATED_IN),  # "студент из Москвы"
            (r'в\s+(.+?)\s+(?:находится|есть|расположен)\s+(.+)', RelationType.LOCATED_IN),
        ]
        
        # Паттерны для работы (works_for)
        self.works_for_patterns = [
            (r'(.+?)\s+работает\s+в\s+(.+)', RelationType.WORKS_FOR),
            (r'(.+?)\s+сотрудник\s+(.+)', RelationType.WORKS_FOR),
            (r'(.+?)\s+директор\s+(.+)', RelationType.WORKS_FOR),
            (r'(.+?)\s+(?:руководит|возглавляет)\s+(.+)', RelationType.WORKS_FOR),
            (r'(.+?)\s+основатель\s+(.+)', RelationType.CREATED_BY),
        ]
        
        # Паттерны для создания (created_by)
        self.created_by_patterns = [
            (r'(.+?)\s+(?:создан|разработан|изобретен)\s+(.+)', RelationType.CREATED_BY),
            (r'(.+?)\s+автор\s+(.+)', RelationType.CREATED_BY),
            (r'(.+?)\s+написал\s+(.+)', RelationType.CREATED_BY),
            (r'(.+?)\s+основал\s+(.+)', RelationType.CREATED_BY),
        ]
        
        # Паттерны для использования (uses)
        self.uses_patterns = [
            (r'(.+?)\s+использует\s+(.+)', RelationType.USES),
            (r'(.+?)\s+применяет\s+(.+)', RelationType.USES),
            (r'(.+?)\s+пользуется\s+(.+)', RelationType.USES),
            (r'(.+?)\s+работает\s+с\s+(.+)', RelationType.USES),
        ]
        
        # Паттерны для причинно-следственных связей (causes)
        self.causes_patterns = [
            (r'(.+?)\s+(?:приводит\s+к|вызывает|является\s+причиной)\s+(.+)', RelationType.CAUSES),
            (r'из-за\s+(.+?)\s+(.+)', RelationType.CAUSES),
            (r'(.+?)\s+влияет\s+на\s+(.+)', RelationType.CAUSES),
            (r'(.+?)\s+способствует\s+(.+)', RelationType.CAUSES),
        ]
        
        # Паттерны для части целого (part_of)
        self.part_of_patterns = [
            (r'(.+?)\s+часть\s+(.+)', RelationType.PART_OF),
            (r'(.+?)\s+компонент\s+(.+)', RelationType.PART_OF),
            (r'(.+?)\s+элемент\s+(.+)', RelationType.PART_OF),
            (r'(.+?)\s+входит\s+в\s+(.+)', RelationType.PART_OF),
        ]
        
        # Все паттерны вместе
        self.all_patterns = [
            *self.is_a_patterns,
            *self.has_patterns,
            *self.location_patterns,
            *self.works_for_patterns,
            *self.created_by_patterns,
            *self.uses_patterns,
            *self.causes_patterns,
            *self.part_of_patterns,
        ]
        
        # Компилируем регулярные выражения
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.UNICODE), relation_type)
            for pattern, relation_type in self.all_patterns
        ]
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Извлечение отношений из текста с учетом найденных сущностей."""
        relations = []
        
        # Сначала ищем отношения по паттернам
        pattern_relations = self._extract_by_patterns(text)
        
        # Затем пытаемся связать с найденными сущностями
        entity_relations = self._extract_with_entities(text, entities)
        
        # Объединяем результаты
        all_relations = pattern_relations + entity_relations
        
        # Фильтруем дублирующиеся отношения
        unique_relations = self._deduplicate_relations(all_relations)
        
        return unique_relations
    
    def extract_relations_from_sentence(self, sentence: Sentence) -> List[Relation]:
        """Извлечение отношений из предложения."""
        if not sentence.entities:
            return []
        
        relations = self.extract_relations(sentence.text, sentence.entities)
        
        # Дополнительно анализируем синтаксические зависимости если есть
        if sentence.tokens and any(token.dependency for token in sentence.tokens):
            dependency_relations = self._extract_from_dependencies(sentence)
            relations.extend(dependency_relations)
        
        return self._deduplicate_relations(relations)
    
    def _extract_by_patterns(self, text: str) -> List[Relation]:
        """Извлечение отношений по лингвистическим паттернам."""
        relations = []
        
        for pattern, relation_type in self.compiled_patterns:
            for match in pattern.finditer(text):
                if len(match.groups()) >= 2:
                    subject_text = match.group(1).strip()
                    object_text = match.group(2).strip()
                    
                    # Очищаем от лишних слов
                    subject_text = self._clean_entity_text(subject_text)
                    object_text = self._clean_entity_text(object_text)
                    
                    if subject_text and object_text:
                        # Создаем временные сущности
                        subject = Entity(
                            text=subject_text,
                            entity_type=EntityType.CONCEPT,
                            start=match.start(1),
                            end=match.end(1),
                            confidence=0.7
                        )
                        
                        obj = Entity(
                            text=object_text,
                            entity_type=EntityType.CONCEPT,
                            start=match.start(2),
                            end=match.end(2),
                            confidence=0.7
                        )
                        
                        relation = Relation(
                            subject=subject,
                            predicate=relation_type,
                            object=obj,
                            confidence=0.8,
                            text_span=match.group(),
                            metadata={'method': 'pattern_matching', 'pattern': pattern.pattern}
                        )
                        
                        relations.append(relation)
        
        return relations
    
    def _extract_with_entities(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Извлечение отношений между найденными сущностями."""
        relations = []
        
        # Ищем отношения между каждой парой сущностей
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i >= j:  # Избегаем дублирования и самоссылок
                    continue
                
                # Извлекаем текст между сущностями
                start_pos = min(entity1.end, entity2.end)
                end_pos = max(entity1.start, entity2.start)
                
                if start_pos < end_pos:
                    between_text = text[start_pos:end_pos].strip()
                    
                    # Ищем паттерны отношений в тексте между сущностями
                    relation_type = self._find_relation_in_text(between_text, entity1, entity2)
                    
                    if relation_type != RelationType.UNKNOWN:
                        # Определяем направление отношения
                        subject, obj = self._determine_relation_direction(
                            entity1, entity2, relation_type, text
                        )
                        
                        relation = Relation(
                            subject=subject,
                            predicate=relation_type,
                            object=obj,
                            confidence=0.75,
                            text_span=between_text,
                            metadata={'method': 'entity_pair_analysis'}
                        )
                        
                        relations.append(relation)
        
        return relations
    
    def _extract_from_dependencies(self, sentence: Sentence) -> List[Relation]:
        """Извлечение отношений на основе синтаксических зависимостей."""
        relations = []
        
        # Анализируем зависимости между токенами
        for token in sentence.tokens:
            if token.dependency and token.head is not None:
                head_token = sentence.tokens[token.head] if token.head < len(sentence.tokens) else None
                
                if head_token:
                    relation_type = self._dependency_to_relation(token.dependency)
                    
                    if relation_type != RelationType.UNKNOWN:
                        # Ищем соответствующие сущности
                        subject_entity = self._find_entity_for_token(head_token, sentence.entities)
                        object_entity = self._find_entity_for_token(token, sentence.entities)
                        
                        if subject_entity and object_entity:
                            relation = Relation(
                                subject=subject_entity,
                                predicate=relation_type,
                                object=object_entity,
                                confidence=0.6,
                                metadata={'method': 'dependency_parsing', 'dependency': token.dependency}
                            )
                            relations.append(relation)
        
        return relations
    
    def _dependency_to_relation(self, dependency: str) -> RelationType:
        """Преобразование синтаксической зависимости в тип отношения."""
        dependency_mapping = {
            'nmod': RelationType.HAS,           # номинальный модификатор
            'nsubj': RelationType.IS_A,        # номинальное подлежащее
            'dobj': RelationType.USES,         # прямое дополнение
            'prep': RelationType.LOCATED_IN,   # предложная связь
            'poss': RelationType.HAS,          # притяжательное отношение
            'compound': RelationType.PART_OF,  # составное слово
            'appos': RelationType.IS_A,        # приложение
        }
        
        return dependency_mapping.get(dependency, RelationType.UNKNOWN)
    
    def _find_entity_for_token(self, token: Token, entities: List[Entity]) -> Optional[Entity]:
        """Поиск сущности, которой принадлежит токен."""
        for entity in entities:
            if entity.start <= token.start < entity.end:
                return entity
        return None
    
    def _clean_entity_text(self, text: str) -> str:
        """Очистка текста сущности от лишних слов."""
        # Удаляем артикли, предлоги и другие служебные слова
        words = text.split()
        cleaned_words = []
        
        for word in words:
            if (word.lower() not in self.stop_words and 
                len(word) > 1 and 
                not word.isdigit()):
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    def _find_relation_in_text(self, text: str, entity1: Entity, entity2: Entity) -> RelationType:
        """Поиск типа отношения в тексте между сущностями."""
        text_lower = text.lower()
        
        # Ключевые слова для разных типов отношений
        relation_keywords = {
            RelationType.IS_A: ['является', 'есть', 'это', 'представляет', 'относится'],
            RelationType.HAS: ['имеет', 'обладает', 'содержит', 'включает'],
            RelationType.LOCATED_IN: ['находится', 'расположен', 'живет', 'из'],
            RelationType.WORKS_FOR: ['работает', 'сотрудник', 'директор', 'руководит'],
            RelationType.CREATED_BY: ['создан', 'разработан', 'изобретен', 'автор'],
            RelationType.USES: ['использует', 'применяет', 'пользуется'],
            RelationType.CAUSES: ['приводит', 'вызывает', 'влияет', 'способствует'],
            RelationType.PART_OF: ['часть', 'компонент', 'элемент', 'входит'],
        }
        
        # Ищем ключевые слова в тексте
        for relation_type, keywords in relation_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return relation_type
        
        return RelationType.UNKNOWN
    
    def _determine_relation_direction(self, entity1: Entity, entity2: Entity, 
                                    relation_type: RelationType, full_text: str) -> Tuple[Entity, Entity]:
        """Определение направления отношения между сущностями."""
        # Простая эвристика: сущность, которая идет первой в тексте, обычно субъект
        if entity1.start < entity2.start:
            return entity1, entity2
        else:
            return entity2, entity1
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Удаление дублирующихся отношений."""
        unique_relations = []
        seen = set()
        
        for relation in relations:
            # Создаем ключ для идентификации дублей
            key = (
                relation.subject.text.lower(),
                relation.predicate.value,
                relation.object.text.lower()
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                # Если дубль, но с большей уверенностью - заменяем
                for i, existing in enumerate(unique_relations):
                    existing_key = (
                        existing.subject.text.lower(),
                        existing.predicate.value,
                        existing.object.text.lower()
                    )
                    if existing_key == key and relation.confidence > existing.confidence:
                        unique_relations[i] = relation
                        break
        
        return unique_relations


class SpacyRelationExtractor(IRelationExtractor):
    """Извлечение отношений с использованием spaCy и синтаксических зависимостей."""
    
    def __init__(self, model_name: str = "ru_core_news_sm"):
        self.logger = get_logger("spacy_relation_extractor")
        self.model_name = model_name
        self.nlp = None
        
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Загружена модель spaCy: {model_name}")
        except ImportError:
            self.logger.warning("spaCy не установлен, используется fallback")
            self.fallback = PatternBasedRelationExtractor()
        except OSError:
            self.logger.warning(f"Модель {model_name} не найдена, используется fallback")
            self.fallback = PatternBasedRelationExtractor()
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Извлечение отношений с использованием spaCy."""
        if self.nlp is None:
            return self.fallback.extract_relations(text, entities)
        
        doc = self.nlp(text)
        relations = []
        
        # Создаем карту позиций сущностей
        entity_map = {}
        for entity in entities:
            for i in range(entity.start, entity.end):
                entity_map[i] = entity
        
        # Анализируем зависимости
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj', 'attr']:
                # Ищем отношения субъект-предикат-объект
                head = token.head
                
                # Ищем соответствующие сущности
                token_entity = self._find_entity_at_position(token.idx, entity_map)
                head_entity = self._find_entity_at_position(head.idx, entity_map)
                
                if token_entity and head_entity and token_entity != head_entity:
                    relation_type = self._spacy_dep_to_relation(token.dep_, head.pos_)
                    
                    # Определяем направление отношения
                    if token.dep_ == 'nsubj':
                        subject, obj = token_entity, head_entity
                    else:
                        subject, obj = head_entity, token_entity
                    
                    relation = Relation(
                        subject=subject,
                        predicate=relation_type,
                        object=obj,
                        confidence=0.7,
                        metadata={
                            'method': 'spacy_dependencies',
                            'dependency': token.dep_,
                            'head_pos': head.pos_
                        }
                    )
                    relations.append(relation)
        
        # Дополняем паттернами
        pattern_relations = self.fallback.extract_relations(text, entities)
        relations.extend(pattern_relations)
        
        return self._deduplicate_relations(relations)
    
    def extract_relations_from_sentence(self, sentence: Sentence) -> List[Relation]:
        """Извлечение отношений из предложения."""
        if self.nlp is None:
            return self.fallback.extract_relations_from_sentence(sentence)
        
        return self.extract_relations(sentence.text, sentence.entities or [])
    
    def _find_entity_at_position(self, pos: int, entity_map: Dict[int, Entity]) -> Optional[Entity]:
        """Поиск сущности в определенной позиции."""
        return entity_map.get(pos)
    
    def _spacy_dep_to_relation(self, dependency: str, head_pos: str) -> RelationType:
        """Преобразование spaCy зависимости в тип отношения."""
        
        # Более точное маппирование с учетом части речи
        if dependency == 'nsubj':
            if head_pos in ['VERB']:
                return RelationType.USES
            elif head_pos in ['NOUN']:
                return RelationType.IS_A
        elif dependency == 'dobj':
            return RelationType.USES
        elif dependency == 'pobj':
            return RelationType.LOCATED_IN
        elif dependency == 'attr':
            return RelationType.IS_A
        elif dependency == 'poss':
            return RelationType.HAS
        elif dependency == 'compound':
            return RelationType.PART_OF
        
        return RelationType.RELATED_TO
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Удаление дублирующихся отношений."""
        # Используем тот же алгоритм, что и в PatternBasedRelationExtractor
        unique_relations = []
        seen = set()
        
        for relation in relations:
            key = (
                relation.subject.text.lower(),
                relation.predicate.value,
                relation.object.text.lower()
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                for i, existing in enumerate(unique_relations):
                    existing_key = (
                        existing.subject.text.lower(),
                        existing.predicate.value,
                        existing.object.text.lower()
                    )
                    if existing_key == key and relation.confidence > existing.confidence:
                        unique_relations[i] = relation
                        break
        
        return unique_relations


class RuleBasedRelationExtractor(IRelationExtractor):
    """Простой экстрактор отношений на основе правил для триплетов."""
    
    def __init__(self):
        self.logger = get_logger("rule_based_relation_extractor")
        
        # Правила для извлечения триплетов
        self.triplet_patterns = [
            # Субъект - связка - объект
            (r'(.+?)\s+(является|есть)\s+(.+)', 'is_a'),
            (r'(.+?)\s+(имеет|содержит)\s+(.+)', 'has'),
            (r'(.+?)\s+(создал|разработал)\s+(.+)', 'created'),
            (r'(.+?)\s+(использует|применяет)\s+(.+)', 'uses'),
            (r'(.+?)\s+(находится в|расположен в)\s+(.+)', 'located_in'),
        ]
        
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), relation)
            for pattern, relation in self.triplet_patterns
        ]
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Извлечение простых триплетов из текста."""
        relations = []
        
        # Разбиваем текст на предложения
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            sentence_relations = self._extract_from_sentence(sentence)
            relations.extend(sentence_relations)
        
        return relations
    
    def extract_relations_from_sentence(self, sentence: Sentence) -> List[Relation]:
        """Извлечение отношений из предложения."""
        return self._extract_from_sentence(sentence.text)
    
    def _extract_from_sentence(self, sentence: str) -> List[Relation]:
        """Извлечение отношений из одного предложения."""
        relations = []
        
        for pattern, relation_name in self.compiled_patterns:
            matches = pattern.finditer(sentence)
            
            for match in matches:
                if len(match.groups()) >= 3:
                    subject_text = match.group(1).strip()
                    predicate_text = match.group(2).strip()
                    object_text = match.group(3).strip()
                    
                    # Создаем сущности
                    subject = Entity(
                        text=subject_text,
                        entity_type=EntityType.CONCEPT,
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.8
                    )
                    
                    obj = Entity(
                        text=object_text,
                        entity_type=EntityType.CONCEPT,
                        start=match.start(3),
                        end=match.end(3),
                        confidence=0.8
                    )
                    
                    # Определяем тип отношения
                    relation_type = self._relation_name_to_type(relation_name)
                    
                    relation = Relation(
                        subject=subject,
                        predicate=relation_type,
                        object=obj,
                        confidence=0.75,
                        text_span=match.group(),
                        metadata={'method': 'rule_based_triplets', 'predicate_text': predicate_text}
                    )
                    
                    relations.append(relation)
        
        return relations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Простое разбиение на предложения."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _relation_name_to_type(self, relation_name: str) -> RelationType:
        """Преобразование имени отношения в тип."""
        mapping = {
            'is_a': RelationType.IS_A,
            'has': RelationType.HAS,
            'created': RelationType.CREATED_BY,
            'uses': RelationType.USES,
            'located_in': RelationType.LOCATED_IN,
        }
        return mapping.get(relation_name, RelationType.RELATED_TO)


class HybridRelationExtractor(IRelationExtractor):
    """Гибридный экстрактор, комбинирующий несколько подходов."""
    
    def __init__(self, use_spacy: bool = True, use_patterns: bool = True, use_rules: bool = True):
        self.logger = get_logger("hybrid_relation_extractor")
        
        self.extractors = []
        
        if use_patterns:
            self.pattern_extractor = PatternBasedRelationExtractor()
            self.extractors.append(('patterns', self.pattern_extractor))
        
        if use_spacy:
            self.spacy_extractor = SpacyRelationExtractor()
            if self.spacy_extractor.nlp is not None:
                self.extractors.append(('spacy', self.spacy_extractor))
        
        if use_rules:
            self.rule_extractor = RuleBasedRelationExtractor()
            self.extractors.append(('rules', self.rule_extractor))
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Извлечение отношений с комбинированием результатов."""
        all_relations = []
        
        # Получаем отношения от всех экстракторов
        for extractor_name, extractor in self.extractors:
            try:
                relations = extractor.extract_relations(text, entities)
                
                # Добавляем информацию об источнике
                for relation in relations:
                    if 'extractors' not in relation.metadata:
                        relation.metadata['extractors'] = []
                    relation.metadata['extractors'].append(extractor_name)
                
                all_relations.extend(relations)
                
            except Exception as e:
                self.logger.warning(f"Ошибка в экстракторе {extractor_name}: {e}")
        
        # Объединяем и улучшаем результаты
        merged_relations = self._merge_relations(all_relations)
        
        return merged_relations
    
    def extract_relations_from_sentence(self, sentence: Sentence) -> List[Relation]:
        """Извлечение отношений из предложения."""
        return self.extract_relations(sentence.text, sentence.entities or [])
    
    def _merge_relations(self, relations: List[Relation]) -> List[Relation]:
        """Объединение отношений от разных экстракторов."""
        if not relations:
            return relations
        
        # Группируем похожие отношения
        groups = {}
        
        for relation in relations:
            key = self._get_relation_key(relation)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(relation)
        
        # Объединяем отношения в каждой группе
        merged = []
        
        for group_relations in groups.values():
            if len(group_relations) == 1:
                merged.append(group_relations[0])
            else:
                merged_relation = self._merge_relation_group(group_relations)
                merged.append(merged_relation)
        
        return merged
    
    def _get_relation_key(self, relation: Relation) -> Tuple[str, str, str]:
        """Получение ключа для группировки отношений."""
        return (
            relation.subject.text.lower().strip(),
            relation.predicate.value,
            relation.object.text.lower().strip()
        )
    
    def _merge_relation_group(self, relations: List[Relation]) -> Relation:
        """Объединение группы похожих отношений."""
        # Выбираем отношение с наибольшей уверенностью как основу
        best_relation = max(relations, key=lambda r: r.confidence)
        
        # Собираем метаданные от всех отношений
        all_extractors = set()
        all_methods = set()
        
        for relation in relations:
            if 'extractors' in relation.metadata:
                all_extractors.update(relation.metadata['extractors'])
            if 'method' in relation.metadata:
                all_methods.add(relation.metadata['method'])
        
        # Повышаем уверенность если несколько экстракторов согласны
        confidence_boost = min(0.2, 0.1 * len(relations))
        final_confidence = min(1.0, best_relation.confidence + confidence_boost)
        
        # Создаем объединенное отношение
        merged_relation = Relation(
            subject=best_relation.subject,
            predicate=best_relation.predicate,
            object=best_relation.object,
            confidence=final_confidence,
            text_span=best_relation.text_span,
            metadata={
                'extractors': list(all_extractors),
                'methods': list(all_methods),
                'merged_from': len(relations),
                'original_confidences': [r.confidence for r in relations]
            }
        )
        
        return merged_relation