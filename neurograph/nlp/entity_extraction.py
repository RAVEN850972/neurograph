"""Реализации извлечения сущностей для модуля NLP."""

import re
from typing import List, Dict, Set, Optional, Tuple
from neurograph.nlp.base import IEntityExtractor, Entity, EntityType, Token
from neurograph.core.logging import get_logger


class RuleBasedEntityExtractor(IEntityExtractor):
    """Извлечение сущностей на основе правил и паттернов."""
    
    def __init__(self, language: str = "ru"):
        self.language = language
        self.logger = get_logger("rule_based_entity_extractor")
        
        # Паттерны для различных типов сущностей
        self._init_patterns()
        
        # Словари имен и организаций
        self._init_dictionaries()
    
    def _init_patterns(self):
        """Инициализация паттернов для извлечения сущностей."""
        
        # Даты
        self.date_patterns = [
            (r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b', 'date'),
            (r'\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}\b', 'date'),
            (r'\b(понедельник|вторник|среда|четверг|пятница|суббота|воскресенье)\b', 'date'),
            (r'\b(сегодня|вчера|завтра|послезавтра)\b', 'date'),
        ]
        
        # Время
        self.time_patterns = [
            (r'\b\d{1,2}:\d{2}(:\d{2})?\b', 'time'),
            (r'\b\d{1,2}\s+(часов|минут|секунд)\b', 'time'),
            (r'\b(утром|днем|вечером|ночью)\b', 'time'),
        ]
        
        # Деньги
        self.money_patterns = [
            (r'\b\d+(\.\d{2})?\s+(рублей|долларов|евро|копеек|центов)\b', 'money'),
            (r'\b\d+\s*(руб|₽|\$|€)\b', 'money'),
            (r'\b\d+\s*(тысяч|миллионов|миллиардов)\s+(рублей|долларов|евро)\b', 'money'),
        ]
        
        # Организации
        self.org_patterns = [
            (r'\b[А-ЯЁ][а-яё]+\s+(ООО|ОАО|ЗАО|ИП|ТОО)\b', 'organization'),
            (r'\b(ООО|ОАО|ЗАО|ИП|ТОО)\s+[«"][А-ЯЁ][а-яё\s]+[»"]\b', 'organization'),
            (r'\b[А-ЯЁ][а-яё]+\s+(банк|компания|корпорация|фирма|предприятие)\b', 'organization'),
            (r'\b(университет|институт|академия|школа|лицей|гимназия)\s+[А-ЯЁ][а-яё\s]+\b', 'organization'),
        ]
        
        # Локации
        self.location_patterns = [
            (r'\b(г\.|город)\s+[А-ЯЁ][а-яё]+\b', 'location'),
            (r'\b[А-ЯЁ][а-яё]+(ск|инск|град|бург|поль)\b', 'location'),
            (r'\b(улица|ул\.|проспект|пр\.|переулок|пер\.)\s+[А-ЯЁ][а-яё\s]+\b', 'location'),
            (r'\b(область|край|район|республика)\s+[А-ЯЁ][а-яё\s]+\b', 'location'),
        ]
        
        # Концепты (абстрактные понятия)
        self.concept_patterns = [
            (r'\b[а-яё]+(ация|ение|ость|изм|логия|граф|метр)\b', 'concept'),
            (r'\b(алгоритм|система|технология|метод|подход|принцип)\s+[а-яё\s]+\b', 'concept'),
        ]
        
        # Компилируем паттерны
        self.compiled_patterns = []
        all_patterns = [
            (self.date_patterns, EntityType.DATE),
            (self.time_patterns, EntityType.TIME),
            (self.money_patterns, EntityType.MONEY),
            (self.org_patterns, EntityType.ORGANIZATION),
            (self.location_patterns, EntityType.LOCATION),
            (self.concept_patterns, EntityType.CONCEPT),
        ]
        
        for pattern_group, entity_type in all_patterns:
            for pattern, _ in pattern_group:
                self.compiled_patterns.append((
                    re.compile(pattern, re.IGNORECASE | re.UNICODE),
                    entity_type
                ))
    
    def _init_dictionaries(self):
        """Инициализация словарей имен и организаций."""
        
        # Русские имена
        self.first_names = {
            'Александр', 'Алексей', 'Андрей', 'Антон', 'Артем', 'Владимир', 'Дмитрий',
            'Евгений', 'Иван', 'Максим', 'Михаил', 'Николай', 'Олег', 'Павел', 'Сергей',
            'Анна', 'Елена', 'Ирина', 'Мария', 'Наталья', 'Ольга', 'Светлана', 'Татьяна'
        }
        
        # Русские фамилии
        self.last_names = {
            'Иванов', 'Петров', 'Сидоров', 'Смирнов', 'Кузнецов', 'Попов', 'Васильев',
            'Соколов', 'Михайлов', 'Новиков', 'Федоров', 'Морозов', 'Волков', 'Алексеев'
        }
        
        # Известные компании
        self.known_organizations = {
            'Яндекс', 'Сбербанк', 'Газпром', 'Роснефть', 'ВТБ', 'Лукойл', 'Магнит',
            'X5 Retail Group', 'Мегафон', 'МТС', 'Билайн', 'Тинькофф', 'Альфа-Банк',
            'Google', 'Microsoft', 'Apple', 'Amazon', 'Facebook', 'Tesla', 'Netflix'
        }
        
        # Известные города
        self.known_cities = {
            'Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань',
            'Нижний Новгород', 'Челябинск', 'Самара', 'Омск', 'Ростов-на-Дону',
            'Уфа', 'Красноярск', 'Воронеж', 'Пермь', 'Волгоград'
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Извлечение сущностей из текста с использованием правил."""
        entities = []
        
        # Извлечение по паттернам
        for pattern, entity_type in self.compiled_patterns:
            for match in pattern.finditer(text):
                entity = Entity(
                    text=match.group(),
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8,  # Средняя уверенность для правил
                    metadata={'method': 'pattern_matching'}
                )
                entities.append(entity)
        
        # Извлечение имен по словарям
        entities.extend(self._extract_person_names(text))
        
        # Извлечение организаций по словарям
        entities.extend(self._extract_known_organizations(text))
        
        # Извлечение городов по словарям
        entities.extend(self._extract_known_locations(text))
        
        # Фильтрация пересекающихся сущностей
        entities = self._filter_overlapping_entities(entities)
        
        return entities
    
    def extract_entities_from_tokens(self, tokens: List[Token]) -> List[Entity]:
        """Извлечение сущностей из токенов."""
        entities = []
        text = ' '.join(token.text for token in tokens)
        
        # Сначала извлекаем из полного текста
        text_entities = self.extract_entities(text)
        
        # Дополнительно анализируем токены
        entities.extend(self._extract_from_token_sequence(tokens))
        
        # Объединяем и фильтруем
        all_entities = text_entities + entities
        return self._filter_overlapping_entities(all_entities)
    
    def _extract_person_names(self, text: str) -> List[Entity]:
        """Извлечение имен людей."""
        entities = []
        words = text.split()
        
        for i, word in enumerate(words):
            # Проверяем имя + фамилия
            if (word in self.first_names and 
                i + 1 < len(words) and 
                words[i + 1] in self.last_names):
                
                full_name = f"{word} {words[i + 1]}"
                start_pos = text.find(full_name)
                
                if start_pos != -1:
                    entity = Entity(
                        text=full_name,
                        entity_type=EntityType.PERSON,
                        start=start_pos,
                        end=start_pos + len(full_name),
                        confidence=0.9,
                        metadata={'method': 'dictionary_lookup', 'type': 'full_name'}
                    )
                    entities.append(entity)
            
            # Проверяем отдельные имена (с осторожностью)
            elif word in self.first_names:
                start_pos = text.find(word)
                if start_pos != -1:
                    entity = Entity(
                        text=word,
                        entity_type=EntityType.PERSON,
                        start=start_pos,
                        end=start_pos + len(word),
                        confidence=0.6,  # Низкая уверенность для одиночных имен
                        metadata={'method': 'dictionary_lookup', 'type': 'first_name'}
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_known_organizations(self, text: str) -> List[Entity]:
        """Извлечение известных организаций."""
        entities = []
        
        for org_name in self.known_organizations:
            # Ищем точные совпадения
            start = 0
            while True:
                pos = text.find(org_name, start)
                if pos == -1:
                    break
                
                entity = Entity(
                    text=org_name,
                    entity_type=EntityType.ORGANIZATION,
                    start=pos,
                    end=pos + len(org_name),
                    confidence=0.95,  # Высокая уверенность для известных организаций
                    metadata={'method': 'dictionary_lookup', 'known_entity': True}
                )
                entities.append(entity)
                start = pos + 1
        
        return entities
    
    def _extract_known_locations(self, text: str) -> List[Entity]:
        """Извлечение известных локаций."""
        entities = []
        
        for city_name in self.known_cities:
            start = 0
            while True:
                pos = text.find(city_name, start)
                if pos == -1:
                    break
                
                entity = Entity(
                    text=city_name,
                    entity_type=EntityType.LOCATION,
                    start=pos,
                    end=pos + len(city_name),
                    confidence=0.9,
                    metadata={'method': 'dictionary_lookup', 'type': 'city'}
                )
                entities.append(entity)
                start = pos + 1
        
        return entities
    
    def _extract_from_token_sequence(self, tokens: List[Token]) -> List[Entity]:
        """Извлечение сущностей на основе последовательности токенов."""
        entities = []
        
        for i, token in enumerate(tokens):
            # Ищем заглавные слова (потенциальные имена собственные)
            if (token.text[0].isupper() and 
                token.is_alpha and 
                not token.is_stop and 
                len(token.text) > 2):
                
                # Проверяем контекст для определения типа
                entity_type = self._determine_entity_type_from_context(tokens, i)
                
                if entity_type != EntityType.UNKNOWN:
                    entity = Entity(
                        text=token.text,
                        entity_type=entity_type,
                        start=token.start,
                        end=token.end,
                        confidence=0.5,  # Низкая уверенность для контекстного анализа
                        metadata={'method': 'context_analysis'}
                    )
                    entities.append(entity)
        
        return entities
    
    def _determine_entity_type_from_context(self, tokens: List[Token], index: int) -> EntityType:
        """Определение типа сущности по контексту."""
        token = tokens[index]
        
        # Смотрим на соседние токены
        context_before = []
        context_after = []
        
        # Берем 2 токена до и после
        for i in range(max(0, index - 2), index):
            context_before.append(tokens[i].text.lower())
        
        for i in range(index + 1, min(len(tokens), index + 3)):
            context_after.append(tokens[i].text.lower())
        
        # Контекстные слова для определения типа
        person_context = {'мистер', 'господин', 'доктор', 'профессор', 'директор'}
        org_context = {'компания', 'корпорация', 'фирма', 'организация', 'банк'}
        location_context = {'город', 'улица', 'область', 'район', 'страна'}
        
        all_context = context_before + context_after
        
        if any(word in person_context for word in all_context):
            return EntityType.PERSON
        elif any(word in org_context for word in all_context):
            return EntityType.ORGANIZATION
        elif any(word in location_context for word in all_context):
            return EntityType.LOCATION
        
        return EntityType.UNKNOWN
    
    def _filter_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Фильтрация пересекающихся сущностей (оставляем с большей уверенностью)."""
        if not entities:
            return entities
        
        # Сортируем по позиции
        sorted_entities = sorted(entities, key=lambda e: (e.start, e.end))
        
        filtered = []
        for entity in sorted_entities:
            # Проверяем пересечение с уже добавленными
            overlaps = False
            for existing in filtered:
                if (entity.start < existing.end and entity.end > existing.start):
                    # Есть пересечение
                    if entity.confidence > existing.confidence:
                        # Новая сущность лучше - заменяем
                        filtered.remove(existing)
                        filtered.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered


class SpacyEntityExtractor(IEntityExtractor):
    """Извлечение сущностей с использованием spaCy."""
    
    def __init__(self, model_name: str = "ru_core_news_sm"):
        self.logger = get_logger("spacy_entity_extractor")
        self.model_name = model_name
        self.nlp = None
        
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Загружена модель spaCy: {model_name}")
        except ImportError:
            self.logger.warning("spaCy не установлен, используется fallback")
            self.fallback = RuleBasedEntityExtractor()
        except OSError:
            self.logger.warning(f"Модель {model_name} не найдена, используется fallback")
            self.fallback = RuleBasedEntityExtractor()
        
        # Маппинг типов spaCy в наши типы
        self.entity_type_mapping = {
            'PERSON': EntityType.PERSON,
            'PER': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'LOC': EntityType.LOCATION,
            'GPE': EntityType.LOCATION,  # Geopolitical entity
            'DATE': EntityType.DATE,
            'TIME': EntityType.TIME,
            'MONEY': EntityType.MONEY,
            'MISC': EntityType.CONCEPT,
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Извлечение сущностей с использованием spaCy NER."""
        if self.nlp is None:
            return self.fallback.extract_entities(text)
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_type = self.entity_type_mapping.get(ent.label_, EntityType.UNKNOWN)
            
            entity = Entity(
                text=ent.text,
                entity_type=entity_type,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.85,  # spaCy обычно дает хорошие результаты
                metadata={
                    'method': 'spacy_ner',
                    'spacy_label': ent.label_,
                    'spacy_kb_id': ent.kb_id_ if hasattr(ent, 'kb_id_') else None
                }
            )
            entities.append(entity)
        
        return entities
    
    def extract_entities_from_tokens(self, tokens: List[Token]) -> List[Entity]:
        """Извлечение сущностей из токенов."""
        # Восстанавливаем текст из токенов
        text = ' '.join(token.text for token in tokens)
        return self.extract_entities(text)


class HybridEntityExtractor(IEntityExtractor):
    """Гибридный экстрактор, комбинирующий несколько подходов."""
    
    def __init__(self, use_spacy: bool = True, use_rules: bool = True):
        self.logger = get_logger("hybrid_entity_extractor")
        
        self.extractors = []
        
        if use_rules:
            self.rule_extractor = RuleBasedEntityExtractor()
            self.extractors.append(('rules', self.rule_extractor))
        
        if use_spacy:
            self.spacy_extractor = SpacyEntityExtractor()
            if self.spacy_extractor.nlp is not None:
                self.extractors.append(('spacy', self.spacy_extractor))
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Извлечение сущностей с комбинированием результатов."""
        all_entities = []
        
        # Получаем сущности от всех экстракторов
        for extractor_name, extractor in self.extractors:
            try:
                entities = extractor.extract_entities(text)
                
                # Добавляем информацию об источнике
                for entity in entities:
                    if 'extractors' not in entity.metadata:
                        entity.metadata['extractors'] = []
                    entity.metadata['extractors'].append(extractor_name)
                
                all_entities.extend(entities)
                
            except Exception as e:
                self.logger.warning(f"Ошибка в экстракторе {extractor_name}: {e}")
        
        # Объединяем и улучшаем результаты
        merged_entities = self._merge_entities(all_entities)
        
        return merged_entities
    
    def extract_entities_from_tokens(self, tokens: List[Token]) -> List[Entity]:
        """Извлечение сущностей из токенов."""
        text = ' '.join(token.text for token in tokens)
        return self.extract_entities(text)
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Объединение сущностей от разных экстракторов."""
        if not entities:
            return entities
        
        # Группируем пересекающиеся сущности
        groups = []
        
        for entity in entities:
            added_to_group = False
            
            for group in groups:
                # Проверяем пересечение с любой сущностью в группе
                for group_entity in group:
                    if self._entities_overlap(entity, group_entity):
                        group.append(entity)
                        added_to_group = True
                        break
                
                if added_to_group:
                    break
            
            if not added_to_group:
                groups.append([entity])
        
        # Объединяем сущности в каждой группе
        merged = []
        
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged_entity = self._merge_entity_group(group)
                merged.append(merged_entity)
        
        return merged
    
    def _entities_overlap(self, e1: Entity, e2: Entity) -> bool:
        """Проверка пересечения двух сущностей."""
        return not (e1.end <= e2.start or e2.end <= e1.start)
    
    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """Объединение группы пересекающихся сущностей."""
        # Выбираем лучшую сущность как основу
        best_entity = max(entities, key=lambda e: e.confidence)
        
        # Собираем метаданные от всех сущностей
        all_extractors = set()
        all_methods = set()
        
        for entity in entities:
            if 'extractors' in entity.metadata:
                all_extractors.update(entity.metadata['extractors'])
            if 'method' in entity.metadata:
                all_methods.add(entity.metadata['method'])
        
        # Повышаем уверенность если несколько экстракторов согласны
        confidence_boost = min(0.2, 0.05 * len(entities))
        final_confidence = min(1.0, best_entity.confidence + confidence_boost)
        
        # Создаем объединенную сущность
        merged_entity = Entity(
            text=best_entity.text,
            entity_type=best_entity.entity_type,
            start=best_entity.start,
            end=best_entity.end,
            confidence=final_confidence,
            metadata={
                'extractors': list(all_extractors),
                'methods': list(all_methods),
                'merged_from': len(entities),
                'original_confidences': [e.confidence for e in entities]
            }
        )
        
        return merged_entity