"""Основная реализация NLP процессора для модуля NLP."""

import time
from typing import Dict, List, Optional, Any, Union
from neurograph.nlp.base import (
    INLProcessor, ProcessingResult, Sentence, Entity, Relation, Token
)
from neurograph.nlp.tokenization import SimpleTokenizer, SpacyTokenizer
from neurograph.nlp.entity_extraction import HybridEntityExtractor
from neurograph.nlp.relation_extraction import HybridRelationExtractor
from neurograph.nlp.text_generation import HybridTextGenerator
from neurograph.core.logging import get_logger
from neurograph.core.cache import cached


class StandardNLProcessor(INLProcessor):
    """Стандартная реализация NLP процессора."""
    
    def __init__(self, 
                 tokenizer_type: str = "simple",
                 use_spacy: bool = True,
                 language: str = "ru",
                 cache_results: bool = True):
        
        self.language = language
        self.cache_results = cache_results
        self.logger = get_logger("standard_nl_processor")
        
        # Инициализация компонентов
        self._init_tokenizer(tokenizer_type, language)
        self._init_entity_extractor(use_spacy, language)
        self._init_relation_extractor(use_spacy, language)
        self._init_text_generator()
        
        # Статистика
        self.stats = {
            'texts_processed': 0,
            'entities_extracted': 0,
            'relations_extracted': 0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info(f"Инициализирован NLP процессор для языка: {language}")
    
    def _init_tokenizer(self, tokenizer_type: str, language: str):
        """Инициализация токенизатора."""
        if tokenizer_type == "spacy":
            model_name = "ru_core_news_sm" if language == "ru" else "en_core_web_sm"
            self.tokenizer = SpacyTokenizer(model_name)
        else:
            self.tokenizer = SimpleTokenizer(language)
        
        self.logger.info(f"Инициализирован токенизатор: {tokenizer_type}")
    
    def _init_entity_extractor(self, use_spacy: bool, language: str):
        """Инициализация экстрактора сущностей."""
        self.entity_extractor = HybridEntityExtractor(
            use_spacy=use_spacy,
            use_rules=True
        )
        self.logger.info("Инициализирован экстрактор сущностей")
    
    def _init_relation_extractor(self, use_spacy: bool, language: str):
        """Инициализация экстрактора отношений."""
        self.relation_extractor = HybridRelationExtractor(
            use_spacy=use_spacy,
            use_patterns=True,
            use_rules=True
        )
        self.logger.info("Инициализирован экстрактор отношений")
    
    def _init_text_generator(self):
        """Инициализация генератора текста."""
        self.text_generator = HybridTextGenerator(
            use_templates=True,
            use_markov=False,
            use_rules=True
        )
        self.logger.info("Инициализирован генератор текста")
    
    @cached(ttl=300)  # Кешируем на 5 минут
    def process_text(self, text: str, extract_entities: bool = True,
                    extract_relations: bool = True,
                    analyze_sentiment: bool = False) -> ProcessingResult:
        """Полная обработка текста с извлечением сущностей и отношений."""
        
        start_time = time.time()
        
        try:
            # Нормализация текста
            normalized_text = self.normalize_text(text)
            
            # Токенизация
            tokens = self.tokenizer.tokenize(normalized_text)
            
            # Разбиение на предложения
            sentence_texts = self.tokenizer.sent_tokenize(normalized_text)
            
            # Обработка предложений
            sentences = []
            all_entities = []
            all_relations = []
            
            char_offset = 0
            
            for sent_text in sentence_texts:
                # Находим позицию предложения в тексте
                sent_start = normalized_text.find(sent_text, char_offset)
                sent_end = sent_start + len(sent_text)
                char_offset = sent_end
                
                # Токены для этого предложения
                sent_tokens = [t for t in tokens if sent_start <= t.start < sent_end]
                
                # Создаем объект предложения
                sentence = Sentence(
                    text=sent_text,
                    tokens=sent_tokens,
                    start=sent_start,
                    end=sent_end
                )
                
                # Извлечение сущностей для предложения
                if extract_entities:
                    sentence.entities = self.entity_extractor.extract_entities(sent_text)
                    all_entities.extend(sentence.entities)
                
                # Анализ тональности
                if analyze_sentiment:
                    sentence.sentiment = self._analyze_sentiment(sent_text)
                
                # Извлечение отношений для предложения
                if extract_relations and sentence.entities:
                    sentence.relations = self.relation_extractor.extract_relations_from_sentence(sentence)
                    all_relations.extend(sentence.relations)
                
                sentences.append(sentence)
            
            # Определение языка
            detected_language = self.get_language(normalized_text)
            
            # Время обработки
            processing_time = time.time() - start_time
            
            # Создание результата
            result = ProcessingResult(
                original_text=text,
                sentences=sentences,
                entities=all_entities,
                relations=all_relations,
                tokens=tokens,
                processing_time=processing_time,
                language=detected_language,
                metadata={
                    'tokenizer': type(self.tokenizer).__name__,
                    'entity_extractor': type(self.entity_extractor).__name__,
                    'relation_extractor': type(self.relation_extractor).__name__,
                    'entities_count': len(all_entities),
                    'relations_count': len(all_relations),
                    'sentences_count': len(sentences)
                }
            )
            
            # Обновление статистики
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки текста: {e}")
            
            # Возвращаем минимальный результат при ошибке
            return ProcessingResult(
                original_text=text,
                sentences=[],
                entities=[],
                relations=[],
                tokens=[],
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def extract_knowledge(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение структурированных знаний из текста."""
        
        # Полная обработка текста
        result = self.process_text(text, extract_entities=True, extract_relations=True)
        
        knowledge_items = []
        
        # Извлекаем знания из отношений
        for relation in result.relations:
            knowledge_item = {
                'type': 'relation',
                'subject': {
                    'text': relation.subject.text,
                    'type': relation.subject.entity_type.value,
                    'confidence': relation.subject.confidence
                },
                'predicate': relation.predicate.value,
                'object': {
                    'text': relation.object.text,
                    'type': relation.object.entity_type.value,
                    'confidence': relation.object.confidence
                },
                'confidence': relation.confidence,
                'source_text': relation.text_span or text,
                'metadata': relation.metadata
            }
            knowledge_items.append(knowledge_item)
        
        # Извлекаем знания из сущностей (факты о сущностях)
        for entity in result.entities:
            if entity.entity_type.value in ['CONCEPT', 'FACT']:
                knowledge_item = {
                    'type': 'entity',
                    'text': entity.text,
                    'entity_type': entity.entity_type.value,
                    'confidence': entity.confidence,
                    'source_text': text,
                    'metadata': entity.metadata
                }
                knowledge_items.append(knowledge_item)
        
        # Извлекаем знания из предложений (определения и факты)
        for sentence in result.sentences:
            # Простая эвристика для определений
            if self._is_definition_sentence(sentence.text):
                knowledge_item = {
                    'type': 'definition',
                    'text': sentence.text,
                    'confidence': 0.8,
                    'source_text': text,
                    'entities': [
                        {
                            'text': e.text,
                            'type': e.entity_type.value,
                            'confidence': e.confidence
                        } for e in sentence.entities
                    ]
                }
                knowledge_items.append(knowledge_item)
        
        self.logger.info(f"Извлечено {len(knowledge_items)} элементов знаний")
        
        return knowledge_items
    
    def normalize_text(self, text: str) -> str:
        """Нормализация текста."""
        if not text:
            return ""
        
        # Удаление лишних пробелов
        import re
        normalized = re.sub(r'\s+', ' ', text.strip())
        
        # Нормализация кавычек
        normalized = normalized.replace('«', '"').replace('»', '"')
        normalized = normalized.replace('"', '"').replace('"', '"')
        
        # Нормализация тире
        normalized = normalized.replace('—', '-').replace('–', '-')
        
        return normalized
    
    def get_language(self, text: str) -> str:
        """Определение языка текста."""
        if not text:
            return "unknown"
        
        # Простая эвристика на основе символов
        cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        latin_count = sum(1 for char in text if 'a' <= char.lower() <= 'z')
        
        total_letters = cyrillic_count + latin_count
        
        if total_letters == 0:
            return "unknown"
        
        cyrillic_ratio = cyrillic_count / total_letters
        
        if cyrillic_ratio > 0.5:
            return "ru"
        elif cyrillic_ratio < 0.1:
            return "en"
        else:
            return "mixed"
    
    def _analyze_sentiment(self, text: str) -> float:
        """Простой анализ тональности текста."""
        
        # Словари для простого анализа тональности
        positive_words = {
            'хорошо', 'отлично', 'прекрасно', 'замечательно', 'великолепно',
            'нравится', 'люблю', 'радует', 'восхищает', 'впечатляет',
            'good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'like'
        }
        
        negative_words = {
            'плохо', 'ужасно', 'отвратительно', 'ненавижу', 'разочарован',
            'грустно', 'печально', 'расстроен', 'злой', 'недоволен',
            'bad', 'terrible', 'awful', 'hate', 'disappointed', 'sad', 'angry'
        }
        
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Простая формула: (положительные - отрицательные) / общее количество
        sentiment = (positive_count - negative_count) / total_words
        
        # Нормализуем в диапазон [-1, 1]
        return max(-1.0, min(1.0, sentiment * 10))
    
    def _is_definition_sentence(self, sentence: str) -> bool:
        """Проверка, является ли предложение определением."""
        
        definition_patterns = [
            r'.+\s+(?:это|есть|является)\s+.+',
            r'.+\s*-\s*это\s+.+',
            r'.+\s+представляет\s+собой\s+.+',
            r'.+\s+называется\s+.+',
            r'.+\s+определяется\s+как\s+.+'
        ]
        
        import re
        for pattern in definition_patterns:
            if re.search(pattern, sentence.lower()):
                return True
        
        return False
    
    def _update_stats(self, result: ProcessingResult):
        """Обновление статистики обработки."""
        self.stats['texts_processed'] += 1
        self.stats['entities_extracted'] += len(result.entities)
        self.stats['relations_extracted'] += len(result.relations)
        
        # Обновление среднего времени обработки
        current_avg = self.stats['avg_processing_time']
        count = self.stats['texts_processed']
        
        self.stats['avg_processing_time'] = (
            (current_avg * (count - 1) + result.processing_time) / count
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики работы процессора."""
        return {
            'processing_stats': self.stats.copy(),
            'components': {
                'tokenizer': type(self.tokenizer).__name__,
                'entity_extractor': type(self.entity_extractor).__name__,
                'relation_extractor': type(self.relation_extractor).__name__,
                'text_generator': type(self.text_generator).__name__
            },
            'language': self.language,
            'cache_enabled': self.cache_results
        }
    
    def generate_text(self, template: str, params: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Генерация текста с использованием встроенного генератора."""
        return self.text_generator.generate_text(template, params, context)
    
    def generate_response(self, query: str, knowledge: Dict[str, Any],
                         style: Optional[str] = None) -> str:
        """Генерация ответа на запрос."""
        return self.text_generator.generate_response(query, knowledge, style)


class LightweightNLProcessor(INLProcessor):
    """Облегченная версия NLP процессора для ограниченных ресурсов."""
    
    def __init__(self, language: str = "ru"):
        self.language = language
        self.logger = get_logger("lightweight_nl_processor")
        
        # Используем только простые компоненты
        from neurograph.nlp.tokenization import SimpleTokenizer
        from neurograph.nlp.entity_extraction import RuleBasedEntityExtractor
        from neurograph.nlp.relation_extraction import RuleBasedRelationExtractor
        from neurograph.nlp.text_generation import RuleBasedTextGenerator
        
        self.tokenizer = SimpleTokenizer(language)
        self.entity_extractor = RuleBasedEntityExtractor(language)
        self.relation_extractor = RuleBasedRelationExtractor()
        self.text_generator = RuleBasedTextGenerator()
        
        self.stats = {'texts_processed': 0}
        
        self.logger.info("Инициализирован облегченный NLP процессор")
    
    def process_text(self, text: str, extract_entities: bool = True,
                    extract_relations: bool = True,
                    analyze_sentiment: bool = False) -> ProcessingResult:
        """Упрощенная обработка текста."""
        
        start_time = time.time()
        
        # Нормализация
        normalized_text = self.normalize_text(text)
        
        # Токенизация
        tokens = self.tokenizer.tokenize(normalized_text)
        
        # Простое разбиение на предложения
        sentence_texts = self.tokenizer.sent_tokenize(normalized_text)
        
        sentences = []
        all_entities = []
        all_relations = []
        
        for i, sent_text in enumerate(sentence_texts):
            sentence = Sentence(
                text=sent_text,
                tokens=[],  # Упрощаем - не привязываем токены к предложениям
                start=0,
                end=len(sent_text)
            )
            
            if extract_entities:
                sentence.entities = self.entity_extractor.extract_entities(sent_text)
                all_entities.extend(sentence.entities)
            
            if extract_relations and sentence.entities:
                sentence.relations = self.relation_extractor.extract_relations_from_sentence(sentence)
                all_relations.extend(sentence.relations)
            
            sentences.append(sentence)
        
        processing_time = time.time() - start_time
        
        result = ProcessingResult(
            original_text=text,
            sentences=sentences,
            entities=all_entities,
            relations=all_relations,
            tokens=tokens,
            processing_time=processing_time,
            language=self.get_language(text),
            metadata={'processor': 'lightweight'}
        )
        
        self.stats['texts_processed'] += 1
        
        return result
    
    def extract_knowledge(self, text: str) -> List[Dict[str, Any]]:
        """Упрощенное извлечение знаний."""
        result = self.process_text(text)
        
        knowledge_items = []
        
        # Простое извлечение из отношений
        for relation in result.relations:
            knowledge_items.append({
                'type': 'relation',
                'subject': relation.subject.text,
                'predicate': relation.predicate.value,
                'object': relation.object.text,
                'confidence': relation.confidence
            })
        
        return knowledge_items
    
    def normalize_text(self, text: str) -> str:
        """Простая нормализация текста."""
        if not text:
            return ""
        
        import re
        return re.sub(r'\s+', ' ', text.strip())
    
    def get_language(self, text: str) -> str:
        """Простое определение языка."""
        if not text:
            return "unknown"
        
        # Подсчет кириллических символов
        cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        
        return "ru" if cyrillic_count > len(text) * 0.3 else "en"


class AdvancedNLProcessor(StandardNLProcessor):
    """Продвинутая версия NLP процессора с дополнительными возможностями."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.logger = get_logger("advanced_nl_processor")
        
        # Дополнительные компоненты
        self._init_advanced_features()
        
        self.logger.info("Инициализирован продвинутый NLP процессор")
    
    def _init_advanced_features(self):
        """Инициализация продвинутых возможностей."""
        
        # Кеш для часто используемых результатов
        self.processing_cache = {}
        
        # Статистика по типам сущностей
        self.entity_type_stats = {}
        
        # Статистика по типам отношений
        self.relation_type_stats = {}
        
        # Пользовательские правила и паттерны
        self.custom_patterns = {
            'entities': [],
            'relations': []
        }
    
    def add_custom_entity_pattern(self, pattern: str, entity_type: str):
        """Добавление пользовательского паттерна для сущностей."""
        self.custom_patterns['entities'].append({
            'pattern': pattern,
            'entity_type': entity_type
        })
        self.logger.info(f"Добавлен паттерн сущности: {pattern} -> {entity_type}")
    
    def add_custom_relation_pattern(self, pattern: str, relation_type: str):
        """Добавление пользовательского паттерна для отношений."""
        self.custom_patterns['relations'].append({
            'pattern': pattern,
            'relation_type': relation_type
        })
        self.logger.info(f"Добавлен паттерн отношения: {pattern} -> {relation_type}")
    
    def process_text(self, text: str, **kwargs) -> ProcessingResult:
        """Продвинутая обработка с применением пользовательских правил."""
        
        # Базовая обработка
        result = super().process_text(text, **kwargs)
        
        # Применение пользовательских паттернов
        if self.custom_patterns['entities']:
            additional_entities = self._apply_custom_entity_patterns(text)
            result.entities.extend(additional_entities)
        
        if self.custom_patterns['relations'] and result.entities:
            additional_relations = self._apply_custom_relation_patterns(text, result.entities)
            result.relations.extend(additional_relations)
        
        # Обновление расширенной статистики
        self._update_advanced_stats(result)
        
        return result
    
    def _apply_custom_entity_patterns(self, text: str) -> List[Entity]:
        """Применение пользовательских паттернов для сущностей."""
        import re
        from neurograph.nlp.base import EntityType
        
        additional_entities = []
        
        for pattern_info in self.custom_patterns['entities']:
            pattern = pattern_info['pattern']
            entity_type_str = pattern_info['entity_type']
            
            try:
                # Преобразуем строку в EntityType
                entity_type = EntityType(entity_type_str) if entity_type_str in [e.value for e in EntityType] else EntityType.CONCEPT
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                        metadata={'source': 'custom_pattern', 'pattern': pattern}
                    )
                    additional_entities.append(entity)
                    
            except Exception as e:
                self.logger.warning(f"Ошибка применения паттерна {pattern}: {e}")
        
        return additional_entities
    
    def _apply_custom_relation_patterns(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Применение пользовательских паттернов для отношений."""
        import re
        from neurograph.nlp.base import RelationType
        
        additional_relations = []
        
        for pattern_info in self.custom_patterns['relations']:
            pattern = pattern_info['pattern']
            relation_type_str = pattern_info['relation_type']
            
            try:
                # Преобразуем строку в RelationType
                relation_type = RelationType(relation_type_str) if relation_type_str in [r.value for r in RelationType] else RelationType.RELATED_TO
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Простая логика: создаем временные сущности из групп паттерна
                    if match.groups() and len(match.groups()) >= 2:
                        subject_text = match.group(1).strip()
                        object_text = match.group(2).strip()
                        
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
                            metadata={'source': 'custom_pattern', 'pattern': pattern}
                        )
                        
                        additional_relations.append(relation)
                        
            except Exception as e:
                self.logger.warning(f"Ошибка применения паттерна отношения {pattern}: {e}")
        
        return additional_relations
    
    def _update_advanced_stats(self, result: ProcessingResult):
        """Обновление расширенной статистики."""
        
        # Статистика по типам сущностей
        for entity in result.entities:
            entity_type = entity.entity_type.value
            if entity_type not in self.entity_type_stats:
                self.entity_type_stats[entity_type] = 0
            self.entity_type_stats[entity_type] += 1
        
        # Статистика по типам отношений
        for relation in result.relations:
            relation_type = relation.predicate.value
            if relation_type not in self.relation_type_stats:
                self.relation_type_stats[relation_type] = 0
            self.relation_type_stats[relation_type] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение расширенной статистики."""
        base_stats = super().get_statistics()
        
        base_stats.update({
            'entity_type_distribution': self.entity_type_stats.copy(),
            'relation_type_distribution': self.relation_type_stats.copy(),
            'custom_patterns': {
                'entities': len(self.custom_patterns['entities']),
                'relations': len(self.custom_patterns['relations'])
            }
        })
        
        return base_stats
    
    def train_custom_patterns(self, training_texts: List[str], annotations: List[Dict[str, Any]]):
        """Обучение пользовательских паттернов на размеченных данных."""
        
        self.logger.info(f"Обучение на {len(training_texts)} текстах с аннотациями")
        
        # Простое извлечение паттернов из аннотированных данных
        for text, annotation in zip(training_texts, annotations):
            
            # Обучение паттернов сущностей
            if 'entities' in annotation:
                for entity_info in annotation['entities']:
                    entity_text = entity_info['text']
                    entity_type = entity_info['type']
                    
                    # Создаем простой паттерн на основе точного текста
                    pattern = re.escape(entity_text)
                    self.add_custom_entity_pattern(pattern, entity_type)
            
            # Обучение паттернов отношений
            if 'relations' in annotation:
                for relation_info in annotation['relations']:
                    subject = relation_info['subject']
                    predicate = relation_info['predicate']
                    obj = relation_info['object']
                    
                    # Создаем паттерн отношения
                    pattern = f"({re.escape(subject)}).+?({re.escape(obj)})"
                    self.add_custom_relation_pattern(pattern, predicate)
        
        self.logger.info("Обучение пользовательских паттернов завершено")