"""Модуль семантической обработки текста."""

import re
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import Counter
from neurograph.core.logging import get_logger
from ..base import ISemanticProcessor, NLPError


class BasicSemanticProcessor(ISemanticProcessor):
    """Базовый семантический процессор."""
    
    def __init__(self, language: str = "ru", vector_provider=None):
        """Инициализация семантического процессора.
        
        Args:
            language: Язык обработки.
            vector_provider: Провайдер векторных представлений.
        """
        self.language = language
        self.vector_provider = vector_provider
        self.logger = get_logger("basic_semantic_processor")
        
        # Стоп-слова
        self.stop_words = self._load_stop_words()
        
        # Кеш для вычислений
        self._similarity_cache = {}
        
        self.logger.info(f"Инициализирован базовый семантический процессор для {language}")
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """Извлекает ключевые слова из текста с использованием TF-IDF."""
        try:
            # Токенизация и очистка
            words = self._tokenize_and_clean(text)
            
            if not words:
                return []
            
            # Подсчет частот слов (TF)
            word_counts = Counter(words)
            total_words = len(words)
            
            # Вычисление TF-IDF (упрощенная версия)
            keyword_scores = {}
            
            for word, count in word_counts.items():
                # Term Frequency
                tf = count / total_words
                
                # Inverse Document Frequency (упрощенная оценка)
                # Используем длину слова и позицию как примитивную IDF
                idf = math.log(len(text) / (len(word) + 1)) + 1
                
                # TF-IDF
                tfidf_score = tf * idf
                
                # Дополнительные веса
                length_bonus = min(len(word) / 10, 1.0)  # Бонус за длину слова
                position_bonus = self._calculate_position_bonus(text, word)
                
                final_score = tfidf_score * (1 + length_bonus + position_bonus)
                keyword_scores[word] = final_score
            
            # Сортировка и возврат топ-N
            sorted_keywords = sorted(
                keyword_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return sorted_keywords[:max_keywords]
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения ключевых слов: {e}")
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет семантическое сходство между текстами."""
        try:
            # Проверяем кеш
            cache_key = (text1, text2)
            if cache_key in self._similarity_cache:
                return self._similarity_cache[cache_key]
            
            # Если есть векторный провайдер, используем его
            if self.vector_provider:
                similarity = self._calculate_vector_similarity(text1, text2)
            else:
                # Лексическое сходство
                similarity = self._calculate_lexical_similarity(text1, text2)
            
            # Кешируем результат
            self._similarity_cache[cache_key] = similarity
            return similarity
            
        except Exception as e:
            self.logger.error(f"Ошибка вычисления сходства: {e}")
            return 0.0
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """Анализирует тональность текста."""
        try:
            # Словари тональности
            positive_words = self._get_positive_words()
            negative_words = self._get_negative_words()
            intensifiers = self._get_intensifiers()
            negations = self._get_negations()
            
            words = self._tokenize_and_clean(text, keep_stop_words=True)
            
            positive_score = 0.0
            negative_score = 0.0
            total_sentiment_words = 0
            
            i = 0
            while i < len(words):
                word = words[i].lower()
                
                # Проверяем интенсификаторы
                intensifier_multiplier = 1.0
                if i > 0 and words[i-1].lower() in intensifiers:
                    intensifier_multiplier = 1.5
                
                # Проверяем отрицания
                negation_found = False
                if i > 0 and words[i-1].lower() in negations:
                    negation_found = True
                elif i > 1 and words[i-2].lower() in negations:
                    negation_found = True
                
                # Оценка тональности слова
                if word in positive_words:
                    score = positive_words[word] * intensifier_multiplier
                    if negation_found:
                        negative_score += score
                    else:
                        positive_score += score
                    total_sentiment_words += 1
                    
                elif word in negative_words:
                    score = negative_words[word] * intensifier_multiplier
                    if negation_found:
                        positive_score += score
                    else:
                        negative_score += score
                    total_sentiment_words += 1
                
                i += 1
            
            # Нормализация и вычисление итоговой тональности
            if total_sentiment_words == 0:
                return 0.0, 0.0
            
            sentiment = (positive_score - negative_score) / total_sentiment_words
            confidence = min(total_sentiment_words / len(words), 1.0)
            
            return sentiment, confidence
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа тональности: {e}")
            return 0.0, 0.0
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Создает краткое содержание текста."""
        try:
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                return ""
            
            if len(sentences) == 1:
                sentence = sentences[0]
                if len(sentence) <= max_length:
                    return sentence
                else:
                    return sentence[:max_length-3] + "..."
            
            # Оценка важности предложений
            sentence_scores = self._score_sentences(sentences, text)
            
            # Выбор наиболее важных предложений
            scored_sentences = list(zip(sentences, sentence_scores))
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Формирование резюме
            summary_sentences = []
            current_length = 0
            
            for sentence, score in scored_sentences:
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence) + 1  # +1 для пробела
                else:
                    break
            
            if not summary_sentences:
                # Если ни одно предложение не помещается, берем первое и обрезаем
                first_sentence = sentences[0]
                return first_sentence[:max_length-3] + "..."
            
            return " ".join(summary_sentences)
            
        except Exception as e:
            self.logger.error(f"Ошибка суммаризации: {e}")
            return text[:max_length-3] + "..." if len(text) > max_length else text
    
    def _tokenize_and_clean(self, text: str, keep_stop_words: bool = False) -> List[str]:
        """Токенизирует и очищает текст."""
        # Приведение к нижнему регистру и удаление пунктуации
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Токенизация
        words = cleaned_text.split()
        
        # Фильтрация
        if not keep_stop_words:
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return words
    
    def _calculate_position_bonus(self, text: str, word: str) -> float:
        """Вычисляет бонус за позицию слова в тексте."""
        first_occurrence = text.lower().find(word.lower())
        if first_occurrence == -1:
            return 0.0
        
        # Бонус за появление в начале текста
        text_length = len(text)
        position_ratio = first_occurrence / text_length
        
        if position_ratio < 0.1:  # Первые 10% текста
            return 0.3
        elif position_ratio < 0.3:  # Первые 30% текста
            return 0.1
        else:
            return 0.0
    
    def _calculate_vector_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет сходство с использованием векторов."""
        try:
            # Получаем векторы текстов
            if hasattr(self.vector_provider, 'encode'):
                vector1 = self.vector_provider.encode(text1)
                vector2 = self.vector_provider.encode(text2)
            else:
                # Пробуем получить векторы для отдельных слов
                words1 = self._tokenize_and_clean(text1)
                words2 = self._tokenize_and_clean(text2)
                
                vector1 = self._get_text_vector_from_words(words1)
                vector2 = self._get_text_vector_from_words(words2)
            
            if vector1 is not None and vector2 is not None:
                return self._cosine_similarity(vector1, vector2)
            else:
                return self._calculate_lexical_similarity(text1, text2)
                
        except Exception as e:
            self.logger.warning(f"Ошибка векторного сходства: {e}")
            return self._calculate_lexical_similarity(text1, text2)
    
    def _get_text_vector_from_words(self, words: List[str]):
        """Получает вектор текста из векторов слов."""
        import numpy as np
        
        word_vectors = []
        for word in words:
            vector = self.vector_provider.get_vector(word)
            if vector is not None:
                word_vectors.append(vector)
        
        if word_vectors:
            # Усредняем векторы слов
            return np.mean(word_vectors, axis=0)
        else:
            return None
    
    def _cosine_similarity(self, vector1, vector2) -> float:
        """Вычисляет косинусное сходство векторов."""
        import numpy as np
        
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет лексическое сходство между текстами."""
        words1 = set(self._tokenize_and_clean(text1))
        words2 = set(self._tokenize_and_clean(text2))
        
        if not words1 or not words2:
            return 0.0
        
        # Коэффициент Жаккара
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Дополнительный учет длины пересечения
        length_factor = len(intersection) / min(len(words1), len(words2))
        
        return (jaccard_similarity + length_factor) / 2
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        # Улучшенное разбиение на предложения
        sentence_endings = re.split(r'[.!?]+', text)
        
        sentences = []
        for sentence in sentence_endings:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Минимальная длина предложения
                sentences.append(sentence)
        
        return sentences
    
    def _score_sentences(self, sentences: List[str], full_text: str) -> List[float]:
        """Оценивает важность предложений для суммаризации."""
        scores = []
        
        # Получаем ключевые слова всего текста
        keywords = dict(self.extract_keywords(full_text, max_keywords=20))
        
        for sentence in sentences:
            score = 0.0
            sentence_words = self._tokenize_and_clean(sentence)
            
            # Оценка на основе ключевых слов
            for word in sentence_words:
                if word in keywords:
                    score += keywords[word]
            
            # Нормализация по длине предложения
            if sentence_words:
                score = score / len(sentence_words)
            
            # Бонус за позицию (первые и последние предложения важнее)
            position_in_text = sentences.index(sentence)
            total_sentences = len(sentences)
            
            if position_in_text == 0:  # Первое предложение
                score *= 1.2
            elif position_in_text == total_sentences - 1:  # Последнее предложение
                score *= 1.1
            
            scores.append(score)
        
        return scores
    
    def _load_stop_words(self) -> Set[str]:
        """Загружает стоп-слова для языка."""
        if self.language == "ru":
            return {
                'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
                'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
                'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
                'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
                'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть'
            }
        else:  # English
            return {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
    
    def _get_positive_words(self) -> Dict[str, float]:
        """Возвращает словарь позитивных слов с весами."""
        if self.language == "ru":
            return {
                'хорошо': 1.0, 'отлично': 1.2, 'прекрасно': 1.1, 'замечательно': 1.1, 'великолепно': 1.2,
                'положительно': 0.8, 'успешно': 1.0, 'эффективно': 0.9, 'качественно': 0.9, 'полезно': 0.8,
                'радость': 1.1, 'счастье': 1.2, 'любовь': 1.1, 'удовольствие': 1.0, 'восторг': 1.2,
                'нравится': 0.8, 'люблю': 1.0, 'восхищает': 1.1, 'впечатляет': 1.0, 'радует': 0.9,
                'интересно': 0.7, 'важно': 0.6, 'значимо': 0.7, 'ценно': 0.8, 'достойно': 0.7
            }
        else:  # English
            return {
                'good': 1.0, 'excellent': 1.2, 'wonderful': 1.1, 'amazing': 1.2, 'great': 1.0,
                'positive': 0.8, 'successful': 1.0, 'effective': 0.9, 'quality': 0.8, 'useful': 0.8,
                'joy': 1.1, 'happiness': 1.2, 'love': 1.1, 'pleasure': 1.0, 'delight': 1.1,
                'like': 0.7, 'enjoy': 0.9, 'admire': 1.0, 'impress': 1.0, 'amazing': 1.2,
                'interesting': 0.6, 'important': 0.5, 'significant': 0.7, 'valuable': 0.8, 'worthy': 0.7
            }
    
    def _get_negative_words(self) -> Dict[str, float]:
        """Возвращает словарь негативных слов с весами."""
        if self.language == "ru":
            return {
                'плохо': 1.0, 'ужасно': 1.2, 'отвратительно': 1.2, 'негативно': 0.8, 'неудачно': 0.9,
                'провал': 1.1, 'ошибка': 0.8, 'проблема': 0.7, 'недостаток': 0.8, 'дефект': 0.9,
                'грусть': 1.0, 'печаль': 1.0, 'злость': 1.1, 'раздражение': 1.0, 'разочарование': 1.0,
                'не нравится': 0.8, 'ненавижу': 1.2, 'отвращает': 1.1, 'расстраивает': 0.9, 'злит': 1.0,
                'сложно': 0.6, 'трудно': 0.6, 'невозможно': 0.9, 'неприемлемо': 1.0, 'недопустимо': 1.0
            }
        else:  # English
            return {
                'bad': 1.0, 'terrible': 1.2, 'awful': 1.2, 'negative': 0.8, 'unsuccessful': 0.9,
                'failure': 1.1, 'error': 0.8, 'problem': 0.7, 'flaw': 0.8, 'defect': 0.9,
                'sadness': 1.0, 'sorrow': 1.0, 'anger': 1.1, 'irritation': 1.0, 'disappointment': 1.0,
                'dislike': 0.8, 'hate': 1.2, 'disgusts': 1.1, 'upsets': 0.9, 'angers': 1.0,
                'difficult': 0.6, 'hard': 0.5, 'impossible': 0.9, 'unacceptable': 1.0, 'inadmissible': 1.0
            }
    
    def _get_intensifiers(self) -> Set[str]:
        """Возвращает набор слов-интенсификаторов."""
        if self.language == "ru":
            return {'очень', 'крайне', 'чрезвычайно', 'исключительно', 'невероятно', 'чертовски', 'довольно'}
        else:  # English
            return {'very', 'extremely', 'incredibly', 'exceptionally', 'remarkably', 'quite', 'really'}
    
    def _get_negations(self) -> Set[str]:
        """Возвращает набор отрицательных слов."""
        if self.language == "ru":
            return {'не', 'ни', 'нет', 'никогда', 'никак', 'нисколько'}
        else:  # English
            return {'not', 'no', 'never', 'none', 'neither', 'nobody', 'nothing'}


class AdvancedSemanticProcessor(BasicSemanticProcessor):
    """Продвинутый семантический процессор с дополнительными возможностями."""
    
    def __init__(self, language: str = "ru", vector_provider=None, enable_topic_modeling: bool = True):
        """Инициализация продвинутого семантического процессора.
        
        Args:
            language: Язык обработки.
            vector_provider: Провайдер векторных представлений.
            enable_topic_modeling: Включить моделирование тем.
        """
        super().__init__(language, vector_provider)
        self.enable_topic_modeling = enable_topic_modeling
        self.logger = get_logger("advanced_semantic_processor")
        
        # Дополнительные возможности
        self.topic_cache = {}
        self.concept_hierarchy = {}
        
    def extract_topics(self, text: str, num_topics: int = 5) -> List[Tuple[str, float]]:
        """Извлекает темы из текста."""
        if not self.enable_topic_modeling:
            return []
        
        try:
            # Простое извлечение тем на основе кластеризации ключевых слов
            keywords = self.extract_keywords(text, max_keywords=20)
            
            if not keywords:
                return []
            
            # Группируем ключевые слова в темы
            topics = self._cluster_keywords_into_topics(keywords, num_topics)
            return topics
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения тем: {e}")
            return []
    
    def find_semantic_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Находит семантические концепты в тексте."""
        try:
            concepts = []
            
            # Извлекаем ключевые слова
            keywords = dict(self.extract_keywords(text, max_keywords=15))
            
            # Анализируем каждое ключевое слово как потенциальный концепт
            for keyword, score in keywords.items():
                concept = {
                    "text": keyword,
                    "score": score,
                    "type": self._classify_concept_type(keyword),
                    "related_words": self._find_related_words(keyword, keywords)
                }
                concepts.append(concept)
            
            return sorted(concepts, key=lambda x: x["score"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска концептов: {e}")
            return []
    
    def _cluster_keywords_into_topics(self, keywords: List[Tuple[str, float]], num_topics: int) -> List[Tuple[str, float]]:
        """Кластеризует ключевые слова в темы."""
        if len(keywords) <= num_topics:
            return [(kw, score) for kw, score in keywords]
        
        # Простая кластеризация на основе семантической близости
        topics = []
        used_keywords = set()
        
        for i in range(min(num_topics, len(keywords))):
            if keywords[i][0] not in used_keywords:
                topic_name = keywords[i][0]
                topic_score = keywords[i][1]
                used_keywords.add(keywords[i][0])
                
                # Найти семантически близкие слова
                for j in range(i + 1, len(keywords)):
                    if keywords[j][0] not in used_keywords:
                        similarity = self.calculate_similarity(topic_name, keywords[j][0])
                        if similarity > 0.3:  # Порог близости
                            topic_score += keywords[j][1] * similarity
                            used_keywords.add(keywords[j][0])
                
                topics.append((topic_name, topic_score))
        
        return sorted(topics, key=lambda x: x[1], reverse=True)
    
    def _classify_concept_type(self, concept: str) -> str:
        """Классифицирует тип концепта."""
        concept_lower = concept.lower()
        
        # Простая классификация на основе морфологии и словарей
        if any(concept_lower.endswith(suffix) for suffix in ['ение', 'ание', 'ство', 'ность', 'tion', 'sion', 'ness']):
            return "abstract"
        elif any(concept_lower.endswith(suffix) for suffix in ['ый', 'ая', 'ое', 'ые', 'ing', 'ed']):
            return "attribute"
        elif concept_lower in ['человек', 'люди', 'персон', 'person', 'people', 'man', 'woman']:
            return "person"
        elif concept_lower in ['место', 'город', 'страна', 'место', 'place', 'city', 'country']:
            return "location"
        else:
            return "object"
    
    def _find_related_words(self, target_word: str, all_keywords: Dict[str, float]) -> List[str]:
        """Находит слова, связанные с целевым словом."""
        related = []
        
        for word, score in all_keywords.items():
            if word != target_word:
                similarity = self.calculate_similarity(target_word, word)
                if similarity > 0.2:  # Порог связанности
                    related.append(word)
        
        return related[:5]  # Максимум 5 связанных слов


def create_semantic_processor(processor_type: str = "basic", **kwargs) -> ISemanticProcessor:
    """Создает экземпляр семантического процессора.
    
    Args:
        processor_type: Тип процессора ("basic", "advanced").
        **kwargs: Параметры для конструктора.
        
    Returns:
        Экземпляр семантического процессора.
    """
    if processor_type == "basic":
        return BasicSemanticProcessor(**kwargs)
    elif processor_type == "advanced":
        return AdvancedSemanticProcessor(**kwargs)
    else:
        raise ValueError(f"Неизвестный тип семантического процессора: {processor_type}")