"""Реализации токенизаторов для модуля NLP."""

import re
from typing import List, Set, Dict, Optional
from neurograph.nlp.base import ITokenizer, Token
from neurograph.core.logging import get_logger


class SimpleTokenizer(ITokenizer):
    """Простой токенизатор на основе регулярных выражений."""
    
    def __init__(self, language: str = "ru"):
        self.language = language
        self.logger = get_logger("simple_tokenizer")
        
        # Русские стоп-слова
        self.stop_words_ru = {
            "а", "в", "и", "к", "о", "с", "у", "я", "на", "не", "по", "то", "за", "же",
            "из", "от", "до", "под", "над", "при", "для", "что", "как", "его", "её",
            "их", "это", "или", "был", "была", "было", "были", "есть", "быть", "иметь"
        }
        
        # Английские стоп-слова
        self.stop_words_en = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
            "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
            "were", "will", "with", "the", "this", "but", "they", "have", "had"
        }
        
        self.stop_words = self.stop_words_ru if language == "ru" else self.stop_words_en
        
        # Регулярные выражения для токенизации
        self.word_pattern = re.compile(r'\b\w+\b')
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.punct_pattern = re.compile(r'[^\w\s]')
        
        # Простые POS теги
        self.pos_patterns = {
            'NOUN': re.compile(r'.*(?:ость|ция|ние|тор|ник|ист)$'),  # Существительные
            'VERB': re.compile(r'.*(?:ать|ить|еть|ывать|овать)$'),   # Глаголы
            'ADJ': re.compile(r'.*(?:ный|ский|ческий|овый)$'),       # Прилагательные
            'NUM': re.compile(r'^\d+$'),                             # Числа
        }
        
    def tokenize(self, text: str) -> List[Token]:
        """Токенизация текста с базовыми лингвистическими атрибутами."""
        tokens = []
        
        # Находим все слова с позициями
        for match in re.finditer(r'\S+', text):
            word = match.group()
            start = match.start()
            end = match.end()
            
            # Базовые атрибуты
            is_alpha = word.isalpha()
            is_punct = bool(self.punct_pattern.match(word))
            is_stop = word.lower() in self.stop_words
            
            # Лемма (упрощенная)
            lemma = self._get_simple_lemma(word.lower())
            
            # POS тег
            pos = self._get_simple_pos(word.lower())
            
            token = Token(
                text=word,
                lemma=lemma,
                pos=pos,
                tag=pos,  # Упрощенно используем тот же тег
                is_alpha=is_alpha,
                is_stop=is_stop,
                is_punct=is_punct,
                start=start,
                end=end
            )
            
            tokens.append(token)
        
        return tokens
    
    def sent_tokenize(self, text: str) -> List[str]:
        """Разбиение на предложения."""
        # Простое разбиение по знакам препинания
        sentences = re.split(r'[.!?]+', text)
        
        # Очищаем и фильтруем пустые
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 2:
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    def _get_simple_lemma(self, word: str) -> str:
        """Упрощенная лемматизация."""
        # Удаляем распространенные окончания
        lemma_rules = [
            (r'ами$', ''),      # творительный падеж множественного числа
            (r'ами$', ''),      # творительный падеж множественного числа
            (r'ов$', ''),       # родительный падеж множественного числа
            (r'ем$', ''),       # творительный падеж единственного числа
            (r'ах$', ''),       # предложный падеж множественного числа
            (r'ый$', ''),       # прилагательные мужского рода
            (r'ая$', ''),       # прилагательные женского рода
            (r'ое$', ''),       # прилагательные среднего рода
            (r'ые$', ''),       # прилагательные множественного числа
            (r'ать$', 'ать'),   # глаголы в инфинитиве
            (r'ить$', 'ить'),   # глаголы в инфинитиве
            (r'еть$', 'еть'),   # глаголы в инфинитиве
        ]
        
        lemma = word
        for pattern, replacement in lemma_rules:
            lemma = re.sub(pattern, replacement, lemma)
            
        return lemma if lemma else word
    
    def _get_simple_pos(self, word: str) -> str:
        """Упрощенное определение части речи."""
        for pos, pattern in self.pos_patterns.items():
            if pattern.match(word):
                return pos
        return 'WORD'  # Общий тег для всех остальных слов


class SpacyTokenizer(ITokenizer):
    """Токенизатор на основе spaCy (если доступен)."""
    
    def __init__(self, model_name: str = "ru_core_news_sm"):
        self.logger = get_logger("spacy_tokenizer")
        self.model_name = model_name
        self.nlp = None
        
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Загружена модель spaCy: {model_name}")
        except ImportError:
            self.logger.warning("spaCy не установлен, используется fallback")
            self.fallback = SimpleTokenizer()
        except OSError:
            self.logger.warning(f"Модель {model_name} не найдена, используется fallback")
            self.fallback = SimpleTokenizer()
    
    def tokenize(self, text: str) -> List[Token]:
        """Токенизация с использованием spaCy."""
        if self.nlp is None:
            return self.fallback.tokenize(text)
        
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            nlp_token = Token(
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                tag=token.tag_,
                is_alpha=token.is_alpha,
                is_stop=token.is_stop,
                is_punct=token.is_punct,
                start=token.idx,
                end=token.idx + len(token.text),
                dependency=token.dep_,
                head=token.head.i if token.head != token else None
            )
            tokens.append(nlp_token)
        
        return tokens
    
    def sent_tokenize(self, text: str) -> List[str]:
        """Разбиение на предложения с spaCy."""
        if self.nlp is None:
            return self.fallback.sent_tokenize(text)
        
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]


class SubwordTokenizer(ITokenizer):
    """Субсловный токенизатор для работы с редкими словами."""
    
    def __init__(self, vocab_size: int = 10000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.logger = get_logger("subword_tokenizer")
        
        # Базовый словарь
        self.vocab = set()
        self.subword_vocab = set()
        self.char_vocab = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')  # Русский алфавит
        self.char_vocab.update('abcdefghijklmnopqrstuvwxyz')  # Английский алфавит
        
        # Простой токенизатор для fallback
        self.simple_tokenizer = SimpleTokenizer()
    
    def build_vocab(self, texts: List[str]) -> None:
        """Построение словаря на основе корпуса текстов."""
        word_freq = {}
        
        # Подсчет частот слов
        for text in texts:
            tokens = self.simple_tokenizer.tokenize(text)
            for token in tokens:
                word = token.text.lower()
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Добавляем часто встречающиеся слова в словарь
        for word, freq in word_freq.items():
            if freq >= self.min_frequency:
                self.vocab.add(word)
        
        # Построение субсловного словаря (BPE-подобный алгоритм)
        self._build_subword_vocab(word_freq)
        
        self.logger.info(f"Построен словарь: {len(self.vocab)} слов, "
                        f"{len(self.subword_vocab)} субслов")
    
    def tokenize(self, text: str) -> List[Token]:
        """Токенизация с субсловной разбивкой."""
        base_tokens = self.simple_tokenizer.tokenize(text)
        
        subword_tokens = []
        for token in base_tokens:
            if token.is_alpha:
                # Разбиваем на субслова если слово не в словаре
                subwords = self._tokenize_word(token.text.lower())
                
                # Создаем токены для каждого субслова
                char_offset = 0
                for subword in subwords:
                    subword_token = Token(
                        text=subword,
                        lemma=subword,  # Для субслов лемма = само слово
                        pos=token.pos,
                        tag=token.tag,
                        is_alpha=subword.isalpha(),
                        is_stop=False,  # Субслова не считаем стоп-словами
                        is_punct=not subword.isalpha(),
                        start=token.start + char_offset,
                        end=token.start + char_offset + len(subword)
                    )
                    subword_tokens.append(subword_token)
                    char_offset += len(subword)
            else:
                subword_tokens.append(token)
        
        return subword_tokens
    
    def sent_tokenize(self, text: str) -> List[str]:
        """Разбиение на предложения (делегируем простому токенизатору)."""
        return self.simple_tokenizer.sent_tokenize(text)
    
    def _build_subword_vocab(self, word_freq: Dict[str, int]) -> None:
        """Построение субсловного словаря."""
        # Инициализируем символами
        self.subword_vocab.update(self.char_vocab)
        
        # Простой алгоритм для создания биграмм
        bigram_freq = {}
        
        for word, freq in word_freq.items():
            if len(word) > 1:
                for i in range(len(word) - 1):
                    bigram = word[i:i+2]
                    bigram_freq[bigram] = bigram_freq.get(bigram, 0) + freq
        
        # Добавляем наиболее частые биграммы
        sorted_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
        for bigram, freq in sorted_bigrams[:1000]:  # Топ-1000 биграмм
            if freq >= self.min_frequency:
                self.subword_vocab.add(bigram)
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Разбивка слова на субслова."""
        if word in self.vocab:
            return [word]
        
        # Жадный алгоритм разбивки
        subwords = []
        i = 0
        
        while i < len(word):
            # Ищем самое длинное субслово
            best_subword = None
            best_length = 0
            
            for length in range(min(len(word) - i, 10), 0, -1):  # Максимум 10 символов
                candidate = word[i:i+length]
                if candidate in self.subword_vocab and length > best_length:
                    best_subword = candidate
                    best_length = length
            
            if best_subword:
                subwords.append(best_subword)
                i += best_length
            else:
                # Если не нашли субслово, берем один символ
                if i < len(word):
                    subwords.append(word[i])
                    i += 1
                else:
                    break
        
        return subwords if subwords else [word]