"""Реализации генерации текста для модуля NLP."""

import re
import random
from typing import Dict, List, Optional, Any, Tuple
from neurograph.nlp.base import ITextGenerator
from neurograph.core.logging import get_logger

try:
    from jinja2 import Template, Environment, BaseLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


class TemplateTextGenerator(ITextGenerator):
    """Генерация текста на основе шаблонов."""
    
    def __init__(self):
        self.logger = get_logger("template_text_generator")
        
        if JINJA2_AVAILABLE:
            self.env = Environment(loader=BaseLoader())
            self.logger.info("Jinja2 доступен - используем продвинутые шаблоны")
        else:
            self.env = None
            self.logger.warning("Jinja2 недоступен - используем простую подстановку")
        
        # Предустановленные шаблоны
        self.predefined_templates = {
            'fact_statement': '{{ subject }} {{ verb }} {{ object }}.',
            'question': 'Что такое {{ concept }}?',
            'definition': '{{ term }} - это {{ definition }}.',
            'comparison': '{{ item1 }} отличается от {{ item2 }} тем, что {{ difference }}.',
            'explanation': 'Для понимания {{ topic }} важно знать, что {{ explanation }}.',
            'instruction': 'Чтобы {{ goal }}, необходимо {{ steps }}.',
            'summary': 'В заключение можно сказать, что {{ main_point }}.',
        }
        
        # Вспомогательные фразы для разных стилей
        self.style_phrases = {
            'formal': {
                'intro': ['Следует отметить, что', 'Необходимо подчеркнуть, что', 'Важно понимать, что'],
                'conclusion': ['Таким образом', 'В заключение', 'Подводя итог'],
                'transition': ['Кроме того', 'Более того', 'Дополнительно']
            },
            'casual': {
                'intro': ['Кстати', 'Между прочим', 'Стоит сказать, что'],
                'conclusion': ['В общем', 'Короче говоря', 'В итоге'],
                'transition': ['Также', 'Еще', 'К тому же']
            },
            'scientific': {
                'intro': ['Исследования показывают, что', 'Согласно данным', 'Установлено, что'],
                'conclusion': ['Результаты свидетельствуют', 'Данные подтверждают', 'Анализ показывает'],
                'transition': ['Аналогично', 'Соответственно', 'При этом']
            }
        }
    
    def generate_text(self, template: str, params: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Генерация текста по шаблону."""
        try:
            # Проверяем, является ли template именем предустановленного шаблона
            if template in self.predefined_templates:
                template = self.predefined_templates[template]
            
            # Объединяем параметры и контекст
            render_context = params.copy()
            if context:
                render_context.update(context)
            
            # Рендерим шаблон
            if JINJA2_AVAILABLE and self.env:
                jinja_template = self.env.from_string(template)
                result = jinja_template.render(**render_context)
            else:
                # Простая подстановка без Jinja2
                result = self._simple_template_render(template, render_context)
            
            # Пост-обработка
            result = self._post_process_text(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации текста: {e}")
            return f"Ошибка генерации: {template}"
    
    def generate_response(self, query: str, knowledge: Dict[str, Any],
                         style: Optional[str] = None) -> str:
        """Генерация ответа на запрос на основе знаний."""
        
        # Анализируем тип запроса
        query_type = self._analyze_query_type(query)
        
        # Выбираем стратегию ответа
        if query_type == 'definition':
            return self._generate_definition_response(query, knowledge, style)
        elif query_type == 'explanation':
            return self._generate_explanation_response(query, knowledge, style)
        elif query_type == 'comparison':
            return self._generate_comparison_response(query, knowledge, style)
        elif query_type == 'howto':
            return self._generate_instruction_response(query, knowledge, style)
        else:
            return self._generate_general_response(query, knowledge, style)
    
    def _simple_template_render(self, template: str, context: Dict[str, Any]) -> str:
        """Простая подстановка переменных без Jinja2."""
        result = template
        for key, value in context.items():
            placeholder = f"{{{{ {key} }}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result
    
    def _analyze_query_type(self, query: str) -> str:
        """Анализ типа запроса для выбора стратегии ответа."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['что такое', 'что это', 'определение']):
            return 'definition'
        elif any(word in query_lower for word in ['как', 'каким образом', 'способ']):
            return 'howto'
        elif any(word in query_lower for word in ['почему', 'зачем', 'причина']):
            return 'explanation'
        elif any(word in query_lower for word in ['отличие', 'разница', 'сравнить']):
            return 'comparison'
        else:
            return 'general'
    
    def _generate_definition_response(self, query: str, knowledge: Dict[str, Any], 
                                    style: Optional[str]) -> str:
        """Генерация ответа-определения."""
        
        # Извлекаем термин из вопроса
        term = self._extract_term_from_query(query)
        
        if term in knowledge:
            definition = knowledge[term]
            
            template = self._get_style_intro(style) + " {{ term }} - это {{ definition }}."
            
            return self.generate_text(template, {
                'term': term,
                'definition': definition
            })
        else:
            return f"К сожалению, у меня нет информации о {term}."
    
    def _generate_explanation_response(self, query: str, knowledge: Dict[str, Any],
                                     style: Optional[str]) -> str:
        """Генерация объяснительного ответа."""
        
        # Находим релевантную информацию в знаниях
        relevant_info = self._find_relevant_knowledge(query, knowledge)
        
        if relevant_info:
            template = ("{{ intro }} {{ topic }}. {{ explanation }} {{ conclusion }}")
            
            return self.generate_text(template, {
                'intro': self._get_style_intro(style),
                'topic': relevant_info.get('topic', 'данный вопрос'),
                'explanation': relevant_info.get('explanation', 'требует дополнительного изучения'),
                'conclusion': self._get_style_conclusion(style)
            })
        else:
            return "Мне нужно больше информации для ответа на этот вопрос."
    
    def _generate_comparison_response(self, query: str, knowledge: Dict[str, Any],
                                    style: Optional[str]) -> str:
        """Генерация сравнительного ответа."""
        
        # Извлекаем объекты сравнения
        items = self._extract_comparison_items(query)
        
        if len(items) >= 2:
            item1, item2 = items[0], items[1]
            
            # Ищем информацию об объектах
            info1 = knowledge.get(item1, {})
            info2 = knowledge.get(item2, {})
            
            if info1 and info2:
                differences = self._find_differences(info1, info2)
                
                template = ("{{ item1 }} и {{ item2 }} имеют различия. "
                           "{{ differences }} {{ conclusion }}")
                
                return self.generate_text(template, {
                    'item1': item1,
                    'item2': item2,
                    'differences': differences,
                    'conclusion': self._get_style_conclusion(style)
                })
        
        return "Для сравнения нужно указать два объекта и иметь о них информацию."
    
    def _generate_instruction_response(self, query: str, knowledge: Dict[str, Any],
                                     style: Optional[str]) -> str:
        """Генерация инструкционного ответа."""
        
        # Извлекаем цель из запроса
        goal = self._extract_goal_from_query(query)
        
        # Ищем инструкции в знаниях
        instructions = knowledge.get('instructions', {}).get(goal, [])
        
        if instructions:
            steps_text = self._format_steps(instructions)
            
            template = ("Для {{ goal }} рекомендуется следующее: {{ steps }}")
            
            return self.generate_text(template, {
                'goal': goal,
                'steps': steps_text
            })
        else:
            return f"У меня нет конкретных инструкций для {goal}."
    
    def _generate_general_response(self, query: str, knowledge: Dict[str, Any],
                                 style: Optional[str]) -> str:
        """Генерация общего ответа."""
        
        # Находим релевантную информацию
        relevant_info = self._find_relevant_knowledge(query, knowledge)
        
        if relevant_info:
            facts = relevant_info.get('facts', [])
            
            if facts:
                # Генерируем ответ на основе фактов
                facts_text = '. '.join(facts[:3])  # Берем первые 3 факта
                
                template = ("{{ intro }} {{ facts }}. {{ conclusion }}")
                
                return self.generate_text(template, {
                    'intro': self._get_style_intro(style),
                    'facts': facts_text,
                    'conclusion': self._get_style_conclusion(style)
                })
        
        return "Мне нужно больше информации для ответа на ваш вопрос."
    
    def _get_style_intro(self, style: Optional[str]) -> str:
        """Получение вступительной фразы в зависимости от стиля."""
        if not style or style not in self.style_phrases:
            style = 'formal'
        
        phrases = self.style_phrases[style]['intro']
        return random.choice(phrases)
    
    def _get_style_conclusion(self, style: Optional[str]) -> str:
        """Получение заключительной фразы в зависимости от стиля."""
        if not style or style not in self.style_phrases:
            style = 'formal'
        
        phrases = self.style_phrases[style]['conclusion']
        return random.choice(phrases)
    
    def _extract_term_from_query(self, query: str) -> str:
        """Извлечение термина из вопроса."""
        # Паттерны для извлечения терминов
        patterns = [
            r'что\s+такое\s+(.+?)[?]?$',
            r'что\s+это\s+(.+?)[?]?$',
            r'определение\s+(.+?)[?]?$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).strip()
        
        # Если паттерн не найден, возвращаем весь запрос
        return query.strip('?')
    
    def _extract_comparison_items(self, query: str) -> List[str]:
        """Извлечение объектов для сравнения."""
        # Паттерны для извлечения объектов сравнения
        patterns = [
            r'сравни\s+(.+?)\s+и\s+(.+?)(?:\s|$|[?])',
            r'отличие\s+(.+?)\s+от\s+(.+?)(?:\s|$|[?])',
            r'разница\s+между\s+(.+?)\s+и\s+(.+?)(?:\s|$|[?])'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        
        return []
    
    def _extract_goal_from_query(self, query: str) -> str:
        """Извлечение цели из инструкционного вопроса."""
        patterns = [
            r'как\s+(.+?)[?]?$',
            r'каким\s+образом\s+(.+?)[?]?$',
            r'способ\s+(.+?)[?]?$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).strip()
        
        return query.strip('?')
    
    def _find_relevant_knowledge(self, query: str, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Поиск релевантной информации в базе знаний."""
        query_words = set(query.lower().split())
        relevant_info = {'facts': [], 'topic': '', 'explanation': ''}
        
        for key, value in knowledge.items():
            if isinstance(value, str):
                # Простой поиск по ключевым словам
                key_words = set(key.lower().split())
                value_words = set(value.lower().split())
                
                # Вычисляем пересечение
                intersection = query_words.intersection(key_words.union(value_words))
                
                if intersection:
                    relevant_info['facts'].append(value)
                    if not relevant_info['topic']:
                        relevant_info['topic'] = key
                    if not relevant_info['explanation'] and len(value) > 50:
                        relevant_info['explanation'] = value
        
        return relevant_info
    
    def _find_differences(self, info1: Dict[str, Any], info2: Dict[str, Any]) -> str:
        """Поиск различий между двумя объектами."""
        differences = []
        
        # Сравниваем атрибуты
        all_keys = set(info1.keys()).union(set(info2.keys()))
        
        for key in all_keys:
            val1 = info1.get(key, 'не указано')
            val2 = info2.get(key, 'не указано')
            
            if val1 != val2:
                differences.append(f"по параметру {key}: {val1} против {val2}")
        
        if differences:
            return ', '.join(differences[:3])  # Максимум 3 различия
        else:
            return "существенных различий не обнаружено"
    
    def _format_steps(self, steps: List[str]) -> str:
        """Форматирование списка шагов."""
        if len(steps) == 1:
            return steps[0]
        elif len(steps) <= 3:
            return ', '.join(steps[:-1]) + ' и ' + steps[-1]
        else:
            return ', '.join(steps[:3]) + ' и другие шаги'
    
    def _post_process_text(self, text: str) -> str:
        """Пост-обработка сгенерированного текста."""
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Исправляем пунктуацию
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*([.!?])', r'\1', text)
        
        # Заглавная буква в начале
        if text:
            text = text[0].upper() + text[1:]
        
        return text


class MarkovTextGenerator(ITextGenerator):
    """Генерация текста на основе цепей Маркова."""
    
    def __init__(self, order: int = 2):
        self.order = order  # Порядок цепи Маркова
        self.logger = get_logger("markov_text_generator")
        self.chains = {}
        self.word_counts = {}
        
    def train(self, texts: List[str]) -> None:
        """Обучение генератора на корпусе текстов."""
        self.logger.info(f"Обучение на {len(texts)} текстах")
        
        for text in texts:
            self._process_text(text)
        
        self.logger.info(f"Построено {len(self.chains)} цепей Маркова")
    
    def generate_text(self, template: str, params: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Генерация текста с использованием цепей Маркова."""
        
        # Определяем длину генерации
        max_length = params.get('max_length', 50)
        seed_words = params.get('seed_words', [])
        
        if not self.chains:
            return "Генератор не обучен"
        
        # Генерируем текст
        generated = self._generate_sequence(seed_words, max_length)
        
        return ' '.join(generated)
    
    def generate_response(self, query: str, knowledge: Dict[str, Any],
                         style: Optional[str] = None) -> str:
        """Генерация ответа с использованием цепей Маркова."""
        
        # Извлекаем ключевые слова из запроса как seed
        seed_words = self._extract_keywords(query)
        
        # Генерируем ответ
        max_length = 30 if style == 'casual' else 50
        
        generated = self._generate_sequence(seed_words[:2], max_length)
        
        if generated:
            return ' '.join(generated)
        else:
            return "Не удалось сгенерировать ответ"
    
    def _process_text(self, text: str) -> None:
        """Обработка текста для построения цепей Маркова."""
        # Очистка и токенизация
        words = self._tokenize_for_markov(text)
        
        if len(words) < self.order + 1:
            return
        
        # Построение цепей
        for i in range(len(words) - self.order):
            # Создаем n-грамму (ключ)
            key = tuple(words[i:i + self.order])
            next_word = words[i + self.order]
            
            if key not in self.chains:
                self.chains[key] = {}
            
            if next_word not in self.chains[key]:
                self.chains[key][next_word] = 0
            
            self.chains[key][next_word] += 1
            
            # Также считаем общую частоту слов
            if next_word not in self.word_counts:
                self.word_counts[next_word] = 0
            self.word_counts[next_word] += 1
    
    def _tokenize_for_markov(self, text: str) -> List[str]:
        """Токенизация текста для цепей Маркова."""
        # Простая токенизация
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
        words = text.split()
        
        # Фильтруем очень короткие слова
        words = [word for word in words if len(word) > 1]
        
        return words
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов из текста."""
        words = self._tokenize_for_markov(text)
        
        # Фильтруем стоп-слова
        stop_words = {'что', 'как', 'где', 'когда', 'почему', 'это', 'есть'}
        keywords = [word for word in words if word not in stop_words]
        
        return keywords[:5]  # Максимум 5 ключевых слов
    
    def _generate_sequence(self, seed_words: List[str], max_length: int) -> List[str]:
        """Генерация последовательности слов."""
        if not self.chains:
            return []
        
        # Инициализация
        if len(seed_words) >= self.order:
            current = tuple(seed_words[-self.order:])
        else:
            # Выбираем случайное начало
            current = random.choice(list(self.chains.keys()))
        
        result = list(current)
        
        for _ in range(max_length - len(result)):
            if current not in self.chains:
                break
            
            # Выбираем следующее слово на основе вероятностей
            next_word = self._choose_next_word(self.chains[current])
            
            if not next_word:
                break
            
            result.append(next_word)
            
            # Обновляем текущую n-грамму
            current = tuple(result[-self.order:])
        
        return result
    
    def _choose_next_word(self, word_counts: Dict[str, int]) -> Optional[str]:
        """Выбор следующего слова на основе вероятностей."""
        if not word_counts:
            return None
        
        # Создаем взвешенный список
        words = []
        weights = []
        
        for word, count in word_counts.items():
            words.append(word)
            weights.append(count)
        
        # Выбираем случайное слово с учетом весов
        return random.choices(words, weights=weights)[0]


class RuleBasedTextGenerator(ITextGenerator):
    """Простой генератор на основе правил и шаблонов."""
    
    def __init__(self):
        self.logger = get_logger("rule_based_text_generator")
        
        # Базовые шаблоны ответов
        self.response_templates = {
            'definition': [
                "{term} - это {definition}.",
                "{term} представляет собой {definition}.",
                "Под {term} понимается {definition}."
            ],
            'explanation': [
                "Это происходит потому, что {reason}.",
                "Причина в том, что {reason}.",
                "Объяснение заключается в {reason}."
            ],
            'instruction': [
                "Для этого нужно {action}.",
                "Рекомендуется {action}.",
                "Следует {action}."
            ],
            'unknown': [
                "К сожалению, у меня нет информации об этом.",
                "Мне нужно больше данных для ответа.",
                "Этот вопрос требует дополнительного изучения."
            ]
        }
        
        # Фразы-связки
        self.connectors = {
            'addition': ['кроме того', 'также', 'более того', 'дополнительно'],
            'contrast': ['однако', 'но', 'в то же время', 'тем не менее'],
            'conclusion': ['таким образом', 'в итоге', 'следовательно', 'в заключение']
        }
    
    def generate_text(self, template: str, params: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Генерация текста по простому шаблону."""
        
        # Простая подстановка параметров
        result = template
        
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        return result
    
    def generate_response(self, query: str, knowledge: Dict[str, Any],
                         style: Optional[str] = None) -> str:
        """Генерация ответа на основе правил."""
        
        # Анализируем тип вопроса
        query_type = self._classify_query(query)
        
        # Ищем релевантную информацию
        relevant_info = self._find_relevant_info(query, knowledge)
        
        if not relevant_info:
            return random.choice(self.response_templates['unknown'])
        
        # Генерируем ответ в зависимости от типа
        if query_type == 'what':
            return self._generate_what_response(relevant_info)
        elif query_type == 'how':
            return self._generate_how_response(relevant_info)
        elif query_type == 'why':
            return self._generate_why_response(relevant_info)
        else:
            return self._generate_general_response(relevant_info)
    
    def _classify_query(self, query: str) -> str:
        """Классификация типа вопроса."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['что', 'что такое', 'какой']):
            return 'what'
        elif any(word in query_lower for word in ['как', 'каким образом']):
            return 'how'
        elif any(word in query_lower for word in ['почему', 'зачем', 'отчего']):
            return 'why'
        else:
            return 'general'
    
    def _find_relevant_info(self, query: str, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Поиск релевантной информации в знаниях."""
        query_words = set(query.lower().split())
        
        best_match = {}
        best_score = 0
        
        for key, value in knowledge.items():
            if isinstance(value, str):
                # Простой подсчет совпадающих слов
                key_words = set(key.lower().split())
                value_words = set(value.lower().split())
                
                score = len(query_words.intersection(key_words.union(value_words)))
                
                if score > best_score:
                    best_score = score
                    best_match = {'key': key, 'value': value, 'score': score}
        
        return best_match if best_score > 0 else {}
    
    def _generate_what_response(self, info: Dict[str, Any]) -> str:
        """Генерация ответа на вопрос 'что'."""
        template = random.choice(self.response_templates['definition'])
        
        return template.format(
            term=info.get('key', 'это'),
            definition=info.get('value', 'неизвестно')
        )
    
    def _generate_how_response(self, info: Dict[str, Any]) -> str:
        """Генерация ответа на вопрос 'как'."""
        template = random.choice(self.response_templates['instruction'])
        
        return template.format(action=info.get('value', 'действовать по ситуации'))
    
    def _generate_why_response(self, info: Dict[str, Any]) -> str:
        """Генерация ответа на вопрос 'почему'."""
        template = random.choice(self.response_templates['explanation'])
        
        return template.format(reason=info.get('value', 'причины могут быть разными'))
    
    def _generate_general_response(self, info: Dict[str, Any]) -> str:
        """Генерация общего ответа."""
        value = info.get('value', '')
        
        if len(value) > 100:
            # Длинный текст - используем первую часть
            return value[:100] + '...'
        else:
            return value


class HybridTextGenerator(ITextGenerator):
    """Гибридный генератор, комбинирующий разные подходы."""
    
    def __init__(self, use_templates: bool = True, use_markov: bool = False, use_rules: bool = True):
        self.logger = get_logger("hybrid_text_generator")
        
        self.generators = []
        
        if use_templates:
            self.template_generator = TemplateTextGenerator()
            self.generators.append(('template', self.template_generator))
        
        if use_markov:
            self.markov_generator = MarkovTextGenerator()
            self.generators.append(('markov', self.markov_generator))
        
        if use_rules:
            self.rule_generator = RuleBasedTextGenerator()
            self.generators.append(('rules', self.rule_generator))
        
        # Приоритеты генераторов для разных задач
        self.task_priorities = {
            'template': ['template', 'rules', 'markov'],
            'response': ['template', 'rules', 'markov'],
            'creative': ['markov', 'template', 'rules']
        }
    
    def generate_text(self, template: str, params: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Генерация текста с выбором лучшего генератора."""
        
        # Определяем задачу
        task_type = context.get('task_type', 'template') if context else 'template'
        
        # Пробуем генераторы в порядке приоритета
        priorities = self.task_priorities.get(task_type, ['template', 'rules'])
        
        for generator_name in priorities:
            generator = self._get_generator(generator_name)
            
            if generator:
                try:
                    result = generator.generate_text(template, params, context)
                    
                    # Проверяем качество результата
                    if self._is_good_result(result):
                        return result
                
                except Exception as e:
                    self.logger.warning(f"Ошибка в генераторе {generator_name}: {e}")
        
        # Fallback
        return f"Не удалось сгенерировать текст для шаблона: {template}"
    
    def generate_response(self, query: str, knowledge: Dict[str, Any],
                         style: Optional[str] = None) -> str:
        """Генерация ответа с выбором лучшего генератора."""
        
        # Пробуем генераторы в порядке приоритета для ответов
        priorities = self.task_priorities['response']
        
        best_result = ""
        best_score = 0
        
        for generator_name in priorities:
            generator = self._get_generator(generator_name)
            
            if generator:
                try:
                    result = generator.generate_response(query, knowledge, style)
                    score = self._score_response(result, query, knowledge)
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                
                except Exception as e:
                    self.logger.warning(f"Ошибка в генераторе {generator_name}: {e}")
        
        return best_result if best_result else "Не удалось сгенерировать ответ."
    
    def train_markov(self, texts: List[str]) -> None:
        """Обучение Марковского генератора."""
        if hasattr(self, 'markov_generator'):
            self.markov_generator.train(texts)
    
    def _get_generator(self, name: str) -> Optional[ITextGenerator]:
        """Получение генератора по имени."""
        for gen_name, generator in self.generators:
            if gen_name == name:
                return generator
        return None
    
    def _is_good_result(self, result: str) -> bool:
        """Проверка качества результата."""
        if not result or len(result.strip()) < 3:
            return False
        
        # Проверяем на наличие незаполненных шаблонов
        if '{' in result and '}' in result:
            return False
        
        # Проверяем на повторяющиеся фразы
        words = result.split()
        if len(set(words)) < len(words) * 0.5:  # Слишком много повторов
            return False
        
        return True
    
    def _score_response(self, response: str, query: str, knowledge: Dict[str, Any]) -> float:
        """Оценка качества ответа."""
        if not response or len(response.strip()) < 5:
            return 0.0
        
        score = 0.5  # Базовая оценка
        
        # Проверяем релевантность
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Пересечение с запросом
        intersection = query_words.intersection(response_words)
        if intersection:
            score += 0.3 * len(intersection) / len(query_words)
        
        # Длина ответа (не слишком короткий, не слишком длинный)
        length_score = min(1.0, len(response) / 200)  # Оптимум около 200 символов
        score += 0.2 * length_score
        
        return min(1.0, score)