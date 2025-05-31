# NeuroGraph ContextVec: Документация для разработчиков

## Обзор модуля

**neurograph-contextvec** — модуль для работы с векторными представлениями (эмбеддингами) слов, фраз и понятий в системе NeuroGraph. Предоставляет единый интерфейс для создания, хранения, поиска и операций над векторными представлениями с поддержкой различных моделей и стратегий.

## Основные компоненты

### 1. Базовые интерфейсы (`neurograph.contextvec.base`)

#### IContextVectors
Основной интерфейс для работы с векторными представлениями.

```python
from neurograph.contextvec import IContextVectors
import numpy as np

# Создание векторных представлений через фабрику
vectors = ContextVectorsFactory.create("dynamic", vector_size=100)

# Создание вектора для понятия
success = vectors.create_vector("apple", np.array([0.1, 0.2, 0.3, ...]))

# Получение вектора
apple_vector = vectors.get_vector("apple")  # np.ndarray или None

# Вычисление сходства
similarity = vectors.similarity("apple", "fruit")  # float в диапазоне [-1, 1]

# Поиск похожих понятий
similar = vectors.get_most_similar("apple", top_n=5)
# Возвращает: [("fruit", 0.8), ("red", 0.6), ...]

# Проверка наличия
has_apple = vectors.has_key("apple")  # bool

# Получение всех ключей
all_keys = vectors.get_all_keys()  # List[str]

# Удаление вектора
removed = vectors.remove_vector("apple")  # bool
```

### 2. Фабрика векторных представлений (`neurograph.contextvec.factory`)

#### ContextVectorsFactory
Центральная фабрика для создания различных типов векторных представлений.

```python
from neurograph.contextvec import ContextVectorsFactory

# Создание динамических векторов (по умолчанию)
vectors = ContextVectorsFactory.create("dynamic", vector_size=384)

# Создание статических векторов
static_vectors = ContextVectorsFactory.create("static", vector_size=100)

# Создание из конфигурации
config = {
    "type": "dynamic",
    "vector_size": 256,
    "use_indexing": True
}
vectors = ContextVectorsFactory.create_from_config(config)

# Регистрация нового типа
class CustomVectors(IContextVectors):
    # реализация интерфейса
    pass

ContextVectorsFactory.register_implementation("custom", CustomVectors)

# Получение доступных типов
available_types = ContextVectorsFactory.get_available_types()
# Возвращает: ["static", "dynamic", "custom"]
```

### 3. Статические векторные представления (`neurograph.contextvec.impl.static`)

#### StaticContextVectors
Простая реализация для статических, неизменяемых векторов.

```python
from neurograph.contextvec.impl.static import StaticContextVectors
import numpy as np

# Создание с заданной размерностью
vectors = StaticContextVectors(vector_size=100)

# Добавление векторов
vectors.create_vector("word1", np.random.random(100))
vectors.create_vector("word2", np.random.random(100))

# Все векторы автоматически нормализуются для косинусного сходства
similarity = vectors.similarity("word1", "word2")

# Поиск похожих
similar = vectors.get_most_similar("word1", top_n=3)
```

### 4. Динамические векторные представления (`neurograph.contextvec.impl.dynamic`)

#### DynamicContextVectors
Продвинутая реализация с поддержкой обновления векторов и индексации.

```python
from neurograph.contextvec.impl.dynamic import DynamicContextVectors
import numpy as np

# Создание с индексацией для быстрого поиска
vectors = DynamicContextVectors(
    vector_size=384, 
    use_indexing=True
)

# Добавление векторов
item_id = vectors.create_vector("concept1", np.random.random(384))

# Обновление существующего вектора с learning rate
vectors.update_vector("concept1", new_vector, learning_rate=0.1)

# Усреднение нескольких векторов
avg_vector = vectors.average_vectors(["concept1", "concept2", "concept3"])

# Поиск с использованием быстрого индекса
similar = vectors.get_most_similar("concept1", top_n=10)

# Сохранение и загрузка
vectors.save("vectors.json")
loaded_vectors = DynamicContextVectors.load("vectors.json")
```

### 5. Адаптеры для внешних моделей

#### Word2VecAdapter
Адаптер для работы с моделями Word2Vec.

```python
from neurograph.contextvec.adapters.word2vec import Word2VecAdapter
import numpy as np

# Загрузка модели Word2Vec
adapter = Word2VecAdapter("path/to/word2vec.bin")  # или .vec для текстового формата

# Создание эмбеддинга для текста
embedding = adapter.encode("apple fruit red")  # np.ndarray

# Пакетная обработка текстов
texts = ["apple fruit", "car vehicle", "house building"]
embeddings = adapter.encode_batch(texts, normalize=True)  # np.ndarray shape (3, vector_size)

# Получение размерности векторов
vector_size = adapter.get_vector_size()

# Поиск похожих слов (если модель поддерживает)
similar_words = adapter.get_most_similar("apple", top_n=5)
# Возвращает: [("fruit", 0.8), ("red", 0.6), ...]
```

#### SentenceTransformerAdapter
Адаптер для моделей Sentence Transformers.

```python
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter

# Создание адаптера с предобученной моделью
adapter = SentenceTransformerAdapter("all-MiniLM-L6-v2")

# Кодирование предложений
text = "Apple is a red fruit"
embedding = adapter.encode(text, normalize=True)

# Пакетная обработка
texts = [
    "Apple is a red fruit",
    "Car is a vehicle",
    "House is a building"
]
embeddings = adapter.encode_batch(texts, batch_size=32, normalize=True)

# Получение размерности
vector_size = adapter.get_vector_size()  # обычно 384 для MiniLM
```

### 6. Легковесные модели (`neurograph.contextvec.models.lightweight`)

#### HashingVectorizer
Векторизатор на основе хеширования без предварительного словаря.

```python
from neurograph.contextvec.models.lightweight import HashingVectorizer

# Создание векторизатора
vectorizer = HashingVectorizer(
    vector_size=1000,
    lowercase=True,
    ngram_range=(1, 2)  # униграммы и биграммы
)

# Преобразование текста в вектор
text = "Apple is a red fruit"
vector = vectorizer.transform(text)  # np.ndarray размерности 1000

# Пакетная обработка
texts = ["Apple is fruit", "Car is vehicle", "House is building"]
vectors = vectorizer.transform_batch(texts)  # shape (3, 1000)
```

#### RandomProjection
Модель для снижения размерности векторов.

```python
from neurograph.contextvec.models.lightweight import RandomProjection
import numpy as np

# Создание модели для снижения размерности с 1000 до 100
projection = RandomProjection(
    input_dim=1000,
    output_dim=100,
    seed=42  # для воспроизводимости
)

# Преобразование вектора
high_dim_vector = np.random.random(1000)
low_dim_vector = projection.transform(high_dim_vector)  # размерность 100

# Пакетная обработка
high_dim_vectors = np.random.random((10, 1000))
low_dim_vectors = projection.transform_batch(high_dim_vectors)  # shape (10, 100)

# Сохранение и загрузка модели
projection.save("projection_model.npz")
loaded_projection = RandomProjection.load("projection_model.npz")
```

## Практические примеры

### Создание системы семантического поиска

```python
from neurograph.contextvec import ContextVectorsFactory
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter
import numpy as np

class SemanticSearchEngine:
    """Система семантического поиска на основе векторных представлений"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Создание адаптера для генерации эмбеддингов
        self.encoder = SentenceTransformerAdapter(model_name)
        
        # Создание хранилища векторов с индексацией
        self.vectors = ContextVectorsFactory.create(
            "dynamic",
            vector_size=self.encoder.get_vector_size(),
            use_indexing=True
        )
        
        # Словарь для хранения оригинальных текстов
        self.documents = {}
    
    def add_document(self, doc_id: str, text: str) -> bool:
        """Добавление документа в индекс"""
        try:
            # Создание эмбеддинга
            embedding = self.encoder.encode(text, normalize=True)
            
            # Сохранение в векторное хранилище
            success = self.vectors.create_vector(doc_id, embedding)
            
            if success:
                self.documents[doc_id] = text
            
            return success
        except Exception as e:
            print(f"Error adding document {doc_id}: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск похожих документов"""
        try:
            # Создание эмбеддинга для запроса
            query_embedding = self.encoder.encode(query, normalize=True)
            
            # Поиск похожих векторов
            similar_docs = self.vectors.get_most_similar_by_vector(
                query_embedding, top_n=top_k
            )
            
            # Формирование результатов
            results = []
            for doc_id, similarity in similar_docs:
                if doc_id in self.documents:
                    results.append({
                        "document_id": doc_id,
                        "text": self.documents[doc_id],
                        "similarity": similarity
                    })
            
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def get_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск документов, похожих на заданный"""
        if not self.vectors.has_key(doc_id):
            return []
        
        similar_docs = self.vectors.get_most_similar(doc_id, top_n=top_k + 1)
        
        # Исключаем сам документ из результатов
        results = []
        for similar_id, similarity in similar_docs:
            if similar_id != doc_id and similar_id in self.documents:
                results.append({
                    "document_id": similar_id,
                    "text": self.documents[similar_id],
                    "similarity": similarity
                })
        
        return results[:top_k]

# Использование
search_engine = SemanticSearchEngine()

# Добавление документов
documents = [
    ("doc1", "Apple is a red fruit that grows on trees"),
    ("doc2", "Car is a vehicle used for transportation"),
    ("doc3", "Red apple is sweet and healthy"),
    ("doc4", "Vehicle maintenance is important for safety"),
]

for doc_id, text in documents:
    search_engine.add_document(doc_id, text)

# Поиск
results = search_engine.search("red fruit", top_k=2)
for result in results:
    print(f"ID: {result['document_id']}, Similarity: {result['similarity']:.3f}")
    print(f"Text: {result['text']}\n")
```

### Создание многоязычной системы векторов

```python
from neurograph.contextvec import ContextVectorsFactory
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter
from typing import Dict, List, Tuple

class MultilingualVectorStore:
    """Многоязычное хранилище векторных представлений"""
    
    def __init__(self):
        self.language_adapters = {
            "en": SentenceTransformerAdapter("all-MiniLM-L6-v2"),
            "multilingual": SentenceTransformerAdapter("paraphrase-multilingual-MiniLM-L12-v2")
        }
        
        # Создание отдельных хранилищ для каждого языка
        self.vector_stores = {}
        for lang, adapter in self.language_adapters.items():
            self.vector_stores[lang] = ContextVectorsFactory.create(
                "dynamic",
                vector_size=adapter.get_vector_size(),
                use_indexing=True
            )
    
    def add_concept(self, concept_id: str, texts: Dict[str, str]) -> bool:
        """Добавление концепта на разных языках
        
        Args:
            concept_id: Идентификатор концепта
            texts: Словарь {язык: текст}
        """
        success_count = 0
        
        for lang, text in texts.items():
            # Используем многоязычный адаптер для неанглийских языков
            adapter_key = "en" if lang == "en" else "multilingual"
            
            if adapter_key in self.language_adapters:
                try:
                    adapter = self.language_adapters[adapter_key]
                    embedding = adapter.encode(text, normalize=True)
                    
                    # Создаем уникальный ключ для каждого языка
                    key = f"{concept_id}_{lang}"
                    
                    if self.vector_stores[adapter_key].create_vector(key, embedding):
                        success_count += 1
                
                except Exception as e:
                    print(f"Error adding {lang} text for {concept_id}: {e}")
        
        return success_count > 0
    
    def find_similar_concepts(self, query_text: str, query_lang: str, 
                            target_langs: List[str] = None, top_k: int = 5) -> List[Dict]:
        """Поиск похожих концептов с возможностью межязыкового поиска"""
        
        # Определяем адаптер для запроса
        adapter_key = "en" if query_lang == "en" else "multilingual"
        adapter = self.language_adapters.get(adapter_key)
        
        if not adapter:
            return []
        
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = adapter.encode(query_text, normalize=True)
            
            results = []
            vector_store = self.vector_stores[adapter_key]
            
            # Поиск в соответствующем векторном хранилище
            similar_items = vector_store.get_most_similar_by_vector(
                query_embedding, top_n=top_k * 2  # берем больше для фильтрации
            )
            
            for item_key, similarity in similar_items:
                # Парсим ключ для извлечения concept_id и языка
                parts = item_key.rsplit('_', 1)
                if len(parts) == 2:
                    concept_id, lang = parts
                    
                    # Фильтруем по целевым языкам если указаны
                    if target_langs is None or lang in target_langs:
                        results.append({
                            "concept_id": concept_id,
                            "language": lang,
                            "similarity": similarity,
                            "key": item_key
                        })
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def get_concept_similarity(self, concept1_id: str, lang1: str,
                             concept2_id: str, lang2: str) -> float:
        """Вычисление сходства между концептами на разных языках"""
        
        key1 = f"{concept1_id}_{lang1}"
        key2 = f"{concept2_id}_{lang2}"
        
        # Определяем хранилища
        adapter_key1 = "en" if lang1 == "en" else "multilingual"
        adapter_key2 = "en" if lang2 == "en" else "multilingual"
        
        # Если языки требуют разных адаптеров, используем многоязычный
        if adapter_key1 != adapter_key2:
            adapter_key = "multilingual"
        else:
            adapter_key = adapter_key1
        
        vector_store = self.vector_stores[adapter_key]
        
        return vector_store.similarity(key1, key2) or 0.0

# Использование
multilingual_store = MultilingualVectorStore()

# Добавление концептов на разных языках
concepts = [
    ("apple", {
        "en": "Apple is a red fruit",
        "es": "La manzana es una fruta roja",
        "fr": "La pomme est un fruit rouge"
    }),
    ("car", {
        "en": "Car is a vehicle for transportation",
        "es": "El coche es un vehículo de transporte",
        "fr": "La voiture est un véhicule de transport"
    })
]

for concept_id, texts in concepts:
    multilingual_store.add_concept(concept_id, texts)

# Межязыковой поиск
results = multilingual_store.find_similar_concepts(
    query_text="fruta roja",
    query_lang="es",
    target_langs=["en", "fr"],
    top_k=3
)

for result in results:
    print(f"Concept: {result['concept_id']} ({result['language']}) - "
          f"Similarity: {result['similarity']:.3f}")
```

### Создание системы с адаптивными векторами

```python
from neurograph.contextvec.impl.dynamic import DynamicContextVectors
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter
import numpy as np
from typing import List, Dict, Any
import time

class AdaptiveVectorSystem:
    """Система векторов, адаптирующаяся к пользовательским предпочтениям"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformerAdapter(model_name)
        self.vectors = DynamicContextVectors(
            vector_size=self.encoder.get_vector_size(),
            use_indexing=True
        )
        
        # Статистика взаимодействий
        self.interaction_history = {}
        self.feedback_weights = {
            "positive": 0.1,    # learning rate для положительной обратной связи
            "negative": -0.05,  # learning rate для отрицательной обратной связи
            "click": 0.02,      # learning rate для кликов
            "time_spent": 0.01  # learning rate для времени просмотра
        }
    
    def add_item(self, item_id: str, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Добавление элемента в систему"""
        try:
            embedding = self.encoder.encode(text, normalize=True)
            success = self.vectors.create_vector(item_id, embedding)
            
            if success and metadata:
                # Сохраняем метаданные
                self.interaction_history[item_id] = {
                    "metadata": metadata,
                    "interactions": [],
                    "last_updated": time.time()
                }
            
            return success
        except Exception as e:
            print(f"Error adding item {item_id}: {e}")
            return False
    
    def record_interaction(self, item_id: str, interaction_type: str, 
                         strength: float = 1.0, context: Dict[str, Any] = None):
        """Запись взаимодействия пользователя с элементом"""
        if item_id not in self.interaction_history:
            self.interaction_history[item_id] = {
                "interactions": [],
                "last_updated": time.time()
            }
        
        interaction = {
            "type": interaction_type,
            "strength": strength,
            "timestamp": time.time(),
            "context": context or {}
        }
        
        self.interaction_history[item_id]["interactions"].append(interaction)
        
        # Адаптация вектора на основе взаимодействия
        self._adapt_vector(item_id, interaction)
    
    def _adapt_vector(self, item_id: str, interaction: Dict[str, Any]):
        """Адаптация вектора на основе взаимодействия"""
        if not self.vectors.has_key(item_id):
            return
        
        interaction_type = interaction["type"]
        strength = interaction["strength"]
        
        if interaction_type in self.feedback_weights:
            learning_rate = self.feedback_weights[interaction_type] * strength
            
            # Получаем текущий вектор
            current_vector = self.vectors.get_vector(item_id)
            if current_vector is None:
                return
            
            # Создаем "целевой" вектор на основе контекста взаимодействия
            target_vector = self._generate_target_vector(interaction, current_vector)
            
            # Обновляем вектор с учетом обратной связи
            if target_vector is not None:
                self.vectors.update_vector(item_id, target_vector, abs(learning_rate))
    
    def _generate_target_vector(self, interaction: Dict[str, Any], 
                              current_vector: np.ndarray) -> np.ndarray:
        """Генерация целевого вектора на основе контекста взаимодействия"""
        context = interaction.get("context", {})
        
        # Если есть связанные элементы в контексте
        if "related_items" in context:
            related_vectors = []
            for related_id in context["related_items"]:
                related_vector = self.vectors.get_vector(related_id)
                if related_vector is not None:
                    related_vectors.append(related_vector)
            
            if related_vectors:
                # Усредняем векторы связанных элементов
                avg_related = np.mean(related_vectors, axis=0)
                
                # Смешиваем с текущим вектором
                interaction_type = interaction["type"]
                if interaction_type in ["positive", "click"]:
                    # Приближаем к связанным элементам
                    return 0.7 * current_vector + 0.3 * avg_related
                elif interaction_type == "negative":
                    # Отдаляем от связанных элементов
                    return 2 * current_vector - avg_related
        
        # Если есть текстовый контекст
        if "text_context" in context:
            try:
                context_embedding = self.encoder.encode(
                    context["text_context"], normalize=True
                )
                
                interaction_type = interaction["type"]
                if interaction_type in ["positive", "click"]:
                    return 0.8 * current_vector + 0.2 * context_embedding
                elif interaction_type == "negative":
                    return 1.2 * current_vector - 0.2 * context_embedding
            except:
                pass
        
        return current_vector
    
    def get_personalized_recommendations(self, query: str, user_context: Dict[str, Any] = None,
                                       top_k: int = 10) -> List[Dict[str, Any]]:
        """Получение персонализированных рекомендаций"""
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self.encoder.encode(query, normalize=True)
            
            # Находим похожие элементы
            similar_items = self.vectors.get_most_similar_by_vector(
                query_embedding, top_n=top_k * 2
            )
            
            # Ранжируем с учетом истории взаимодействий
            ranked_items = []
            for item_id, base_similarity in similar_items:
                # Получаем историю взаимодействий
                history = self.interaction_history.get(item_id, {})
                interactions = history.get("interactions", [])
                
                # Вычисляем персонализированный скор
                personalization_boost = self._calculate_personalization_boost(
                    interactions, user_context
                )
                
                final_score = base_similarity + personalization_boost
                
                ranked_items.append({
                    "item_id": item_id,
                    "base_similarity": base_similarity,
                    "personalization_boost": personalization_boost,
                    "final_score": final_score,
                    "metadata": history.get("metadata", {})
                })
            
            # Сортируем по финальному скору
            ranked_items.sort(key=lambda x: x["final_score"], reverse=True)
            
            return ranked_items[:top_k]
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def _calculate_personalization_boost(self, interactions: List[Dict], 
                                       user_context: Dict[str, Any] = None) -> float:
        """Вычисление персонализированного буста на основе истории"""
        if not interactions:
            return 0.0
        
        boost = 0.0
        recent_weight = 2.0  # вес для недавних взаимодействий
        time_decay = 86400   # день в секундах
        current_time = time.time()
        
        for interaction in interactions:
            interaction_type = interaction["type"]
            strength = interaction["strength"]
            timestamp = interaction["timestamp"]
            
            # Временное затухание
            time_diff = current_time - timestamp
            decay_factor = max(0.1, 1.0 - (time_diff / time_decay))
            
            # Базовый вес взаимодействия
            if interaction_type == "positive":
                base_weight = 0.3
            elif interaction_type == "click":
                base_weight = 0.1
            elif interaction_type == "time_spent":
                base_weight = 0.05 * min(strength, 10)  # ограничиваем влияние времени
            elif interaction_type == "negative":
                base_weight = -0.2
            else:
                base_weight = 0.0
            
            boost += base_weight * strength * decay_factor
        
        # Ограничиваем максимальный буст
        return max(-0.5, min(0.5, boost))
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Получение статистики системы"""
        total_items = len(self.vectors.get_all_keys())
        items_with_interactions = len(self.interaction_history)
        
        interaction_types = {}
        total_interactions = 0
        
        for item_data in self.interaction_history.values():
            for interaction in item_data.get("interactions", []):
                interaction_type = interaction["type"]
                interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
                total_interactions += 1
        
        return {
            "total_items": total_items,
            "items_with_interactions": items_with_interactions,
            "total_interactions": total_interactions,
            "interaction_types": interaction_types,
            "average_interactions_per_item": total_interactions / max(items_with_interactions, 1)
        }

# Использование
adaptive_system = AdaptiveVectorSystem()

# Добавление элементов
items = [
    ("article1", "Machine learning fundamentals", {"category": "tech", "difficulty": "beginner"}),
    ("article2", "Advanced neural networks", {"category": "tech", "difficulty": "advanced"}),
    ("article3", "Cooking pasta recipes", {"category": "food", "difficulty": "easy"}),
    ("article4", "Python programming guide", {"category": "tech", "difficulty": "intermediate"}),
]

for item_id, text, metadata in items:
    adaptive_system.add_item(item_id, text, metadata)

# Симуляция взаимодействий пользователя
adaptive_system.record_interaction("article1", "positive", strength=1.0)
adaptive_system.record_interaction("article1", "click", strength=1.0)
adaptive_system.record_interaction("article4", "positive", strength=0.8)
adaptive_system.record_interaction("article2", "negative", strength=1.0)

# Получение персонализированных рекомендаций
recommendations = adaptive_system.get_personalized_recommendations(
    query="programming tutorial",
    top_k=3
)

print("Персонализированные рекомендации:")
for rec in recommendations:
    print(f"Item: {rec['item_id']}")
    print(f"  Base similarity: {rec['base_similarity']:.3f}")
    print(f"  Personalization boost: {rec['personalization_boost']:.3f}")
    print(f"  Final score: {rec['final_score']:.3f}")
    print(f"  Category: {rec['metadata'].get('category', 'unknown')}\n")

# Статистика системы
stats = adaptive_system.get_system_statistics()
print(f"Статистика системы: {stats}")
```

## Интеграция с другими модулями

### Интеграция с SemGraph

```python
from neurograph.contextvec import ContextVectorsFactory
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter

class GraphVectorIntegration:
    """Интеграция векторных представлений с семантическим графом"""
    
    def __init__(self, graph, vector_size: int = 384):
        self.graph = graph  # экземпляр ISemGraph
        self.encoder = SentenceTransformerAdapter("all-MiniLM-L6-v2")
        self.vectors = ContextVectorsFactory.create(
            "dynamic", 
            vector_size=vector_size,
            use_indexing=True
        )
    
    def add_node_with_vector(self, node_id: str, text_description: str, **attributes):
        """Добавление узла в граф с автоматическим созданием вектора"""
        # Добавляем узел в граф
        self.graph.add_node(node_id, description=text_description, **attributes)
        
        # Создаем векторное представление
        embedding = self.encoder.encode(text_description, normalize=True)
        self.vectors.create_vector(node_id, embedding)
        
        return True
    
    def find_semantically_similar_nodes(self, query_text: str, top_k: int = 5) -> List[str]:
        """Поиск семантически похожих узлов в графе"""
        query_embedding = self.encoder.encode(query_text, normalize=True)
        similar_vectors = self.vectors.get_most_similar_by_vector(query_embedding, top_n=top_k)
        
        # Фильтруем только существующие в графе узлы
        existing_nodes = []
        for node_id, similarity in similar_vectors:
            if self.graph.has_node(node_id):
                existing_nodes.append(node_id)
        
        return existing_nodes
    
    def get_node_vector_similarity(self, node1: str, node2: str) -> float:
        """Получение векторного сходства между узлами"""
        return self.vectors.similarity(node1, node2) or 0.0
    
    def update_node_vector_from_graph(self, node_id: str):
        """Обновление вектора узла на основе его связей в графе"""
        if not self.graph.has_node(node_id):
            return False
        
        # Получаем соседей узла
        neighbors = self.graph.get_neighbors(node_id)
        if not neighbors:
            return False
        
        # Собираем векторы соседей
        neighbor_vectors = []
        for neighbor_id in neighbors:
            neighbor_vector = self.vectors.get_vector(neighbor_id)
            if neighbor_vector is not None:
                neighbor_vectors.append(neighbor_vector)
        
        if neighbor_vectors:
            # Усредняем векторы соседей
            avg_neighbor_vector = np.mean(neighbor_vectors, axis=0)
            
            # Обновляем вектор узла с учетом соседей
            self.vectors.update_vector(node_id, avg_neighbor_vector, learning_rate=0.1)
            return True
        
        return False

# Использование
# graph = SemGraphFactory.create("memory_efficient")  # предполагаем наличие графа
# integration = GraphVectorIntegration(graph)
# 
# integration.add_node_with_vector("apple", "Apple is a red fruit that grows on trees")
# integration.add_node_with_vector("fruit", "Fruit is a plant food that contains seeds")
# 
# similar_nodes = integration.find_semantically_similar_nodes("red food", top_k=3)
```

### Интеграция с Memory

```python
from neurograph.contextvec import ContextVectorsFactory
from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter

class MemoryVectorEnhancer:
    """Улучшение системы памяти с помощью векторных представлений"""
    
    def __init__(self, memory_system):
        self.memory = memory_system  # экземпляр IMemory
        self.encoder = SentenceTransformerAdapter("all-MiniLM-L6-v2")
        self.vectors = ContextVectorsFactory.create("dynamic", vector_size=384)
    
    def add_memory_item_with_vector(self, content: str, content_type: str = "text", 
                                  metadata: Dict[str, Any] = None) -> str:
        """Добавление элемента в память с автоматическим созданием вектора"""
        from neurograph.memory.base import MemoryItem
        
        # Создаем векторное представление
        embedding = self.encoder.encode(content, normalize=True)
        
        # Создаем элемент памяти
        memory_item = MemoryItem(
            content=content,
            embedding=embedding,
            content_type=content_type,
            metadata=metadata or {}
        )
        
        # Добавляем в память
        item_id = self.memory.add(memory_item)
        
        # Дублируем в векторное хранилище для быстрого поиска
        self.vectors.create_vector(item_id, embedding)
        
        return item_id
    
    def semantic_memory_search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Семантический поиск в памяти"""
        query_embedding = self.encoder.encode(query, normalize=True)
        
        # Поиск в векторном хранилище
        similar_items = self.vectors.get_most_similar_by_vector(query_embedding, top_n=limit)
        
        # Проверяем, что элементы еще существуют в памяти
        valid_results = []
        for item_id, similarity in similar_items:
            memory_item = self.memory.get(item_id)
            if memory_item is not None:
                valid_results.append((item_id, similarity))
        
        return valid_results
    
    def consolidate_similar_memories(self, similarity_threshold: float = 0.9):
        """Консолидация похожих воспоминаний"""
        all_items = self.vectors.get_all_keys()
        consolidated_pairs = []
        
        for i, item1_id in enumerate(all_items):
            for item2_id in all_items[i+1:]:
                similarity = self.vectors.similarity(item1_id, item2_id)
                
                if similarity and similarity > similarity_threshold:
                    # Получаем элементы памяти
                    item1 = self.memory.get(item1_id)
                    item2 = self.memory.get(item2_id)
                    
                    if item1 and item2:
                        # Создаем консолидированный элемент
                        consolidated_content = f"{item1.content} | {item2.content}"
                        consolidated_embedding = np.mean([item1.embedding, item2.embedding], axis=0)
                        
                        from neurograph.memory.base import MemoryItem
                        consolidated_item = MemoryItem(
                            content=consolidated_content,
                            embedding=consolidated_embedding,
                            content_type="consolidated",
                            metadata={
                                "source_items": [item1_id, item2_id],
                                "consolidation_similarity": similarity
                            }
                        )
                        
                        # Добавляем консолидированный элемент
                        new_item_id = self.memory.add(consolidated_item)
                        self.vectors.create_vector(new_item_id, consolidated_embedding)
                        
                        # Удаляем оригинальные элементы
                        self.memory.remove(item1_id)
                        self.memory.remove(item2_id)
                        self.vectors.remove_vector(item1_id)
                        self.vectors.remove_vector(item2_id)
                        
                        consolidated_pairs.append((item1_id, item2_id, new_item_id))
        
        return consolidated_pairs

# Использование
# memory = MemoryFactory.create("biomorphic")  # предполагаем наличие памяти
# enhancer = MemoryVectorEnhancer(memory)
# 
# # Добавление элементов
# item1_id = enhancer.add_memory_item_with_vector("Apple is a red fruit")
# item2_id = enhancer.add_memory_item_with_vector("Red apple is sweet")
# 
# # Семантический поиск
# results = enhancer.semantic_memory_search("fruit", limit=5)
```

## Конфигурация и настройки

### Конфигурация по умолчанию

```python
default_contextvec_config = {
    "static_vectors": {
        "vector_size": 100,
        "normalize_vectors": True
    },
    "dynamic_vectors": {
        "vector_size": 384,
        "use_indexing": True,
        "max_items": 100000,
        "learning_rate": 0.1
    },
    "adapters": {
        "word2vec": {
            "normalize": True,
            "handle_oov": "zero"  # "zero", "random", "skip"
        },
        "sentence_transformer": {
            "model_name": "all-MiniLM-L6-v2",
            "normalize_embeddings": True,
            "batch_size": 32
        }
    },
    "lightweight_models": {
        "hashing_vectorizer": {
            "vector_size": 1000,
            "ngram_range": [1, 2],
            "lowercase": True
        },
        "random_projection": {
            "preserve_distances": True,
            "random_seed": 42
        }
    },
    "indexing": {
        "hnsw": {
            "ef_construction": 200,
            "m": 16,
            "max_m": 16,
            "max_m0": 32
        }
    }
}
```

### Создание конфигурируемой системы

```python
from neurograph.core.config import Configuration
from neurograph.contextvec import ContextVectorsFactory

class ConfigurableVectorSystem:
    """Конфигурируемая система векторных представлений"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.systems = {}
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Инициализация систем на основе конфигурации"""
        
        # Инициализация основного хранилища векторов
        main_config = self.config.get("main_vectors", {})
        vector_type = main_config.get("type", "dynamic")
        vector_params = main_config.get("params", {})
        
        self.main_vectors = ContextVectorsFactory.create(vector_type, **vector_params)
        
        # Инициализация адаптеров
        adapters_config = self.config.get("adapters", {})
        self.adapters = {}
        
        for adapter_name, adapter_config in adapters_config.items():
            if adapter_config.get("enabled", False):
                self.adapters[adapter_name] = self._create_adapter(adapter_name, adapter_config)
        
        # Инициализация специализированных хранилищ
        specialized_config = self.config.get("specialized_vectors", {})
        self.specialized_vectors = {}
        
        for name, spec_config in specialized_config.items():
            if spec_config.get("enabled", False):
                vector_type = spec_config.get("type", "static")
                vector_params = spec_config.get("params", {})
                self.specialized_vectors[name] = ContextVectorsFactory.create(vector_type, **vector_params)
    
    def _create_adapter(self, adapter_name: str, adapter_config: Dict[str, Any]):
        """Создание адаптера на основе конфигурации"""
        if adapter_name == "sentence_transformer":
            from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter
            model_name = adapter_config.get("model_name", "all-MiniLM-L6-v2")
            return SentenceTransformerAdapter(model_name)
        
        elif adapter_name == "word2vec":
            from neurograph.contextvec.adapters.word2vec import Word2VecAdapter
            model_path = adapter_config.get("model_path")
            if model_path:
                return Word2VecAdapter(model_path)
        
        elif adapter_name == "hashing":
            from neurograph.contextvec.models.lightweight import HashingVectorizer
            params = adapter_config.get("params", {})
            return HashingVectorizer(**params)
        
        return None
    
    def encode_text(self, text: str, adapter_name: str = "default") -> np.ndarray:
        """Кодирование текста с использованием указанного адаптера"""
        if adapter_name == "default" and "sentence_transformer" in self.adapters:
            adapter_name = "sentence_transformer"
        
        adapter = self.adapters.get(adapter_name)
        if adapter:
            return adapter.encode(text, normalize=True)
        else:
            raise ValueError(f"Adapter '{adapter_name}' not found or not enabled")
    
    def add_to_system(self, system_name: str, key: str, text: str) -> bool:
        """Добавление текста в указанную систему векторов"""
        if system_name == "main":
            target_system = self.main_vectors
        elif system_name in self.specialized_vectors:
            target_system = self.specialized_vectors[system_name]
        else:
            return False
        
        # Определяем адаптер для кодирования
        encoding_config = self.config.get(f"{system_name}_vectors.encoding", {})
        adapter_name = encoding_config.get("adapter", "default")
        
        try:
            embedding = self.encode_text(text, adapter_name)
            return target_system.create_vector(key, embedding)
        except Exception as e:
            print(f"Error adding to {system_name}: {e}")
            return False
    
    def search_in_system(self, system_name: str, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Поиск в указанной системе векторов"""
        if system_name == "main":
            target_system = self.main_vectors
        elif system_name in self.specialized_vectors:
            target_system = self.specialized_vectors[system_name]
        else:
            return []
        
        try:
            # Определяем адаптер для кодирования запроса
            encoding_config = self.config.get(f"{system_name}_vectors.encoding", {})
            adapter_name = encoding_config.get("adapter", "default")
            
            query_embedding = self.encode_text(query, adapter_name)
            return target_system.get_most_similar_by_vector(query_embedding, top_n=top_k)
        except Exception as e:
            print(f"Error searching in {system_name}: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Получение статистики всех систем"""
        stats = {
            "main_vectors": {
                "size": len(self.main_vectors.get_all_keys()),
                "type": type(self.main_vectors).__name__
            },
            "specialized_vectors": {},
            "adapters": list(self.adapters.keys())
        }
        
        for name, system in self.specialized_vectors.items():
            stats["specialized_vectors"][name] = {
                "size": len(system.get_all_keys()),
                "type": type(system).__name__
            }
        
        return stats

# Пример конфигурации
config = Configuration({
    "main_vectors": {
        "type": "dynamic",
        "params": {
            "vector_size": 384,
            "use_indexing": True
        },
        "encoding": {
            "adapter": "sentence_transformer"
        }
    },
    "specialized_vectors": {
        "keywords": {
            "enabled": True,
            "type": "static",
            "params": {"vector_size": 100},
            "encoding": {"adapter": "hashing"}
        },
        "documents": {
            "enabled": True,
            "type": "dynamic",
            "params": {"vector_size": 384},
            "encoding": {"adapter": "sentence_transformer"}
        }
    },
    "adapters": {
        "sentence_transformer": {
            "enabled": True,
            "model_name": "all-MiniLM-L6-v2"
        },
        "hashing": {
            "enabled": True,
            "params": {
                "vector_size": 100,
                "ngram_range": [1, 2]
            }
        }
    }
})

# Использование
vector_system = ConfigurableVectorSystem(config)

# Добавление в разные системы
vector_system.add_to_system("main", "concept1", "Apple is a red fruit")
vector_system.add_to_system("keywords", "kw1", "apple fruit red")
vector_system.add_to_system("documents", "doc1", "This document discusses various fruits including apples")

# Поиск в разных системах
main_results = vector_system.search_in_system("main", "red food", top_k=5)
keyword_results = vector_system.search_in_system("keywords", "fruit", top_k=3)

# Статистика
stats = vector_system.get_system_stats()
print(f"System statistics: {stats}")
```

## Производительность и оптимизация

### Профилирование и бенчмарки

```python
import time
import numpy as np
from typing import Dict, List, Callable
from neurograph.contextvec import ContextVectorsFactory
from neurograph.core.utils.metrics import timed, global_metrics

class VectorPerformanceBenchmark:
    """Бенчмарк для оценки производительности векторных операций"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_vector_systems(self, vector_sizes: List[int], item_counts: List[int]) -> Dict[str, Any]:
        """Бенчмарк различных систем векторов"""
        results = {}
        
        for vector_size in vector_sizes:
            for item_count in item_counts:
                print(f"Benchmarking vector_size={vector_size}, items={item_count}")
                
                # Тестируем статические векторы
                static_results = self._benchmark_static_vectors(vector_size, item_count)
                
                # Тестируем динамические векторы
                dynamic_results = self._benchmark_dynamic_vectors(vector_size, item_count)
                
                key = f"size_{vector_size}_items_{item_count}"
                results[key] = {
                    "static": static_results,
                    "dynamic": dynamic_results,
                    "vector_size": vector_size,
                    "item_count": item_count
                }
        
        return results
    
    @timed("static_vector_benchmark")
    def _benchmark_static_vectors(self, vector_size: int, item_count: int) -> Dict[str, float]:
        """Бенчмарк статических векторов"""
        vectors = ContextVectorsFactory.create("static", vector_size=vector_size)
        
        # Генерация тестовых данных
        test_vectors = [np.random.random(vector_size) for _ in range(item_count)]
        test_keys = [f"item_{i}" for i in range(item_count)]
        
        # Тест добавления
        start_time = time.time()
        for i, (key, vector) in enumerate(zip(test_keys, test_vectors)):
            vectors.create_vector(key, vector)
        add_time = time.time() - start_time
        
        # Тест получения
        start_time = time.time()
        for key in test_keys[:100]:  # тестируем первые 100
            vectors.get_vector(key)
        get_time = time.time() - start_time
        
        # Тест поиска похожих
        start_time = time.time()
        for i in range(min(10, item_count)):  # тестируем 10 поисков
            vectors.get_most_similar(test_keys[i], top_n=5)
        search_time = time.time() - start_time
        
        # Тест вычисления сходства
        start_time = time.time()
        for i in range(min(100, item_count-1)):
            vectors.similarity(test_keys[i], test_keys[i+1])
        similarity_time = time.time() - start_time
        
        return {
            "add_time": add_time,
            "get_time": get_time,
            "search_time": search_time,
            "similarity_time": similarity_time,
            "add_rate": item_count / add_time,
            "get_rate": 100 / get_time if get_time > 0 else float('inf'),
            "search_rate": 10 / search_time if search_time > 0 else float('inf')
        }
    
    @timed("dynamic_vector_benchmark")
    def _benchmark_dynamic_vectors(self, vector_size: int, item_count: int) -> Dict[str, float]:
        """Бенчмарк динамических векторов"""
        vectors = ContextVectorsFactory.create("dynamic", vector_size=vector_size, use_indexing=True)
        
        # Аналогичные тесты для динамических векторов
        test_vectors = [np.random.random(vector_size) for _ in range(item_count)]
        test_keys = [f"item_{i}" for i in range(item_count)]
        
        # Тест добавления
        start_time = time.time()
        for key, vector in zip(test_keys, test_vectors):
            vectors.create_vector(key, vector)
        add_time = time.time() - start_time
        
        # Тест обновления векторов
        start_time = time.time()
        for i in range(min(100, item_count)):
            new_vector = np.random.random(vector_size)
            vectors.update_vector(test_keys[i], new_vector, learning_rate=0.1)
        update_time = time.time() - start_time
        
        # Тест поиска с индексом
        start_time = time.time()
        for i in range(min(10, item_count)):
            vectors.get_most_similar(test_keys[i], top_n=5)
        search_time = time.time() - start_time
        
        # Тест усреднения векторов
        start_time = time.time()
        for i in range(0, min(50, item_count), 5):
            keys_to_average = test_keys[i:i+5]
            vectors.average_vectors(keys_to_average)
        average_time = time.time() - start_time
        
        return {
            "add_time": add_time,
            "update_time": update_time,
            "search_time": search_time,
            "average_time": average_time,
            "add_rate": item_count / add_time,
            "update_rate": 100 / update_time if update_time > 0 else float('inf'),
            "search_rate": 10 / search_time if search_time > 0 else float('inf')
        }
    
    def benchmark_adapters(self, texts: List[str]) -> Dict[str, Any]:
        """Бенчмарк различных адаптеров"""
        results = {}
        
        # Тест HashingVectorizer
        start_time = time.time()
        try:
            from neurograph.contextvec.models.lightweight import HashingVectorizer
            vectorizer = HashingVectorizer(vector_size=1000)
            
            for text in texts:
                vectorizer.transform(text)
            
            hashing_time = time.time() - start_time
            results["hashing"] = {
                "total_time": hashing_time,
                "rate": len(texts) / hashing_time,
                "status": "success"
            }
        except Exception as e:
            results["hashing"] = {"status": "error", "error": str(e)}
        
        # Тест SentenceTransformer (если доступен)
        start_time = time.time()
        try:
            from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter
            adapter = SentenceTransformerAdapter("all-MiniLM-L6-v2")
            
            # Пакетная обработка для эффективности
            adapter.encode_batch(texts, batch_size=32)
            
            sentence_time = time.time() - start_time
            results["sentence_transformer"] = {
                "total_time": sentence_time,
                "rate": len(texts) / sentence_time,
                "status": "success"
            }
        except Exception as e:
            results["sentence_transformer"] = {"status": "error", "error": str(e)}
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Запуск комплексного бенчмарка"""
        print("Starting comprehensive vector benchmark...")
        
        # Тест различных размеров векторов и количества элементов
        vector_benchmark = self.benchmark_vector_systems(
            vector_sizes=[100, 384, 768],
            item_counts=[1000, 5000, 10000]
        )
        
        # Тест адаптеров
        test_texts = [
            f"This is test document number {i} with some content"
            for i in range(1000)
        ]
        adapter_benchmark = self.benchmark_adapters(test_texts)
        
        # Сводная статистика
        summary = {
            "vector_systems": vector_benchmark,
            "adapters": adapter_benchmark,
            "system_metrics": global_metrics.get_all_metrics()
        }
        
        return summary

# Использование
benchmark = VectorPerformanceBenchmark()
results = benchmark.run_comprehensive_benchmark()

# Анализ результатов
print("\n=== Benchmark Results ===")
for config, data in results["vector_systems"].items():
    print(f"\nConfiguration: {config}")
    print(f"Static vectors - Add rate: {data['static']['add_rate']:.2f} items/sec")
    print(f"Dynamic vectors - Add rate: {data['dynamic']['add_rate']:.2f} items/sec")
    print(f"Dynamic vectors - Search rate: {data['dynamic']['search_rate']:.2f} searches/sec")

print(f"\n=== Adapter Performance ===")
for adapter, data in results["adapters"].items():
    if data["status"] == "success":
        print(f"{adapter}: {data['rate']:.2f} texts/sec")
    else:
        print(f"{adapter}: {data['status']} - {data.get('error', 'Unknown error')}")
```

## Лучшие практики

### 1. Выбор типа векторных представлений

```python
# Для статических словарей и простых случаев
static_vectors = ContextVectorsFactory.create("static", vector_size=100)

# Для динамически обновляемых систем с большим объемом данных
dynamic_vectors = ContextVectorsFactory.create("dynamic", vector_size=384, use_indexing=True)

# Для экспериментов и быстрого прототипирования
from neurograph.contextvec.models.lightweight import HashingVectorizer
vectorizer = HashingVectorizer(vector_size=1000, ngram_range=(1, 2))
```

### 2. Оптимизация производительности

```python
# Используйте пакетную обработку когда возможно
texts = ["text1", "text2", "text3", ...]
embeddings = adapter.encode_batch(texts, batch_size=32)

# Нормализуйте векторы для косинусного сходства
vectors.create_vector("key", embedding, normalize=True)

# Используйте индексацию для больших коллекций
large_vectors = ContextVectorsFactory.create("dynamic", use_indexing=True)

# Кешируйте часто используемые вычисления
from neurograph.core.cache import cached

@cached(ttl=300)
def expensive_similarity_computation(key1, key2):
    return vectors.similarity(key1, key2)
```

### 3. Управление памятью

```python
# Регулярно очищайте неиспользуемые векторы
def cleanup_old_vectors(vectors, max_age_seconds=86400):
    current_time = time.time()
    keys_to_remove = []
    
    for key in vectors.get_all_keys():
        # Предполагаем, что в метаданных есть timestamp
        if should_remove_vector(key, current_time, max_age_seconds):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        vectors.remove_vector(key)
    
    return len(keys_to_remove)

# Используйте проекции для снижения размерности
from neurograph.contextvec.models.lightweight import RandomProjection

projection = RandomProjection(input_dim=768, output_dim=128)
reduced_vector = projection.transform(high_dim_vector)
```

### 4. Обработка ошибок

```python
from neurograph.core.errors import VectorError, InvalidVectorDimensionError

try:
    vectors.create_vector("key", wrong_size_vector)
except InvalidVectorDimensionError as e:
    logger.error(f"Vector dimension mismatch: {e}")
    # Обработка ошибки размерности
except VectorError as e:
    logger.error(f"Vector operation failed: {e}")
    # Общая обработка векторных ошибок
```

### 5. Мониторинг и логирование

```python
from neurograph.core.logging import get_logger
from neurograph.core.utils.metrics import global_metrics

logger = get_logger("contextvec.operations")

def monitored_vector_operation(vectors, operation_name, *args, **kwargs):
    """Обертка для мониторинга векторных операций"""
    start_time = time.time()
    
    try:
        result = getattr(vectors, operation_name)(*args, **kwargs)
        
        # Записываем успешную операцию
        global_metrics.increment_counter(f"vector_operations_{operation_name}_success")
        execution_time = time.time() - start_time
        global_metrics.record_time(f"vector_operations_{operation_name}", execution_time)
        
        logger.debug(f"Vector operation {operation_name} completed in {execution_time:.3f}s")
        return result
        
    except Exception as e:
        global_metrics.increment_counter(f"vector_operations_{operation_name}_error")
        logger.error(f"Vector operation {operation_name} failed: {e}")
        raise

# Использование
result = monitored_vector_operation(vectors, "get_most_similar", "apple", top_n=5)
```

## Расширение функциональности

### Создание пользовательского адаптера

```python
from neurograph.contextvec.base import IContextVectors
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class CustomEmbeddingAdapter:
    """Пользовательский адаптер для специфической модели эмбеддингов"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.vector_size = model_config.get("vector_size", 512)
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация пользовательской модели"""
        # Здесь может быть загрузка специфической модели
        # Например, своя fine-tuned модель или внешний API
        pass
    
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Кодирование текста в вектор"""
        # Пример простой реализации
        # В реальности здесь будет вызов вашей модели
        
        # Простой хеш-основанный подход для демонстрации
        hash_value = hash(text.lower())
        
        # Создаем детерминированный вектор на основе хеша
        np.random.seed(abs(hash_value) % (2**31))
        vector = np.random.normal(0, 1, self.vector_size).astype(np.float32)
        
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        return vector
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, 
                    normalize: bool = True) -> np.ndarray:
        """Пакетное кодирование текстов"""
        vectors = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        
        for i, text in enumerate(texts):
            vectors[i] = self.encode(text, normalize=normalize)
        
        return vectors
    
    def get_vector_size(self) -> int:
        return self.vector_size
    
    def fine_tune_on_pairs(self, positive_pairs: List[Tuple[str, str]], 
                          negative_pairs: List[Tuple[str, str]]):
        """Дообучение на парах похожих/непохожих текстов"""
        # Здесь может быть реализована логика дообучения
        # на основе положительных и отрицательных пар
        pass

# Регистрация пользовательского адаптера
def create_custom_adapter(config: Dict[str, Any]) -> CustomEmbeddingAdapter:
    return CustomEmbeddingAdapter(config)

# Использование
custom_config = {
    "vector_size": 256,
    "model_type": "custom_hash",
    "normalization": True
}

adapter = create_custom_adapter(custom_config)
embedding = adapter.encode("Test text for custom adapter")
```

### Создание специализированного векторного хранилища

```python
from neurograph.contextvec.base import IContextVectors
from neurograph.core.cache import Cache
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
import time

class TimedVectorStore(IContextVectors):
    """Векторное хранилище с поддержкой времени жизни векторов"""
    
    def __init__(self, vector_size: int, default_ttl: float = 3600.0):
        self.vector_size = vector_size
        self.default_ttl = default_ttl
        self.vectors: Dict[str, np.ndarray] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttls: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Запускаем поток для очистки устаревших векторов
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def create_vector(self, key: str, vector: np.ndarray, ttl: Optional[float] = None) -> bool:
        """Создание вектора с указанным временем жизни"""
        if vector.shape != (self.vector_size,):
            return False
        
        # Нормализация вектора
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        with self._lock:
            self.vectors[key] = vector.copy()
            self.timestamps[key] = time.time()
            self.ttls[key] = ttl or self.default_ttl
        
        return True
    
    def get_vector(self, key: str) -> Optional[np.ndarray]:
        """Получение вектора с проверкой времени жизни"""
        with self._lock:
            if not self._is_valid(key):
                return None
            
            # Обновляем время последнего доступа
            self.timestamps[key] = time.time()
            return self.vectors[key].copy()
    
    def similarity(self, key1: str, key2: str) -> Optional[float]:
        """Вычисление сходства с проверкой валидности ключей"""
        with self._lock:
            if not self._is_valid(key1) or not self._is_valid(key2):
                return None
            
            vector1 = self.vectors[key1]
            vector2 = self.vectors[key2]
            
            # Косинусное сходство
            return float(np.dot(vector1, vector2))
    
    def get_most_similar(self, key: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Поиск похожих векторов среди валидных"""
        with self._lock:
            if not self._is_valid(key):
                return []
            
            query_vector = self.vectors[key]
            similarities = []
            
            for other_key, other_vector in self.vectors.items():
                if other_key != key and self._is_valid(other_key):
                    similarity = float(np.dot(query_vector, other_vector))
                    similarities.append((other_key, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_n]
    
    def has_key(self, key: str) -> bool:
        """Проверка наличия валидного ключа"""
        with self._lock:
            return self._is_valid(key)
    
    def get_all_keys(self) -> List[str]:
        """Получение всех валидных ключей"""
        with self._lock:
            return [key for key in self.vectors.keys() if self._is_valid(key)]
    
    def remove_vector(self, key: str) -> bool:
        """Удаление вектора"""
        with self._lock:
            if key in self.vectors:
                del self.vectors[key]
                del self.timestamps[key]
                del self.ttls[key]
                return True
            return False
    
    def _is_valid(self, key: str) -> bool:
        """Проверка валидности ключа по времени жизни"""
        if key not in self.vectors:
            return False
        
        current_time = time.time()
        creation_time = self.timestamps[key]
        ttl = self.ttls[key]
        
        return (current_time - creation_time) < ttl
    
    def _cleanup_loop(self):
        """Цикл очистки устаревших векторов"""
        while True:
            try:
                with self._lock:
                    expired_keys = []
                    for key in list(self.vectors.keys()):
                        if not self._is_valid(key):
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.vectors[key]
                        del self.timestamps[key]
                        del self.ttls[key]
                
                # Спим 60 секунд между проверками
                time.sleep(60)
                
            except Exception as e:
                # Логируем ошибки, но продолжаем работу
                pass
    
    def extend_ttl(self, key: str, additional_time: float) -> bool:
        """Продление времени жизни вектора"""
        with self._lock:
            if key in self.ttls:
                self.ttls[key] += additional_time
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики хранилища"""
        with self._lock:
            current_time = time.time()
            valid_count = 0
            expired_count = 0
            
            for key in self.vectors.keys():
                if self._is_valid(key):
                    valid_count += 1
                else:
                    expired_count += 1
            
            return {
                "total_vectors": len(self.vectors),
                "valid_vectors": valid_count,
                "expired_vectors": expired_count,
                "vector_size": self.vector_size,
                "default_ttl": self.default_ttl
            }

# Регистрация в фабрике
ContextVectorsFactory.register_implementation("timed", TimedVectorStore)

# Использование
timed_vectors = ContextVectorsFactory.create("timed", vector_size=100, default_ttl=1800.0)

# Добавление вектора с коротким временем жизни
vector = np.random.random(100)
timed_vectors.create_vector("short_lived", vector, ttl=60.0)  # 1 минута

# Добавление вектора с долгим временем жизни
timed_vectors.create_vector("long_lived", vector, ttl=3600.0)  # 1 час
```

## Устранение неполадок

### Частые проблемы и решения

#### 1. Ошибки размерности векторов

```python
# Проблема: InvalidVectorDimensionError при создании вектора
try:
    vectors.create_vector("key", wrong_size_vector)
except InvalidVectorDimensionError as e:
    # Решение: проверяйте размерность перед добавлением
    expected_size = vectors.get_vector_size() if hasattr(vectors, 'get_vector_size') else 384
    
    if vector.shape[0] != expected_size:
        # Приведение к нужной размерности
        if vector.shape[0] > expected_size:
            # Обрезание
            vector = vector[:expected_size]
        else:
            # Дополнение нулями
            vector = np.pad(vector, (0, expected_size - vector.shape[0]))
    
    vectors.create_vector("key", vector)
```

#### 2. Проблемы с производительностью

```python
# Проблема: медленный поиск в больших коллекциях
# Решение: используйте индексацию

# Неэффективно для больших данных
slow_vectors = ContextVectorsFactory.create("static", vector_size=384)

# Эффективно с индексацией
fast_vectors = ContextVectorsFactory.create("dynamic", vector_size=384, use_indexing=True)

# Также используйте пакетную обработку
texts = ["text1", "text2", ...]
embeddings = adapter.encode_batch(texts, batch_size=64)
```

#### 3. Проблемы с памятью

```python
# Проблема: утечки памяти при работе с большими коллекциями
# Решение: регулярная очистка и использование контекстных менеджеров

class VectorContextManager:
    def __init__(self, vector_type: str, **kwargs):
        self.vector_type = vector_type
        self.kwargs = kwargs
        self.vectors = None
    
    def __enter__(self):
        self.vectors = ContextVectorsFactory.create(self.vector_type, **self.kwargs)
        return self.vectors
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Очистка ресурсов
        if hasattr(self.vectors, 'clear'):
            self.vectors.clear()
        self.vectors = None

# Использование
with VectorContextManager("dynamic", vector_size=384) as vectors:
    # Работа с векторами
    vectors.create_vector("key", vector)
    # Автоматическая очистка при выходе из контекста
```

#### 4. Проблемы с кодировкой текста

```python
# Проблема: ошибки при обработке текста с различными кодировками
def safe_encode_text(adapter, text: str, encoding: str = 'utf-8') -> Optional[np.ndarray]:
    """Безопасное кодирование текста с обработкой ошибок кодировки"""
    try:
        # Попытка нормализации текста
        if isinstance(text, bytes):
            text = text.decode(encoding, errors='replace')
        
        # Удаление невидимых символов
        import unicodedata
        text = ''.join(char for char in text if unicodedata.category(char) != 'Cc')
        
        # Кодирование
        return adapter.encode(text, normalize=True)
        
    except UnicodeDecodeError:
        # Попытка с другой кодировкой
        try:
            text = text.decode('latin-1', errors='replace')
            return adapter.encode(text, normalize=True)
        except:
            return None
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        return None
```

### Диагностика и отладка

```python
from neurograph.contextvec import ContextVectorsFactory
from neurograph.core.logging import get_logger
import numpy as np

def diagnose_vector_system(vectors: IContextVectors) -> Dict[str, Any]:
    """Диагностика состояния векторной системы"""
    logger = get_logger("contextvec.diagnostics")
    
    diagnostics = {
        "system_type": type(vectors).__name__,
        "total_vectors": len(vectors.get_all_keys()),
        "vector_statistics": {},
        "performance_test": {},
        "health_check": {}
    }
    
    try:
        # Анализ векторов
        all_keys = vectors.get_all_keys()
        if all_keys:
            sample_vector = vectors.get_vector(all_keys[0])
            if sample_vector is not None:
                diagnostics["vector_statistics"] = {
                    "vector_size": len(sample_vector),
                    "sample_norm": float(np.linalg.norm(sample_vector)),
                    "sample_mean": float(np.mean(sample_vector)),
                    "sample_std": float(np.std(sample_vector))
                }
        
        # Тест производительности
        test_vector = np.random.random(diagnostics["vector_statistics"].get("vector_size", 100))
        
        start_time = time.time()
        vectors.create_vector("_test_vector", test_vector)
        create_time = time.time() - start_time
        
        start_time = time.time()
        retrieved = vectors.get_vector("_test_vector")
        get_time = time.time() - start_time
        
        start_time = time.time()
        if len(all_keys) > 0:
            similar = vectors.get_most_similar(all_keys[0], top_n=5)
        search_time = time.time() - start_time
        
        # Очистка тестового вектора
        vectors.remove_vector("_test_vector")
        
        diagnostics["performance_test"] = {
            "create_time": create_time,
            "get_time": get_time,
            "search_time": search_time
        }
        
        # Проверка здоровья системы
        diagnostics["health_check"] = {
            "can_create": True,
            "can_retrieve": retrieved is not None,
            "can_search": len(similar) >= 0 if 'similar' in locals() else False,
            "status": "healthy"
        }
        
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        diagnostics["health_check"] = {
            "status": "error",
            "error": str(e)
        }
    
    return diagnostics

# Использование
vectors = ContextVectorsFactory.create("dynamic", vector_size=384)
diagnostics = diagnose_vector_system(vectors)
print(f"Диагностика системы: {diagnostics}")
```

## Заключение

Модуль **neurograph-contextvec** предоставляет мощную и гибкую систему для работы с векторными представлениями в рамках проекта NeuroGraph. Ключевые возможности включают:

- **Единый интерфейс** для различных типов векторных хранилищ
- **Адаптеры** для интеграции с популярными моделями эмбеддингов
- **Легковесные модели** для ресурсоограниченных сценариев
- **Масштабируемые решения** с поддержкой индексации
- **Гибкую конфигурацию** и расширяемость

**Рекомендации для разработчиков:**

1. **Выбирайте подходящий тип хранилища** в зависимости от ваших потребностей
2. **Используйте индексацию** для больших коллекций векторов
3. **Применяйте пакетную обработку** для повышения производительности
4. **Мониторьте ресурсы** при работе с большими объемами данных
5. **Тестируйте производительность** на реальных данных

Модуль интегрируется с другими компонентами NeuroGraph через единые интерфейсы, обеспечивая бесшовную работу в составе общей системы.