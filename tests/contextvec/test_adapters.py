"""Тесты для адаптеров векторных представлений."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List


class TestWord2VecAdapter:
    """Тесты для адаптера Word2Vec."""
    
    @pytest.mark.skipif(
        pytest.importorskip("gensim", reason="gensim is not installed") is None,
        reason="gensim is not installed"
    )
    def test_preprocess_text(self):
        """Проверка предварительной обработки текста."""
        # Импортируем адаптер только если тесты не пропускаются
        from neurograph.contextvec.adapters.word2vec import Word2VecAdapter
        
        # Создаем мок для KeyedVectors
        with patch("gensim.models.KeyedVectors.load_word2vec_format") as mock_load:
            # Создаем мок для KeyedVectors
            keyed_vectors_mock = MagicMock()
            keyed_vectors_mock.vector_size = 100
            mock_load.return_value = keyed_vectors_mock
            
            # Создаем адаптер
            adapter = Word2VecAdapter(model_path="dummy.bin")
            
            # Проверяем обработку текста
            tokens = adapter._preprocess_text("Hello, World! This is a TEST.")
            assert tokens == ["hello", "world", "this", "is", "a", "test"]


class TestSentenceTransformerAdapter:
    """Тесты для адаптера SentenceTransformer."""
    
    @pytest.mark.skipif(
        pytest.importorskip("sentence_transformers", reason="sentence_transformers is not installed") is None,
        reason="sentence_transformers is not installed"
    )
    def test_encode_with_mock(self):
        """Проверка кодирования текста с использованием мока."""
        # Импортируем адаптер только если тесты не пропускаются
        from neurograph.contextvec.adapters.sentence import SentenceTransformerAdapter
        
        # Создаем мок для SentenceTransformer
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            # Создаем мок для модели
            model_mock = MagicMock()
            model_mock.get_sentence_embedding_dimension.return_value = 384
            model_mock.encode.return_value = np.ones(384)
            mock_transformer.return_value = model_mock
            
            # Создаем адаптер
            adapter = SentenceTransformerAdapter(model_name="dummy-model")
            
            # Проверяем размерность вектора
            assert adapter.get_vector_size() == 384
            
            # Проверяем кодирование текста
            vector = adapter.encode("Test text")
            assert vector.shape == (384,)
            assert np.all(vector == 1.0)
            
            # Проверяем, что encode был вызван с правильными параметрами
            model_mock.encode.assert_called_once_with("Test text", normalize_embeddings=True)
            
            # Проверяем кодирование нескольких текстов
            model_mock.encode.return_value = np.ones((2, 384))
            vectors = adapter.encode_batch(["Text 1", "Text 2"])
            assert vectors.shape == (2, 384)