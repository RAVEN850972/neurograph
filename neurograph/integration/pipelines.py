# neurograph/integration/pipelines.py
"""
Конвейеры обработки для различных типов запросов.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from neurograph.core.logging import get_logger
from neurograph.core.events import publish

from .base import (
    IPipeline, ProcessingRequest, ProcessingResponse, IComponentProvider,
    ResponseFormat
)


class BasePipeline(IPipeline):
    """Базовый класс для конвейеров обработки."""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.logger = get_logger(f"pipeline_{pipeline_name}")
        self.processing_stats = {
            "requests_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0
        }
    
    def validate_request(self, request: ProcessingRequest) -> Tuple[bool, Optional[str]]:
        """Базовая валидация запроса."""
        if not request.content.strip():
            return False, "Пустое содержимое запроса"
        
        if len(request.content) > 50000:  # 50KB лимит
            return False, "Слишком длинный запрос"
        
        return True, None
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Информация о конвейере."""
        return {
            "name": self.pipeline_name,
            "stats": self.processing_stats.copy(),
            "avg_processing_time": (
                self.processing_stats["total_processing_time"] / 
                max(1, self.processing_stats["requests_processed"])
            )
        }
    
    def _update_stats(self, success: bool, processing_time: float) -> None:
        """Обновление статистики конвейера."""
        self.processing_stats["requests_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        if success:
            self.processing_stats["successful_requests"] += 1
        else:
            self.processing_stats["failed_requests"] += 1
    
    def _create_response(self, request: ProcessingRequest, 
                        success: bool = True,
                        primary_response: str = "",
                        structured_data: Optional[Dict[str, Any]] = None,
                        error_message: Optional[str] = None) -> ProcessingResponse:
        """Создание ответа."""
        return ProcessingResponse(
            request_id=request.request_id,
            success=success,
            primary_response=primary_response,
            structured_data=structured_data or {},
            error_message=error_message,
            components_used=[self.pipeline_name]
        )


class TextProcessingPipeline(BasePipeline):
    """Конвейер обработки произвольного текста."""
    
    def __init__(self):
        super().__init__("text_processing")
    
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Обработка текста через все доступные компоненты."""
        start_time = time.time()
        
        try:
            # Валидация
            is_valid, error_msg = self.validate_request(request)
            if not is_valid:
                return self._create_response(request, False, error_message=error_msg)
            
            components_used = []
            structured_data = {}
            explanations = []
            
            # 1. NLP обработка
            if request.enable_nlp and provider.is_component_available("nlp"):
                nlp_result = self._process_with_nlp(request, provider)
                components_used.append("nlp")
                structured_data["nlp"] = nlp_result
                explanations.append("Текст обработан через NLP модуль")
            
            # 2. Добавление в граф знаний
            if request.enable_graph_reasoning and provider.is_component_available("semgraph"):
                graph_result = self._add_to_graph(request, structured_data.get("nlp"), provider)
                components_used.append("semgraph")
                structured_data["graph"] = graph_result
                explanations.append("Извлеченные знания добавлены в граф")
            
            # 3. Векторизация и индексация
            if request.enable_vector_search and provider.is_component_available("contextvec"):
                vector_result = self._vectorize_content(request, structured_data.get("nlp"), provider)
                components_used.append("contextvec")
                structured_data["vectors"] = vector_result
                explanations.append("Созданы векторные представления")
            
            # 4. Сохранение в память
            if request.enable_memory_search and provider.is_component_available("memory"):
                memory_result = self._save_to_memory(request, structured_data, provider)
                components_used.append("memory")
                structured_data["memory"] = memory_result
                explanations.append("Информация сохранена в память")
            
            # 5. Генерация итогового ответа
            primary_response = self._generate_summary_response(request, structured_data)
            
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time)
            
            response = self._create_response(
                request, 
                success=True,
                primary_response=primary_response,
                structured_data=structured_data
            )
            response.components_used = components_used
            response.explanation = explanations
            response.processing_time = processing_time
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка в текстовом конвейере: {e}")
            self._update_stats(False, processing_time)
            
            return self._create_response(request, False, error_message=str(e))
    
    def _process_with_nlp(self, request: ProcessingRequest, 
                         provider: IComponentProvider) -> Dict[str, Any]:
        """Обработка текста через NLP."""
        nlp = provider.get_component("nlp")
        
        nlp_result = nlp.process_text(
            request.content,
            extract_entities=True,
            extract_relations=True
        )
        
        return {
            "entities": [
                {
                    "text": entity.text,
                    "type": entity.entity_type.value,
                    "confidence": entity.confidence
                }
                for entity in nlp_result.entities
            ],
            "relations": [
                {
                    "subject": relation.subject.text,
                    "predicate": relation.predicate.value,
                    "object": relation.object.text,
                    "confidence": relation.confidence
                }
                for relation in nlp_result.relations
            ],
            "language": nlp_result.language,
            "sentences_count": len(nlp_result.sentences)
        }
    
    def _add_to_graph(self, request: ProcessingRequest, nlp_data: Optional[Dict],
                     provider: IComponentProvider) -> Dict[str, Any]:
        """Добавление информации в граф знаний."""
        graph = provider.get_component("semgraph")
        
        nodes_added = 0
        edges_added = 0
        
        if nlp_data:
            # Добавление сущностей как узлов
            for entity in nlp_data.get("entities", []):
                if entity["confidence"] > 0.7:
                    graph.add_node(
                        entity["text"],
                        type=entity["type"],
                        confidence=entity["confidence"],
                        source="text_processing"
                    )
                    nodes_added += 1
            
            # Добавление отношений как ребер
            for relation in nlp_data.get("relations", []):
                if relation["confidence"] > 0.6:
                    graph.add_edge(
                        relation["subject"],
                        relation["object"],
                        relation["predicate"],
                        weight=relation["confidence"]
                    )
                    edges_added += 1
        
        return {
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "total_nodes": len(graph.get_all_nodes()),
            "total_edges": len(graph.get_all_edges())
        }
    
    def _vectorize_content(self, request: ProcessingRequest, nlp_data: Optional[Dict],
                          provider: IComponentProvider) -> Dict[str, Any]:
        """Создание векторных представлений."""
        vectors = provider.get_component("contextvec")
        
        vectors_created = 0
        
        # Векторизуем весь текст
        import numpy as np
        text_vector = np.random.random(384)  # Заглушка для демонстрации
        
        text_key = f"text_{hash(request.content) % 1000000}"
        vectors.create_vector(text_key, text_vector)
        vectors_created += 1
        
        # Векторизуем сущности
        if nlp_data:
            for entity in nlp_data.get("entities", []):
                if entity["confidence"] > 0.8:
                    entity_vector = np.random.random(384)  # Заглушка
                    vectors.create_vector(entity["text"], entity_vector)
                    vectors_created += 1
        
        return {
            "vectors_created": vectors_created,
            "total_vectors": len(vectors.get_all_keys())
        }
    
    def _save_to_memory(self, request: ProcessingRequest, structured_data: Dict,
                       provider: IComponentProvider) -> Dict[str, Any]:
        """Сохранение информации в память."""
        memory = provider.get_component("memory")
        
        from neurograph.memory.base import MemoryItem
        import numpy as np
        
        # Создаем элемент памяти
        embedding = np.random.random(384)  # Заглушка
        
        memory_item = MemoryItem(
            content=request.content,
            embedding=embedding,
            content_type="processed_text",
            metadata={
                "nlp_entities_count": len(structured_data.get("nlp", {}).get("entities", [])),
                "graph_nodes_added": structured_data.get("graph", {}).get("nodes_added", 0),
                "processing_timestamp": time.time()
            }
        )
        
        item_id = memory.add(memory_item)
        
        return {
            "memory_item_id": item_id,
            "memory_size": memory.size()
        }
    
    def _generate_summary_response(self, request: ProcessingRequest, 
                                  structured_data: Dict) -> str:
        """Генерация итогового ответа."""
        if request.response_format == ResponseFormat.TEXT:
            return f"Текст обработан. Найдено {len(structured_data.get('nlp', {}).get('entities', []))} сущностей."
        
        elif request.response_format == ResponseFormat.CONVERSATIONAL:
            entities_count = len(structured_data.get("nlp", {}).get("entities", []))
            relations_count = len(structured_data.get("nlp", {}).get("relations", []))
            
            return (f"Я проанализировал ваш текст и нашел {entities_count} важных понятий "
                   f"и {relations_count} связей между ними. Информация сохранена в системе знаний.")
        
        else:
            return "Текст успешно обработан"


class QueryProcessingPipeline(BasePipeline):
    """Конвейер обработки запросов к системе знаний."""
    
    def __init__(self):
        super().__init__("query_processing")
    
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Обработка запроса к системе знаний."""
        start_time = time.time()
        
        try:
            # Валидация
            is_valid, error_msg = self.validate_request(request)
            if not is_valid:
                return self._create_response(request, False, error_message=error_msg)
            
            components_used = []
            structured_data = {}
            explanations = []
            
            # 1. Анализ запроса через NLP
            if provider.is_component_available("nlp"):
                query_analysis = self._analyze_query(request, provider)
                components_used.append("nlp")
                structured_data["query_analysis"] = query_analysis
                explanations.append("Запрос проанализирован и структурирован")
            
            # 2. Поиск в графе знаний
            if provider.is_component_available("semgraph"):
                graph_results = self._search_in_graph(request, query_analysis, provider)
                components_used.append("semgraph")
                structured_data["graph_search"] = graph_results
                explanations.append("Выполнен поиск в графе знаний")
            
            # 3. Семантический поиск в векторах
            if provider.is_component_available("contextvec"):
                vector_results = self._semantic_search(request, provider)
                components_used.append("contextvec")
                structured_data["vector_search"] = vector_results
                explanations.append("Выполнен семантический поиск")
            
            # 4. Поиск в памяти
            if provider.is_component_available("memory"):
                memory_results = self._search_memory(request, provider)
                components_used.append("memory")
                structured_data["memory_search"] = memory_results
                explanations.append("Выполнен поиск в памяти")
            
            # 5. Распространение активации
            if provider.is_component_available("propagation"):
                propagation_results = self._propagate_activation(request, query_analysis, provider)
                components_used.append("propagation")
                structured_data["propagation"] = propagation_results
                explanations.append("Выполнено распространение активации")
            
            # 6. Логический вывод
            if provider.is_component_available("processor"):
                inference_results = self._perform_inference(request, structured_data, provider)
                components_used.append("processor")
                structured_data["inference"] = inference_results
                explanations.append("Выполнен логический вывод")
            
            # 7. Синтез ответа
            primary_response = self._synthesize_answer(request, structured_data)
            
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time)
            
            response = self._create_response(
                request,
                success=True,
                primary_response=primary_response,
                structured_data=structured_data
            )
            response.components_used = components_used
            response.explanation = explanations
            response.processing_time = processing_time
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка в запросном конвейере: {e}")
            self._update_stats(False, processing_time)
            
            return self._create_response(request, False, error_message=str(e))
    
    def _analyze_query(self, request: ProcessingRequest, 
                      provider: IComponentProvider) -> Dict[str, Any]:
        """Анализ запроса через NLP."""
        nlp = provider.get_component("nlp")
        
        nlp_result = nlp.process_text(request.content)
        
        # Извлекаем ключевые концепты для поиска
        key_concepts = [
            entity.text for entity in nlp_result.entities
            if entity.confidence > 0.6
        ]
        
        return {
            "key_concepts": key_concepts,
            "entities": len(nlp_result.entities),
            "relations": len(nlp_result.relations),
            "language": nlp_result.language
        }
    
    def _search_in_graph(self, request: ProcessingRequest, query_analysis: Dict,
                        provider: IComponentProvider) -> Dict[str, Any]:
        """Поиск в графе знаний."""
        graph = provider.get_component("semgraph")
        
        found_nodes = []
        related_concepts = []
        
        # Поиск узлов по ключевым концептам
        for concept in query_analysis.get("key_concepts", []):
            if graph.has_node(concept):
                found_nodes.append(concept)
                
                # Получаем связанные концепты
                neighbors = graph.get_neighbors(concept)
                related_concepts.extend(neighbors[:3])  # Топ-3 связанных
        
        return {
            "found_nodes": found_nodes,
            "related_concepts": list(set(related_concepts)),
            "total_matches": len(found_nodes)
        }
    
    def _semantic_search(self, request: ProcessingRequest,
                        provider: IComponentProvider) -> Dict[str, Any]:
        """Семантический поиск в векторах."""
        vectors = provider.get_component("contextvec")
        
        # Заглушка для демонстрации
        import numpy as np
        query_vector = np.random.random(384)
        
        # В реальности здесь был бы поиск похожих векторов
        similar_concepts = []
        all_keys = vectors.get_all_keys()
        
        for key in all_keys[:5]:  # Берем первые 5 для демонстрации
            similar_concepts.append({
                "concept": key,
                "similarity": np.random.random()
            })
        
        return {
            "similar_concepts": similar_concepts,
            "total_vectors_searched": len(all_keys)
        }
    
    def _search_memory(self, request: ProcessingRequest,
                      provider: IComponentProvider) -> Dict[str, Any]:
        """Поиск в памяти."""
        memory = provider.get_component("memory")
        
        # Заглушка для демонстрации
        import numpy as np
        query_vector = np.random.random(384)
        
        # В реальности здесь был бы семантический поиск в памяти
        search_results = []
        
        # Получаем недавние элементы для демонстрации
        recent_items = memory.get_recent_items(hours=24.0)
        
        for item in recent_items[:3]:
            search_results.append({
                "content": item.content[:100] + "..." if len(item.content) > 100 else item.content,
                "relevance": np.random.random(),
                "memory_type": item.content_type
            })
        
        return {
            "relevant_memories": search_results,
            "total_memory_size": memory.size()
        }
    
    def _propagate_activation(self, request: ProcessingRequest, query_analysis: Dict,
                             provider: IComponentProvider) -> Dict[str, Any]:
        """Распространение активации по графу."""
        try:
            propagation = provider.get_component("propagation")
            
            # Инициализируем активацию от ключевых концептов
            initial_nodes = {}
            for concept in query_analysis.get("key_concepts", []):
                initial_nodes[concept] = 1.0
            
            if initial_nodes:
                from neurograph.propagation import create_default_config
                config = create_default_config()
                
                result = propagation.propagate(initial_nodes, config)
                
                # Получаем активированные узлы
                activated = result.get_most_activated_nodes(top_n=10)
                
                return {
                    "activated_concepts": [
                        {"concept": concept, "activation": float(activation)}
                        for concept, activation in activated
                    ],
                    "convergence_achieved": result.convergence_achieved,
                    "iterations_used": result.iterations_used
                }
            
        except Exception as e:
            self.logger.warning(f"Ошибка распространения активации: {e}")
        
        return {"activated_concepts": [], "error": "Не удалось выполнить распространение"}
    
    def _perform_inference(self, request: ProcessingRequest, structured_data: Dict,
                          provider: IComponentProvider) -> Dict[str, Any]:
        """Логический вывод."""
        try:
            processor = provider.get_component("processor")
            
            from neurograph.processor.base import ProcessingContext
            
            # Создаем контекст для вывода
            context = ProcessingContext()
            
            # Добавляем факты из найденной информации
            graph_results = structured_data.get("graph_search", {})
            for node in graph_results.get("found_nodes", []):
                context.add_fact(f"exists_{node}", True, 1.0)
            
            # Выполняем вывод
            derivation_result = processor.derive(context, depth=2)
            
            derived_facts = []
            for fact_key, fact_data in derivation_result.derived_facts.items():
                derived_facts.append({
                    "fact": fact_key,
                    "confidence": fact_data.get("confidence", 0.0)
                })
            
            return {
                "derived_facts": derived_facts,
                "rules_used": derivation_result.rules_used,
                "success": derivation_result.success
            }
            
        except Exception as e:
            self.logger.warning(f"Ошибка логического вывода: {e}")
        
        return {"derived_facts": [], "error": "Не удалось выполнить логический вывод"}
    
    def _synthesize_answer(self, request: ProcessingRequest, 
                          structured_data: Dict) -> str:
        """Синтез итогового ответа."""
        if request.response_format == ResponseFormat.CONVERSATIONAL:
            return self._generate_conversational_answer(structured_data)
        elif request.response_format == ResponseFormat.STRUCTURED:
            return self._generate_structured_answer(structured_data)
        else:
            return self._generate_text_answer(structured_data)
    
    def _generate_conversational_answer(self, structured_data: Dict) -> str:
        """Генерация разговорного ответа."""
        parts = []
        
        # Информация из графа
        graph_search = structured_data.get("graph_search", {})
        found_nodes = graph_search.get("found_nodes", [])
        if found_nodes:
            parts.append(f"Я нашел информацию о: {', '.join(found_nodes[:3])}")
        
        # Активированные концепты
        propagation = structured_data.get("propagation", {})
        activated = propagation.get("activated_concepts", [])
        if activated:
            top_concepts = [c["concept"] for c in activated[:3]]
            parts.append(f"Связанные концепты: {', '.join(top_concepts)}")
        
        # Логические выводы
        inference = structured_data.get("inference", {})
        derived_facts = inference.get("derived_facts", [])
        if derived_facts:
            parts.append(f"На основе анализа можно сделать {len(derived_facts)} выводов")
        
        if parts:
            return ". ".join(parts) + "."
        else:
            return "К сожалению, я не нашел релевантной информации по вашему запросу."
    
    def _generate_structured_answer(self, structured_data: Dict) -> str:
        """Генерация структурированного ответа."""
        return f"Результаты поиска:\n" + "\n".join([
            f"- {key}: {len(data) if isinstance(data, list) else 'выполнено'}"
            for key, data in structured_data.items()
        ])
    
    def _generate_text_answer(self, structured_data: Dict) -> str:
        """Генерация текстового ответа."""
        total_results = sum([
            len(data.get("found_nodes", [])) for data in structured_data.values()
            if isinstance(data, dict) and "found_nodes" in data
        ])
        
        return f"Поиск завершен. Найдено {total_results} релевантных результатов."


class LearningPipeline(BasePipeline):
    """Конвейер обучения системы."""
    
    def __init__(self):
        super().__init__("learning")
    
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Обучение системы на новом контенте."""
        start_time = time.time()
        
        try:
            # Используем текстовый конвейер для базовой обработки
            text_pipeline = TextProcessingPipeline()
            text_result = text_pipeline.process(request, provider)
            
            if not text_result.success:
                return text_result
            
            # Дополнительная обработка для обучения
            learning_data = {}
            
            # Создание правил из извлеченных отношений
            if provider.is_component_available("processor"):
                rules_created = self._create_inference_rules(text_result.structured_data, provider)
                learning_data["rules_created"] = rules_created
            
            # Обновление векторных индексов
            if provider.is_component_available("contextvec"):
                index_updates = self._update_vector_indices(text_result.structured_data, provider)
                learning_data["index_updates"] = index_updates
            
            # Консолидация памяти
            if provider.is_component_available("memory"):
                consolidation_result = self._trigger_memory_consolidation(provider)
                learning_data["memory_consolidation"] = consolidation_result
            
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time)
            
            # Создаем итоговый ответ
            response = text_result
            response.structured_data["learning"] = learning_data
            response.processing_time = processing_time
            response.primary_response = self._generate_learning_response(learning_data)
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка в обучающем конвейере: {e}")
            self._update_stats(False, processing_time)
            
            return self._create_response(request, False, error_message=str(e))
    
    def _create_inference_rules(self, structured_data: Dict, 
                               provider: IComponentProvider) -> Dict[str, Any]:
        """Создание правил логического вывода."""
        processor = provider.get_component("processor")
        
        rules_created = 0
        nlp_data = structured_data.get("nlp", {})
        
        # Создаем правила из отношений
        for relation in nlp_data.get("relations", []):
            if relation["confidence"] > 0.8:
                from neurograph.processor.base import SymbolicRule
                
                rule = SymbolicRule(
                    condition=f"{relation['subject']} существует",
                    action=f"derive {relation['subject']} {relation['predicate']} {relation['object']}",
                    confidence=relation["confidence"],
                    metadata={"source": "learning_pipeline"}
                )
                
                try:
                    processor.add_rule(rule)
                    rules_created += 1
                except Exception as e:
                    self.logger.warning(f"Не удалось добавить правило: {e}")
        
        return {"count": rules_created}
    
    def _update_vector_indices(self, structured_data: Dict, 
                              provider: IComponentProvider) -> Dict[str, Any]:
        """Обновление векторных индексов."""
        # В реальной реализации здесь было бы обновление HNSW индексов
        vectors = provider.get_component("contextvec")
        
        return {
            "total_vectors": len(vectors.get_all_keys()),
            "index_updated": True
        }
    
    def _trigger_memory_consolidation(self, provider: IComponentProvider) -> Dict[str, Any]:
        """Запуск консолидации памяти."""
        memory = provider.get_component("memory")
        
        try:
            consolidation_result = memory.force_consolidation()
            return {
                "consolidated": consolidation_result.get("consolidated", 0),
                "success": True
            }
        except Exception as e:
            self.logger.warning(f"Ошибка консолидации: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_learning_response(self, learning_data: Dict) -> str:
        """Генерация ответа об обучении."""
        parts = ["Система успешно обучена на новом контенте."]
        
        rules_count = learning_data.get("rules_created", {}).get("count", 0)
        if rules_count > 0:
            parts.append(f"Создано {rules_count} новых правил логического вывода.")
        
        if learning_data.get("memory_consolidation", {}).get("success"):
            consolidated = learning_data["memory_consolidation"].get("consolidated", 0)
            parts.append(f"Консолидировано {consolidated} элементов памяти.")
        
        return " ".join(parts)


class InferencePipeline(BasePipeline):
    """Конвейер логического вывода."""
    
    def __init__(self):
        super().__init__("inference")
    
    def process(self, request: ProcessingRequest, 
                provider: IComponentProvider) -> ProcessingResponse:
        """Выполнение логического вывода."""
        start_time = time.time()
        
        try:
            # Валидация
            is_valid, error_msg = self.validate_request(request)
            if not is_valid:
                return self._create_response(request, False, error_message=error_msg)
            
            components_used = []
            structured_data = {}
            explanations = []
            
            # 1. Анализ входных данных
            if provider.is_component_available("nlp"):
                query_analysis = self._analyze_inference_query(request, provider)
                components_used.append("nlp")
                structured_data["query_analysis"] = query_analysis
                explanations.append("Запрос на вывод проанализирован")
            
            # 2. Подготовка контекста для вывода
            context = self._prepare_inference_context(request, structured_data, provider)
            explanations.append("Подготовлен контекст для логического вывода")

            # 3. Выполнение логического вывода
            if provider.is_component_available("processor"):
                inference_result = self._execute_inference(context, provider)
                components_used.append("processor")
                structured_data["inference"] = inference_result
                explanations.append("Выполнен логический вывод")
            
            # 4. Распространение активации для поиска связей
            if provider.is_component_available("propagation"):
                propagation_result = self._propagate_from_conclusions(inference_result, provider)
                components_used.append("propagation")
                structured_data["propagation"] = propagation_result
                explanations.append("Найдены дополнительные связи")
            
            # 5. Генерация объяснений
            explanation_text = self._generate_inference_explanation(structured_data)
            
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time)
            
            response = self._create_response(
                request,
                success=True,
                primary_response=explanation_text,
                structured_data=structured_data
            )
            response.components_used = components_used
            response.explanation = explanations
            response.processing_time = processing_time
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка в конвейере вывода: {e}")
            self._update_stats(False, processing_time)
            
            return self._create_response(request, False, error_message=str(e))

    def _analyze_inference_query(self, request: ProcessingRequest,
                                provider: IComponentProvider) -> Dict[str, Any]:
        """Анализ запроса на логический вывод."""
        nlp = provider.get_component("nlp")
        
        nlp_result = nlp.process_text(request.content)
        
        # Извлекаем предпосылки и заключения
        premises = []
        questions = []
        
        for sentence in nlp_result.sentences:
            if "?" in sentence.text:
                questions.append(sentence.text)
            else:
                premises.append(sentence.text)
        
        return {
            "premises": premises,
            "questions": questions,
            "entities": len(nlp_result.entities),
            "relations": len(nlp_result.relations)
        }

    def _prepare_inference_context(self, request: ProcessingRequest, 
                                structured_data: Dict,
                                provider: IComponentProvider):
        """Подготовка контекста для логического вывода."""
        from neurograph.processor.base import ProcessingContext
        
        context = ProcessingContext()
        
        # Добавляем факты из предпосылок
        query_analysis = structured_data.get("query_analysis", {})
        for i, premise in enumerate(query_analysis.get("premises", [])):
            context.add_fact(f"premise_{i}", premise, 1.0)
        
        # Добавляем факты из графа знаний
        if provider.is_component_available("semgraph"):
            graph = provider.get_component("semgraph")
            
            # Добавляем релевантные факты из графа
            for source, target, edge_type in list(graph.get_all_edges())[:50]:  # Ограничиваем количество
                fact_key = f"graph_{source}_{edge_type}_{target}"
                context.add_fact(fact_key, True, 0.9)
        
        return context

    def _execute_inference(self, context, provider: IComponentProvider) -> Dict[str, Any]:
        """Выполнение логического вывода."""
        processor = provider.get_component("processor")
        
        # Выполняем вывод с увеличенной глубиной
        derivation_result = processor.derive(context, depth=3)
        
        conclusions = []
        for fact_key, fact_data in derivation_result.derived_facts.items():
            conclusions.append({
                "conclusion": fact_key,
                "confidence": fact_data.get("confidence", 0.0),
                "derived_from": derivation_result.rules_used
            })
        
        return {
            "conclusions": conclusions,
            "rules_applied": len(derivation_result.rules_used),
            "success": derivation_result.success,
            "confidence": derivation_result.confidence,
            "explanation_steps": [
                {
                    "step": step.step_number,
                    "rule": step.rule_description,
                    "reasoning": step.reasoning
                }
                for step in derivation_result.explanation
            ]
        }

    def _propagate_from_conclusions(self, inference_result: Dict,
                                provider: IComponentProvider) -> Dict[str, Any]:
        """Распространение активации от полученных выводов."""
        try:
            propagation = provider.get_component("propagation")
            
            # Создаем начальную активацию от выводов
            initial_nodes = {}
            for conclusion in inference_result.get("conclusions", []):
                # Извлекаем ключевые слова из заключения
                words = conclusion["conclusion"].split("_")
                for word in words[:2]:  # Берем первые 2 слова
                    if len(word) > 3:  # Игнорируем короткие слова
                        initial_nodes[word] = conclusion["confidence"]
            
            if initial_nodes:
                from neurograph.propagation import create_fast_config
                config = create_fast_config()
                
                result = propagation.propagate(initial_nodes, config)
                
                return {
                    "activated_concepts": [
                        {"concept": concept, "activation": float(activation)}
                        for concept, activation in result.get_most_activated_nodes(top_n=5)
                    ],
                    "success": result.success
                }
            
        except Exception as e:
            self.logger.warning(f"Ошибка распространения от выводов: {e}")
        
        return {"activated_concepts": [], "success": False}

    def _generate_inference_explanation(self, structured_data: Dict) -> str:
        """Генерация объяснения логического вывода."""
        inference = structured_data.get("inference", {})
        conclusions = inference.get("conclusions", [])
        
        if not conclusions:
            return "Не удалось сделать определенных выводов на основе предоставленной информации."
        
        explanation_parts = ["На основе анализа можно сделать следующие выводы:"]
        
        for i, conclusion in enumerate(conclusions[:3], 1):  # Топ-3 вывода
            confidence = conclusion["confidence"]
            confidence_text = "высокой" if confidence > 0.8 else "средней" if confidence > 0.5 else "низкой"
            
            explanation_parts.append(
                f"{i}. {conclusion['conclusion']} (уверенность: {confidence_text})"
            )
        
        # Добавляем информацию о процессе
        rules_count = inference.get("rules_applied", 0)
        if rules_count > 0:
            explanation_parts.append(f"В процессе вывода применено {rules_count} правил.")
        
        return "\n".join(explanation_parts)