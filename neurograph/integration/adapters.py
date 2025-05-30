"""
Адаптеры для интеграции между различными компонентами системы.
"""

from typing import Any, Dict, List, Optional
from neurograph.core.logging import get_logger

from .base import IComponentAdapter


class BaseAdapter(IComponentAdapter):
    """Базовый класс для адаптеров."""
    
    def __init__(self, adapter_name: str):
        self.adapter_name = adapter_name
        self.logger = get_logger(f"adapter_{adapter_name}")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Базовая реализация поддерживаемых форматов."""
        return {
            "input": ["generic"],
            "output": ["generic"]
        }


class GraphMemoryAdapter(BaseAdapter):
    """Адаптер между графом знаний и системой памяти."""
    
    def __init__(self):
        super().__init__("graph_memory")
    
    def adapt(self, source_data: Any, target_format: str) -> Any:
        """Адаптация данных между графом и памятью."""
        if target_format == "memory_items":
            return self._graph_to_memory_items(source_data)
        elif target_format == "graph_structure":
            return self._memory_to_graph_structure(source_data)
        else:
            raise ValueError(f"Неподдерживаемый формат: {target_format}")
    
    def _graph_to_memory_items(self, graph_data: Dict) -> List[Dict]:
        """Преобразование данных графа в элементы памяти."""
        memory_items = []
        
        # Преобразуем узлы графа
        for node_id, node_data in graph_data.get("nodes", {}).items():
            memory_items.append({
                "content": f"Концепт: {node_id}",
                "content_type": "concept",
                "metadata": {
                    "source": "graph",
                    "node_id": node_id,
                    "node_type": node_data.get("type", "unknown"),
                    **node_data
                }
            })
        
        # Преобразуем ребра графа
        for edge in graph_data.get("edges", []):
            source, target, edge_type = edge[:3]
            memory_items.append({
                "content": f"Связь: {source} {edge_type} {target}",
                "content_type": "relation",
                "metadata": {
                    "source": "graph",
                    "relation_type": edge_type,
                    "subject": source,
                    "object": target
                }
            })
        
        return memory_items
    
    def _memory_to_graph_structure(self, memory_items: List) -> Dict:
        """Преобразование элементов памяти в структуру графа."""
        nodes = {}
        edges = []
        
        for item in memory_items:
            if item.get("content_type") == "concept":
                node_id = item["metadata"].get("node_id") or item["content"]
                nodes[node_id] = {
                    "type": item["metadata"].get("node_type", "concept"),
                    "source": "memory"
                }
            
            elif item.get("content_type") == "relation":
                metadata = item["metadata"]
                subject = metadata.get("subject")
                obj = metadata.get("object")
                relation_type = metadata.get("relation_type")
                
                if subject and obj and relation_type:
                    edges.append([subject, obj, relation_type])
        
        return {"nodes": nodes, "edges": edges}
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Поддерживаемые форматы адаптера."""
        return {
            "input": ["graph_data", "memory_items"],
            "output": ["memory_items", "graph_structure"]
        }


class VectorProcessorAdapter(BaseAdapter):
    """Адаптер между векторными представлениями и процессором."""
    
    def __init__(self):
        super().__init__("vector_processor")
    
    def adapt(self, source_data: Any, target_format: str) -> Any:
        """Адаптация данных между векторами и процессором."""
        if target_format == "processing_context":
            return self._vectors_to_context(source_data)
        elif target_format == "vector_rules":
            return self._processor_to_vectors(source_data)
        else:
            raise ValueError(f"Неподдерживаемый формат: {target_format}")
    
    def _vectors_to_context(self, vector_data: Dict) -> Dict:
        """Преобразование векторных данных в контекст процессора."""
        from neurograph.processor.base import ProcessingContext
        
        context = ProcessingContext()
        
        # Добавляем факты о существовании концептов
        for concept_id in vector_data.get("vector_keys", []):
            context.add_fact(f"has_vector_{concept_id}", True, 1.0)
        
        # Добавляем факты о семантической близости
        for similarity in vector_data.get("similarities", []):
            concept1 = similarity.get("concept1")
            concept2 = similarity.get("concept2")
            score = similarity.get("score", 0.0)
            
            if score > 0.7:  # Высокое сходство
                context.add_fact(f"similar_{concept1}_{concept2}", True, score)
        
        return {"context": context}
    
    def _processor_to_vectors(self, processor_data: Dict) -> Dict:
        """Преобразование данных процессора в векторные правила."""
        vector_rules = []
        
        # Создаем векторные правила из логических правил
        for rule in processor_data.get("rules", []):
            condition = rule.get("condition", "")
            action = rule.get("action", "")
            
            vector_rules.append({
                "rule_vector": f"rule_{hash(condition + action) % 1000000}",
                "condition_concepts": self._extract_concepts(condition),
                "action_concepts": self._extract_concepts(action),
                "confidence": rule.get("confidence", 0.0)
            })
        
        return {"vector_rules": vector_rules}
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Извлечение концептов из текста."""
        # Простое извлечение - разбиение по словам
        words = text.lower().split()
        # Фильтруем служебные слова
        stop_words = {"и", "или", "не", "если", "то", "является", "имеет"}
        concepts = [word for word in words if word not in stop_words and len(word) > 2]
        return concepts[:5]  # Ограничиваем количество
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Поддерживаемые форматы адаптера."""
        return {
            "input": ["vector_data", "processor_data"],
            "output": ["processing_context", "vector_rules"]
        }


class NLPGraphAdapter(BaseAdapter):
    """Адаптер между NLP модулем и графом знаний."""
    
    def __init__(self):
        super().__init__("nlp_graph")
    
    def adapt(self, source_data: Any, target_format: str) -> Any:
        """Адаптация данных между NLP и графом."""
        if target_format == "graph_updates":
            return self._nlp_to_graph_updates(source_data)
        elif target_format == "nlp_context":
            return self._graph_to_nlp_context(source_data)
        else:
            raise ValueError(f"Неподдерживаемый формат: {target_format}")
    
    def _nlp_to_graph_updates(self, nlp_data: Dict) -> Dict:
        """Преобразование результатов NLP в обновления графа."""
        nodes_to_add = []
        edges_to_add = []
        
        # Обрабатываем сущности
        for entity in nlp_data.get("entities", []):
            if entity.get("confidence", 0) > 0.7:
                nodes_to_add.append({
                    "id": entity["text"],
                    "type": entity["entity_type"],
                    "confidence": entity["confidence"],
                    "source": "nlp_extraction"
                })
        
        # Обрабатываем отношения
        for relation in nlp_data.get("relations", []):
            if relation.get("confidence", 0) > 0.6:
                edges_to_add.append({
                    "source": relation["subject"]["text"],
                    "target": relation["object"]["text"],
                    "type": relation["predicate"],
                    "weight": relation["confidence"],
                    "source": "nlp_extraction"
                })
        
        return {
            "nodes_to_add": nodes_to_add,
            "edges_to_add": edges_to_add,
            "update_metadata": {
                "source": "nlp",
                "extraction_time": nlp_data.get("processing_time", 0)
            }
        }
    
    def _graph_to_nlp_context(self, graph_data: Dict) -> Dict:
        """Преобразование данных графа в контекст для NLP."""
        nlp_context = {
            "known_entities": [],
            "known_relations": [],
            "domain_vocabulary": set()
        }
        
        # Извлекаем известные сущности
        for node_id, node_data in graph_data.get("nodes", {}).items():
            nlp_context["known_entities"].append({
                "text": node_id,
                "type": node_data.get("type", "UNKNOWN"),
                "confidence": node_data.get("confidence", 1.0)
            })
            
            # Добавляем в словарь
            nlp_context["domain_vocabulary"].add(node_id.lower())
        
        # Извлекаем известные отношения
        for edge in graph_data.get("edges", []):
            if len(edge) >= 3:
                source, target, edge_type = edge[:3]
                nlp_context["known_relations"].append({
                    "subject": source,
                    "predicate": edge_type,
                    "object": target
                })
                
                # Добавляем типы отношений в словарь
                nlp_context["domain_vocabulary"].add(edge_type.lower())
        
        # Преобразуем множество в список
        nlp_context["domain_vocabulary"] = list(nlp_context["domain_vocabulary"])
        
        return nlp_context
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Поддерживаемые форматы адаптера."""
        return {
            "input": ["nlp_result", "graph_data"],
            "output": ["graph_updates", "nlp_context"]
        }


class MemoryProcessorAdapter(BaseAdapter):
    """Адаптер между системой памяти и процессором."""
    
    def __init__(self):
        super().__init__("memory_processor")
    
    def adapt(self, source_data: Any, target_format: str) -> Any:
        """Адаптация данных между памятью и процессором."""
        if target_format == "inference_context":
            return self._memory_to_inference_context(source_data)
        elif target_format == "memory_updates":
            return self._processor_to_memory_updates(source_data)
        else:
            raise ValueError(f"Неподдерживаемый формат: {target_format}")
    
    def _memory_to_inference_context(self, memory_data: List) -> Dict:
        """Преобразование элементов памяти в контекст для вывода."""
        from neurograph.processor.base import ProcessingContext
        
        context = ProcessingContext()
        
        # Обрабатываем элементы памяти
        for item in memory_data:
            content_type = item.get("content_type", "unknown")
            confidence = item.get("metadata", {}).get("confidence", 0.8)
            
            if content_type == "fact":
                # Факты добавляем как есть
                fact_key = f"memory_fact_{hash(item['content']) % 1000000}"
                context.add_fact(fact_key, item["content"], confidence)
            
            elif content_type == "rule":
                # Правила обрабатываем отдельно
                metadata = item.get("metadata", {})
                if "condition" in metadata and "action" in metadata:
                    rule_fact = f"rule_available_{metadata['condition']}"
                    context.add_fact(rule_fact, True, confidence)
            
            elif content_type in ["concept", "entity"]:
                # Концепты и сущности
                concept_fact = f"concept_known_{item['content']}"
                context.add_fact(concept_fact, True, confidence)
        
        return {"inference_context": context}
    
    def _processor_to_memory_updates(self, processor_data: Dict) -> List[Dict]:
        """Преобразование результатов процессора в обновления памяти."""
        memory_updates = []
        
        # Обрабатываем выведенные факты
        for fact in processor_data.get("derived_facts", []):
            memory_updates.append({
                "content": fact.get("fact", ""),
                "content_type": "derived_fact",
                "metadata": {
                    "confidence": fact.get("confidence", 0.0),
                    "source": "logical_inference",
                    "derivation_rules": processor_data.get("rules_used", [])
                }
            })
        
        # Обрабатываем использованные правила (повышаем их важность)
        for rule_id in processor_data.get("rules_used", []):
            memory_updates.append({
                "content": f"Применено правило {rule_id}",
                "content_type": "rule_usage",
                "metadata": {
                    "rule_id": rule_id,
                    "usage_context": processor_data.get("context_summary", ""),
                    "importance_boost": 0.1
                }
            })
        
        return memory_updates
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Поддерживаемые форматы адаптера."""
        return {
            "input": ["memory_items", "processor_results"],
            "output": ["inference_context", "memory_updates"]
        }
