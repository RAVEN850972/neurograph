"""
Конфигурационные утилиты и предустановленные конфигурации.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from neurograph.core import Configuration
from neurograph.core.logging import get_logger

from .base import IntegrationConfig


class IntegrationConfigManager:
    """Менеджер конфигураций интеграционного слоя."""
    
    def __init__(self):
        self.logger = get_logger("integration_config")
        self.config_cache = {}
    
    def load_config(self, config_path: str) -> IntegrationConfig:
        """Загрузка конфигурации из файла."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Валидация и преобразование
            validated_config = self._validate_config(config_data)
            
            # Создание объекта конфигурации
            integration_config = IntegrationConfig(**validated_config)
            
            # Кеширование
            self.config_cache[config_path] = integration_config
            
            self.logger.info(f"Конфигурация загружена: {config_path}")
            return integration_config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки конфигурации: {e}")
    
    def save_config(self, config: IntegrationConfig, config_path: str) -> None:
        """Сохранение конфигурации в файл."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Преобразование в словарь
            config_data = self._config_to_dict(config)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Конфигурация сохранена: {config_path}")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка сохранения конфигурации: {e}")
    
    def _validate_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация конфигурации."""
        # Обязательные поля
        required_fields = ["engine_name", "components"]
        for field in required_fields:
            if field not in config_data:
                raise ValueError(f"Отсутствует обязательное поле: {field}")
        
        # Валидация компонентов
        components = config_data.get("components", {})
        valid_components = [
            "semgraph", "contextvec", "memory", 
            "processor", "propagation", "nlp"
        ]
        
        for component_name in components:
            if component_name not in valid_components:
                self.logger.warning(f"Неизвестный компонент: {component_name}")
        
        # Значения по умолчанию
        defaults = {
            "version": "1.0.0",
            "max_concurrent_requests": 10,
            "default_timeout": 30.0,
            "enable_caching": True,
            "cache_ttl": 300,
            "enable_fallbacks": True,
            "max_retries": 3,
            "circuit_breaker_threshold": 5,
            "enable_metrics": True,
            "enable_health_checks": True,
            "log_level": "INFO",
            "enable_input_validation": True,
            "max_input_length": 10000,
            "rate_limiting": False
        }
        
        # Применяем значения по умолчанию
        for key, default_value in defaults.items():
            if key not in config_data:
                config_data[key] = default_value
        
        return config_data
    
    def _config_to_dict(self, config: IntegrationConfig) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь."""
        return {
            "engine_name": config.engine_name,
            "version": config.version,
            "components": config.components,
            "max_concurrent_requests": config.max_concurrent_requests,
            "default_timeout": config.default_timeout,
            "enable_caching": config.enable_caching,
            "cache_ttl": config.cache_ttl,
            "enable_fallbacks": config.enable_fallbacks,
            "max_retries": config.max_retries,
            "circuit_breaker_threshold": config.circuit_breaker_threshold,
            "enable_metrics": config.enable_metrics,
            "enable_health_checks": config.enable_health_checks,
            "log_level": config.log_level,
            "enable_input_validation": config.enable_input_validation,
            "max_input_length": config.max_input_length,
            "rate_limiting": config.rate_limiting
        }
    
    def create_template_config(self, config_type: str = "default") -> Dict[str, Any]:
        """Создание шаблона конфигурации."""
        templates = {
            "default": self._get_default_template(),
            "lightweight": self._get_lightweight_template(),
            "research": self._get_research_template(),
            "production": self._get_production_template()
        }
        
        return templates.get(config_type, templates["default"])
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Шаблон конфигурации по умолчанию."""
        return {
            "engine_name": "default_neurograph",
            "version": "1.0.0",
            "components": {
                "semgraph": {
                    "type": "memory_efficient",
                    "params": {
                        "auto_save": False
                    }
                },
                "contextvec": {
                    "type": "dynamic",
                    "params": {
                        "vector_size": 384,
                        "use_indexing": True
                    }
                },
                "memory": {
                    "params": {
                        "stm_capacity": 100,
                        "ltm_capacity": 10000,
                        "use_semantic_indexing": True,
                        "auto_consolidation": True,
                        "consolidation_interval": 300.0
                    }
                },
                "processor": {
                    "type": "pattern_matching",
                    "params": {
                        "confidence_threshold": 0.5,
                        "max_depth": 5,
                        "enable_explanations": True
                    }
                },
                "propagation": {
                    "params": {
                        "max_iterations": 100,
                        "convergence_threshold": 0.001,
                        "activation_threshold": 0.1
                    }
                },
                "nlp": {
                    "params": {
                        "language": "ru",
                        "use_spacy": False,
                        "confidence_threshold": 0.5
                    }
                }
            },
            "max_concurrent_requests": 10,
            "default_timeout": 30.0,
            "enable_caching": True,
            "cache_ttl": 300,
            "enable_fallbacks": True,
            "max_retries": 3,
            "circuit_breaker_threshold": 5,
            "enable_metrics": True,
            "enable_health_checks": True,
            "log_level": "INFO",
            "enable_input_validation": True,
            "max_input_length": 10000,
            "rate_limiting": False
        }
    
    def _get_lightweight_template(self) -> Dict[str, Any]:
        """Шаблон облегченной конфигурации."""
        return {
            "engine_name": "lightweight_neurograph",
            "version": "1.0.0",
            "components": {
                "semgraph": {
                    "type": "memory_efficient",
                    "params": {}
                },
                "contextvec": {
                    "type": "static",
                    "params": {
                        "vector_size": 100
                    }
                },
                "memory": {
                    "params": {
                        "stm_capacity": 25,
                        "ltm_capacity": 500,
                        "use_semantic_indexing": False,
                        "auto_consolidation": False
                    }
                },
                "processor": {
                    "type": "pattern_matching",
                    "params": {
                        "confidence_threshold": 0.7,
                        "max_depth": 2,
                        "enable_explanations": False
                    }
                },
                "nlp": {
                    "params": {
                        "language": "ru",
                        "use_spacy": False
                    }
                }
            },
            "max_concurrent_requests": 2,
            "default_timeout": 10.0,
            "enable_caching": False,
            "enable_fallbacks": True,
            "max_retries": 1,
            "enable_metrics": False,
            "enable_health_checks": False,
            "log_level": "WARNING",
            "max_input_length": 5000
        }
    
    def _get_research_template(self) -> Dict[str, Any]:
        """Шаблон исследовательской конфигурации."""
        return {
            "engine_name": "research_neurograph",
            "version": "1.0.0",
            "components": {
                "semgraph": {
                    "type": "persistent",
                    "params": {
                        "file_path": "research_graph.json",
                        "auto_save_interval": 300.0
                    }
                },
                "contextvec": {
                    "type": "dynamic",
                    "params": {
                        "vector_size": 768,
                        "use_indexing": True
                    }
                },
                "memory": {
                    "params": {
                        "stm_capacity": 300,
                        "ltm_capacity": 50000,
                        "use_semantic_indexing": True,
                        "auto_consolidation": True,
                        "consolidation_interval": 120.0
                    }
                },
                "processor": {
                    "type": "graph_based",
                    "params": {
                        "confidence_threshold": 0.3,
                        "max_depth": 10,
                        "enable_explanations": True,
                        "use_graph_structure": True
                    }
                },
                "propagation": {
                    "params": {
                        "max_iterations": 500,
                        "convergence_threshold": 0.0001,
                        "activation_threshold": 0.05,
                        "lateral_inhibition": True
                    }
                },
                "nlp": {
                    "params": {
                        "language": "ru",
                        "use_spacy": True,
                        "extract_entities": True,
                        "extract_relations": True,
                        "confidence_threshold": 0.3
                    }
                }
            },
            "max_concurrent_requests": 20,
            "default_timeout": 120.0,
            "enable_caching": True,
            "cache_ttl": 600,
            "enable_fallbacks": True,
            "max_retries": 5,
            "circuit_breaker_threshold": 10,
            "enable_metrics": True,
            "enable_health_checks": True,
            "log_level": "DEBUG",
            "enable_input_validation": True,
            "max_input_length": 50000,
            "rate_limiting": False
        }
    
    def _get_production_template(self) -> Dict[str, Any]:
        """Шаблон продакшн конфигурации."""
        return {
            "engine_name": "production_neurograph",
            "version": "1.0.0",
            "components": {
                "semgraph": {
                    "type": "persistent",
                    "params": {
                        "file_path": "prod_graph.json",
                        "auto_save_interval": 600.0
                    }
                },
                "contextvec": {
                    "type": "dynamic",
                    "params": {
                        "vector_size": 384,
                        "use_indexing": True
                    }
                },
                "memory": {
                    "params": {
                        "stm_capacity": 200,
                        "ltm_capacity": 20000,
                        "use_semantic_indexing": True,
                        "auto_consolidation": True,
                        "consolidation_interval": 300.0
                    }
                },
                "processor": {
                    "type": "pattern_matching",
                    "params": {
                        "confidence_threshold": 0.6,
                        "max_depth": 7,
                        "enable_explanations": True
                    }
                },
                "propagation": {
                    "params": {
                        "max_iterations": 200,
                        "convergence_threshold": 0.001,
                        "activation_threshold": 0.1
                    }
                },
                "nlp": {
                    "params": {
                        "language": "ru",
                        "use_spacy": True,
                        "confidence_threshold": 0.6
                    }
                }
            },
            "max_concurrent_requests": 50,
            "default_timeout": 45.0,
            "enable_caching": True,
            "cache_ttl": 900,
            "enable_fallbacks": True,
            "max_retries": 3,
            "circuit_breaker_threshold": 15,
            "enable_metrics": True,
            "enable_health_checks": True,
            "log_level": "INFO",
            "enable_input_validation": True,
            "max_input_length": 20000,
            "rate_limiting": True
        }