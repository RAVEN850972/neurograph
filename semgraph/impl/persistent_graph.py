"""Реализация графа с постоянным хранением на диске."""

import os
import threading
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import json

from neurograph.semgraph.base import ISemGraph
from neurograph.semgraph.impl.memory_graph import MemoryEfficientSemGraph
from neurograph.core.logging import get_logger
from neurograph.core.errors import NeuroGraphError

logger = get_logger("semgraph.persistent_graph")

class PersistentSemGraph(ISemGraph):
    """Семантический граф с постоянным хранением на диске."""
    
    def __init__(self, file_path: str, auto_save_interval: float = 300.0):
        """Инициализирует граф.
        
        Args:
            file_path: Путь к файлу для хранения графа.
            auto_save_interval: Интервал автоматического сохранения в секундах.
        """
        self.file_path = file_path
        self.auto_save_interval = auto_save_interval
        
        # Граф в памяти
        self.graph = MemoryEfficientSemGraph()
        
        # Флаг для отслеживания изменений с момента последнего сохранения
        self._modified = False
        
        # Блокировка для потокобезопасности
        self._lock = threading.RLock()
        
        # Загружаем граф из файла, если он существует
        if os.path.exists(file_path):
            try:
                self._load()
            except Exception as e:
                logger.error(f"Ошибка при загрузке графа из файла {file_path}: {str(e)}")
                raise NeuroGraphError(f"Не удалось загрузить граф из файла {file_path}", {"error": str(e)})
        
        # Запускаем поток для автоматического сохранения
        self._auto_save_thread = None
        self._auto_save_running = False
        
        if auto_save_interval > 0:
            self._start_auto_save()
    
    def _start_auto_save(self) -> None:
        """Запускает поток для автоматического сохранения."""
        with self._lock:
            if self._auto_save_running:
                return
                
            self._auto_save_running = True
            self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
            self._auto_save_thread.start()
            logger.info(f"Запущен поток автоматического сохранения с интервалом {self.auto_save_interval} секунд")
    
    def _stop_auto_save(self) -> None:
        """Останавливает поток автоматического сохранения."""
        with self._lock:
            if not self._auto_save_running:
                return
                
            self._auto_save_running = False
            
            if self._auto_save_thread and self._auto_save_thread.is_alive():
                self._auto_save_thread.join(timeout=1.0)
            
            logger.info("Остановлен поток автоматического сохранения")
    
    def _auto_save_loop(self) -> None:
        """Цикл автоматического сохранения."""
        while self._auto_save_running:
            try:
                time.sleep(self.auto_save_interval)
                
                # Сохраняем граф, если он был изменен
                with self._lock:
                    if self._modified:
                        self._save()
                        self._modified = False
                        logger.debug("Выполнено автоматическое сохранение графа")
            except Exception as e:
                logger.error(f"Ошибка при автоматическом сохранении графа: {str(e)}")
    
    def _load(self) -> None:
        """Загружает граф из файла."""
        with self._lock:
            try:
                self.graph = MemoryEfficientSemGraph.load(self.file_path)
                logger.info(f"Граф успешно загружен из файла {self.file_path}: {len(self.graph.get_all_nodes())} узлов, {len(self.graph.get_all_edges())} ребер")
            except Exception as e:
                logger.error(f"Ошибка при загрузке графа из файла {self.file_path}: {str(e)}")
                raise
    
    def _save(self) -> None:
        """Сохраняет граф в файл."""
        with self._lock:
            try:
                # Создаем директорию, если она не существует
                os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
                
                self.graph.save(self.file_path)
                logger.info(f"Граф успешно сохранен в файл {self.file_path}: {len(self.graph.get_all_nodes())} узлов, {len(self.graph.get_all_edges())} ребер")
            except Exception as e:
                logger.error(f"Ошибка при сохранении графа в файл {self.file_path}: {str(e)}")
                raise
    
    def add_node(self, node_id: str, **attributes) -> None:
        """Добавляет узел в граф с заданными атрибутами."""
        with self._lock:
            self.graph.add_node(node_id, **attributes)
            self._modified = True
    
    def add_edge(self, source: str, target: str, edge_type: str = "default", 
                weight: float = 1.0, **attributes) -> None:
        """Добавляет направленное ребро между узлами."""
        with self._lock:
            self.graph.add_edge(source, target, edge_type, weight, **attributes)
            self._modified = True
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает узел с его атрибутами."""
        with self._lock:
            return self.graph.get_node(node_id)
    
    def get_edge(self, source: str, target: str, edge_type: str = "default") -> Optional[Dict[str, Any]]:
        """Возвращает ребро с его атрибутами."""
        with self._lock:
            return self.graph.get_edge(source, target, edge_type)
    
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Возвращает соседей узла."""
        with self._lock:
            return self.graph.get_neighbors(node_id, edge_type)
    
    def get_edge_weight(self, source: str, target: str, edge_type: str = "default") -> Optional[float]:
        """Возвращает вес ребра."""
        with self._lock:
            return self.graph.get_edge_weight(source, target, edge_type)
    
    def update_edge_weight(self, source: str, target: str, weight: float, 
                         edge_type: str = "default") -> bool:
        """Обновляет вес ребра."""
        with self._lock:
            result = self.graph.update_edge_weight(source, target, weight, edge_type)
            if result:
                self._modified = True
            return result
    
    def has_node(self, node_id: str) -> bool:
        """Проверяет наличие узла в графе."""
        with self._lock:
            return self.graph.has_node(node_id)
    
    def has_edge(self, source: str, target: str, edge_type: str = "default") -> bool:
        """Проверяет наличие ребра в графе."""
        with self._lock:
            return self.graph.has_edge(source, target, edge_type)
    
    def get_all_nodes(self) -> List[str]:
        """Возвращает список всех узлов в графе."""
        with self._lock:
            return self.graph.get_all_nodes()
    
    def get_all_edges(self) -> List[Tuple[str, str, str]]:
        """Возвращает список всех ребер в графе."""
        with self._lock:
            return self.graph.get_all_edges()
    
    def save_now(self) -> None:
        """Принудительно сохраняет граф в файл."""
        with self._lock:
            self._save()
            self._modified = False
    
    def reload(self) -> None:
        """Перезагружает граф из файла, отбрасывая все несохраненные изменения."""
        with self._lock:
            self._load()
            self._modified = False
    
    def close(self) -> None:
        """Закрывает граф, сохраняя все изменения."""
        self._stop_auto_save()
        with self._lock:
            if self._modified:
                self._save()