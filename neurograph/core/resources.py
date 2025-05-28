# neurograph/core/resources.py

import os
import psutil
import threading
from typing import Dict, Any, Optional

from neurograph.core.logging import get_logger

logger = get_logger("resources")

class ResourceMonitor:
    """Монитор системных ресурсов."""
    
    def __init__(self, check_interval: float = 5.0):
        """Инициализирует монитор ресурсов.
        
        Args:
            check_interval: Интервал проверки ресурсов в секундах.
        """
        self.check_interval = check_interval
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        self._process = psutil.Process(os.getpid())
        
        # Последние измерения
        self._last_measurements: Dict[str, Any] = {}
    
    def start_monitoring(self) -> None:
        """Запускает мониторинг ресурсов."""
        with self._lock:
            if self._monitoring:
                return
                
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            
            logger.info("Мониторинг ресурсов запущен")
    
    def stop_monitoring(self) -> None:
        """Останавливает мониторинг ресурсов."""
        with self._lock:
            if not self._monitoring:
                return
                
            self._monitoring = False
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=1.0)
                
            logger.info("Мониторинг ресурсов остановлен")
    
    def _monitor_loop(self) -> None:
        """Цикл мониторинга ресурсов."""
        import time
        
        while self._monitoring:
            try:
                self._update_measurements()
            except Exception as e:
                logger.error(f"Ошибка при обновлении измерений ресурсов: {str(e)}")
                
            time.sleep(self.check_interval)
    
    def _update_measurements(self) -> None:
        """Обновляет измерения ресурсов."""
        with self._lock:
            # Использование CPU
            self._last_measurements["cpu_percent"] = self._process.cpu_percent()
            
            # Использование памяти
            memory_info = self._process.memory_info()
            self._last_measurements["memory_rss"] = memory_info.rss
            self._last_measurements["memory_vms"] = memory_info.vms
            
            # Количество потоков
            self._last_measurements["thread_count"] = self._process.num_threads()
            
            # Системная информация
            self._last_measurements["system_cpu_percent"] = psutil.cpu_percent()
            self._last_measurements["system_memory_percent"] = psutil.virtual_memory().percent
            
            # Количество открытых файлов
            try:
                self._last_measurements["open_files"] = len(self._process.open_files())
            except Exception:
                self._last_measurements["open_files"] = -1
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Возвращает текущее использование ресурсов.
        
        Returns:
            Словарь с информацией об использовании ресурсов.
        """
        with self._lock:
            # Если мониторинг не запущен, обновляем измерения один раз
            if not self._monitoring:
                try:
                    self._update_measurements()
                except Exception as e:
                    logger.error(f"Ошибка при обновлении измерений ресурсов: {str(e)}")
            
            return dict(self._last_measurements)

# Глобальный монитор ресурсов
global_resource_monitor = ResourceMonitor()

def start_resource_monitoring(check_interval: float = 5.0) -> None:
    """Запускает глобальный мониторинг ресурсов.
    
    Args:
        check_interval: Интервал проверки ресурсов в секундах.
    """
    global_resource_monitor.check_interval = check_interval
    global_resource_monitor.start_monitoring()

def stop_resource_monitoring() -> None:
    """Останавливает глобальный мониторинг ресурсов."""
    global_resource_monitor.stop_monitoring()

def get_resource_usage() -> Dict[str, Any]:
    """Возвращает текущее использование ресурсов.
    
    Returns:
        Словарь с информацией об использовании ресурсов.
    """
    return global_resource_monitor.get_resource_usage()