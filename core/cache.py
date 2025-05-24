# neurograph/core/cache.py

import time
import threading
from typing import Dict, Any, Callable, TypeVar, Optional, Tuple, List, Union
from functools import wraps
import json
import pickle
import hashlib

from neurograph.core.logging import get_logger

logger = get_logger("core.cache")

T = TypeVar('T')

class CacheEntry:
    """Запись в кеше."""
    
    def __init__(self, key: str, value: Any, expires_at: Optional[float] = None):
        """Инициализирует запись в кеше.
        
        Args:
            key: Ключ записи.
            value: Значение записи.
            expires_at: Время истечения срока действия записи (в секундах с начала эпохи).
        """
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.created_at = time.time()
        self.last_accessed_at = self.created_at
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Проверяет, истек ли срок действия записи.
        
        Returns:
            True, если срок действия истек, иначе False.
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def access(self) -> None:
        """Отмечает, что запись была доступна."""
        self.last_accessed_at = time.time()
        self.access_count += 1


class Cache:
    """Класс для кеширования данных в памяти."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """Инициализирует кеш.
        
        Args:
            max_size: Максимальный размер кеша.
            ttl: Время жизни записей в кеше в секундах (если None, записи не истекают).
        """
        self.max_size = max_size
        self.ttl = ttl
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Устанавливает значение в кеше.
        
        Args:
            key: Ключ для доступа к значению.
            value: Значение для кеширования.
            ttl: Время жизни записи в секундах (если None, используется стандартное TTL).
        """
        with self._lock:
            # Если кеш полон, удаляем самую старую или наименее используемую запись
            if len(self._entries) >= self.max_size and key not in self._entries:
                self._evict()
            
            # Вычисляем время истечения срока действия
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            elif self.ttl is not None:
                expires_at = time.time() + self.ttl
            
            # Создаем новую запись
            self._entries[key] = CacheEntry(key, value, expires_at)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Возвращает значение из кеша.
        
        Args:
            key: Ключ для доступа к значению.
            default: Значение по умолчанию, если ключ не найден или запись истекла.
            
        Returns:
            Значение из кеша или значение по умолчанию.
        """
        with self._lock:
            if key not in self._entries:
                return default
            
            entry = self._entries[key]
            
            # Проверяем, не истек ли срок действия записи
            if entry.is_expired():
                del self._entries[key]
                return default
            
            # Отмечаем доступ к записи
            entry.access()
            
            return entry.value
    
    def delete(self, key: str) -> bool:
        """Удаляет запись из кеша.
        
        Args:
            key: Ключ записи для удаления.
            
        Returns:
            True, если запись была удалена, иначе False.
        """
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False
    
    def clear(self) -> None:
        """Очищает кеш."""
        with self._lock:
            self._entries.clear()
    
    def _evict(self) -> None:
        """Удаляет записи из кеша, используя стратегию вытеснения."""
        # Сначала удаляем все истекшие записи
        current_time = time.time()
        expired_keys = [key for key, entry in self._entries.items() 
                      if entry.expires_at is not None and entry.expires_at <= current_time]
        
        for key in expired_keys:
            del self._entries[key]
        
        # Если после удаления истекших записей кеш все еще полон,
        # удаляем наименее недавно использованную запись
        if len(self._entries) >= self.max_size:
            oldest_key = min(self._entries.keys(), 
                           key=lambda k: self._entries[k].last_accessed_at)
            del self._entries[oldest_key]
    
    def size(self) -> int:
        """Возвращает текущий размер кеша.
        
        Returns:
            Количество записей в кеше.
        """
        with self._lock:
            return len(self._entries)
    
    def stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования кеша.
        
        Returns:
            Словарь со статистикой.
        """
        with self._lock:
            hit_count = sum(entry.access_count for entry in self._entries.values())
            active_entries = sum(1 for entry in self._entries.values() if not entry.is_expired())
            expired_entries = len(self._entries) - active_entries
            
            return {
                "size": len(self._entries),
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "hit_count": hit_count,
                "max_size": self.max_size,
                "ttl": self.ttl
            }


# Глобальный кеш
global_cache = Cache()

def cached(key_fn: Optional[Callable[..., str]] = None, ttl: Optional[float] = None, 
          cache: Optional[Cache] = None) -> Callable:
    """Декоратор для кеширования результатов функции.
    
    Args:
        key_fn: Функция для генерации ключа кеша (если None, используется хеш аргументов).
        ttl: Время жизни записи в секундах.
        cache: Кеш для использования (если None, используется глобальный кеш).
        
    Returns:
        Декоратор.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Определяем, какой кеш использовать
            target_cache = cache or global_cache
            
            # Генерируем ключ кеша
            if key_fn is not None:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Создаем хеш аргументов в качестве ключа
                key_parts = []
                for arg in args:
                    try:
                        key_parts.append(str(arg))
                    except Exception:
                        key_parts.append(str(hash(arg)))
                
                for key, value in sorted(kwargs.items()):
                    try:
                        key_parts.append(f"{key}={value}")
                    except Exception:
                        key_parts.append(f"{key}={hash(value)}")
                
                # Добавляем имя функции к ключу
                cache_key = f"{func.__module__}.{func.__name__}:{','.join(key_parts)}"
                
                # Хешируем ключ, если он слишком длинный
                if len(cache_key) > 100:
                    cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            # Проверяем, есть ли результат в кеше
            cached_result = target_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Если результата нет в кеше, вычисляем его
            result = func(*args, **kwargs)
            
            # Сохраняем результат в кеше
            target_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    # Если декоратор используется без аргументов, key_fn на самом деле является функцией
    if callable(key_fn) and ttl is None and cache is None:
        func = key_fn
        key_fn = None
        return decorator(func)
    
    return decorator