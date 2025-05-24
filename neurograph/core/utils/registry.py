"""Система регистрации компонентов."""

from typing import Dict, Any, Type, Callable, Optional, TypeVar, Generic
import inspect

T = TypeVar('T')


class Registry(Generic[T]):
    """Хранилище компонентов для динамической регистрации и создания."""
    
    def __init__(self, name: str):
        """Инициализирует регистр с заданным именем.
        
        Args:
            name: Имя регистра.
        """
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
        
    def register(self, name: str, component_class: Type[T]) -> None:
        """Регистрирует компонент в регистре.
        
        Args:
            name: Уникальное имя компонента.
            component_class: Класс компонента.
            
        Raises:
            ValueError: Если компонент с указанным именем уже зарегистрирован.
        """
        if name in self._registry:
            raise ValueError(f"Компонент с именем '{name}' уже зарегистрирован в регистре '{self.name}'")
            
        self._registry[name] = component_class
        
    def decorator(self, name: Optional[str] = None) -> Callable[[Type[T]], Type[T]]:
        """Декоратор для регистрации компонентов.
        
        Args:
            name: Имя для регистрации компонента. Если None, используется имя класса.
            
        Returns:
            Декоратор для регистрации компонента.
        """
        def wrapper(cls: Type[T]) -> Type[T]:
            component_name = name or cls.__name__
            self.register(component_name, cls)
            return cls
            
        return wrapper
        
    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Создает экземпляр компонента с указанным именем.
        
        Args:
            name: Имя зарегистрированного компонента.
            *args: Позиционные аргументы для конструктора компонента.
            **kwargs: Именованные аргументы для конструктора компонента.
            
        Returns:
            Экземпляр компонента.
            
        Raises:
            ValueError: Если компонент с указанным именем не зарегистрирован.
        """
        if name not in self._registry:
            raise ValueError(f"Компонент с именем '{name}' не найден в регистре '{self.name}'")
            
        component_class = self._registry[name]
        return component_class(*args, **kwargs)
        
    def get_names(self) -> list[str]:
        """Возвращает список имен зарегистрированных компонентов.
        
        Returns:
            Список имен зарегистрированных компонентов.
        """
        return list(self._registry.keys())
        
    def get_class(self, name: str) -> Type[T]:
        """Возвращает класс компонента по имени.
        
        Args:
            name: Имя зарегистрированного компонента.
            
        Returns:
            Класс компонента.
            
        Raises:
            ValueError: Если компонент с указанным именем не зарегистрирован.
        """
        if name not in self._registry:
            raise ValueError(f"Компонент с именем '{name}' не найден в регистре '{self.name}'")
            
        return self._registry[name]