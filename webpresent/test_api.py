#!/usr/bin/env python3
"""
Тестирование API NeuroGraph для выявления доступных методов
"""

import inspect
from typing import Any, Dict, List

def test_neurograph_api():
    """Тестирование реального API NeuroGraph."""
    
    try:
        from neurograph.integration import create_default_engine
        print("✅ NeuroGraph успешно импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта NeuroGraph: {e}")
        return
    
    try:
        # Создаем движок
        engine = create_default_engine()
        print("✅ Движок NeuroGraph создан")
        
        # Исследуем движок
        print("\n" + "="*50)
        print("МЕТОДЫ ДВИЖКА:")
        print("="*50)
        
        engine_methods = [method for method in dir(engine) if not method.startswith('_')]
        for method in engine_methods:
            try:
                attr = getattr(engine, method)
                if callable(attr):
                    sig = inspect.signature(attr)
                    print(f"  ✓ {method}{sig}")
                else:
                    print(f"  • {method}: {type(attr).__name__}")
            except Exception as e:
                print(f"  ⚠ {method}: ошибка получения сигнатуры - {e}")
        
        # Тестируем компоненты
        print("\n" + "="*50)
        print("ТЕСТИРОВАНИЕ КОМПОНЕНТОВ:")
        print("="*50)
        
        try:
            memory = engine.provider.get_component('memory')
            print("\n📊 MEMORY КОМПОНЕНТ:")
            print("-" * 30)
            
            memory_methods = [method for method in dir(memory) if not method.startswith('_')]
            for method in memory_methods:
                try:
                    attr = getattr(memory, method)
                    if callable(attr):
                        sig = inspect.signature(attr)
                        print(f"  ✓ {method}{sig}")
                    else:
                        print(f"  • {method}: {type(attr).__name__}")
                except Exception as e:
                    print(f"  ⚠ {method}: ошибка - {e}")
            
            # Тестируем методы памяти
            print("\n🧪 ТЕСТИРОВАНИЕ МЕТОДОВ ПАМЯТИ:")
            print("-" * 35)
            
            # Тест get_memory_statistics
            try:
                stats = memory.get_memory_statistics()
                print(f"  ✅ get_memory_statistics(): работает")
                print(f"     Структура: {list(stats.keys())}")
            except Exception as e:
                print(f"  ❌ get_memory_statistics(): {e}")
            
            # Тест get_recent_items
            try:
                recent = memory.get_recent_items()
                print(f"  ✅ get_recent_items(): работает, получено {len(recent) if recent else 0} элементов")
                if recent:
                    print(f"     Первый элемент: {type(recent[0])}")
                    print(f"     Атрибуты: {dir(recent[0])}")
            except Exception as e:
                print(f"  ❌ get_recent_items(): {e}")
            
            # Тест get_most_accessed_items
            try:
                accessed = memory.get_most_accessed_items()
                print(f"  ✅ get_most_accessed_items(): работает, получено {len(accessed) if accessed else 0} элементов")
            except Exception as e:
                print(f"  ❌ get_most_accessed_items(): {e}")
            
            # Тест search
            try:
                # Сначала добавим что-то в память через обучение
                print("\n  📝 Добавляем тестовые данные...")
                engine.learn("Тестовые данные для поиска в памяти")
                
                search_result = memory.search("тестовые", limit=5)
                print(f"  ✅ search('тестовые', limit=5): работает, найдено {len(search_result) if search_result else 0} элементов")
            except Exception as e:
                try:
                    # Пробуем без параметра limit
                    search_result = memory.search("тестовые")
                    print(f"  ✅ search('тестовые'): работает, найдено {len(search_result) if search_result else 0} элементов")
                except Exception as e2:
                    print(f"  ❌ search(): {e2}")
            
            # Исследуем STM и LTM
            print("\n🔍 ИССЛЕДОВАНИЕ STM/LTM:")
            print("-" * 25)
            
            if hasattr(memory, 'stm'):
                print(f"  ✓ STM доступен: {type(memory.stm)}")
                stm_methods = [m for m in dir(memory.stm) if not m.startswith('_')]
                print(f"    Методы STM: {stm_methods[:5]}...")  # Показываем первые 5
            
            if hasattr(memory, 'ltm'):
                print(f"  ✓ LTM доступен: {type(memory.ltm)}")
                ltm_methods = [m for m in dir(memory.ltm) if not m.startswith('_')]
                print(f"    Методы LTM: {ltm_methods[:5]}...")  # Показываем первые 5
                
        except Exception as e:
            print(f"❌ Ошибка работы с memory: {e}")
        
        # Тестируем граф
        try:
            graph = engine.provider.get_component('semgraph')
            print("\n🕸️ SEMGRAPH КОМПОНЕНТ:")
            print("-" * 30)
            
            graph_methods = [method for method in dir(graph) if not method.startswith('_')]
            print(f"  Методы: {graph_methods[:10]}...")  # Показываем первые 10
            
        except Exception as e:
            print(f"❌ Ошибка работы с semgraph: {e}")
        
        # Тестируем NLP
        try:
            nlp = engine.provider.get_component('nlp')
            print("\n🔤 NLP КОМПОНЕНТ:")
            print("-" * 20)
            
            nlp_methods = [method for method in dir(nlp) if not method.startswith('_')]
            print(f"  Методы: {nlp_methods[:10]}...")  # Показываем первые 10
            
        except Exception as e:
            print(f"❌ Ошибка работы с nlp: {e}")
        
        # Тестируем основные методы движка
        print("\n" + "="*50)
        print("ТЕСТИРОВАНИЕ ОСНОВНЫХ МЕТОДОВ ДВИЖКА:")
        print("="*50)
        
        # Тест learn
        try:
            learn_result = engine.learn("Тестовое обучение")
            print(f"  ✅ learn(): работает")
            print(f"     Результат: {type(learn_result)}")
            print(f"     Success: {getattr(learn_result, 'success', 'нет атрибута')}")
        except Exception as e:
            print(f"  ❌ learn(): {e}")
        
        # Тест query
        try:
            query_result = engine.query("Тестовый запрос")
            print(f"  ✅ query(): работает")
            print(f"     Результат: {type(query_result)}")
            print(f"     Primary response: {getattr(query_result, 'primary_response', 'нет атрибута')}")
        except Exception as e:
            print(f"  ❌ query(): {e}")
        
        # Тест process_text
        try:
            process_result = engine.process_text("Тестовый текст для обработки")
            print(f"  ✅ process_text(): работает")
            print(f"     Результат: {type(process_result)}")
        except Exception as e:
            print(f"  ❌ process_text(): {e}")
        
        print("\n✅ Тестирование завершено!")
        
        # Финальная статистика памяти
        try:
            final_stats = memory.get_memory_statistics()
            print(f"\n📊 Финальная статистика памяти:")
            print(f"  STM: {final_stats['memory_levels']['stm']['size']} элементов")
            print(f"  LTM: {final_stats['memory_levels']['ltm']['size']} элементов")
            print(f"  Всего: {final_stats.get('total_items', 'неизвестно')}")
        except Exception as e:
            print(f"❌ Ошибка получения финальной статистики: {e}")
            
        # Корректное завершение
        try:
            engine.shutdown()
            print("✅ Движок корректно завершен")
        except Exception as e:
            print(f"⚠️ Ошибка при завершении: {e}")
    
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 ТЕСТИРОВАНИЕ API NEUROGRAPH")
    print("=" * 60)
    test_neurograph_api()