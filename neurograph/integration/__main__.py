"""
Точка входа для запуска модуля Integration как standalone приложения.
"""

import sys
import argparse
from pathlib import Path

from neurograph.integration import (
    create_default_engine,
    create_lightweight_engine,
    create_research_engine,
    ProcessingRequest,
    ComponentMonitor
)
from neurograph.integration.examples.basic_usage import main as run_basic_examples
from neurograph.integration.examples.advanced_integration import main as run_advanced_examples
from neurograph.integration.examples.performance_testing import main as run_performance_tests


def main():
    """Главная функция для CLI интерфейса."""
    parser = argparse.ArgumentParser(
        description="NeuroGraph Integration Module CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m neurograph.integration --demo basic
  python -m neurograph.integration --interactive
  python -m neurograph.integration --test performance
  python -m neurograph.integration --process "Ваш текст здесь"
        """
    )
    
    # Основные команды
    parser.add_argument(
        "--demo", 
        choices=["basic", "advanced", "all"],
        help="Запуск демонстрационных примеров"
    )
    
    parser.add_argument(
        "--test",
        choices=["unit", "integration", "performance", "all"],
        help="Запуск тестов"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Интерактивный режим"
    )
    
    parser.add_argument(
        "--process",
        type=str,
        help="Обработка указанного текста"
    )
    
    # Настройки
    parser.add_argument(
        "--config",
        choices=["default", "lightweight", "research"],
        default="default",
        help="Тип конфигурации движка"
    )
    
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Включить мониторинг"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод"
    )
    
    args = parser.parse_args()
    
    # Обработка команд
    if args.demo:
        run_demo(args.demo)
    elif args.test:
        run_tests(args.test)
    elif args.interactive:
        run_interactive_mode(args.config, args.monitor, args.verbose)
    elif args.process:
        process_text(args.process, args.config, args.verbose)
    else:
        parser.print_help()


def run_demo(demo_type: str):
    """Запуск демонстрационных примеров."""
    print(f"🚀 Запуск демонстрации: {demo_type}")
    
    try:
        if demo_type == "basic":
            run_basic_examples()
        elif demo_type == "advanced":
            run_advanced_examples()
        elif demo_type == "all":
            print("📋 Базовые примеры:")
            run_basic_examples()
            print("\n📋 Продвинутые примеры:")
            run_advanced_examples()
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")
        return False
    
    return True


def run_tests(test_type: str):
    """Запуск тестов."""
    print(f"🧪 Запуск тестов: {test_type}")
    
    try:
        if test_type == "unit":
            from tests.integration.test_integration_module import run_integration_tests
            return run_integration_tests()
        elif test_type == "performance":
            run_performance_tests()
        elif test_type == "all":
            print("🧪 Unit тесты:")
            from tests.integration.test_integration_module import run_integration_tests
            unit_success = run_integration_tests()
            
            print("\n🚀 Тесты производительности:")
            run_performance_tests()
            
            return unit_success
        else:
            print(f"❌ Неизвестный тип тестов: {test_type}")
            return False
    except Exception as e:
        print(f"❌ Ошибка в тестах: {e}")
        return False


def create_engine(config_type: str):
    """Создание движка по типу конфигурации."""
    if config_type == "lightweight":
        return create_lightweight_engine()
    elif config_type == "research":
        return create_research_engine()
    else:
        return create_default_engine()


def process_text(text: str, config_type: str, verbose: bool):
    """Обработка текста."""
    print(f"📝 Обработка текста (конфигурация: {config_type})")
    if verbose:
        print(f"Текст: {text}")
    
    engine = create_engine(config_type)
    
    try:
        start_time = time.time()
        response = engine.process_text(text)
        processing_time = time.time() - start_time
        
        print(f"\n📊 Результат:")
        print(f"✅ Успех: {response.success}")
        print(f"⏱️ Время: {processing_time:.3f}с")
        print(f"🔧 Компоненты: {', '.join(response.components_used)}")
        print(f"💬 Ответ: {response.primary_response}")
        
        if verbose and response.structured_data:
            print(f"\n📋 Структурированные данные:")
            for key, data in response.structured_data.items():
                if isinstance(data, dict):
                    print(f"  {key}: {len(data)} элементов")
                else:
                    print(f"  {key}: {type(data).__name__}")
        
        if response.explanation:
            print(f"\n📝 Объяснение:")
            for explanation in response.explanation:
                print(f"  - {explanation}")
        
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
    finally:
        engine.shutdown()


def run_interactive_mode(config_type: str, enable_monitor: bool, verbose: bool):
    """Интерактивный режим работы."""
    print(f"🤖 Интерактивный режим NeuroGraph (конфигурация: {config_type})")
    print("Введите 'help' для справки, 'quit' для выхода")
    
    engine = create_engine(config_type)
    monitor = None
    
    if enable_monitor:
        monitor = ComponentMonitor()
        monitor.start_monitoring(engine.provider)
        print("📊 Мониторинг включен")
    
    try:
        while True:
            try:
                user_input = input("\n🤖 > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 До свидания!")
                    break
                
                if user_input.lower() == 'help':
                    print_help()
                    continue
                
                if user_input.lower() == 'status':
                    print_system_status(engine, monitor, verbose)
                    continue
                
                if user_input.lower().startswith('learn:'):
                    text = user_input[6:].strip()
                    if text:
                        process_learning(engine, text, verbose)
                    else:
                        print("❌ Укажите текст для изучения: learn: <текст>")
                    continue
                
                if user_input.lower().startswith('query:'):
                    text = user_input[6:].strip()
                    if text:
                        process_query(engine, text, verbose)
                    else:
                        print("❌ Укажите запрос: query: <запрос>")
                    continue
                
                # Обычная обработка текста
                process_interactive_input(engine, user_input, verbose)
                
            except KeyboardInterrupt:
                print("\n\n👋 Выход по Ctrl+C")
                break
            except EOFError:
                print("\n\n👋 Выход по EOF")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
    
    finally:
        if monitor:
            monitor.stop_monitoring()
        engine.shutdown()


def print_help():
    """Вывод справки для интерактивного режима."""
    help_text = """
📚 Справка по командам:

Основные команды:
  <текст>           - Обработка произвольного текста
  learn: <текст>    - Изучение новой информации  
  query: <запрос>   - Запрос к системе знаний
  
Системные команды:
  status            - Состояние системы
  help              - Эта справка
  quit/exit/q       - Выход

Примеры:
  Python - это язык программирования
  learn: Django - веб-фреймворк для Python
  query: Что такое Python?
  status
    """
    print(help_text)


def print_system_status(engine, monitor, verbose):
    """Вывод статуса системы."""
    print("📊 Статус системы:")
    
    # Здоровье системы
    health = engine.get_health_status()
    status_emoji = {
        "healthy": "✅",
        "degraded": "🟡", 
        "critical": "❌"
    }.get(health.get("overall_status", "unknown"), "❓")
    
    print(f"  {status_emoji} Общий статус: {health.get('overall_status', 'unknown')}")
    print(f"  🔧 Компонентов: {len(health.get('components', {}))}")
    
    # Метрики
    metrics = health.get("metrics", {})
    if metrics:
        print(f"  📈 Запросов обработано: {metrics.get('requests_processed', 0)}")
        print(f"  ✅ Успешных: {metrics.get('successful_requests', 0)}")
        print(f"  ❌ Неудачных: {metrics.get('failed_requests', 0)}")
        
        avg_time = metrics.get('average_processing_time', 0)
        if avg_time > 0:
            print(f"  ⏱️ Среднее время: {avg_time:.3f}с")
    
    # Подробности о компонентах
    if verbose:
        print(f"\n🔧 Детали по компонентам:")
        for comp_name, comp_status in health.get("components", {}).items():
            status = comp_status.get("status", "unknown")
            emoji = {"healthy": "✅", "degraded": "🟡", "unhealthy": "❌"}.get(status, "❓")
            print(f"    {emoji} {comp_name}: {status}")
    
    # Мониторинг
    if monitor:
        monitor_report = monitor.get_monitoring_report()
        print(f"  📊 Мониторинг активен: {monitor_report['monitoring_active']}")
        
        alerts = monitor_report.get('recent_alerts', [])
        if alerts:
            print(f"  ⚠️ Последних алертов: {len(alerts)}")


def process_learning(engine, text, verbose):
    """Обработка команды изучения."""
    print(f"📚 Изучаю: {text}")
    
    try:
        response = engine.learn(text)
        
        if response.success:
            print(f"✅ Информация изучена!")
            if verbose:
                learning_data = response.structured_data.get("learning", {})
                if learning_data:
                    print(f"📋 Детали: {learning_data}")
        else:
            print(f"❌ Ошибка изучения: {response.error_message}")
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def process_query(engine, query, verbose):
    """Обработка запроса к знаниям."""
    print(f"❓ Запрос: {query}")
    
    try:
        response = engine.query(query)
        
        if response.success:
            print(f"💡 Ответ: {response.primary_response}")
            
            if verbose and response.structured_data:
                graph_search = response.structured_data.get("graph_search", {})
                if graph_search:
                    found = len(graph_search.get("found_nodes", []))
                    related = len(graph_search.get("related_concepts", []))
                    print(f"📊 Найдено в графе: {found} узлов, {related} связанных концептов")
                
                memory_search = response.structured_data.get("memory_search", {})
                if memory_search:
                    memories = len(memory_search.get("relevant_memories", []))
                    print(f"🧠 Найдено в памяти: {memories} релевантных воспоминаний")
        else:
            print(f"❌ Ошибка запроса: {response.error_message}")
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def process_interactive_input(engine, text, verbose):
    """Обработка обычного ввода."""
    try:
        start_time = time.time()
        response = engine.process_text(text)
        processing_time = time.time() - start_time
        
        if response.success:
            print(f"💬 {response.primary_response}")
            
            if verbose:
                print(f"⏱️ Время: {processing_time:.3f}с")
                print(f"🔧 Компоненты: {', '.join(response.components_used)}")
                
                if response.structured_data:
                    nlp_data = response.structured_data.get("nlp", {})
                    if nlp_data:
                        entities = len(nlp_data.get("entities", []))
                        relations = len(nlp_data.get("relations", []))
                        print(f"📋 NLP: {entities} сущностей, {relations} отношений")
        else:
            print(f"❌ Ошибка: {response.error_message}")
    
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")


if __name__ == "__main__":
    import time
    main()