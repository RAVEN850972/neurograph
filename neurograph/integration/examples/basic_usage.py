"""
Базовые примеры использования модуля Integration.
"""

import time
from neurograph.integration import (
    create_default_engine,
    create_lightweight_engine,
    create_research_engine,
    ProcessingRequest,
    ProcessingResponse,
    ProcessingMode,
    ResponseFormat,
    ComponentMonitor,
    IntegrationMetrics,
    HealthChecker
)


def example_basic_text_processing():
    """Пример базовой обработки текста."""
    print("=== Базовая обработка текста ===")
    
    # Создание движка
    engine = create_default_engine()
    
    # Простая обработка текста
    response = engine.process_text(
        "Python - это высокоуровневый язык программирования. "
        "Он используется для веб-разработки и машинного обучения."
    )
    
    print(f"Успех: {response.success}")
    print(f"Ответ: {response.primary_response}")
    print(f"Время обработки: {response.processing_time:.3f}с")
    print(f"Использованы компоненты: {', '.join(response.components_used)}")
    
    if response.structured_data:
        nlp_data = response.structured_data.get("nlp", {})
        print(f"Найдено сущностей: {len(nlp_data.get('entities', []))}")
        print(f"Найдено отношений: {len(nlp_data.get('relations', []))}")
    
    engine.shutdown()


def example_query_processing():
    """Пример обработки запросов к системе знаний."""
    print("\n=== Обработка запросов ===")
    
    engine = create_default_engine()
    
    # Сначала обучаем систему
    learning_texts = [
        "Python создан Гвидо ван Россумом в 1991 году",
        "Django - это веб-фреймворк для Python",
        "TensorFlow - библиотека машинного обучения",
        "Машинное обучение - раздел искусственного интеллекта"
    ]
    
    print("Обучение системы...")
    for text in learning_texts:
        response = engine.learn(text)
        print(f"Обучено: {text[:30]}... (успех: {response.success})")
    
    # Теперь задаем вопросы
    questions = [
        "Что такое Python?",
        "Кто создал Python?",
        "Какие фреймворки есть для Python?",
        "Расскажи о машинном обучении"
    ]
    
    print("\nОтветы на вопросы:")
    for question in questions:
        response = engine.query(question)
        print(f"Q: {question}")
        print(f"A: {response.primary_response}")
        print(f"Уверенность: {response.confidence:.2f}")
        print()
    
    engine.shutdown()


def example_advanced_processing():
    """Пример продвинутой обработки с настройками."""
    print("=== Продвинутая обработка ===")
    
    engine = create_research_engine()  # Исследовательская конфигурация
    
    # Создание расширенного запроса
    request = ProcessingRequest(
        content="Нейронные сети используются в глубоком обучении. "
                "Глубокое обучение является частью машинного обучения. "
                "Машинное обучение применяется в искусственном интеллекте.",
        request_type="learning",
        mode=ProcessingMode.SYNCHRONOUS,
        response_format=ResponseFormat.CONVERSATIONAL,
        enable_nlp=True,
        enable_memory_search=True,
        enable_graph_reasoning=True,
        enable_vector_search=True,
        enable_logical_inference=True,
        confidence_threshold=0.3,
        max_results=10,
        explanation_level="detailed"
    )
    
    # Обработка
    response = engine.process_request(request)
    
    print(f"Успех: {response.success}")
    print(f"Время: {response.processing_time:.3f}с")
    print(f"Компоненты: {', '.join(response.components_used)}")
    print(f"Ответ: {response.primary_response}")
    
    # Детали обработки
    if response.explanation:
        print("\nШаги обработки:")
        for step in response.explanation:
            print(f"- {step}")
    
    # Структурированные данные
    if response.structured_data:
        print("\nДетали:")
        for component, data in response.structured_data.items():
            if isinstance(data, dict) and "count" in str(data):
                print(f"- {component}: {data}")
    
    engine.shutdown()


def example_monitoring_and_health():
    """Пример мониторинга и проверки здоровья системы."""
    print("\n=== Мониторинг системы ===")
    
    engine = create_default_engine()
    
    # Создание монитора
    monitor = ComponentMonitor(check_interval=10.0)
    monitor.start_monitoring(engine.provider)
    
    # Создание проверщика здоровья
    health_checker = HealthChecker()
    
    # Выполнение нескольких операций для сбора метрик
    test_requests = [
        "Python - язык программирования",
        "Что такое Python?",
        "Django связан с Python",
        "Машинное обучение использует алгоритмы"
    ]
    
    print("Выполнение тестовых запросов...")
    for i, text in enumerate(test_requests):
        try:
            start_time = time.time()
            response = engine.process_text(text)
            processing_time = time.time() - start_time
            
            # Записываем метрики
            monitor.metrics.record_request("test", processing_time, response.success)
            
            print(f"Запрос {i+1}: {'✓' if response.success else '✗'} "
                  f"({processing_time:.3f}с)")
            
        except Exception as e:
            print(f"Запрос {i+1}: ✗ (ошибка: {e})")
            monitor.metrics.record_request("test", 0.0, False)
    
    # Проверка здоровья компонентов
    health_results = health_checker.check_all_components(engine.provider)
    
    print(f"\n{health_checker.get_health_summary()}")
    
    # Детальный отчет о здоровье
    overall_health = health_checker.get_overall_health()
    print(f"Статус системы: {overall_health['status']}")
    print(f"Компонентов здоровых: {overall_health['healthy']}")
    print(f"Компонентов с проблемами: {overall_health['degraded'] + overall_health['unhealthy']}")
    
    # Метрики производительности
    metrics_summary = monitor.metrics.get_summary()
    print(f"\nМетрики производительности:")
    print(f"- Всего запросов: {metrics_summary['requests']['total']}")
    print(f"- Успешность: {metrics_summary['requests']['success_rate']:.1%}")
    print(f"- Среднее время: {metrics_summary['requests']['average_response_time']:.3f}с")
    print(f"- Запросов в минуту: {metrics_summary['requests']['requests_per_minute']:.1f}")
    
    # Отчет мониторинга
    monitor_report = monitor.get_monitoring_report()
    if monitor_report['recent_alerts']:
        print(f"\nПоследние алерты: {len(monitor_report['recent_alerts'])}")
        for alert in monitor_report['recent_alerts'][-3:]:
            print(f"- {alert['type']}: {alert['message']}")
    
    # Завершение
    monitor.stop_monitoring()
    engine.shutdown()


def example_configuration_management():
    """Пример управления конфигурациями."""
    print("\n=== Управление конфигурациями ===")
    
    from neurograph.integration.config import IntegrationConfigManager
    
    config_manager = IntegrationConfigManager()
    
    # Создание шаблонов конфигураций
    templates = ["default", "lightweight", "research", "production"]
    
    for template_name in templates:
        print(f"\nШаблон '{template_name}':")
        template = config_manager.create_template_config(template_name)
        
        print(f"- Движок: {template['engine_name']}")
        print(f"- Макс. запросов: {template['max_concurrent_requests']}")
        print(f"- Таймаут: {template['default_timeout']}с")
        print(f"- Кеширование: {template['enable_caching']}")
        print(f"- Метрики: {template['enable_metrics']}")
        
        # Компоненты
        components = template['components']
        print(f"- Компонентов: {len(components)}")
        
        for comp_name, comp_config in components.items():
            comp_type = comp_config.get('type', 'default')
            params_count = len(comp_config.get('params', {}))
            print(f"  - {comp_name}: {comp_type} ({params_count} параметров)")


def example_pipeline_comparison():
    """Пример сравнения различных конвейеров обработки."""
    print("\n=== Сравнение конвейеров ===")
    
    engine = create_default_engine()
    
    test_text = ("Искусственный интеллект революционизирует технологии. "
                "Машинное обучение позволяет компьютерам обучаться. "
                "Нейронные сети моделируют работу мозга.")
    
    # Тестируем разные типы запросов
    request_types = [
        ("text_processing", "Обработка текста"),
        ("learning", "Обучение"),
        ("query", "Запрос"),
        ("inference", "Логический вывод")
    ]
    
    results = {}
    
    for request_type, description in request_types:
        print(f"\n{description} ({request_type}):")
        
        try:
            request = ProcessingRequest(
                content=test_text,
                request_type=request_type,
                response_format=ResponseFormat.STRUCTURED
            )
            
            start_time = time.time()
            response = engine.process_request(request)
            processing_time = time.time() - start_time
            
            results[request_type] = {
                "success": response.success,
                "time": processing_time,
                "components": len(response.components_used),
                "structured_keys": len(response.structured_data.keys())
            }
            
            print(f"  Успех: {'✓' if response.success else '✗'}")
            print(f"  Время: {processing_time:.3f}с")
            print(f"  Компонентов: {len(response.components_used)}")
            print(f"  Данных: {len(response.structured_data.keys())} блоков")
            
            if response.explanation:
                print(f"  Шагов: {len(response.explanation)}")
            
        except Exception as e:
            print(f"  Ошибка: {e}")
            results[request_type] = {"success": False, "error": str(e)}
    
    # Сводка
    print(f"\n=== Сводка по конвейерам ===")
    successful = sum(1 for r in results.values() if r.get("success", False))
    total = len(results)
    
    print(f"Успешных: {successful}/{total}")
    
    if successful > 0:
        avg_time = sum(r.get("time", 0) for r in results.values() if r.get("success")) / successful
        print(f"Среднее время: {avg_time:.3f}с")
        
        fastest = min((r for r in results.items() if r[1].get("success")), 
                     key=lambda x: x[1]["time"])
        print(f"Самый быстрый: {fastest[0]} ({fastest[1]['time']:.3f}с)")
    
    engine.shutdown()


def main():
    """Запуск всех примеров."""
    print("🚀 Примеры использования модуля Integration NeuroGraph\n")
    
    examples = [
        ("Базовая обработка текста", example_basic_text_processing),
        ("Обработка запросов", example_query_processing),
        ("Продвинутая обработка", example_advanced_processing),
        ("Мониторинг и здоровье", example_monitoring_and_health),
        ("Управление конфигурациями", example_configuration_management),
        ("Сравнение конвейеров", example_pipeline_comparison)
    ]
    
    for name, example_func in examples:
        print(f"📋 {name}")
        try:
            example_func()
        except Exception as e:
            print(f"❌ Ошибка в примере '{name}': {e}")
        
        print("\n" + "="*60)
    
    print("✅ Все примеры завершены!")


if __name__ == "__main__":
    main()