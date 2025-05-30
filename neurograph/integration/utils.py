# neurograph/integration/utils.py
"""
Утилиты для мониторинга и диагностики интеграционного слоя.
"""

import time
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from neurograph.core.logging import get_logger


@dataclass
class ComponentHealth:
    """Состояние здоровья компонента."""
    name: str
    status: str  # healthy, degraded, unhealthy, unknown
    response_time: float = 0.0
    error_rate: float = 0.0
    last_check: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class IntegrationMetrics:
    """Сбор и анализ метрик интеграционного слоя."""
    
    def __init__(self):
        self.logger = get_logger("integration_metrics")
        self.start_time = time.time()
        
        # Метрики запросов
        self.request_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0,
            "request_types": {}
        }
        
        # Метрики компонентов
        self.component_metrics = {}
        
        # Метрики производительности
        self.performance_metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": []
        }
        
        # История метрик
        self.metrics_history = []
        self.max_history_size = 1000
    
    def record_request(self, request_type: str, response_time: float, success: bool) -> None:
        """Запись метрик запроса."""
        self.request_metrics["total_requests"] += 1
        self.request_metrics["total_response_time"] += response_time
        
        if success:
            self.request_metrics["successful_requests"] += 1
        else:
            self.request_metrics["failed_requests"] += 1
        
        # Обновляем среднее время ответа
        if self.request_metrics["total_requests"] > 0:
            self.request_metrics["average_response_time"] = (
                self.request_metrics["total_response_time"] / 
                self.request_metrics["total_requests"]
            )
        
        # Метрики по типам запросов
        if request_type not in self.request_metrics["request_types"]:
            self.request_metrics["request_types"][request_type] = {
                "count": 0,
                "success_count": 0,
                "total_time": 0.0
            }
        
        type_metrics = self.request_metrics["request_types"][request_type]
        type_metrics["count"] += 1
        type_metrics["total_time"] += response_time
        
        if success:
            type_metrics["success_count"] += 1
    
    def record_component_metrics(self, component_name: str, metrics: Dict[str, Any]) -> None:
        """Запись метрик компонента."""
        if component_name not in self.component_metrics:
            self.component_metrics[component_name] = {
                "operations": 0,
                "errors": 0,
                "total_time": 0.0,
                "last_update": time.time()
            }
        
        comp_metrics = self.component_metrics[component_name]
        comp_metrics.update(metrics)
        comp_metrics["last_update"] = time.time()
    
    def record_system_metrics(self) -> None:
        """Запись системных метрик."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.performance_metrics["cpu_usage"].append({
                "timestamp": time.time(),
                "value": cpu_percent
            })
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.performance_metrics["memory_usage"].append({
                "timestamp": time.time(),
                "percent": memory.percent,
                "available": memory.available,
                "used": memory.used
            })
            
            # Ограничиваем размер истории
            for metric_type in self.performance_metrics:
                if len(self.performance_metrics[metric_type]) > 100:
                    self.performance_metrics[metric_type] = (
                        self.performance_metrics[metric_type][-50:]
                    )
                    
        except Exception as e:
            self.logger.warning(f"Ошибка сбора системных метрик: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик."""
        uptime = time.time() - self.start_time
        
        # Расчет коэффициентов
        total_requests = self.request_metrics["total_requests"]
        success_rate = 0.0
        if total_requests > 0:
            success_rate = (
                self.request_metrics["successful_requests"] / total_requests
            )
        
        # Последние системные метрики
        latest_cpu = 0.0
        latest_memory = 0.0
        
        if self.performance_metrics["cpu_usage"]:
            latest_cpu = self.performance_metrics["cpu_usage"][-1]["value"]
        
        if self.performance_metrics["memory_usage"]:
            latest_memory = self.performance_metrics["memory_usage"][-1]["percent"]
        
        return {
            "uptime_seconds": uptime,
            "requests": {
                "total": total_requests,
                "success_rate": success_rate,
                "average_response_time": self.request_metrics["average_response_time"],
                "requests_per_minute": (total_requests / uptime * 60) if uptime > 0 else 0
            },
            "system": {
                "cpu_percent": latest_cpu,
                "memory_percent": latest_memory
            },
            "components": {
                name: {
                    "operations": metrics.get("operations", 0),
                    "error_rate": (
                        metrics.get("errors", 0) / max(1, metrics.get("operations", 1))
                    )
                }
                for name, metrics in self.component_metrics.items()
            }
        }
    
    def get_detailed_report(self) -> str:
        """Получение детального отчета о метриках."""
        summary = self.get_summary()
        
        report_parts = [
            "=== Отчет о состоянии системы NeuroGraph ===",
            f"Время работы: {summary['uptime_seconds']:.1f} секунд",
            "",
            "Запросы:",
            f"  Всего: {summary['requests']['total']}",
            f"  Успешность: {summary['requests']['success_rate']:.1%}",
            f"  Среднее время ответа: {summary['requests']['average_response_time']:.3f}с",
            f"  Запросов в минуту: {summary['requests']['requests_per_minute']:.1f}",
            "",
            "Система:",
            f"  CPU: {summary['system']['cpu_percent']:.1f}%",
            f"  Память: {summary['system']['memory_percent']:.1f}%",
            "",
            "Компоненты:"
        ]
        
        for name, metrics in summary["components"].items():
            report_parts.extend([
                f"  {name}:",
                f"    Операций: {metrics['operations']}",
                f"    Ошибок: {metrics['error_rate']:.1%}"
            ])
        
        # Статистика по типам запросов
        if self.request_metrics["request_types"]:
            report_parts.extend(["", "Типы запросов:"])
            for request_type, type_metrics in self.request_metrics["request_types"].items():
                success_rate = (
                    type_metrics["success_count"] / max(1, type_metrics["count"])
                )
                avg_time = (
                    type_metrics["total_time"] / max(1, type_metrics["count"])
                )
                report_parts.extend([
                    f"  {request_type}:",
                    f"    Количество: {type_metrics['count']}",
                    f"    Успешность: {success_rate:.1%}",
                    f"    Среднее время: {avg_time:.3f}с"
                ])
        
        return "\n".join(report_parts)


class HealthChecker:
    """Проверка состояния здоровья компонентов системы."""
    
    def __init__(self):
        self.logger = get_logger("health_checker")
        self.component_health: Dict[str, ComponentHealth] = {}
        self.check_interval = 30.0  # Интервал проверок в секундах
        self.last_full_check = 0.0
    
    def check_component_health(self, component_name: str, component: Any) -> ComponentHealth:
        """Проверка здоровья отдельного компонента."""
        start_time = time.time()
        
        try:
            # Базовая проверка доступности
            if not component:
                return ComponentHealth(
                    name=component_name,
                    status="unhealthy",
                    details={"error": "Component not available"}
                )
            
            # Проверка методов статистики
            details = {}
            if hasattr(component, 'get_statistics'):
                try:
                    stats = component.get_statistics()
                    details["statistics"] = stats
                except Exception as e:
                    details["statistics_error"] = str(e)
            
            # Проверка специфичных методов
            status = "healthy"
            if hasattr(component, 'size'):
                try:
                    size = component.size()
                    details["size"] = size
                    
                    # Проверка переполнения для компонентов с ограниченным размером
                    if hasattr(component, 'capacity'):
                        capacity = getattr(component, 'capacity', None)
                        if capacity and size > capacity * 0.9:
                            status = "degraded"
                            details["warning"] = "Near capacity limit"
                            
                except Exception as e:
                    status = "degraded"
                    details["size_error"] = str(e)
            
            response_time = time.time() - start_time
            
            return ComponentHealth(
                name=component_name,
                status=status,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка проверки здоровья {component_name}: {e}")
            return ComponentHealth(
                name=component_name,
                status="unhealthy",
                response_time=time.time() - start_time,
                details={"error": str(e)}
            )
    
    def check_all_components(self, provider: Any) -> Dict[str, ComponentHealth]:
        """Проверка здоровья всех компонентов."""
        self.last_full_check = time.time()
        
        # Получаем все компоненты
        if hasattr(provider, 'get_all_components_status'):
            components_status = provider.get_all_components_status()
        else:
            components_status = {}
        
        health_results = {}
        
        # Проверяем каждый компонент
        for component_name in components_status:
            try:
                component = provider.get_component(component_name)
                health = self.check_component_health(component_name, component)
                health_results[component_name] = health
                self.component_health[component_name] = health
                
            except Exception as e:
                health = ComponentHealth(
                    name=component_name,
                    status="unhealthy",
                    details={"error": f"Failed to get component: {e}"}
                )
                health_results[component_name] = health
                self.component_health[component_name] = health
        
        return health_results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Получение общего состояния здоровья системы."""
        if not self.component_health:
            return {
                "status": "unknown",
                "message": "No health checks performed yet"
            }
        
        # Анализируем состояния компонентов
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        
        for health in self.component_health.values():
            if health.status == "healthy":
                healthy_count += 1
            elif health.status == "degraded":
                degraded_count += 1
            else:
                unhealthy_count += 1
        
        total_components = len(self.component_health)
        
        # Определяем общий статус
        if unhealthy_count > 0:
            overall_status = "critical"
        elif degraded_count > total_components // 2:
            overall_status = "degraded"
        elif degraded_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "total_components": total_components,
            "healthy": healthy_count,
            "degraded": degraded_count,
            "unhealthy": unhealthy_count,
            "last_check": self.last_full_check,
            "details": {
                name: {
                    "status": health.status,
                    "response_time": health.response_time
                }
                for name, health in self.component_health.items()
            }
        }
    
    def get_health_summary(self) -> str:
        """Получение краткой сводки о здоровье системы."""
        overall = self.get_overall_health()
        
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "degraded": "🟡",
            "critical": "❌",
            "unknown": "❓"
        }
        
        emoji = status_emoji.get(overall["status"], "❓")
        
        return (
            f"{emoji} Система: {overall['status'].upper()}\n"
            f"Компонентов: {overall['total_components']} "
            f"(✅{overall['healthy']} ⚠️{overall['degraded']} ❌{overall['unhealthy']})"
        )


class ComponentMonitor:
    """Мониторинг компонентов в реальном времени."""
    
    def __init__(self, check_interval: float = 60.0):
        self.logger = get_logger("component_monitor")
        self.check_interval = check_interval
        self.metrics = IntegrationMetrics()
        self.health_checker = HealthChecker()
        self.monitoring_active = False
        
        # Пороги для алертов
        self.alert_thresholds = {
            "response_time": 5.0,      # секунды
            "error_rate": 0.1,         # 10%
            "cpu_usage": 80.0,         # 80%
            "memory_usage": 85.0       # 85%
        }
        
        # История алертов
        self.alerts_history = []
        self.max_alerts_history = 100
    
    def start_monitoring(self, provider: Any) -> None:
        """Запуск мониторинга."""
        if self.monitoring_active:
            self.logger.warning("Мониторинг уже активен")
            return
        
        self.monitoring_active = True
        self.logger.info("Запуск мониторинга компонентов")
        
        # В реальной реализации здесь был бы запуск в отдельном потоке
        # Для демонстрации выполняем однократную проверку
        self._perform_monitoring_cycle(provider)
    
    def stop_monitoring(self) -> None:
        """Остановка мониторинга."""
        self.monitoring_active = False
        self.logger.info("Остановка мониторинга компонентов")
    
    def _perform_monitoring_cycle(self, provider: Any) -> None:
        """Выполнение цикла мониторинга."""
        try:
            # Сбор системных метрик
            self.metrics.record_system_metrics()
            
            # Проверка здоровья компонентов
            health_results = self.health_checker.check_all_components(provider)
            
            # Анализ на предмет алертов
            self._check_for_alerts(health_results)
            
            # Обновление метрик компонентов
            for name, health in health_results.items():
                self.metrics.record_component_metrics(name, {
                    "response_time": health.response_time,
                    "status": health.status,
                    "last_check": health.last_check
                })
            
            self.logger.debug("Цикл мониторинга завершен")
            
        except Exception as e:
            self.logger.error(f"Ошибка в цикле мониторинга: {e}")
    
    def _check_for_alerts(self, health_results: Dict[str, ComponentHealth]) -> None:
        """Проверка условий для алертов."""
        current_time = time.time()
        
        # Проверка времени ответа компонентов
        for name, health in health_results.items():
            if health.response_time > self.alert_thresholds["response_time"]:
                self._create_alert(
                    "high_response_time",
                    f"Компонент {name} отвечает медленно: {health.response_time:.2f}с",
                    {"component": name, "response_time": health.response_time}
                )
        
        # Проверка статуса компонентов
        overall_health = self.health_checker.get_overall_health()
        if overall_health["status"] in ["critical", "degraded"]:
            self._create_alert(
                "system_health",
                f"Система в состоянии: {overall_health['status']}",
                overall_health
            )
        
        # Проверка системных ресурсов
        if self.metrics.performance_metrics["cpu_usage"]:
            latest_cpu = self.metrics.performance_metrics["cpu_usage"][-1]["value"]
            if latest_cpu > self.alert_thresholds["cpu_usage"]:
                self._create_alert(
                    "high_cpu",
                    f"Высокая загрузка CPU: {latest_cpu:.1f}%",
                    {"cpu_usage": latest_cpu}
                )
        
        if self.metrics.performance_metrics["memory_usage"]:
            latest_memory = self.metrics.performance_metrics["memory_usage"][-1]["percent"]
            if latest_memory > self.alert_thresholds["memory_usage"]:
                self._create_alert(
                    "high_memory",
                    f"Высокое использование памяти: {latest_memory:.1f}%",
                    {"memory_usage": latest_memory}
                )
    
    def _create_alert(self, alert_type: str, message: str, details: Dict[str, Any]) -> None:
        """Создание алерта."""
        alert = {
            "type": alert_type,
            "message": message,
            "details": details,
            "timestamp": time.time(),
            "severity": self._get_alert_severity(alert_type)
        }
        
        self.alerts_history.append(alert)
        
        # Ограничиваем размер истории
        if len(self.alerts_history) > self.max_alerts_history:
            self.alerts_history = self.alerts_history[-self.max_alerts_history//2:]
        
        # Логируем алерт
        severity = alert["severity"]
        if severity == "critical":
            self.logger.error(f"ALERT: {message}")
        elif severity == "warning":
            self.logger.warning(f"ALERT: {message}")
        else:
            self.logger.info(f"ALERT: {message}")
        
        # Можно добавить отправку уведомлений
        self._send_alert_notification(alert)
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Определение уровня серьезности алерта."""
        severity_map = {
            "high_response_time": "warning",
            "system_health": "critical",
            "high_cpu": "warning",
            "high_memory": "critical",
            "component_failure": "critical"
        }
        return severity_map.get(alert_type, "info")
    
    def _send_alert_notification(self, alert: Dict[str, Any]) -> None:
        """Отправка уведомления об алерте."""
        # В реальной реализации здесь была бы отправка уведомлений
        # (email, Slack, webhook и т.д.)
        pass
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Получение отчета мониторинга."""
        return {
            "monitoring_active": self.monitoring_active,
            "metrics_summary": self.metrics.get_summary(),
            "health_status": self.health_checker.get_overall_health(),
            "recent_alerts": self.alerts_history[-10:],  # Последние 10 алертов
            "alert_thresholds": self.alert_thresholds.copy()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Получение данных для дашборда."""
        metrics_summary = self.metrics.get_summary()
        health_status = self.health_checker.get_overall_health()
        
        return {
            "status": {
                "overall": health_status["status"],
                "uptime": metrics_summary["uptime_seconds"],
                "last_check": health_status.get("last_check", 0)
            },
            "performance": {
                "requests_per_minute": metrics_summary["requests"]["requests_per_minute"],
                "success_rate": metrics_summary["requests"]["success_rate"],
                "average_response_time": metrics_summary["requests"]["average_response_time"],
                "cpu_usage": metrics_summary["system"]["cpu_percent"],
                "memory_usage": metrics_summary["system"]["memory_percent"]
            },
            "components": {
                name: {
                    "status": details["status"],
                    "response_time": details["response_time"]
                }
                for name, details in health_status.get("details", {}).items()
            },
            "alerts": {
                "total": len(self.alerts_history),
                "recent": len([
                    alert for alert in self.alerts_history
                    if time.time() - alert["timestamp"] < 3600  # За последний час
                ])
            }
        }