#!/usr/bin/env python3
"""
Health Monitor
==============

Advanced health monitoring system for MCP servers.
Provides real-time health checking, performance monitoring,
alerting, and automated recovery capabilities.

Integrates with Session A foundation for caching, validation, and metrics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import json
import statistics
from collections import deque, defaultdict

import aiohttp
from urllib.parse import urljoin

# Session A Foundation imports
from shared.utils.caching import cache_health
from shared.utils.metrics import get_metrics_manager, track_performance
from shared.utils.validation import sanitize_input
from shared.config.settings import ConfigManager
from shared.base.error_handling import BaseException, ErrorCode

# Registry imports
from .registry import ServerRegistry, ServerInfo, ServerStatus


class HealthLevel(str, Enum):
    """Health level enumeration."""
    CRITICAL = "critical"
    WARNING = "warning"
    HEALTHY = "healthy"
    UNKNOWN = "unknown"


class MetricType(str, Enum):
    """Health metric type enumeration."""
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    CONNECTION_COUNT = "connection_count"
    QUEUE_SIZE = "queue_size"


@dataclass
class HealthMetric:
    """Health metric data class."""
    metric_type: MetricType
    value: float
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    @property
    def health_level(self) -> HealthLevel:
        """Determine health level based on thresholds."""
        if self.threshold_critical is not None and self.value >= self.threshold_critical:
            return HealthLevel.CRITICAL
        elif self.threshold_warning is not None and self.value >= self.threshold_warning:
            return HealthLevel.WARNING
        else:
            return HealthLevel.HEALTHY


@dataclass
class HealthCheck:
    """Health check result data class."""
    server_id: str
    timestamp: datetime
    status: ServerStatus
    health_level: HealthLevel
    response_time_ms: float
    metrics: List[HealthMetric]
    error_message: Optional[str] = None
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "server_id": self.server_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "health_level": self.health_level.value,
            "response_time_ms": self.response_time_ms,
            "metrics": [asdict(m) for m in self.metrics],
            "error_message": self.error_message,
            "additional_data": self.additional_data
        }


class HealthMonitor:
    """
    Advanced health monitoring system for MCP servers.
    
    Provides comprehensive health checking, performance monitoring,
    alerting, and automated recovery capabilities. Monitors multiple
    health metrics and provides real-time status reporting.
    """
    
    def __init__(self, registry: ServerRegistry, config_manager: ConfigManager = None):
        self.logger = logging.getLogger("HealthMonitor")
        self.metrics = get_metrics_manager()
        self.config = config_manager or ConfigManager()
        self.registry = registry
        
        # Health monitoring configuration
        self.check_interval = 15  # seconds
        self.detailed_check_interval = 60  # seconds for detailed metrics
        self.timeout = 10  # seconds
        self.history_size = 1000  # number of health checks to keep
        
        # Health thresholds
        self.default_thresholds = {
            MetricType.RESPONSE_TIME: {"warning": 1000, "critical": 5000},  # ms
            MetricType.ERROR_RATE: {"warning": 5.0, "critical": 10.0},  # percentage
            MetricType.CPU_USAGE: {"warning": 80.0, "critical": 95.0},  # percentage
            MetricType.MEMORY_USAGE: {"warning": 85.0, "critical": 95.0},  # percentage
            MetricType.DISK_USAGE: {"warning": 90.0, "critical": 98.0},  # percentage
        }
        
        # State
        self._health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_size))
        self._metric_history: Dict[str, Dict[MetricType, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        self._alert_handlers: List[Callable] = []
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._detailed_monitor_task: Optional[asyncio.Task] = None
        
        # HTTP session
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        # State
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the health monitor."""
        if self._initialized:
            return
        
        self.logger.info("Initializing health monitor...")
        
        # Initialize HTTP session
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._http_session = aiohttp.ClientSession(timeout=timeout)
        
        # Start monitoring tasks
        await self._start_monitoring()
        
        self._initialized = True
        self.logger.info("Health monitor initialized")
    
    async def cleanup(self) -> None:
        """Cleanup the health monitor."""
        self.logger.info("Cleaning up health monitor...")
        
        # Stop monitoring tasks
        await self._stop_monitoring()
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        
        # Clear data
        self._health_history.clear()
        self._metric_history.clear()
        self._alert_handlers.clear()
        
        self._initialized = False
        self.logger.info("Health monitor cleanup complete")
    
    def add_alert_handler(self, handler: Callable[[HealthCheck], None]) -> None:
        """
        Add an alert handler function.
        
        Args:
            handler: Async function that handles health alerts
        """
        self._alert_handlers.append(handler)
        self.logger.info(f"Added alert handler: {handler.__name__}")
    
    @track_performance(tags={"component": "health_monitor", "operation": "check_server_health"})
    async def check_server_health(self, server_id: str, detailed: bool = False) -> Optional[HealthCheck]:
        """
        Perform health check on a specific server.
        
        Args:
            server_id: ID of the server to check
            detailed: Whether to perform detailed health checks
            
        Returns:
            Health check result or None if server not found
        """
        try:
            # Get server info from registry
            server_info_dict = await self.registry.get_server(server_id)
            if not server_info_dict:
                self.logger.warning(f"Server {server_id} not found in registry")
                return None
            
            start_time = datetime.utcnow()
            
            # Perform basic health check
            health_check = await self._perform_basic_health_check(server_id, server_info_dict)
            
            # Perform detailed checks if requested and server is responsive
            if detailed and health_check.status != ServerStatus.OFFLINE:
                detailed_metrics = await self._perform_detailed_health_check(server_id, server_info_dict)
                health_check.metrics.extend(detailed_metrics)
            
            # Calculate overall health level
            health_check.health_level = self._calculate_overall_health_level(health_check.metrics)
            
            # Store in history
            self._health_history[server_id].append(health_check)
            
            # Update metric history
            for metric in health_check.metrics:
                self._metric_history[server_id][metric.metric_type].append(
                    (metric.timestamp, metric.value)
                )
            
            # Check for alerts
            await self._check_and_trigger_alerts(health_check)
            
            # Update metrics
            self.metrics.counter("health_monitor.checks.performed").increment()
            self.metrics.histogram("health_monitor.response_time").update(health_check.response_time_ms)
            
            return health_check
            
        except Exception as e:
            self.logger.error(f"Health check failed for server {server_id}: {str(e)}")
            self.metrics.counter("health_monitor.checks.errors").increment()
            
            # Create error health check
            return HealthCheck(
                server_id=server_id,
                timestamp=datetime.utcnow(),
                status=ServerStatus.UNKNOWN,
                health_level=HealthLevel.CRITICAL,
                response_time_ms=float('inf'),
                metrics=[],
                error_message=str(e)
            )
    
    @cache_health(ttl=30)
    async def get_server_health_status(self, server_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current health status for a server.
        
        Args:
            server_id: ID of the server
            
        Returns:
            Current health status or None if not found
        """
        if server_id not in self._health_history:
            return None
        
        if not self._health_history[server_id]:
            return None
        
        latest_check = self._health_history[server_id][-1]
        return latest_check.to_dict()
    
    async def get_server_health_history(
        self,
        server_id: str,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get health history for a server.
        
        Args:
            server_id: ID of the server
            limit: Maximum number of records to return
            since: Only return records after this timestamp
            
        Returns:
            List of health check records
        """
        if server_id not in self._health_history:
            return []
        
        history = list(self._health_history[server_id])
        
        # Filter by timestamp if provided
        if since:
            history = [check for check in history if check.timestamp > since]
        
        # Apply limit
        history = history[-limit:]
        
        return [check.to_dict() for check in history]
    
    async def get_health_metrics_summary(self, server_id: str) -> Dict[str, Any]:
        """
        Get aggregated health metrics for a server.
        
        Args:
            server_id: ID of the server
            
        Returns:
            Aggregated health metrics and statistics
        """
        if server_id not in self._metric_history:
            return {}
        
        summary = {}
        
        for metric_type, metric_data in self._metric_history[server_id].items():
            if not metric_data:
                continue
            
            # Extract values (ignore timestamps for aggregation)
            values = [value for _, value in metric_data]
            
            if values:
                summary[metric_type.value] = {
                    "current": values[-1],
                    "average": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
                
                # Add percentiles for certain metrics
                if len(values) >= 10:
                    sorted_values = sorted(values)
                    summary[metric_type.value]["p50"] = sorted_values[len(sorted_values) // 2]
                    summary[metric_type.value]["p90"] = sorted_values[int(len(sorted_values) * 0.9)]
                    summary[metric_type.value]["p99"] = sorted_values[int(len(sorted_values) * 0.99)]
        
        return summary
    
    async def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall health status across all monitored servers."""
        all_servers = await self.registry.discover_servers(healthy_only=False)
        
        total_servers = len(all_servers)
        healthy_count = 0
        warning_count = 0
        critical_count = 0
        unknown_count = 0
        
        server_statuses = {}
        
        for server in all_servers:
            server_id = server["server_id"]
            latest_health = await self.get_server_health_status(server_id)
            
            if latest_health:
                health_level = latest_health["health_level"]
                
                if health_level == HealthLevel.HEALTHY.value:
                    healthy_count += 1
                elif health_level == HealthLevel.WARNING.value:
                    warning_count += 1
                elif health_level == HealthLevel.CRITICAL.value:
                    critical_count += 1
                else:
                    unknown_count += 1
                
                server_statuses[server_id] = {
                    "name": server["name"],
                    "type": server["server_type"],
                    "status": latest_health["status"],
                    "health_level": health_level,
                    "last_check": latest_health["timestamp"]
                }
            else:
                unknown_count += 1
                server_statuses[server_id] = {
                    "name": server["name"],
                    "type": server["server_type"],
                    "status": "unknown",
                    "health_level": "unknown",
                    "last_check": None
                }
        
        # Determine overall health
        if critical_count > 0:
            overall_health = HealthLevel.CRITICAL
        elif warning_count > 0:
            overall_health = HealthLevel.WARNING
        elif healthy_count == total_servers:
            overall_health = HealthLevel.HEALTHY
        else:
            overall_health = HealthLevel.UNKNOWN
        
        return {
            "overall_health": overall_health.value,
            "total_servers": total_servers,
            "healthy": healthy_count,
            "warning": warning_count,
            "critical": critical_count,
            "unknown": unknown_count,
            "health_percentage": (healthy_count / total_servers * 100) if total_servers > 0 else 0,
            "server_statuses": server_statuses,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _perform_basic_health_check(self, server_id: str, server_info: Dict[str, Any]) -> HealthCheck:
        """Perform basic health check (response time and availability)."""
        start_time = datetime.utcnow()
        
        try:
            # Build health check URL
            base_url = f"{server_info['protocol']}://{server_info['host']}:{server_info['port']}"
            health_url = urljoin(base_url, "/health")
            
            # Perform HTTP request
            request_start = datetime.utcnow()
            
            if self._http_session:
                async with self._http_session.get(health_url) as response:
                    response_time = (datetime.utcnow() - request_start).total_seconds() * 1000
                    
                    if response.status == 200:
                        status = ServerStatus.HEALTHY
                        
                        # Try to parse response for additional metrics
                        try:
                            response_data = await response.json()
                            additional_data = response_data
                        except:
                            additional_data = {}
                    else:
                        status = ServerStatus.UNHEALTHY
                        additional_data = {"http_status": response.status}
            else:
                raise Exception("HTTP session not available")
        
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            status = ServerStatus.OFFLINE
            additional_data = {"error": str(e)}
        
        # Create basic metrics
        metrics = [
            HealthMetric(
                metric_type=MetricType.RESPONSE_TIME,
                value=response_time,
                threshold_warning=self.default_thresholds[MetricType.RESPONSE_TIME]["warning"],
                threshold_critical=self.default_thresholds[MetricType.RESPONSE_TIME]["critical"],
                unit="ms"
            )
        ]
        
        return HealthCheck(
            server_id=server_id,
            timestamp=start_time,
            status=status,
            health_level=HealthLevel.UNKNOWN,  # Will be calculated later
            response_time_ms=response_time,
            metrics=metrics,
            additional_data=additional_data
        )
    
    async def _perform_detailed_health_check(self, server_id: str, server_info: Dict[str, Any]) -> List[HealthMetric]:
        """Perform detailed health check to gather system metrics."""
        detailed_metrics = []
        
        try:
            # Build metrics endpoint URL
            base_url = f"{server_info['protocol']}://{server_info['host']}:{server_info['port']}"
            metrics_url = urljoin(base_url, "/metrics")
            
            if self._http_session:
                async with self._http_session.get(metrics_url) as response:
                    if response.status == 200:
                        try:
                            metrics_data = await response.json()
                            
                            # Parse system metrics
                            if "system" in metrics_data:
                                system_metrics = metrics_data["system"]
                                
                                # CPU usage
                                if "cpu_usage" in system_metrics:
                                    detailed_metrics.append(HealthMetric(
                                        metric_type=MetricType.CPU_USAGE,
                                        value=float(system_metrics["cpu_usage"]),
                                        threshold_warning=self.default_thresholds[MetricType.CPU_USAGE]["warning"],
                                        threshold_critical=self.default_thresholds[MetricType.CPU_USAGE]["critical"],
                                        unit="%"
                                    ))
                                
                                # Memory usage
                                if "memory_usage" in system_metrics:
                                    detailed_metrics.append(HealthMetric(
                                        metric_type=MetricType.MEMORY_USAGE,
                                        value=float(system_metrics["memory_usage"]),
                                        threshold_warning=self.default_thresholds[MetricType.MEMORY_USAGE]["warning"],
                                        threshold_critical=self.default_thresholds[MetricType.MEMORY_USAGE]["critical"],
                                        unit="%"
                                    ))
                                
                                # Disk usage
                                if "disk_usage" in system_metrics:
                                    detailed_metrics.append(HealthMetric(
                                        metric_type=MetricType.DISK_USAGE,
                                        value=float(system_metrics["disk_usage"]),
                                        threshold_warning=self.default_thresholds[MetricType.DISK_USAGE]["warning"],
                                        threshold_critical=self.default_thresholds[MetricType.DISK_USAGE]["critical"],
                                        unit="%"
                                    ))
                            
                            # Parse application metrics
                            if "application" in metrics_data:
                                app_metrics = metrics_data["application"]
                                
                                # Error rate
                                if "error_rate" in app_metrics:
                                    detailed_metrics.append(HealthMetric(
                                        metric_type=MetricType.ERROR_RATE,
                                        value=float(app_metrics["error_rate"]),
                                        threshold_warning=self.default_thresholds[MetricType.ERROR_RATE]["warning"],
                                        threshold_critical=self.default_thresholds[MetricType.ERROR_RATE]["critical"],
                                        unit="%"
                                    ))
                                
                                # Throughput
                                if "throughput" in app_metrics:
                                    detailed_metrics.append(HealthMetric(
                                        metric_type=MetricType.THROUGHPUT,
                                        value=float(app_metrics["throughput"]),
                                        unit="req/sec"
                                    ))
                                
                                # Connection count
                                if "connection_count" in app_metrics:
                                    detailed_metrics.append(HealthMetric(
                                        metric_type=MetricType.CONNECTION_COUNT,
                                        value=float(app_metrics["connection_count"]),
                                        unit="connections"
                                    ))
                                
                                # Queue size
                                if "queue_size" in app_metrics:
                                    detailed_metrics.append(HealthMetric(
                                        metric_type=MetricType.QUEUE_SIZE,
                                        value=float(app_metrics["queue_size"]),
                                        unit="items"
                                    ))
                        
                        except (json.JSONDecodeError, ValueError, KeyError) as e:
                            self.logger.warning(f"Failed to parse metrics for server {server_id}: {str(e)}")
                    
                    else:
                        self.logger.warning(f"Metrics endpoint returned status {response.status} for server {server_id}")
        
        except Exception as e:
            self.logger.warning(f"Failed to fetch detailed metrics for server {server_id}: {str(e)}")
        
        return detailed_metrics
    
    def _calculate_overall_health_level(self, metrics: List[HealthMetric]) -> HealthLevel:
        """Calculate overall health level based on individual metrics."""
        if not metrics:
            return HealthLevel.UNKNOWN
        
        health_levels = [metric.health_level for metric in metrics]
        
        # If any metric is critical, overall is critical
        if HealthLevel.CRITICAL in health_levels:
            return HealthLevel.CRITICAL
        
        # If any metric is warning, overall is warning
        if HealthLevel.WARNING in health_levels:
            return HealthLevel.WARNING
        
        # If all metrics are healthy, overall is healthy
        if all(level == HealthLevel.HEALTHY for level in health_levels):
            return HealthLevel.HEALTHY
        
        # Otherwise unknown
        return HealthLevel.UNKNOWN
    
    async def _check_and_trigger_alerts(self, health_check: HealthCheck) -> None:
        """Check for alert conditions and trigger alerts."""
        # Check if health level has deteriorated
        previous_checks = list(self._health_history[health_check.server_id])
        
        if len(previous_checks) > 1:
            previous_check = previous_checks[-2]  # Second to last (before current)
            
            # Alert if health deteriorated to warning or critical
            if (previous_check.health_level in [HealthLevel.HEALTHY, HealthLevel.UNKNOWN] and
                health_check.health_level in [HealthLevel.WARNING, HealthLevel.CRITICAL]):
                
                await self._trigger_alerts(health_check, "health_deteriorated")
            
            # Alert if server went offline
            if (previous_check.status != ServerStatus.OFFLINE and
                health_check.status == ServerStatus.OFFLINE):
                
                await self._trigger_alerts(health_check, "server_offline")
    
    async def _trigger_alerts(self, health_check: HealthCheck, alert_type: str) -> None:
        """Trigger alerts by calling registered alert handlers."""
        self.logger.warning(f"Triggering {alert_type} alert for server {health_check.server_id}")
        
        # Update metrics
        self.metrics.counter(f"health_monitor.alerts.{alert_type}").increment()
        
        # Call alert handlers
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(health_check)
                else:
                    handler(health_check)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {str(e)}")
    
    async def _start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        # Basic health check task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        # Detailed health check task
        self._detailed_monitor_task = asyncio.create_task(self._detailed_monitor_loop())
        
        self.logger.info("Health monitoring tasks started")
    
    async def _stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        tasks = [self._monitor_task, self._detailed_monitor_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Health monitoring tasks stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop for basic health checks."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Get all registered servers
                servers = await self.registry.discover_servers(healthy_only=False)
                
                # Check each server
                check_tasks = []
                for server in servers:
                    task = asyncio.create_task(
                        self.check_server_health(server["server_id"], detailed=False)
                    )
                    check_tasks.append(task)
                
                # Wait for all checks to complete
                if check_tasks:
                    await asyncio.gather(*check_tasks, return_exceptions=True)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitor loop error: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _detailed_monitor_loop(self) -> None:
        """Detailed monitoring loop for comprehensive health checks."""
        while True:
            try:
                await asyncio.sleep(self.detailed_check_interval)
                
                # Get healthy servers for detailed monitoring
                servers = await self.registry.discover_servers(healthy_only=True)
                
                # Perform detailed checks on a subset of servers to avoid overload
                max_detailed_checks = min(len(servers), 5)  # Limit to 5 servers per cycle
                
                for i, server in enumerate(servers[:max_detailed_checks]):
                    await self.check_server_health(server["server_id"], detailed=True)
                    
                    # Small delay between detailed checks
                    if i < max_detailed_checks - 1:
                        await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Detailed monitor loop error: {str(e)}")
                await asyncio.sleep(10)  # Longer pause for detailed checks 