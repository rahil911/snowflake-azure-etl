"""
Metrics and performance monitoring for the multi-agent data intelligence platform.

This module provides comprehensive metrics collection, performance monitoring,
and observability tools for tracking system health and performance.
"""

import asyncio
import logging
import psutil
import time
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Union, Callable, Deque
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ConfigDict


@dataclass
class MetricSample:
    """A single metric measurement."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


class MetricType:
    """Metric type constants."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class Counter:
    """A counter metric that only increases."""
    
    def __init__(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._value = 0
        self._created_at = datetime.utcnow()
    
    def increment(self, amount: int = 1) -> None:
        """Increment the counter."""
        if amount < 0:
            raise ValueError("Counter can only be incremented by positive values")
        self._value += amount
    
    def get_value(self) -> int:
        """Get current counter value."""
        return self._value
    
    def reset(self) -> None:
        """Reset counter to zero."""
        self._value = 0
    
    def to_sample(self) -> MetricSample:
        """Convert to metric sample."""
        return MetricSample(
            name=self.name,
            value=self._value,
            timestamp=datetime.utcnow(),
            tags=self.tags
        )


class Gauge:
    """A gauge metric that can go up and down."""
    
    def __init__(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._value = 0.0
        self._created_at = datetime.utcnow()
    
    def set(self, value: Union[int, float]) -> None:
        """Set the gauge value."""
        self._value = float(value)
    
    def increment(self, amount: Union[int, float] = 1) -> None:
        """Increment the gauge."""
        self._value += amount
    
    def decrement(self, amount: Union[int, float] = 1) -> None:
        """Decrement the gauge."""
        self._value -= amount
    
    def get_value(self) -> float:
        """Get current gauge value."""
        return self._value
    
    def to_sample(self) -> MetricSample:
        """Convert to metric sample."""
        return MetricSample(
            name=self.name,
            value=self._value,
            timestamp=datetime.utcnow(),
            tags=self.tags
        )


class Histogram:
    """A histogram metric for tracking value distributions."""
    
    def __init__(self, name: str, description: str = "", buckets: Optional[List[float]] = None, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float('inf')]
        self._bucket_counts = {bucket: 0 for bucket in self.buckets}
        self._sum = 0.0
        self._count = 0
        self._created_at = datetime.utcnow()
    
    def observe(self, value: Union[int, float]) -> None:
        """Record an observation."""
        value = float(value)
        self._sum += value
        self._count += 1
        
        # Increment bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[bucket] += 1
    
    def get_count(self) -> int:
        """Get total number of observations."""
        return self._count
    
    def get_sum(self) -> float:
        """Get sum of all observations."""
        return self._sum
    
    def get_mean(self) -> float:
        """Get mean of all observations."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count
    
    def get_buckets(self) -> Dict[float, int]:
        """Get bucket counts."""
        return self._bucket_counts.copy()
    
    def to_samples(self) -> List[MetricSample]:
        """Convert to metric samples."""
        samples = []
        timestamp = datetime.utcnow()
        
        # Count sample
        samples.append(MetricSample(
            name=f"{self.name}_count",
            value=self._count,
            timestamp=timestamp,
            tags=self.tags
        ))
        
        # Sum sample
        samples.append(MetricSample(
            name=f"{self.name}_sum",
            value=self._sum,
            timestamp=timestamp,
            tags=self.tags
        ))
        
        # Bucket samples
        for bucket, count in self._bucket_counts.items():
            bucket_tags = self.tags.copy()
            bucket_tags['le'] = str(bucket)
            samples.append(MetricSample(
                name=f"{self.name}_bucket",
                value=count,
                timestamp=timestamp,
                tags=bucket_tags
            ))
        
        return samples


class Timer:
    """A timer metric for measuring durations."""
    
    def __init__(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._histogram = Histogram(f"{name}_duration_seconds", description, tags=tags)
        self._active_timers: Dict[str, float] = {}
    
    def start(self, timer_id: Optional[str] = None) -> str:
        """Start a timer and return timer ID."""
        timer_id = timer_id or f"timer_{time.time()}"
        self._active_timers[timer_id] = time.time()
        return timer_id
    
    def stop(self, timer_id: str) -> float:
        """Stop a timer and return duration."""
        if timer_id not in self._active_timers:
            raise ValueError(f"Timer {timer_id} not found")
        
        start_time = self._active_timers.pop(timer_id)
        duration = time.time() - start_time
        self._histogram.observe(duration)
        return duration
    
    def time_it(self, func: Callable) -> float:
        """Time a function execution."""
        start_time = time.time()
        try:
            result = func()
            return result
        finally:
            duration = time.time() - start_time
            self._histogram.observe(duration)
    
    @contextmanager
    def time_context(self):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._histogram.observe(duration)
    
    @asynccontextmanager
    async def time_async_context(self):
        """Async context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._histogram.observe(duration)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get timer statistics."""
        return {
            'count': self._histogram.get_count(),
            'sum': self._histogram.get_sum(),
            'mean': self._histogram.get_mean(),
            'active_timers': len(self._active_timers)
        }
    
    def to_samples(self) -> List[MetricSample]:
        """Convert to metric samples."""
        return self._histogram.to_samples()


class MetricsCollector:
    """Central metrics collection and management."""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Metric storage
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._timers: Dict[str, Timer] = {}
        
        # Sample storage
        self._samples: Deque[MetricSample] = deque(maxlen=10000)
        self._aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Collection task
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
    
    def counter(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create a counter metric."""
        if name not in self._counters:
            self._counters[name] = Counter(name, description, tags)
        return self._counters[name]
    
    def gauge(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create a gauge metric."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description, tags)
        return self._gauges[name]
    
    def histogram(self, name: str, description: str = "", buckets: Optional[List[float]] = None, tags: Optional[Dict[str, str]] = None) -> Histogram:
        """Get or create a histogram metric."""
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, description, buckets, tags)
        return self._histograms[name]
    
    def timer(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Timer:
        """Get or create a timer metric."""
        if name not in self._timers:
            self._timers[name] = Timer(name, description, tags)
        return self._timers[name]
    
    def record_sample(self, sample: MetricSample) -> None:
        """Record a metric sample."""
        self._samples.append(sample)
    
    def record_value(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value directly."""
        sample = MetricSample(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {}
        )
        self.record_sample(sample)
    
    async def start_collection(self) -> None:
        """Start periodic metric collection."""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Started metric collection")
    
    async def stop_collection(self) -> None:
        """Stop metric collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped metric collection")
    
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metric collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_metrics(self) -> None:
        """Collect metrics from all sources."""
        # Collect from registered metrics
        for counter in self._counters.values():
            self.record_sample(counter.to_sample())
        
        for gauge in self._gauges.values():
            self.record_sample(gauge.to_sample())
        
        for histogram in self._histograms.values():
            for sample in histogram.to_samples():
                self.record_sample(sample)
        
        for timer in self._timers.values():
            for sample in timer.to_samples():
                self.record_sample(sample)
        
        # Collect system metrics
        system_metrics = get_system_metrics()
        for name, value in system_metrics.items():
            self.record_value(f"system_{name}", value, {"source": "system"})
    
    def get_samples(self, since: Optional[datetime] = None, metric_name: Optional[str] = None) -> List[MetricSample]:
        """Get metric samples with optional filtering."""
        samples = list(self._samples)
        
        if since:
            samples = [s for s in samples if s.timestamp >= since]
        
        if metric_name:
            samples = [s for s in samples if s.name == metric_name]
        
        return samples
    
    def get_metric_names(self) -> List[str]:
        """Get all metric names."""
        names = set()
        names.update(self._counters.keys())
        names.update(self._gauges.keys())
        names.update(self._histograms.keys())
        names.update(self._timers.keys())
        return sorted(names)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            'total_samples': len(self._samples),
            'metric_counts': {
                'counters': len(self._counters),
                'gauges': len(self._gauges),
                'histograms': len(self._histograms),
                'timers': len(self._timers)
            },
            'collection_interval': self.collection_interval,
            'is_running': self._running,
            'metric_names': self.get_metric_names()
        }
    
    def reset_all(self) -> None:
        """Reset all metrics."""
        for counter in self._counters.values():
            counter.reset()
        
        # Clear samples
        self._samples.clear()
        self._aggregated_metrics.clear()


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience functions

def counter(name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Counter:
    """Get or create a counter metric."""
    return get_metrics_collector().counter(name, description, tags)


def gauge(name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Gauge:
    """Get or create a gauge metric."""
    return get_metrics_collector().gauge(name, description, tags)


def histogram(name: str, description: str = "", buckets: Optional[List[float]] = None, tags: Optional[Dict[str, str]] = None) -> Histogram:
    """Get or create a histogram metric."""
    return get_metrics_collector().histogram(name, description, buckets, tags)


def timer(name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Timer:
    """Get or create a timer metric."""
    return get_metrics_collector().timer(name, description, tags)


def record_metric(name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
    """Record a metric value."""
    get_metrics_collector().record_value(name, value, tags)


# Performance decorators

def track_performance(metric_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Decorator to track function performance.
    
    Usage:
        @track_performance()
        def slow_function():
            time.sleep(1)
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        perf_timer = timer(f"{name}_duration", f"Execution time for {name}", tags)
        perf_counter = counter(f"{name}_calls", f"Call count for {name}", tags)
        error_counter = counter(f"{name}_errors", f"Error count for {name}", tags)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            perf_counter.increment()
            async with perf_timer.time_async_context():
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_counter.increment()
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            perf_counter.increment()
            with perf_timer.time_context():
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_counter.increment()
                    raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def measure_time(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Decorator to measure execution time.
    
    Usage:
        @measure_time("api_call_duration")
        def api_call():
            pass
    """
    def decorator(func: Callable) -> Callable:
        perf_timer = timer(metric_name, tags=tags)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with perf_timer.time_async_context():
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with perf_timer.time_context():
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def count_calls(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Decorator to count function calls.
    
    Usage:
        @count_calls("api_calls")
        def api_call():
            pass
    """
    def decorator(func: Callable) -> Callable:
        call_counter = counter(metric_name, tags=tags)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            call_counter.increment()
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# System metrics

def get_system_metrics() -> Dict[str, float]:
    """Get current system metrics."""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        memory_used_gb = memory.used / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        # Network metrics (basic)
        network = psutil.net_io_counters()
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory_percent,
            'memory_available_gb': memory_available_gb,
            'memory_used_gb': memory_used_gb,
            'disk_percent': disk_percent,
            'disk_free_gb': disk_free_gb,
            'network_bytes_sent': network.bytes_sent,
            'network_bytes_recv': network.bytes_recv,
            'network_packets_sent': network.packets_sent,
            'network_packets_recv': network.packets_recv
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error collecting system metrics: {e}")
        return {}


class PerformanceProfiler:
    """Performance profiler for detailed analysis."""
    
    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @contextmanager
    def profile(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Profile an operation."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            profile_data = {
                'operation': operation_name,
                'duration_ms': (end_time - start_time) * 1000,
                'memory_delta_mb': (end_memory - start_memory) / (1024**2),
                'timestamp': datetime.utcnow().isoformat(),
                'tags': tags or {}
            }
            
            self.profiles[operation_name].append(profile_data)
    
    def get_profile_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get profile summary for an operation."""
        profiles = self.profiles.get(operation_name, [])
        if not profiles:
            return {}
        
        durations = [p['duration_ms'] for p in profiles]
        memory_deltas = [p['memory_delta_mb'] for p in profiles]
        
        return {
            'operation': operation_name,
            'call_count': len(profiles),
            'duration_stats': {
                'min_ms': min(durations),
                'max_ms': max(durations),
                'avg_ms': sum(durations) / len(durations),
                'total_ms': sum(durations)
            },
            'memory_stats': {
                'min_delta_mb': min(memory_deltas),
                'max_delta_mb': max(memory_deltas),
                'avg_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'total_delta_mb': sum(memory_deltas)
            }
        }


# Global profiler instance
_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler 