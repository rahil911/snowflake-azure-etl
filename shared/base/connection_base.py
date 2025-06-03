"""
Base connection classes for the multi-agent data intelligence platform.

This module provides foundational classes for managing connections to
databases and external services with pooling, retry logic, and monitoring.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, AsyncContextManager
from contextlib import asynccontextmanager
from dataclasses import dataclass

from pydantic import BaseModel, Field, ConfigDict


class ConnectionStatus(str, Enum):
    """Connection status states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


class ConnectionType(str, Enum):
    """Types of connections."""
    DATABASE = "database"
    WEB_API = "web_api"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    CACHE = "cache"
    CUSTOM = "custom"


@dataclass
class ConnectionMetrics:
    """Connection performance metrics."""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    average_connection_time_ms: float = 0.0
    total_queries: int = 0
    failed_queries: int = 0
    average_query_time_ms: float = 0.0
    last_connection_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    last_error_message: Optional[str] = None


class ConnectionConfig(BaseModel):
    """Base configuration for connections."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='allow'
    )
    
    # Connection identity
    connection_id: str
    connection_type: ConnectionType
    name: str
    description: str = ""
    
    # Connection parameters
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    
    # Behavior configuration
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    connection_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    query_timeout_seconds: float = Field(default=60.0, ge=1.0, le=3600.0)
    
    # Health check configuration
    health_check_interval_seconds: int = Field(default=300, ge=30, le=3600)
    health_check_query: Optional[str] = None
    
    # Additional parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    
    def get_connection_string(self) -> str:
        """Build connection string (override in subclasses)."""
        return f"{self.connection_type}://{self.host}:{self.port}/{self.database}"


class PoolConfig(BaseModel):
    """Configuration for connection pooling."""
    model_config = ConfigDict(extra='forbid')
    
    min_size: int = Field(default=1, ge=0, le=100)
    max_size: int = Field(default=10, ge=1, le=100)
    max_idle_time_seconds: int = Field(default=3600, ge=60, le=86400)
    acquire_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    
    def __post_init__(self):
        if self.min_size > self.max_size:
            raise ValueError("min_size cannot be greater than max_size")


class BaseConnection(ABC):
    """
    Abstract base class for all connections in the multi-agent system.
    
    Provides common functionality for connection management, including:
    - Connection lifecycle management
    - Retry logic and error handling
    - Health monitoring
    - Performance metrics
    """
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection_id = config.connection_id
        self.logger = logging.getLogger(f"Connection[{self.connection_id}]")
        
        # Connection state
        self._status = ConnectionStatus.DISCONNECTED
        self._connection = None
        self._last_health_check = None
        
        # Metrics
        self.metrics = ConnectionMetrics()
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Retry state
        self._retry_count = 0
        self._last_retry_at = None
    
    @property
    def status(self) -> ConnectionStatus:
        """Current connection status."""
        return self._status
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is established and healthy."""
        return self._status == ConnectionStatus.CONNECTED
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy based on recent health check."""
        if not self.is_connected:
            return False
        
        if self._last_health_check is None:
            return True  # No health check configured or performed yet
        
        # Consider healthy if last check was within 2x the interval
        max_age = timedelta(seconds=self.config.health_check_interval_seconds * 2)
        return (datetime.utcnow() - self._last_health_check) <= max_age
    
    async def connect(self) -> bool:
        """Establish connection with retry logic."""
        if self.is_connected:
            return True
        
        self._status = ConnectionStatus.CONNECTING
        start_time = time.time()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(f"Connecting (attempt {attempt + 1}/{self.config.max_retries + 1})")
                
                # Perform the actual connection
                self._connection = await self._connect_implementation()
                
                # Update metrics and status
                connection_time = (time.time() - start_time) * 1000
                self.metrics.total_connections += 1
                self.metrics.active_connections += 1
                self.metrics.last_connection_at = datetime.utcnow()
                
                # Update average connection time
                if self.metrics.average_connection_time_ms == 0:
                    self.metrics.average_connection_time_ms = connection_time
                else:
                    self.metrics.average_connection_time_ms = (
                        (self.metrics.average_connection_time_ms + connection_time) / 2
                    )
                
                self._status = ConnectionStatus.CONNECTED
                self._retry_count = 0
                
                # Start health monitoring
                await self._start_health_monitoring()
                
                self.logger.info(f"Connected successfully in {connection_time:.2f}ms")
                return True
                
            except Exception as e:
                self.metrics.failed_connections += 1
                self.metrics.last_error_at = datetime.utcnow()
                self.metrics.last_error_message = str(e)
                
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    self._status = ConnectionStatus.ERROR
                    self.logger.error("All connection attempts failed")
                    return False
        
        return False
    
    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        if self._status == ConnectionStatus.DISCONNECTED:
            return
        
        self.logger.info("Disconnecting")
        self._status = ConnectionStatus.CLOSING
        
        try:
            # Stop health monitoring
            await self._stop_health_monitoring()
            
            # Close the actual connection
            if self._connection:
                await self._disconnect_implementation()
                self._connection = None
            
            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)
            self._status = ConnectionStatus.DISCONNECTED
            
            self.logger.info("Disconnected successfully")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}", exc_info=True)
            self._status = ConnectionStatus.ERROR
            raise
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """Execute a query with automatic reconnection."""
        if not self.is_connected:
            if not await self.connect():
                raise RuntimeError("Cannot establish connection")
        
        timeout = timeout or self.config.query_timeout_seconds
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self._execute_query_implementation(query, parameters),
                timeout=timeout
            )
            
            # Update metrics
            query_time = (time.time() - start_time) * 1000
            self.metrics.total_queries += 1
            
            if self.metrics.average_query_time_ms == 0:
                self.metrics.average_query_time_ms = query_time
            else:
                self.metrics.average_query_time_ms = (
                    (self.metrics.average_query_time_ms + query_time) / 2
                )
            
            return result
            
        except Exception as e:
            self.metrics.failed_queries += 1
            self.metrics.last_error_at = datetime.utcnow()
            self.metrics.last_error_message = str(e)
            
            # Check if we should attempt reconnection
            if self._should_reconnect(e):
                self.logger.warning("Query failed, attempting reconnection")
                await self._attempt_reconnection()
                
                # Retry the query once after reconnection
                if self.is_connected:
                    return await self._execute_query_implementation(query, parameters)
            
            raise
    
    async def health_check(self) -> bool:
        """Perform health check on the connection."""
        if not self.is_connected:
            return False
        
        try:
            if self.config.health_check_query:
                await self.execute_query(self.config.health_check_query)
            else:
                # Use implementation-specific health check
                await self._health_check_implementation()
            
            self._last_health_check = datetime.utcnow()
            return True
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    async def _start_health_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self.config.health_check_interval_seconds <= 0:
            return
        
        async def health_monitor():
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(self.config.health_check_interval_seconds)
                    
                    if not self._shutdown_event.is_set():
                        healthy = await self.health_check()
                        if not healthy:
                            self.logger.warning("Health check failed, marking connection as unhealthy")
                            
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in health monitor: {e}")
        
        self._health_check_task = asyncio.create_task(health_monitor())
    
    async def _stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self._shutdown_event.set()
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
    
    async def _attempt_reconnection(self) -> None:
        """Attempt to reconnect after connection failure."""
        self._status = ConnectionStatus.RECONNECTING
        self._retry_count += 1
        self._last_retry_at = datetime.utcnow()
        
        try:
            if self._connection:
                await self._disconnect_implementation()
                self._connection = None
            
            await self.connect()
            
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            self._status = ConnectionStatus.ERROR
    
    def _should_reconnect(self, error: Exception) -> bool:
        """Determine if reconnection should be attempted based on error type."""
        # Override in subclasses for specific error handling
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive connection status."""
        return {
            'connection_id': self.connection_id,
            'name': self.config.name,
            'type': self.config.connection_type.value,
            'status': self.status.value,
            'is_healthy': self.is_healthy,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'retry_count': self._retry_count,
            'last_retry_at': self._last_retry_at.isoformat() if self._last_retry_at else None,
            'metrics': {
                'total_connections': self.metrics.total_connections,
                'active_connections': self.metrics.active_connections,
                'failed_connections': self.metrics.failed_connections,
                'connection_success_rate': (
                    self.metrics.total_connections - self.metrics.failed_connections
                ) / max(self.metrics.total_connections, 1),
                'average_connection_time_ms': self.metrics.average_connection_time_ms,
                'total_queries': self.metrics.total_queries,
                'failed_queries': self.metrics.failed_queries,
                'query_success_rate': (
                    self.metrics.total_queries - self.metrics.failed_queries
                ) / max(self.metrics.total_queries, 1),
                'average_query_time_ms': self.metrics.average_query_time_ms,
                'last_connection_at': self.metrics.last_connection_at.isoformat() if self.metrics.last_connection_at else None,
                'last_error_at': self.metrics.last_error_at.isoformat() if self.metrics.last_error_at else None,
                'last_error_message': self.metrics.last_error_message
            }
        }
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions (override in subclasses)."""
        # Default implementation - no transaction support
        yield self._connection
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def _connect_implementation(self) -> Any:
        """Establish the actual connection."""
        pass
    
    @abstractmethod
    async def _disconnect_implementation(self) -> None:
        """Close the actual connection."""
        pass
    
    @abstractmethod
    async def _execute_query_implementation(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute query using the connection."""
        pass
    
    async def _health_check_implementation(self) -> None:
        """Implementation-specific health check (optional override)."""
        pass


class ConnectionPool:
    """
    Connection pool for managing multiple connections of the same type.
    
    Provides connection pooling with automatic scaling, health monitoring,
    and load balancing across connections.
    """
    
    def __init__(
        self,
        connection_factory: Callable[[], BaseConnection],
        pool_config: PoolConfig
    ):
        self.connection_factory = connection_factory
        self.pool_config = pool_config
        self.logger = logging.getLogger("ConnectionPool")
        
        # Pool state
        self._available_connections: asyncio.Queue = asyncio.Queue()
        self._all_connections: List[BaseConnection] = []
        self._connection_semaphore = asyncio.Semaphore(pool_config.max_size)
        
        # Monitoring
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Metrics
        self.total_created = 0
        self.total_destroyed = 0
        self.total_acquired = 0
        self.total_released = 0
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        self.logger.info(f"Initializing connection pool (min: {self.pool_config.min_size}, max: {self.pool_config.max_size})")
        
        # Create minimum connections
        for _ in range(self.pool_config.min_size):
            connection = await self._create_connection()
            if connection:
                await self._available_connections.put(connection)
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_idle_connections())
        
        self.logger.info("Connection pool initialized")
    
    async def cleanup(self) -> None:
        """Cleanup all connections and stop the pool."""
        self.logger.info("Cleaning up connection pool")
        self._shutdown_event.set()
        
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connection in self._all_connections[:]:
            await self._destroy_connection(connection)
        
        self.logger.info("Connection pool cleanup completed")
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None) -> AsyncContextManager[BaseConnection]:
        """Acquire a connection from the pool."""
        timeout = timeout or self.pool_config.acquire_timeout_seconds
        connection = None
        
        try:
            # Wait for semaphore (connection slot)
            await asyncio.wait_for(
                self._connection_semaphore.acquire(),
                timeout=timeout
            )
            
            # Try to get existing connection
            try:
                connection = self._available_connections.get_nowait()
                
                # Check if connection is still healthy
                if not connection.is_healthy:
                    await self._destroy_connection(connection)
                    connection = None
                    
            except asyncio.QueueEmpty:
                connection = None
            
            # Create new connection if needed
            if connection is None:
                connection = await self._create_connection()
                if connection is None:
                    raise RuntimeError("Failed to create connection")
            
            self.total_acquired += 1
            yield connection
            
        finally:
            # Return connection to pool
            if connection and connection.is_connected:
                await self._available_connections.put(connection)
                self.total_released += 1
            elif connection:
                await self._destroy_connection(connection)
            
            # Release semaphore
            self._connection_semaphore.release()
    
    async def _create_connection(self) -> Optional[BaseConnection]:
        """Create and initialize a new connection."""
        try:
            connection = self.connection_factory()
            
            if await connection.connect():
                self._all_connections.append(connection)
                self.total_created += 1
                self.logger.debug(f"Created new connection {connection.connection_id}")
                return connection
            else:
                self.logger.warning("Failed to connect new connection")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating connection: {e}")
            return None
    
    async def _destroy_connection(self, connection: BaseConnection) -> None:
        """Destroy a connection and remove from pool."""
        try:
            await connection.disconnect()
            
            if connection in self._all_connections:
                self._all_connections.remove(connection)
                self.total_destroyed += 1
                self.logger.debug(f"Destroyed connection {connection.connection_id}")
                
        except Exception as e:
            self.logger.error(f"Error destroying connection: {e}")
    
    async def _cleanup_idle_connections(self) -> None:
        """Periodically cleanup idle connections."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self._shutdown_event.is_set():
                    break
                
                current_time = datetime.utcnow()
                idle_threshold = timedelta(seconds=self.pool_config.max_idle_time_seconds)
                
                # Find idle connections
                idle_connections = []
                for connection in self._all_connections:
                    if (
                        not connection.is_connected or
                        (
                            connection.metrics.last_connection_at and
                            (current_time - connection.metrics.last_connection_at) > idle_threshold
                        )
                    ):
                        idle_connections.append(connection)
                
                # Remove excess idle connections (keep minimum)
                connections_to_remove = max(
                    0,
                    len(self._all_connections) - len(idle_connections) - self.pool_config.min_size
                )
                
                for connection in idle_connections[:connections_to_remove]:
                    await self._destroy_connection(connection)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pool status information."""
        return {
            'pool_config': self.pool_config.model_dump(),
            'connections': {
                'total': len(self._all_connections),
                'available': self._available_connections.qsize(),
                'active': len(self._all_connections) - self._available_connections.qsize(),
                'healthy': sum(1 for conn in self._all_connections if conn.is_healthy)
            },
            'metrics': {
                'total_created': self.total_created,
                'total_destroyed': self.total_destroyed,
                'total_acquired': self.total_acquired,
                'total_released': self.total_released,
                'acquire_success_rate': self.total_released / max(self.total_acquired, 1)
            }
        } 