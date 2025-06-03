#!/usr/bin/env python3
"""
Server Discovery Registry
==========================

Server discovery and registration system for MCP servers.
Provides service discovery, health monitoring, and load balancing
capabilities for the multi-agent platform.

Integrates with Session A foundation for caching, validation, and metrics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set
from enum import Enum
from dataclasses import dataclass, asdict
import json
import hashlib

import aiohttp
import asyncio
from urllib.parse import urljoin

# Session A Foundation imports
from shared.utils.caching import cache_registry
from shared.utils.metrics import get_metrics_manager, track_performance
from shared.utils.validation import sanitize_input
from shared.config.settings import ConfigManager
from shared.base.error_handling import BaseException, ErrorCode


class ServerStatus(str, Enum):
    """Server status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"


class ServerType(str, Enum):
    """Server type enumeration."""
    SNOWFLAKE = "snowflake"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"
    ETL = "etl"
    CUSTOM = "custom"


@dataclass
class ServerInfo:
    """Server information data class."""
    server_id: str
    name: str
    server_type: ServerType
    host: str
    port: int
    protocol: str = "http"
    version: str = "1.0.0"
    endpoints: List[str] = None
    metadata: Dict[str, Any] = None
    status: ServerStatus = ServerStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    registration_time: Optional[datetime] = None
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = []
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = set()
        if self.registration_time is None:
            self.registration_time = datetime.utcnow()
    
    @property
    def base_url(self) -> str:
        """Get the base URL for the server."""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def health_url(self) -> str:
        """Get the health check URL."""
        return urljoin(self.base_url, "/health")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["tags"] = list(self.tags)
        data["last_health_check"] = self.last_health_check.isoformat() if self.last_health_check else None
        data["registration_time"] = self.registration_time.isoformat() if self.registration_time else None
        return data


class ServerRegistry:
    """
    Server discovery and registration system.
    
    Manages service discovery, health monitoring, and load balancing
    for MCP servers in the multi-agent platform. Provides centralized
    registry with automatic health checking and failover capabilities.
    """
    
    def __init__(self, config_manager: ConfigManager = None):
        self.logger = logging.getLogger("ServerRegistry")
        self.metrics = get_metrics_manager()
        self.config = config_manager or ConfigManager()
        
        # Registry state
        self._servers: Dict[str, ServerInfo] = {}
        self._server_types: Dict[ServerType, List[str]] = {}
        self._tags_index: Dict[str, Set[str]] = {}
        
        # Health monitoring
        self._health_check_interval = 30  # seconds
        self._health_check_timeout = 5  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.max_servers = 100
        self.default_health_check_interval = 30
        self.unhealthy_threshold = 3  # consecutive failures
        
        # State tracking
        self._failure_counts: Dict[str, int] = {}
        self._last_health_checks: Dict[str, datetime] = {}
        
        # HTTP session for health checks
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        # State
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the server registry."""
        if self._initialized:
            return
        
        self.logger.info("Initializing server registry...")
        
        # Initialize HTTP session for health checks
        timeout = aiohttp.ClientTimeout(total=self._health_check_timeout)
        self._http_session = aiohttp.ClientSession(timeout=timeout)
        
        # Initialize server type indexes
        for server_type in ServerType:
            self._server_types[server_type] = []
        
        # Start health monitoring
        await self._start_health_monitoring()
        
        self._initialized = True
        self.logger.info("Server registry initialized")
    
    async def cleanup(self) -> None:
        """Cleanup the server registry."""
        self.logger.info("Cleaning up server registry...")
        
        # Stop health monitoring
        await self._stop_health_monitoring()
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        
        # Clear all data
        self._servers.clear()
        self._server_types.clear()
        self._tags_index.clear()
        self._failure_counts.clear()
        self._last_health_checks.clear()
        
        self._initialized = False
        self.logger.info("Server registry cleanup complete")
    
    @track_performance(tags={"component": "registry", "operation": "register_server"})
    async def register_server(
        self,
        name: str,
        server_type: Union[str, ServerType],
        host: str,
        port: int,
        protocol: str = "http",
        version: str = "1.0.0",
        endpoints: List[str] = None,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> str:
        """
        Register a new server in the registry.
        
        Args:
            name: Human-readable server name
            server_type: Type of server (snowflake, analytics, etc.)
            host: Server host address
            port: Server port number
            protocol: Communication protocol (http, https)
            version: Server version
            endpoints: List of available endpoints
            metadata: Additional server metadata
            tags: Server tags for categorization
            
        Returns:
            Server ID for the registered server
        """
        try:
            self.logger.info(f"Registering server: {name} ({server_type}) at {host}:{port}")
            
            # Validate inputs
            if not name or not host or port <= 0:
                raise ValueError("Invalid server registration parameters")
            
            # Convert server type
            if isinstance(server_type, str):
                try:
                    server_type = ServerType(server_type.lower())
                except ValueError:
                    server_type = ServerType.CUSTOM
            
            # Generate server ID
            server_id = self._generate_server_id(name, host, port)
            
            # Check if server already exists
            if server_id in self._servers:
                self.logger.warning(f"Server {server_id} already registered, updating...")
                return await self._update_server(server_id, name, server_type, host, port, 
                                               protocol, version, endpoints, metadata, tags)
            
            # Check registry capacity
            if len(self._servers) >= self.max_servers:
                raise ValueError(f"Registry capacity exceeded (max: {self.max_servers})")
            
            # Create server info
            server_info = ServerInfo(
                server_id=server_id,
                name=name,
                server_type=server_type,
                host=host,
                port=port,
                protocol=protocol,
                version=version,
                endpoints=endpoints or [],
                metadata=metadata or {},
                tags=set(tags or [])
            )
            
            # Add to registry
            self._servers[server_id] = server_info
            
            # Update indexes
            self._server_types[server_type].append(server_id)
            
            for tag in server_info.tags:
                if tag not in self._tags_index:
                    self._tags_index[tag] = set()
                self._tags_index[tag].add(server_id)
            
            # Initialize health tracking
            self._failure_counts[server_id] = 0
            
            # Perform initial health check
            asyncio.create_task(self._health_check_server(server_id))
            
            # Update metrics
            self.metrics.counter("registry.servers.registered").increment()
            self.metrics.counter(f"registry.servers.{server_type.value}.registered").increment()
            self.metrics.gauge("registry.servers.total").set(len(self._servers))
            
            self.logger.info(f"Server {server_id} registered successfully")
            return server_id
            
        except Exception as e:
            self.logger.error(f"Failed to register server: {str(e)}")
            self.metrics.counter("registry.servers.registration_errors").increment()
            raise
    
    @track_performance(tags={"component": "registry", "operation": "unregister_server"})
    async def unregister_server(self, server_id: str) -> bool:
        """
        Unregister a server from the registry.
        
        Args:
            server_id: ID of the server to unregister
            
        Returns:
            True if server was unregistered, False if not found
        """
        try:
            if server_id not in self._servers:
                self.logger.warning(f"Server {server_id} not found for unregistration")
                return False
            
            server_info = self._servers[server_id]
            
            # Remove from main registry
            del self._servers[server_id]
            
            # Remove from type index
            if server_id in self._server_types[server_info.server_type]:
                self._server_types[server_info.server_type].remove(server_id)
            
            # Remove from tags index
            for tag in server_info.tags:
                if tag in self._tags_index and server_id in self._tags_index[tag]:
                    self._tags_index[tag].remove(server_id)
                    # Clean up empty tag sets
                    if not self._tags_index[tag]:
                        del self._tags_index[tag]
            
            # Clean up health tracking
            self._failure_counts.pop(server_id, None)
            self._last_health_checks.pop(server_id, None)
            
            # Update metrics
            self.metrics.counter("registry.servers.unregistered").increment()
            self.metrics.counter(f"registry.servers.{server_info.server_type.value}.unregistered").increment()
            self.metrics.gauge("registry.servers.total").set(len(self._servers))
            
            self.logger.info(f"Server {server_id} unregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister server {server_id}: {str(e)}")
            self.metrics.counter("registry.servers.unregistration_errors").increment()
            raise
    
    @cache_registry(ttl=30)  # Cache for 30 seconds
    async def discover_servers(
        self,
        server_type: Optional[Union[str, ServerType]] = None,
        tags: Optional[List[str]] = None,
        status: Optional[Union[str, ServerStatus]] = None,
        healthy_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Discover servers matching specified criteria.
        
        Args:
            server_type: Filter by server type
            tags: Filter by tags (AND operation)
            status: Filter by status
            healthy_only: Only return healthy servers
            
        Returns:
            List of matching server information
        """
        try:
            self.logger.debug(f"Discovering servers with filters: type={server_type}, tags={tags}, status={status}")
            
            # Start with all servers
            candidate_servers = set(self._servers.keys())
            
            # Filter by server type
            if server_type:
                if isinstance(server_type, str):
                    try:
                        server_type = ServerType(server_type.lower())
                    except ValueError:
                        server_type = ServerType.CUSTOM
                
                type_servers = set(self._server_types.get(server_type, []))
                candidate_servers &= type_servers
            
            # Filter by tags (AND operation)
            if tags:
                for tag in tags:
                    if tag in self._tags_index:
                        candidate_servers &= self._tags_index[tag]
                    else:
                        # Tag not found, no servers match
                        candidate_servers = set()
                        break
            
            # Filter by status
            if status:
                if isinstance(status, str):
                    try:
                        status = ServerStatus(status.lower())
                    except ValueError:
                        status = None
                
                if status:
                    status_servers = {
                        server_id for server_id, server_info in self._servers.items()
                        if server_info.status == status
                    }
                    candidate_servers &= status_servers
            
            # Filter healthy only
            if healthy_only:
                healthy_servers = {
                    server_id for server_id, server_info in self._servers.items()
                    if server_info.status == ServerStatus.HEALTHY
                }
                candidate_servers &= healthy_servers
            
            # Build result list
            result = []
            for server_id in candidate_servers:
                server_info = self._servers[server_id]
                result.append(server_info.to_dict())
            
            # Sort by registration time (newest first)
            result.sort(key=lambda x: x["registration_time"], reverse=True)
            
            # Update metrics
            self.metrics.counter("registry.discoveries.performed").increment()
            self.metrics.histogram("registry.discoveries.results_count").update(len(result))
            
            self.logger.debug(f"Discovery returned {len(result)} servers")
            return result
            
        except Exception as e:
            self.logger.error(f"Server discovery failed: {str(e)}")
            self.metrics.counter("registry.discoveries.errors").increment()
            raise
    
    async def get_server(self, server_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific server.
        
        Args:
            server_id: ID of the server
            
        Returns:
            Server information or None if not found
        """
        server_info = self._servers.get(server_id)
        return server_info.to_dict() if server_info else None
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics and health information."""
        total_servers = len(self._servers)
        
        # Count servers by status
        status_counts = {}
        for status in ServerStatus:
            count = sum(1 for server in self._servers.values() if server.status == status)
            status_counts[status.value] = count
        
        # Count servers by type
        type_counts = {}
        for server_type in ServerType:
            type_counts[server_type.value] = len(self._server_types[server_type])
        
        # Calculate uptime statistics
        now = datetime.utcnow()
        uptimes = []
        for server in self._servers.values():
            if server.registration_time:
                uptime = (now - server.registration_time).total_seconds()
                uptimes.append(uptime)
        
        avg_uptime = sum(uptimes) / len(uptimes) if uptimes else 0
        
        return {
            "total_servers": total_servers,
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "average_uptime_seconds": avg_uptime,
            "health_check_interval": self._health_check_interval,
            "registry_capacity": self.max_servers,
            "capacity_utilization": (total_servers / self.max_servers) * 100,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_server_id(self, name: str, host: str, port: int) -> str:
        """Generate a unique server ID."""
        data = f"{name}:{host}:{port}:{datetime.utcnow().timestamp()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    async def _update_server(
        self,
        server_id: str,
        name: str,
        server_type: ServerType,
        host: str,
        port: int,
        protocol: str,
        version: str,
        endpoints: List[str],
        metadata: Dict[str, Any],
        tags: List[str]
    ) -> str:
        """Update an existing server registration."""
        server_info = self._servers[server_id]
        
        # Update server info
        server_info.name = name
        server_info.server_type = server_type
        server_info.host = host
        server_info.port = port
        server_info.protocol = protocol
        server_info.version = version
        server_info.endpoints = endpoints or []
        server_info.metadata = metadata or {}
        
        # Update tags
        old_tags = server_info.tags
        new_tags = set(tags or [])
        
        # Remove from old tag indexes
        for tag in old_tags - new_tags:
            if tag in self._tags_index and server_id in self._tags_index[tag]:
                self._tags_index[tag].remove(server_id)
        
        # Add to new tag indexes
        for tag in new_tags - old_tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(server_id)
        
        server_info.tags = new_tags
        
        self.logger.info(f"Server {server_id} updated successfully")
        return server_id
    
    async def _start_health_monitoring(self) -> None:
        """Start the health monitoring background task."""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        self.logger.info("Health monitoring started")
    
    async def _stop_health_monitoring(self) -> None:
        """Stop the health monitoring background task."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _health_monitor_loop(self) -> None:
        """Main health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                # Check all registered servers
                health_check_tasks = []
                for server_id in list(self._servers.keys()):
                    task = asyncio.create_task(self._health_check_server(server_id))
                    health_check_tasks.append(task)
                
                # Wait for all health checks to complete
                if health_check_tasks:
                    await asyncio.gather(*health_check_tasks, return_exceptions=True)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _health_check_server(self, server_id: str) -> None:
        """Perform health check on a specific server."""
        if server_id not in self._servers:
            return
        
        server_info = self._servers[server_id]
        
        try:
            # Skip if server is marked as stopping or offline
            if server_info.status in [ServerStatus.STOPPING, ServerStatus.OFFLINE]:
                return
            
            # Perform HTTP health check
            if self._http_session:
                async with self._http_session.get(server_info.health_url) as response:
                    if response.status == 200:
                        # Health check successful
                        server_info.status = ServerStatus.HEALTHY
                        server_info.last_health_check = datetime.utcnow()
                        self._failure_counts[server_id] = 0
                        
                        self.metrics.counter("registry.health_checks.successful").increment()
                    else:
                        raise aiohttp.ClientError(f"HTTP {response.status}")
            
        except Exception as e:
            # Health check failed
            self._failure_counts[server_id] = self._failure_counts.get(server_id, 0) + 1
            
            if self._failure_counts[server_id] >= self.unhealthy_threshold:
                server_info.status = ServerStatus.UNHEALTHY
            else:
                server_info.status = ServerStatus.UNKNOWN
            
            server_info.last_health_check = datetime.utcnow()
            
            self.metrics.counter("registry.health_checks.failed").increment()
            self.logger.warning(f"Health check failed for server {server_id}: {str(e)}")
        
        self._last_health_checks[server_id] = datetime.utcnow() 