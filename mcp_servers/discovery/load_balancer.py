#!/usr/bin/env python3
"""
Load Balancer
=============

Load balancing system for MCP servers.
Provides intelligent request distribution across healthy servers
with multiple load balancing algorithms and automatic failover.

Integrates with Session A foundation for caching, validation, and metrics.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import hashlib
from collections import defaultdict, deque

import aiohttp
from urllib.parse import urljoin

# Session A Foundation imports
from shared.utils.caching import cache_load_balancer
from shared.utils.metrics import get_metrics_manager, track_performance
from shared.utils.validation import sanitize_input
from shared.config.settings import ConfigManager
from shared.base.error_handling import BaseException, ErrorCode

# Registry and health monitor imports
from .registry import ServerRegistry, ServerInfo, ServerStatus
from .health_monitor import HealthMonitor, HealthLevel


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategy enumeration."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    CONSISTENT_HASH = "consistent_hash"
    IP_HASH = "ip_hash"


class FailoverStrategy(str, Enum):
    """Failover strategy enumeration."""
    IMMEDIATE = "immediate"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRADUAL_RECOVERY = "gradual_recovery"


@dataclass
class ServerWeight:
    """Server weight configuration."""
    server_id: str
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    response_time_avg: float = 0.0
    last_request: Optional[datetime] = None


@dataclass
class LoadBalancerStats:
    """Load balancer statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    requests_per_second: float = 0.0
    server_distributions: Dict[str, int] = None
    
    def __post_init__(self):
        if self.server_distributions is None:
            self.server_distributions = {}


class LoadBalancer:
    """
    Intelligent load balancer for MCP servers.
    
    Provides request distribution across healthy servers using multiple
    load balancing algorithms, automatic failover, and performance optimization.
    Integrates with service registry and health monitoring for dynamic
    server management.
    """
    
    def __init__(
        self,
        registry: ServerRegistry,
        health_monitor: HealthMonitor,
        config_manager: ConfigManager = None
    ):
        self.logger = logging.getLogger("LoadBalancer")
        self.metrics = get_metrics_manager()
        self.config = config_manager or ConfigManager()
        self.registry = registry
        self.health_monitor = health_monitor
        
        # Load balancing configuration
        self.default_strategy = LoadBalancingStrategy.ROUND_ROBIN
        self.failover_strategy = FailoverStrategy.CIRCUIT_BREAKER
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.circuit_breaker_threshold = 5  # failures
        self.circuit_breaker_timeout = 60  # seconds
        
        # Server state tracking
        self._server_weights: Dict[str, ServerWeight] = {}
        self._round_robin_indices: Dict[str, int] = {}  # Per server type
        self._connection_counts: Dict[str, int] = defaultdict(int)
        self._response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Circuit breaker state
        self._circuit_breaker_failures: Dict[str, int] = defaultdict(int)
        self._circuit_breaker_timeouts: Dict[str, datetime] = {}
        
        # Statistics
        self._stats = LoadBalancerStats()
        self._request_history: deque = deque(maxlen=1000)
        
        # HTTP session for requests
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        # Background tasks
        self._stats_task: Optional[asyncio.Task] = None
        
        # State
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the load balancer."""
        if self._initialized:
            return
        
        self.logger.info("Initializing load balancer...")
        
        # Initialize HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        self._http_session = aiohttp.ClientSession(timeout=timeout)
        
        # Start background tasks
        self._stats_task = asyncio.create_task(self._stats_calculation_loop())
        
        self._initialized = True
        self.logger.info("Load balancer initialized")
    
    async def cleanup(self) -> None:
        """Cleanup the load balancer."""
        self.logger.info("Cleaning up load balancer...")
        
        # Stop background tasks
        if self._stats_task and not self._stats_task.done():
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        
        # Clear state
        self._server_weights.clear()
        self._round_robin_indices.clear()
        self._connection_counts.clear()
        self._response_times.clear()
        self._circuit_breaker_failures.clear()
        self._circuit_breaker_timeouts.clear()
        self._request_history.clear()
        
        self._initialized = False
        self.logger.info("Load balancer cleanup complete")
    
    @track_performance(tags={"component": "load_balancer", "operation": "select_server"})
    async def select_server(
        self,
        server_type: str,
        strategy: Optional[LoadBalancingStrategy] = None,
        client_id: Optional[str] = None,
        sticky_session: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Select an optimal server for a request.
        
        Args:
            server_type: Type of server needed
            strategy: Load balancing strategy to use
            client_id: Client identifier for consistent hashing
            sticky_session: Whether to use sticky sessions
            
        Returns:
            Selected server information or None if no servers available
        """
        try:
            strategy = strategy or self.default_strategy
            
            # Get healthy servers of the requested type
            servers = await self.registry.discover_servers(
                server_type=server_type,
                healthy_only=True
            )
            
            if not servers:
                self.logger.warning(f"No healthy servers found for type: {server_type}")
                return None
            
            # Filter out circuit breaker blocked servers
            available_servers = []
            for server in servers:
                server_id = server["server_id"]
                
                # Check circuit breaker
                if await self._is_circuit_breaker_open(server_id):
                    continue
                
                available_servers.append(server)
            
            if not available_servers:
                self.logger.warning(f"All servers for type {server_type} are circuit breaker blocked")
                return None
            
            # Select server based on strategy
            selected_server = await self._select_by_strategy(
                available_servers, strategy, server_type, client_id
            )
            
            if selected_server:
                # Update connection count
                server_id = selected_server["server_id"]
                self._connection_counts[server_id] += 1
                
                # Update server weight tracking
                if server_id not in self._server_weights:
                    self._server_weights[server_id] = ServerWeight(server_id=server_id)
                
                self._server_weights[server_id].current_connections += 1
                self._server_weights[server_id].last_request = datetime.utcnow()
                
                # Update metrics
                self.metrics.counter("load_balancer.server_selections").increment()
                self.metrics.counter(f"load_balancer.selections.{strategy.value}").increment()
                
                self.logger.debug(f"Selected server {server_id} using {strategy.value} strategy")
            
            return selected_server
            
        except Exception as e:
            self.logger.error(f"Server selection failed: {str(e)}")
            self.metrics.counter("load_balancer.selection_errors").increment()
            raise
    
    @track_performance(tags={"component": "load_balancer", "operation": "execute_request"})
    async def execute_request(
        self,
        server_type: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        strategy: Optional[LoadBalancingStrategy] = None,
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a request with automatic load balancing and failover.
        
        Args:
            server_type: Type of server to use
            endpoint: API endpoint to call
            method: HTTP method
            data: Request data
            headers: Request headers
            strategy: Load balancing strategy
            client_id: Client identifier
            
        Returns:
            Response data and metadata
        """
        request_start = datetime.utcnow()
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            try:
                # Select server
                server = await self.select_server(
                    server_type=server_type,
                    strategy=strategy,
                    client_id=client_id
                )
                
                if not server:
                    raise Exception(f"No available servers for type: {server_type}")
                
                server_id = server["server_id"]
                
                # Execute request
                response_data = await self._execute_server_request(
                    server, endpoint, method, data, headers
                )
                
                # Record successful request
                response_time = (datetime.utcnow() - request_start).total_seconds() * 1000
                await self._record_request_success(server_id, response_time)
                
                # Update statistics
                self._stats.total_requests += 1
                self._stats.successful_requests += 1
                
                return {
                    "success": True,
                    "data": response_data,
                    "server_id": server_id,
                    "server_info": server,
                    "response_time_ms": response_time,
                    "attempts": attempt + 1,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                last_error = e
                attempt += 1
                
                # Record failure if we had a server
                if 'server_id' in locals():
                    await self._record_request_failure(server_id)
                
                self.logger.warning(f"Request attempt {attempt} failed: {str(e)}")
                
                # Wait before retry (except on last attempt)
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
        
        # All retries exhausted
        response_time = (datetime.utcnow() - request_start).total_seconds() * 1000
        
        # Update statistics
        self._stats.total_requests += 1
        self._stats.failed_requests += 1
        
        # Update metrics
        self.metrics.counter("load_balancer.requests.failed").increment()
        
        return {
            "success": False,
            "error": str(last_error),
            "response_time_ms": response_time,
            "attempts": attempt,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def release_connection(self, server_id: str) -> None:
        """Release a connection for a server."""
        if server_id in self._connection_counts:
            self._connection_counts[server_id] = max(0, self._connection_counts[server_id] - 1)
        
        if server_id in self._server_weights:
            self._server_weights[server_id].current_connections = max(
                0, self._server_weights[server_id].current_connections - 1
            )
    
    @cache_load_balancer(ttl=10)
    async def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        # Calculate additional stats
        total_requests = self._stats.total_requests
        success_rate = (
            (self._stats.successful_requests / total_requests * 100)
            if total_requests > 0 else 0
        )
        
        # Get server distribution
        server_distributions = {}
        for server_id, weight in self._server_weights.items():
            server_info = await self.registry.get_server(server_id)
            if server_info:
                server_distributions[server_id] = {
                    "name": server_info["name"],
                    "current_connections": weight.current_connections,
                    "requests_handled": self._stats.server_distributions.get(server_id, 0),
                    "avg_response_time": weight.response_time_avg,
                    "last_request": weight.last_request.isoformat() if weight.last_request else None
                }
        
        return {
            "total_requests": total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "success_rate_percentage": success_rate,
            "average_response_time_ms": self._stats.average_response_time,
            "requests_per_second": self._stats.requests_per_second,
            "server_distributions": server_distributions,
            "circuit_breaker_status": {
                server_id: {
                    "failures": self._circuit_breaker_failures[server_id],
                    "is_open": await self._is_circuit_breaker_open(server_id),
                    "timeout_until": self._circuit_breaker_timeouts.get(server_id, datetime.min).isoformat()
                }
                for server_id in self._circuit_breaker_failures.keys()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _select_by_strategy(
        self,
        servers: List[Dict[str, Any]],
        strategy: LoadBalancingStrategy,
        server_type: str,
        client_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Select server based on the specified strategy."""
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin_selection(servers, server_type)
        
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin_selection(servers, server_type)
        
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections_selection(servers)
        
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return await self._least_response_time_selection(servers)
        
        elif strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(servers)
        
        elif strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return await self._weighted_random_selection(servers)
        
        elif strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return await self._consistent_hash_selection(servers, client_id or "default")
        
        elif strategy == LoadBalancingStrategy.IP_HASH:
            return await self._ip_hash_selection(servers, client_id or "default")
        
        else:
            # Default to round robin
            return await self._round_robin_selection(servers, server_type)
    
    async def _round_robin_selection(
        self,
        servers: List[Dict[str, Any]],
        server_type: str
    ) -> Dict[str, Any]:
        """Round robin server selection."""
        if server_type not in self._round_robin_indices:
            self._round_robin_indices[server_type] = 0
        
        index = self._round_robin_indices[server_type]
        selected_server = servers[index % len(servers)]
        
        self._round_robin_indices[server_type] = (index + 1) % len(servers)
        
        return selected_server
    
    async def _weighted_round_robin_selection(
        self,
        servers: List[Dict[str, Any]],
        server_type: str
    ) -> Dict[str, Any]:
        """Weighted round robin server selection."""
        # Create weighted server list
        weighted_servers = []
        for server in servers:
            server_id = server["server_id"]
            weight = self._server_weights.get(server_id, ServerWeight(server_id)).weight
            weighted_servers.extend([server] * weight)
        
        return await self._round_robin_selection(weighted_servers, f"{server_type}_weighted")
    
    async def _least_connections_selection(
        self,
        servers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Least connections server selection."""
        min_connections = float('inf')
        selected_server = None
        
        for server in servers:
            server_id = server["server_id"]
            connections = self._connection_counts.get(server_id, 0)
            
            if connections < min_connections:
                min_connections = connections
                selected_server = server
        
        return selected_server or servers[0]
    
    async def _least_response_time_selection(
        self,
        servers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Least response time server selection."""
        min_response_time = float('inf')
        selected_server = None
        
        for server in servers:
            server_id = server["server_id"]
            
            if server_id in self._response_times and self._response_times[server_id]:
                avg_response_time = sum(self._response_times[server_id]) / len(self._response_times[server_id])
            else:
                avg_response_time = 0  # Prefer servers with no history (new servers)
            
            if avg_response_time < min_response_time:
                min_response_time = avg_response_time
                selected_server = server
        
        return selected_server or servers[0]
    
    async def _weighted_random_selection(
        self,
        servers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Weighted random server selection."""
        weights = []
        for server in servers:
            server_id = server["server_id"]
            weight = self._server_weights.get(server_id, ServerWeight(server_id)).weight
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(servers)
        
        r = random.uniform(0, total_weight)
        upto = 0
        for i, weight in enumerate(weights):
            if upto + weight >= r:
                return servers[i]
            upto += weight
        
        return servers[-1]  # Fallback
    
    async def _consistent_hash_selection(
        self,
        servers: List[Dict[str, Any]],
        key: str
    ) -> Dict[str, Any]:
        """Consistent hash server selection."""
        # Simple consistent hashing implementation
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        server_index = hash_value % len(servers)
        return servers[server_index]
    
    async def _ip_hash_selection(
        self,
        servers: List[Dict[str, Any]],
        client_ip: str
    ) -> Dict[str, Any]:
        """IP hash server selection."""
        return await self._consistent_hash_selection(servers, client_ip)
    
    async def _execute_server_request(
        self,
        server: Dict[str, Any],
        endpoint: str,
        method: str,
        data: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]]
    ) -> Any:
        """Execute request to a specific server."""
        base_url = f"{server['protocol']}://{server['host']}:{server['port']}"
        url = urljoin(base_url, endpoint)
        
        request_headers = headers or {}
        request_headers.setdefault("Content-Type", "application/json")
        
        if not self._http_session:
            raise Exception("HTTP session not available")
        
        async with self._http_session.request(
            method=method,
            url=url,
            json=data if data else None,
            headers=request_headers
        ) as response:
            
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")
            
            try:
                return await response.json()
            except:
                return await response.text()
    
    async def _record_request_success(self, server_id: str, response_time: float) -> None:
        """Record a successful request."""
        # Update response time tracking
        self._response_times[server_id].append(response_time)
        
        # Update server weight
        if server_id in self._server_weights:
            weight = self._server_weights[server_id]
            
            # Calculate running average
            if weight.response_time_avg == 0:
                weight.response_time_avg = response_time
            else:
                weight.response_time_avg = (weight.response_time_avg * 0.9) + (response_time * 0.1)
        
        # Reset circuit breaker failures
        self._circuit_breaker_failures[server_id] = 0
        if server_id in self._circuit_breaker_timeouts:
            del self._circuit_breaker_timeouts[server_id]
        
        # Update server distribution stats
        self._stats.server_distributions[server_id] = (
            self._stats.server_distributions.get(server_id, 0) + 1
        )
        
        # Add to request history
        self._request_history.append({
            "timestamp": datetime.utcnow(),
            "server_id": server_id,
            "response_time": response_time,
            "success": True
        })
    
    async def _record_request_failure(self, server_id: str) -> None:
        """Record a failed request."""
        # Increment circuit breaker failures
        self._circuit_breaker_failures[server_id] += 1
        
        # Open circuit breaker if threshold reached
        if self._circuit_breaker_failures[server_id] >= self.circuit_breaker_threshold:
            timeout_until = datetime.utcnow() + timedelta(seconds=self.circuit_breaker_timeout)
            self._circuit_breaker_timeouts[server_id] = timeout_until
            
            self.logger.warning(f"Circuit breaker opened for server {server_id}")
            self.metrics.counter("load_balancer.circuit_breaker.opened").increment()
        
        # Add to request history
        self._request_history.append({
            "timestamp": datetime.utcnow(),
            "server_id": server_id,
            "response_time": None,
            "success": False
        })
    
    async def _is_circuit_breaker_open(self, server_id: str) -> bool:
        """Check if circuit breaker is open for a server."""
        if server_id not in self._circuit_breaker_timeouts:
            return False
        
        timeout_time = self._circuit_breaker_timeouts[server_id]
        
        if datetime.utcnow() > timeout_time:
            # Circuit breaker timeout expired, close it
            del self._circuit_breaker_timeouts[server_id]
            self._circuit_breaker_failures[server_id] = 0
            
            self.logger.info(f"Circuit breaker closed for server {server_id}")
            self.metrics.counter("load_balancer.circuit_breaker.closed").increment()
            
            return False
        
        return True
    
    async def _stats_calculation_loop(self) -> None:
        """Background task to calculate statistics."""
        while True:
            try:
                await asyncio.sleep(10)  # Calculate every 10 seconds
                
                # Calculate requests per second
                now = datetime.utcnow()
                recent_requests = [
                    req for req in self._request_history
                    if (now - req["timestamp"]).total_seconds() <= 60
                ]
                
                self._stats.requests_per_second = len(recent_requests) / 60.0
                
                # Calculate average response time
                successful_requests = [
                    req for req in recent_requests
                    if req["success"] and req["response_time"] is not None
                ]
                
                if successful_requests:
                    self._stats.average_response_time = sum(
                        req["response_time"] for req in successful_requests
                    ) / len(successful_requests)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stats calculation error: {str(e)}")
                await asyncio.sleep(5) 