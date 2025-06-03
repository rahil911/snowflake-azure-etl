"""
MCP server connection and management for the coordinator agent.

This module provides connectivity to MCP (Model Context Protocol) servers
from Session B, including the Snowflake and Analytics servers.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable, Protocol
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ConfigDict
import google.genai as genai

from shared.schemas.agent_communication import (
    AgentMessage, AgentRole, MessageType, ConversationContext
)
from shared.config.logging_config import setup_logging
from shared.utils.metrics import get_metrics_collector
from shared.config.settings import get_settings


class ServerStatus(str, Enum):
    """MCP server status."""
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ServerCapability(str, Enum):
    """MCP server capabilities."""
    DATABASE_QUERY = "database_query"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"
    DATA_DISCOVERY = "data_discovery"
    ETL_OPERATIONS = "etl_operations"
    REPORTING = "reporting"


class ConnectionType(str, Enum):
    """Connection types for MCP servers."""
    HTTP = "http"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    STDIO = "stdio"
    UNIX_SOCKET = "unix_socket"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    server_id: str
    name: str
    description: str
    
    # Connection details
    connection_type: ConnectionType
    endpoint: str
    port: Optional[int] = None
    
    # Authentication
    auth_required: bool = False
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Capabilities
    capabilities: Set[ServerCapability] = field(default_factory=set)
    
    # Configuration
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30
    retry_attempts: int = 3
    health_check_interval: int = 60
    
    # Metadata
    version: str = "1.0.0"
    vendor: str = "unknown"
    created_at: datetime = field(default_factory=datetime.utcnow)


class MCPRequest(BaseModel):
    """MCP protocol request."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str = Field(..., description="MCP method name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Method parameters")
    
    # Metadata
    server_id: str = Field(..., description="Target server ID")
    timeout: Optional[int] = Field(default=None, description="Request timeout")
    priority: int = Field(default=0, description="Request priority")
    
    # Context
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MCPResponse(BaseModel):
    """MCP protocol response."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    id: str = Field(..., description="Request ID")
    success: bool = Field(..., description="Success status")
    
    # Response data
    result: Optional[Any] = Field(default=None, description="Success result")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    
    # Metadata
    server_id: str = Field(..., description="Source server ID")
    execution_time_ms: float = Field(..., description="Execution time")
    
    # Timestamps
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class ServerHealthMetrics(BaseModel):
    """Health metrics for an MCP server."""
    server_id: str
    status: ServerStatus
    
    # Performance metrics
    average_response_time_ms: float = 0.0
    success_rate: float = 1.0
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    
    # Resource usage
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    
    # Timestamps
    last_health_check: datetime = Field(default_factory=datetime.utcnow)
    uptime_start: datetime = Field(default_factory=datetime.utcnow)


class MCPServerConnection:
    """
    Individual MCP server connection management.
    
    Handles connection lifecycle, request/response handling,
    and health monitoring for a single MCP server.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.logger = setup_logging(f"MCPServer-{config.server_id}")
        
        # Connection state
        self.status = ServerStatus.DISCONNECTED
        self.connection = None
        self.last_error: Optional[str] = None
        
        # Request tracking
        self.active_requests: Dict[str, MCPRequest] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        
        # Health metrics
        self.health_metrics = ServerHealthMetrics(
            server_id=config.server_id,
            status=self.status
        )
        
        # Background tasks
        self.connection_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.request_processor_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"MCP server connection initialized: {config.name}")
    
    async def connect(self) -> bool:
        """
        Establish connection to the MCP server.
        
        Returns:
            True if connection successful
        """
        try:
            self.status = ServerStatus.CONNECTING
            self.logger.info(f"Connecting to {self.config.name} at {self.config.endpoint}")
            
            # Implementation depends on connection type
            if self.config.connection_type == ConnectionType.HTTP:
                success = await self._connect_http()
            elif self.config.connection_type == ConnectionType.WEBSOCKET:
                success = await self._connect_websocket()
            elif self.config.connection_type == ConnectionType.STDIO:
                success = await self._connect_stdio()
            else:
                self.logger.error(f"Unsupported connection type: {self.config.connection_type}")
                success = False
            
            if success:
                self.status = ServerStatus.CONNECTED
                self.health_metrics.uptime_start = datetime.utcnow()
                
                # Start background tasks
                await self._start_background_tasks()
                
                self.logger.info(f"Successfully connected to {self.config.name}")
            else:
                self.status = ServerStatus.ERROR
                self.logger.error(f"Failed to connect to {self.config.name}")
            
            return success
            
        except Exception as e:
            self.status = ServerStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f"Connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the MCP server.
        
        Returns:
            True if disconnection successful
        """
        try:
            self.logger.info(f"Disconnecting from {self.config.name}")
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Close connection
            if self.connection:
                if hasattr(self.connection, 'close'):
                    await self.connection.close()
                self.connection = None
            
            self.status = ServerStatus.DISCONNECTED
            self.logger.info(f"Disconnected from {self.config.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Disconnection error: {e}")
            return False
    
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """
        Send a request to the MCP server.
        
        Args:
            request: MCP request to send
            
        Returns:
            MCP response from the server
        """
        if self.status != ServerStatus.CONNECTED:
            raise CoordinatorError(f"Server {self.config.server_id} not connected")
        
        start_time = time.time()
        
        try:
            # Add to active requests
            self.active_requests[request.id] = request
            
            # Send request based on connection type
            if self.config.connection_type == ConnectionType.HTTP:
                response = await self._send_http_request(request)
            elif self.config.connection_type == ConnectionType.WEBSOCKET:
                response = await self._send_websocket_request(request)
            elif self.config.connection_type == ConnectionType.STDIO:
                response = await self._send_stdio_request(request)
            else:
                raise CoordinatorError(f"Unsupported connection type: {self.config.connection_type}")
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, True)
            
            # Create response
            return MCPResponse(
                id=request.id,
                success=True,
                result=response,
                server_id=self.config.server_id,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            # Update metrics for failure
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(execution_time, False)
            
            # Create error response
            return MCPResponse(
                id=request.id,
                success=False,
                error={
                    "type": type(e).__name__,
                    "message": str(e),
                    "code": getattr(e, 'code', 'UNKNOWN_ERROR')
                },
                server_id=self.config.server_id,
                execution_time_ms=execution_time
            )
            
        finally:
            # Remove from active requests
            self.active_requests.pop(request.id, None)
    
    async def _connect_http(self) -> bool:
        """Connect via HTTP."""
        # Simulate HTTP connection setup
        # In real implementation, this would create HTTP client session
        import aiohttp
        
        try:
            self.connection = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )
            
            # Test connection with health check
            health_url = f"{self.config.endpoint}/health"
            async with self.connection.get(health_url) as response:
                if response.status == 200:
                    return True
                else:
                    self.logger.warning(f"Health check failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"HTTP connection failed: {e}")
            return False
    
    async def _connect_websocket(self) -> bool:
        """Connect via WebSocket."""
        # Simulate WebSocket connection
        # In real implementation, this would establish WebSocket connection
        try:
            import websockets
            
            self.connection = await websockets.connect(
                self.config.endpoint,
                timeout=self.config.timeout_seconds
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def _connect_stdio(self) -> bool:
        """Connect via STDIO."""
        # Simulate STDIO connection
        # In real implementation, this would spawn subprocess and connect to stdio
        try:
            # This would typically spawn the MCP server process
            # For now, simulate successful connection
            self.connection = {"type": "stdio", "process": None}
            return True
            
        except Exception as e:
            self.logger.error(f"STDIO connection failed: {e}")
            return False
    
    async def _send_http_request(self, request: MCPRequest) -> Any:
        """Send HTTP request."""
        if not self.connection:
            raise CoordinatorError("No HTTP connection available")
        
        try:
            # Construct request URL and payload
            url = f"{self.config.endpoint}/mcp/{request.method}"
            payload = {
                "id": request.id,
                "params": request.params
            }
            
            # Add authentication if required
            headers = {}
            if self.config.auth_required and self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            # Send request
            timeout = request.timeout or self.config.timeout_seconds
            async with self.connection.post(
                url, 
                json=payload, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise CoordinatorError(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            self.logger.error(f"HTTP request failed: {e}")
            raise
    
    async def _send_websocket_request(self, request: MCPRequest) -> Any:
        """Send WebSocket request."""
        if not self.connection:
            raise CoordinatorError("No WebSocket connection available")
        
        try:
            # Construct MCP message
            message = {
                "jsonrpc": "2.0",
                "id": request.id,
                "method": request.method,
                "params": request.params
            }
            
            # Send message
            await self.connection.send(json.dumps(message))
            
            # Wait for response
            timeout = request.timeout or self.config.timeout_seconds
            response_data = await asyncio.wait_for(
                self.connection.recv(), 
                timeout=timeout
            )
            
            response = json.loads(response_data)
            
            if "error" in response:
                raise CoordinatorError(f"MCP error: {response['error']}")
            
            return response.get("result")
            
        except Exception as e:
            self.logger.error(f"WebSocket request failed: {e}")
            raise
    
    async def _send_stdio_request(self, request: MCPRequest) -> Any:
        """Send STDIO request."""
        # Simulate STDIO request
        # In real implementation, this would write to process stdin and read from stdout
        try:
            # Mock response based on method
            if request.method == "database/query":
                return {"rows": [], "columns": [], "execution_time": 0.1}
            elif request.method == "analytics/analyze":
                return {"insights": [], "metrics": {}, "confidence": 0.8}
            else:
                return {"status": "ok", "message": f"Processed {request.method}"}
                
        except Exception as e:
            self.logger.error(f"STDIO request failed: {e}")
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Request processor task
        self.request_processor_task = asyncio.create_task(self._request_processor_loop())
    
    async def _stop_background_tasks(self) -> None:
        """Stop background tasks."""
        tasks = [self.health_check_task, self.request_processor_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    async def _request_processor_loop(self) -> None:
        """Background request processor loop."""
        while True:
            try:
                # This would process queued requests in order
                # For now, just maintain the loop
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Request processor error: {e}")
                await asyncio.sleep(1)
    
    async def _perform_health_check(self) -> None:
        """Perform health check on the server."""
        try:
            # Create health check request
            health_request = MCPRequest(
                method="health/check",
                params={},
                server_id=self.config.server_id,
                timeout=10
            )
            
            # Send health check
            start_time = time.time()
            response = await self.send_request(health_request)
            health_time = (time.time() - start_time) * 1000
            
            if response.success:
                self.status = ServerStatus.CONNECTED
                self.last_error = None
            else:
                self.status = ServerStatus.ERROR
                self.last_error = str(response.error)
            
            # Update health metrics
            self.health_metrics.last_health_check = datetime.utcnow()
            
        except Exception as e:
            self.status = ServerStatus.ERROR
            self.last_error = str(e)
            self.logger.warning(f"Health check failed: {e}")
    
    def _update_metrics(self, execution_time: float, success: bool) -> None:
        """Update performance metrics."""
        # Update response time (exponential moving average)
        alpha = 0.1
        current_avg = self.health_metrics.average_response_time_ms
        self.health_metrics.average_response_time_ms = (
            (1 - alpha) * current_avg + alpha * execution_time
        )
        
        # Update counters
        self.health_metrics.total_requests += 1
        if not success:
            self.health_metrics.failed_requests += 1
        
        # Update success rate
        self.health_metrics.success_rate = (
            (self.health_metrics.total_requests - self.health_metrics.failed_requests) /
            self.health_metrics.total_requests
        )
        
        # Update active connections
        self.health_metrics.active_connections = len(self.active_requests)
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status and metrics."""
        return {
            "server_id": self.config.server_id,
            "name": self.config.name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.config.capabilities],
            "connection_type": self.config.connection_type.value,
            "health_metrics": self.health_metrics.dict(),
            "last_error": self.last_error,
            "active_requests": len(self.active_requests)
        }


class MCPServerConnector:
    """
    Multi-server MCP connection manager.
    
    Manages connections to multiple MCP servers, handles
    load balancing, failover, and provides unified interface.
    """
    
    def __init__(self, settings=None):
        self.logger = setup_logging(self.__class__.__name__)
        self.metrics = get_metrics_collector()
        self.settings = settings or get_settings()
        
        # Server management
        self.servers: Dict[str, MCPServerConnection] = {}
        self.server_configs: Dict[str, MCPServerConfig] = {}
        
        # Capability mapping
        self.servers_by_capability: Dict[ServerCapability, List[str]] = {}
        
        # Load balancing
        self.request_counts: Dict[str, int] = {}
        
        self.logger.info("MCPServerConnector initialized")
    
    async def initialize_from_config(self) -> None:
        """Initialize MCP servers from configuration."""
        # settings instance is already available via self.settings, initialized in __init__
        
        # Snowflake server config from settings
        snowflake_endpoint = self.settings.MCP_SNOWFLAKE_SERVER_ENDPOINT # Assuming this setting exists
        snowflake_api_key_secret = self.settings.MCP_SNOWFLAKE_API_KEY # Assuming this setting exists
        snowflake_api_key = snowflake_api_key_secret.get_secret_value() if snowflake_api_key_secret else None

        snowflake_config = MCPServerConfig(
            server_id="snowflake_server",
            name="Snowflake Data Server",
            description="MCP server for Snowflake database operations",
            connection_type=ConnectionType.HTTP,
            endpoint=snowflake_endpoint,
            capabilities={
                ServerCapability.DATABASE_QUERY,
                ServerCapability.DATA_DISCOVERY,
                ServerCapability.ETL_OPERATIONS
            },
            auth_required=bool(snowflake_api_key),
            api_key=snowflake_api_key,
            timeout_seconds=self.settings.MCP_SERVER_DEFAULT_TIMEOUT_SECONDS, # Assuming a general timeout setting
            max_concurrent_requests=self.settings.MCP_SERVER_DEFAULT_MAX_REQUESTS # Assuming general max requests
        )
        
        # Analytics server config from settings
        analytics_endpoint = self.settings.MCP_ANALYTICS_SERVER_ENDPOINT # Assuming this setting exists
        # Assuming analytics server might also have an API key, add if needed:
        # analytics_api_key_secret = self.settings.MCP_ANALYTICS_API_KEY
        # analytics_api_key = analytics_api_key_secret.get_secret_value() if analytics_api_key_secret else None

        analytics_config = MCPServerConfig(
            server_id="analytics_server",
            name="Analytics Server",
            description="MCP server for data analytics and insights",
            connection_type=ConnectionType.HTTP,
            endpoint=analytics_endpoint,
            capabilities={
                ServerCapability.ANALYTICS,
                ServerCapability.REPORTING,
                ServerCapability.DATA_DISCOVERY
            },
            # auth_required=bool(analytics_api_key), # If analytics server needs auth
            # api_key=analytics_api_key,             # If analytics server needs auth
            timeout_seconds=self.settings.MCP_SERVER_DEFAULT_TIMEOUT_SECONDS,
            max_concurrent_requests=self.settings.MCP_SERVER_DEFAULT_MAX_REQUESTS
        )
        
        # Register servers
        await self.register_server(snowflake_config)
        await self.register_server(analytics_config)
        
        self.logger.info("MCP servers initialized from configuration")
    
    async def register_server(self, config: MCPServerConfig) -> bool:
        """
        Register a new MCP server.
        
        Args:
            config: Server configuration
            
        Returns:
            True if registration successful
        """
        try:
            # Create server connection
            server = MCPServerConnection(config)
            
            # Store server and config
            self.servers[config.server_id] = server
            self.server_configs[config.server_id] = config
            
            # Update capability mapping
            for capability in config.capabilities:
                if capability not in self.servers_by_capability:
                    self.servers_by_capability[capability] = []
                self.servers_by_capability[capability].append(config.server_id)
            
            # Initialize request counter
            self.request_counts[config.server_id] = 0
            
            self.logger.info(f"Registered MCP server: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register server {config.server_id}: {e}")
            return False
    
    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all registered servers.
        
        Returns:
            Dictionary of server_id -> connection_success
        """
        results = {}
        
        # Connect to all servers in parallel
        tasks = []
        for server_id, server in self.servers.items():
            task = asyncio.create_task(server.connect())
            tasks.append((server_id, task))
        
        # Wait for all connections
        for server_id, task in tasks:
            try:
                success = await task
                results[server_id] = success
                
                if success:
                    self.logger.info(f"Connected to server: {server_id}")
                else:
                    self.logger.error(f"Failed to connect to server: {server_id}")
                    
            except Exception as e:
                self.logger.error(f"Connection error for {server_id}: {e}")
                results[server_id] = False
        
        return results
    
    async def disconnect_all(self) -> Dict[str, bool]:
        """
        Disconnect from all servers.
        
        Returns:
            Dictionary of server_id -> disconnection_success
        """
        results = {}
        
        # Disconnect from all servers in parallel
        tasks = []
        for server_id, server in self.servers.items():
            task = asyncio.create_task(server.disconnect())
            tasks.append((server_id, task))
        
        # Wait for all disconnections
        for server_id, task in tasks:
            try:
                success = await task
                results[server_id] = success
            except Exception as e:
                self.logger.error(f"Disconnection error for {server_id}: {e}")
                results[server_id] = False
        
        return results
    
    async def send_request(
        self,
        method: str,
        params: Dict[str, Any],
        capability: Optional[ServerCapability] = None,
        server_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> MCPResponse:
        """
        Send a request to an appropriate MCP server.
        
        Args:
            method: MCP method name
            params: Method parameters
            capability: Required capability (for server selection)
            server_id: Specific server ID (overrides capability selection)
            timeout: Request timeout
            
        Returns:
            Response from the server
        """
        # Select target server
        if server_id:
            if server_id not in self.servers:
                raise ValueError(f"Server {server_id} not found")
            target_server_id = server_id
        elif capability:
            target_server_id = self._select_server_by_capability(capability)
        else:
            # Default to first available server
            available_servers = [
                sid for sid, server in self.servers.items()
                if server.status == ServerStatus.CONNECTED
            ]
            if not available_servers:
                raise ValueError("No servers available")
            target_server_id = available_servers[0]
        
        # Create request
        request = MCPRequest(
            method=method,
            params=params,
            server_id=target_server_id,
            timeout=timeout
        )
        
        # Send request
        server = self.servers[target_server_id]
        response = await server.send_request(request)
        
        # Update request counter
        self.request_counts[target_server_id] += 1
        
        return response
    
    def _select_server_by_capability(self, capability: ServerCapability) -> str:
        """Select the best server for a given capability."""
        if capability not in self.servers_by_capability:
            raise CoordinatorError(f"No servers available for capability: {capability.value}")
        
        candidate_servers = self.servers_by_capability[capability]
        
        # Filter to connected servers
        available_servers = [
            server_id for server_id in candidate_servers
            if (server_id in self.servers and 
                self.servers[server_id].status == ServerStatus.CONNECTED)
        ]
        
        if not available_servers:
            raise CoordinatorError(f"No connected servers for capability: {capability.value}")
        
        # Simple load balancing: select server with fewest requests
        return min(available_servers, key=lambda sid: self.request_counts.get(sid, 0))
    
    def get_server_status(self, server_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of specific server or all servers."""
        if server_id:
            if server_id not in self.servers:
                return None
            return self.servers[server_id].get_status()
        else:
            return [server.get_status() for server in self.servers.values()]
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get available capabilities and supporting servers."""
        return {
            capability.value: servers
            for capability, servers in self.servers_by_capability.items()
        }
    
    def get_connection_metrics(self) -> Dict[str, Any]:
        """Get comprehensive connection metrics."""
        total_servers = len(self.servers)
        connected_servers = len([
            s for s in self.servers.values() 
            if s.status == ServerStatus.CONNECTED
        ])
        
        return {
            "total_servers": total_servers,
            "connected_servers": connected_servers,
            "connection_rate": connected_servers / total_servers if total_servers > 0 else 0,
            "total_requests": sum(self.request_counts.values()),
            "servers_by_capability": {
                cap.value: len(servers) 
                for cap, servers in self.servers_by_capability.items()
            },
            "request_distribution": dict(self.request_counts)
        } 