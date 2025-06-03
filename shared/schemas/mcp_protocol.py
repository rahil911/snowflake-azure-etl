"""
Model Context Protocol (MCP) Schema Definitions
================================================

Defines the schemas and interfaces for MCP-compatible tool servers
that will be implemented in Session B. This follows MCP protocol
specifications while being compatible with Python 3.9.

Based on requirements in AGENT_DEPLOYMENT_GUIDE.md and BIG_PICTURE.md.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, validator


class MCPMessageType(str, Enum):
    """MCP message types for tool server communication."""
    INITIALIZE = "initialize"
    LIST_TOOLS = "list_tools"
    CALL_TOOL = "call_tool"
    GET_SCHEMA = "get_schema"
    HEALTH_CHECK = "health_check"
    ERROR = "error"


class MCPStatus(str, Enum):
    """MCP server status states."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class MCPToolParameterType(str, Enum):
    """Parameter types for MCP tools."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class MCPToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str = Field(..., description="Parameter name")
    type: MCPToolParameterType = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value if not required")
    enum: Optional[List[Any]] = Field(None, description="Allowed values for the parameter")
    minimum: Optional[Union[int, float]] = Field(None, description="Minimum value for numbers")
    maximum: Optional[Union[int, float]] = Field(None, description="Maximum value for numbers")
    pattern: Optional[str] = Field(None, description="Regex pattern for strings")


class MCPToolSchema(BaseModel):
    """Schema definition for an MCP tool."""
    name: str = Field(..., description="Tool name (must be unique)")
    description: str = Field(..., description="Tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    
    # Input schema
    parameters: List[MCPToolParameter] = Field(default_factory=list, description="Tool parameters")
    
    # Output schema
    returns: MCPToolParameterType = Field(default=MCPToolParameterType.OBJECT, description="Return type")
    return_description: str = Field(default="Tool execution result", description="Return value description")
    
    # Metadata
    category: str = Field(default="general", description="Tool category")
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Usage examples")
    
    # Performance characteristics
    estimated_runtime_ms: Optional[int] = Field(None, description="Estimated execution time")
    max_timeout_ms: int = Field(default=30000, description="Maximum execution timeout")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate tool name follows naming conventions."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Tool name must contain only alphanumeric characters, hyphens, and underscores")
        return v.lower()


class MCPToolRequest(BaseModel):
    """Request to execute an MCP tool."""
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique request ID")
    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    
    # Request metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    timeout_ms: Optional[int] = Field(None, description="Request timeout in milliseconds")
    priority: int = Field(default=5, ge=1, le=10, description="Request priority (1=highest, 10=lowest)")
    
    # Context
    agent_id: Optional[str] = Field(None, description="Requesting agent ID")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking")


class MCPToolResponse(BaseModel):
    """Response from MCP tool execution."""
    request_id: str = Field(..., description="Original request ID")
    tool_name: str = Field(..., description="Executed tool name")
    success: bool = Field(..., description="Whether execution succeeded")
    
    # Results
    result: Optional[Any] = Field(None, description="Tool execution result")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Execution metadata
    execution_time_ms: float = Field(..., description="Actual execution time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    # Resource usage
    memory_used_mb: Optional[float] = Field(None, description="Memory used during execution")
    cpu_time_ms: Optional[float] = Field(None, description="CPU time used")


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""
    server_id: str = Field(..., description="Unique server identifier")
    name: str = Field(..., description="Server name")
    description: str = Field(..., description="Server description")
    version: str = Field(..., description="Server version")
    
    # Connection info
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(..., ge=1, le=65535, description="Server port")
    protocol: Literal["http", "https", "ws", "wss"] = Field(default="http", description="Protocol")
    
    # Capabilities
    available_tools: List[str] = Field(default_factory=list, description="List of available tool names")
    max_concurrent_requests: int = Field(default=10, ge=1, le=1000, description="Max concurrent requests")
    
    # Status
    status: MCPStatus = Field(default=MCPStatus.INITIALIZING, description="Current server status")
    last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Server uptime in seconds")


class MCPHealthCheck(BaseModel):
    """Health check response from MCP server."""
    server_id: str = Field(..., description="Server identifier")
    status: MCPStatus = Field(..., description="Current status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    
    # Health metrics
    uptime_seconds: float = Field(..., description="Server uptime")
    active_requests: int = Field(..., description="Currently active requests")
    total_requests: int = Field(default=0, description="Total requests processed")
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Error rate (0.0-1.0)")
    average_response_time_ms: float = Field(default=0.0, description="Average response time")
    
    # Resource usage
    memory_usage_mb: Optional[float] = Field(None, description="Current memory usage")
    cpu_usage_percent: Optional[float] = Field(None, description="Current CPU usage")
    
    # Additional health data
    health_data: Dict[str, Any] = Field(default_factory=dict, description="Additional health information")


class MCPMessage(BaseModel):
    """Generic MCP message structure."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Message ID")
    type: MCPMessageType = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    # Content
    payload: Union[MCPToolRequest, MCPToolResponse, MCPServerInfo, MCPHealthCheck, Dict[str, Any]] = Field(
        ..., description="Message payload")
    
    # Routing
    source: Optional[str] = Field(None, description="Source identifier")
    destination: Optional[str] = Field(None, description="Destination identifier")


class MCPRegistry(BaseModel):
    """Registry of available MCP servers and tools."""
    
    servers: Dict[str, MCPServerInfo] = Field(default_factory=dict, description="Registered servers")
    tools: Dict[str, MCPToolSchema] = Field(default_factory=dict, description="Available tools")
    
    def register_server(self, server_info: MCPServerInfo) -> None:
        """Register a new MCP server."""
        self.servers[server_info.server_id] = server_info
    
    def register_tool(self, tool_schema: MCPToolSchema, server_id: str) -> None:
        """Register a tool with its server."""
        # Add server reference to tool
        tool_key = f"{server_id}:{tool_schema.name}"
        self.tools[tool_key] = tool_schema
        
        # Update server's available tools
        if server_id in self.servers:
            if tool_schema.name not in self.servers[server_id].available_tools:
                self.servers[server_id].available_tools.append(tool_schema.name)
    
    def find_tool(self, tool_name: str) -> Optional[MCPToolSchema]:
        """Find a tool by name (returns first match)."""
        for tool_key, tool_schema in self.tools.items():
            if tool_schema.name == tool_name:
                return tool_schema
        return None
    
    def find_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Find which server provides a specific tool."""
        for tool_key in self.tools:
            if tool_key.endswith(f":{tool_name}"):
                return tool_key.split(":")[0]
        return None
    
    def get_healthy_servers(self) -> List[MCPServerInfo]:
        """Get list of servers that are currently healthy."""
        return [
            server for server in self.servers.values()
            if server.status == MCPStatus.READY
        ]


# Factory functions for common MCP operations

def create_tool_request(
    tool_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
    timeout_ms: Optional[int] = None
) -> MCPToolRequest:
    """Create a standard tool request."""
    return MCPToolRequest(
        tool_name=tool_name,
        parameters=parameters or {},
        agent_id=agent_id,
        timeout_ms=timeout_ms
    )


def create_tool_response(
    request_id: str,
    tool_name: str,
    success: bool,
    result: Optional[Any] = None,
    error_message: Optional[str] = None,
    execution_time_ms: float = 0.0
) -> MCPToolResponse:
    """Create a standard tool response."""
    return MCPToolResponse(
        request_id=request_id,
        tool_name=tool_name,
        success=success,
        result=result,
        error_message=error_message,
        execution_time_ms=execution_time_ms
    )


def create_server_info(
    server_id: str,
    name: str,
    port: int,
    description: str = "",
    version: str = "1.0.0"
) -> MCPServerInfo:
    """Create server info for registration."""
    return MCPServerInfo(
        server_id=server_id,
        name=name,
        description=description,
        version=version,
        port=port
    ) 