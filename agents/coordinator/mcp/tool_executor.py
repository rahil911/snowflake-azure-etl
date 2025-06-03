"""
Tool execution system for MCP (Model Context Protocol) tools.

This module provides the ToolExecutor class that manages execution of
tools from MCP servers, handles parameter validation, and provides
execution tracking and error handling.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ConfigDict, ValidationError
import google.genai as genai

from shared.schemas.agent_communication import (
    AgentMessage, AgentRole, MessageType, ConversationContext
)
from shared.config.logging_config import setup_logging
from shared.utils.metrics import get_metrics_collector
from .server_connector import MCPServerConnector, MCPRequest, MCPResponse, ServerCapability


class ExecutionStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolCategory(str, Enum):
    """Tool categories for organization."""
    DATABASE = "database"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"
    ETL = "etl"
    DISCOVERY = "discovery"
    REPORTING = "reporting"
    UTILITY = "utility"


class ParameterType(str, Enum):
    """Parameter data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATE = "date"
    DATETIME = "datetime"


@dataclass
class ParameterDefinition:
    """Tool parameter definition."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    
    # Validation constraints
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    
    # Metadata
    examples: List[Any] = field(default_factory=list)
    sensitive: bool = False  # For logging/security


class ToolDefinition(BaseModel):
    """Tool definition from MCP server."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    tool_id: str = Field(..., description="Unique tool identifier")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    
    # Categorization
    category: ToolCategory = Field(..., description="Tool category")
    server_id: str = Field(..., description="Source MCP server")
    
    # Parameters
    parameters: List[ParameterDefinition] = Field(default_factory=list)
    
    # Execution constraints
    timeout_seconds: int = Field(default=300, description="Default timeout")
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    concurrent_limit: int = Field(default=5, description="Max concurrent executions")
    
    # Capabilities and requirements
    required_capabilities: Set[ServerCapability] = Field(default_factory=set)
    
    # Metadata
    version: str = Field(default="1.0.0")
    deprecated: bool = Field(default=False)
    experimental: bool = Field(default=False)
    
    # Examples and documentation
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    documentation_url: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ExecutionContext(BaseModel):
    """Context for tool execution."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str = Field(..., description="Tool being executed")
    
    # Parameters and configuration
    parameters: Dict[str, Any] = Field(..., description="Execution parameters")
    timeout: Optional[int] = Field(default=None, description="Execution timeout")
    
    # Context information
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_role: Optional[AgentRole] = None
    
    # Execution metadata
    priority: int = Field(default=0, description="Execution priority")
    retry_count: int = Field(default=0, description="Current retry count")
    tags: Set[str] = Field(default_factory=set, description="Execution tags")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ExecutionResult(BaseModel):
    """Result of tool execution."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    execution_id: str = Field(..., description="Execution identifier")
    tool_id: str = Field(..., description="Tool identifier")
    status: ExecutionStatus = Field(..., description="Execution status")
    
    # Results
    result: Optional[Any] = Field(default=None, description="Success result")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    
    # Performance metrics
    execution_time_ms: float = Field(..., description="Total execution time")
    server_time_ms: float = Field(default=0.0, description="Server processing time")
    
    # Resource usage
    memory_used_mb: Optional[float] = None
    cpu_time_ms: Optional[float] = None
    
    # Metadata
    server_id: str = Field(..., description="Executing server")
    retry_count: int = Field(default=0, description="Number of retries")
    
    # Timestamps
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class ParameterValidator:
    """Validates tool parameters against definitions."""
    
    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)
    
    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        parameter_definitions: List[ParameterDefinition]
    ) -> Dict[str, Any]:
        """
        Validate and normalize parameters.
        
        Args:
            parameters: Raw parameters to validate
            parameter_definitions: Parameter definitions
            
        Returns:
            Validated and normalized parameters
            
        Raises:
            ValidationError: If validation fails
        """
        validated = {}
        definition_map = {p.name: p for p in parameter_definitions}
        
        # Check required parameters
        for param_def in parameter_definitions:
            if param_def.required and param_def.name not in parameters:
                if param_def.default is not None:
                    validated[param_def.name] = param_def.default
                else:
                    raise ValidationError(f"Required parameter '{param_def.name}' missing")
        
        # Validate provided parameters
        for param_name, param_value in parameters.items():
            if param_name not in definition_map:
                # Allow extra parameters (might be used by server)
                validated[param_name] = param_value
                continue
            
            param_def = definition_map[param_name]
            validated_value = self._validate_parameter_value(param_value, param_def)
            validated[param_name] = validated_value
        
        return validated
    
    def _validate_parameter_value(self, value: Any, param_def: ParameterDefinition) -> Any:
        """Validate a single parameter value."""
        if value is None:
            if param_def.required and param_def.default is None:
                raise ValidationError(f"Parameter '{param_def.name}' cannot be null")
            return param_def.default
        
        # Type validation
        if param_def.type == ParameterType.STRING:
            if not isinstance(value, str):
                raise ValidationError(f"Parameter '{param_def.name}' must be a string")
            return self._validate_string(value, param_def)
        
        elif param_def.type == ParameterType.INTEGER:
            if not isinstance(value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    raise ValidationError(f"Parameter '{param_def.name}' must be an integer")
            return self._validate_numeric(value, param_def)
        
        elif param_def.type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise ValidationError(f"Parameter '{param_def.name}' must be a number")
            return self._validate_numeric(value, param_def)
        
        elif param_def.type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                if isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    value = bool(value)
            return value
        
        elif param_def.type == ParameterType.ARRAY:
            if not isinstance(value, list):
                raise ValidationError(f"Parameter '{param_def.name}' must be an array")
            return self._validate_array(value, param_def)
        
        elif param_def.type == ParameterType.OBJECT:
            if not isinstance(value, dict):
                raise ValidationError(f"Parameter '{param_def.name}' must be an object")
            return value
        
        elif param_def.type == ParameterType.DATE:
            return self._validate_date(value, param_def)
        
        elif param_def.type == ParameterType.DATETIME:
            return self._validate_datetime(value, param_def)
        
        return value
    
    def _validate_string(self, value: str, param_def: ParameterDefinition) -> str:
        """Validate string parameter."""
        if param_def.min_length is not None and len(value) < param_def.min_length:
            raise ValidationError(
                f"Parameter '{param_def.name}' must be at least {param_def.min_length} characters"
            )
        
        if param_def.max_length is not None and len(value) > param_def.max_length:
            raise ValidationError(
                f"Parameter '{param_def.name}' must be at most {param_def.max_length} characters"
            )
        
        if param_def.pattern is not None:
            import re
            if not re.match(param_def.pattern, value):
                raise ValidationError(
                    f"Parameter '{param_def.name}' does not match required pattern"
                )
        
        if param_def.enum_values is not None and value not in param_def.enum_values:
            raise ValidationError(
                f"Parameter '{param_def.name}' must be one of: {param_def.enum_values}"
            )
        
        return value
    
    def _validate_numeric(self, value: Union[int, float], param_def: ParameterDefinition) -> Union[int, float]:
        """Validate numeric parameter."""
        if param_def.min_value is not None and value < param_def.min_value:
            raise ValidationError(
                f"Parameter '{param_def.name}' must be at least {param_def.min_value}"
            )
        
        if param_def.max_value is not None and value > param_def.max_value:
            raise ValidationError(
                f"Parameter '{param_def.name}' must be at most {param_def.max_value}"
            )
        
        if param_def.enum_values is not None and value not in param_def.enum_values:
            raise ValidationError(
                f"Parameter '{param_def.name}' must be one of: {param_def.enum_values}"
            )
        
        return value
    
    def _validate_array(self, value: List[Any], param_def: ParameterDefinition) -> List[Any]:
        """Validate array parameter."""
        if param_def.min_length is not None and len(value) < param_def.min_length:
            raise ValidationError(
                f"Parameter '{param_def.name}' must have at least {param_def.min_length} items"
            )
        
        if param_def.max_length is not None and len(value) > param_def.max_length:
            raise ValidationError(
                f"Parameter '{param_def.name}' must have at most {param_def.max_length} items"
            )
        
        return value
    
    def _validate_date(self, value: Any, param_def: ParameterDefinition) -> str:
        """Validate date parameter."""
        if isinstance(value, str):
            try:
                datetime.strptime(value, "%Y-%m-%d")
                return value
            except ValueError:
                raise ValidationError(f"Parameter '{param_def.name}' must be a valid date (YYYY-MM-DD)")
        
        elif isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        
        else:
            raise ValidationError(f"Parameter '{param_def.name}' must be a date string or datetime object")
    
    def _validate_datetime(self, value: Any, param_def: ParameterDefinition) -> str:
        """Validate datetime parameter."""
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return value
            except ValueError:
                raise ValidationError(f"Parameter '{param_def.name}' must be a valid ISO datetime")
        
        elif isinstance(value, datetime):
            return value.isoformat()
        
        else:
            raise ValidationError(f"Parameter '{param_def.name}' must be a datetime string or datetime object")


class ToolExecutor:
    """
    Tool execution manager for MCP tools.
    
    Features:
    - Parameter validation
    - Execution tracking
    - Retry logic
    - Concurrent execution limits
    - Performance monitoring
    - Error handling
    """
    
    def __init__(
        self,
        server_connector: MCPServerConnector,
        settings = None
    ):
        self.logger = setup_logging(self.__class__.__name__)
        self.server_connector = server_connector
        self.metrics = get_metrics_collector()
        
        # Tool management
        self.tools: Dict[str, ToolDefinition] = {}
        self.tools_by_category: Dict[ToolCategory, List[str]] = {}
        self.tools_by_server: Dict[str, List[str]] = {}
        
        # Execution tracking
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_history: List[ExecutionResult] = []
        self.concurrent_counts: Dict[str, int] = {}  # tool_id -> count
        
        # Components
        self.parameter_validator = ParameterValidator()
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        
        self.logger.info("ToolExecutor initialized")
    
    async def initialize(self) -> None:
        """Initialize the tool executor by discovering available tools."""
        await self._discover_tools()
        
        # Start background cleanup
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info(f"ToolExecutor initialized with {len(self.tools)} tools")
    
    async def _discover_tools(self) -> None:
        """Discover tools from all connected MCP servers."""
        try:
            # Get server status
            server_statuses = self.server_connector.get_server_status()
            
            for server_status in server_statuses:
                if server_status["status"] != "connected":
                    continue
                
                server_id = server_status["server_id"]
                
                try:
                    # Request tool list from server
                    response = await self.server_connector.send_request(
                        method="tools/list",
                        params={},
                        server_id=server_id,
                        timeout=30
                    )
                    
                    if response.success and response.result:
                        tools_data = response.result.get("tools", [])
                        
                        for tool_data in tools_data:
                            tool_def = self._create_tool_definition(tool_data, server_id)
                            await self._register_tool(tool_def)
                            
                except Exception as e:
                    self.logger.error(f"Failed to discover tools from {server_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Tool discovery failed: {e}")
    
    def _create_tool_definition(self, tool_data: Dict[str, Any], server_id: str) -> ToolDefinition:
        """Create tool definition from server data."""
        # Parse parameters
        parameters = []
        for param_data in tool_data.get("parameters", []):
            param = ParameterDefinition(
                name=param_data["name"],
                type=ParameterType(param_data.get("type", "string")),
                description=param_data.get("description", ""),
                required=param_data.get("required", True),
                default=param_data.get("default"),
                min_value=param_data.get("min_value"),
                max_value=param_data.get("max_value"),
                min_length=param_data.get("min_length"),
                max_length=param_data.get("max_length"),
                pattern=param_data.get("pattern"),
                enum_values=param_data.get("enum_values"),
                examples=param_data.get("examples", []),
                sensitive=param_data.get("sensitive", False)
            )
            parameters.append(param)
        
        # Determine category
        category = ToolCategory.UTILITY
        if "database" in tool_data["name"].lower() or "sql" in tool_data["name"].lower():
            category = ToolCategory.DATABASE
        elif "analyt" in tool_data["name"].lower() or "insight" in tool_data["name"].lower():
            category = ToolCategory.ANALYTICS
        elif "visual" in tool_data["name"].lower() or "chart" in tool_data["name"].lower():
            category = ToolCategory.VISUALIZATION
        elif "etl" in tool_data["name"].lower() or "transform" in tool_data["name"].lower():
            category = ToolCategory.ETL
        elif "discover" in tool_data["name"].lower() or "schema" in tool_data["name"].lower():
            category = ToolCategory.DISCOVERY
        elif "report" in tool_data["name"].lower():
            category = ToolCategory.REPORTING
        
        return ToolDefinition(
            tool_id=f"{server_id}:{tool_data['name']}",
            name=tool_data["name"],
            description=tool_data.get("description", ""),
            category=category,
            server_id=server_id,
            parameters=parameters,
            timeout_seconds=tool_data.get("timeout", 300),
            max_retries=tool_data.get("max_retries", 2),
            concurrent_limit=tool_data.get("concurrent_limit", 5),
            examples=tool_data.get("examples", []),
            documentation_url=tool_data.get("documentation_url")
        )
    
    async def _register_tool(self, tool_def: ToolDefinition) -> None:
        """Register a tool definition."""
        self.tools[tool_def.tool_id] = tool_def
        
        # Update indexes
        if tool_def.category not in self.tools_by_category:
            self.tools_by_category[tool_def.category] = []
        self.tools_by_category[tool_def.category].append(tool_def.tool_id)
        
        if tool_def.server_id not in self.tools_by_server:
            self.tools_by_server[tool_def.server_id] = []
        self.tools_by_server[tool_def.server_id].append(tool_def.tool_id)
        
        # Initialize concurrent counter
        self.concurrent_counts[tool_def.tool_id] = 0
        
        self.logger.debug(f"Registered tool: {tool_def.name} ({tool_def.tool_id})")
    
    async def execute_tool(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_id: Tool identifier
            parameters: Tool parameters
            context: Execution context
            timeout: Execution timeout
            
        Returns:
            Execution result
        """
        if tool_id not in self.tools:
            raise CoordinatorError(f"Tool {tool_id} not found")
        
        tool_def = self.tools[tool_id]
        
        # Check concurrent limit
        if self.concurrent_counts[tool_id] >= tool_def.concurrent_limit:
            raise CoordinatorError(f"Tool {tool_id} concurrent limit exceeded")
        
        # Create execution context
        exec_context = ExecutionContext(
            tool_id=tool_id,
            parameters=parameters,
            timeout=timeout or tool_def.timeout_seconds,
            conversation_id=context.get("conversation_id") if context else None,
            user_id=context.get("user_id") if context else None,
            agent_role=context.get("agent_role") if context else None
        )
        
        try:
            # Validate parameters
            validated_params = self.parameter_validator.validate_parameters(
                parameters, tool_def.parameters
            )
            exec_context.parameters = validated_params
            
            # Execute with retry logic
            result = await self._execute_with_retry(tool_def, exec_context)
            
            # Store in history
            self.execution_history.append(result)
            
            # Limit history size
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
            return result
            
        except Exception as e:
            # Create error result
            execution_time = 0.0
            if exec_context.started_at:
                execution_time = (datetime.utcnow() - exec_context.started_at).total_seconds() * 1000
            
            error_result = ExecutionResult(
                execution_id=exec_context.execution_id,
                tool_id=tool_id,
                status=ExecutionStatus.FAILED,
                error={
                    "type": type(e).__name__,
                    "message": str(e),
                    "code": getattr(e, 'code', 'EXECUTION_ERROR')
                },
                execution_time_ms=execution_time,
                server_id=tool_def.server_id,
                retry_count=exec_context.retry_count,
                started_at=exec_context.started_at or datetime.utcnow()
            )
            
            self.execution_history.append(error_result)
            return error_result
    
    async def _execute_with_retry(
        self,
        tool_def: ToolDefinition,
        exec_context: ExecutionContext
    ) -> ExecutionResult:
        """Execute tool with retry logic."""
        last_error = None
        
        for attempt in range(tool_def.max_retries + 1):
            exec_context.retry_count = attempt
            
            try:
                return await self._execute_single(tool_def, exec_context)
            
            except Exception as e:
                last_error = e
                self.logger.warning(f"Tool execution attempt {attempt + 1} failed: {e}")
                
                if attempt < tool_def.max_retries:
                    # Wait before retry
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed
                    break
        
        # All retries exhausted
        raise last_error or CoordinatorError("Tool execution failed after all retries")
    
    async def _execute_single(
        self,
        tool_def: ToolDefinition,
        exec_context: ExecutionContext
    ) -> ExecutionResult:
        """Execute tool once."""
        start_time = time.time()
        exec_context.started_at = datetime.utcnow()
        
        try:
            # Track active execution
            self.active_executions[exec_context.execution_id] = exec_context
            self.concurrent_counts[tool_def.tool_id] += 1
            
            # Create MCP request
            method = f"tools/{tool_def.name}"
            
            # Send request to server
            response = await self.server_connector.send_request(
                method=method,
                params=exec_context.parameters,
                server_id=tool_def.server_id,
                timeout=exec_context.timeout
            )
            
            execution_time = (time.time() - start_time) * 1000
            exec_context.completed_at = datetime.utcnow()
            
            if response.success:
                return ExecutionResult(
                    execution_id=exec_context.execution_id,
                    tool_id=tool_def.tool_id,
                    status=ExecutionStatus.COMPLETED,
                    result=response.result,
                    execution_time_ms=execution_time,
                    server_time_ms=response.execution_time_ms,
                    server_id=tool_def.server_id,
                    retry_count=exec_context.retry_count,
                    started_at=exec_context.started_at,
                    completed_at=exec_context.completed_at
                )
            else:
                return ExecutionResult(
                    execution_id=exec_context.execution_id,
                    tool_id=tool_def.tool_id,
                    status=ExecutionStatus.FAILED,
                    error=response.error,
                    execution_time_ms=execution_time,
                    server_time_ms=response.execution_time_ms,
                    server_id=tool_def.server_id,
                    retry_count=exec_context.retry_count,
                    started_at=exec_context.started_at,
                    completed_at=exec_context.completed_at
                )
            
        finally:
            # Cleanup
            self.active_executions.pop(exec_context.execution_id, None)
            self.concurrent_counts[tool_def.tool_id] -= 1
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution.
        
        Args:
            execution_id: Execution to cancel
            
        Returns:
            True if cancellation successful
        """
        if execution_id not in self.active_executions:
            return False
        
        exec_context = self.active_executions[execution_id]
        
        try:
            # In a real implementation, this would signal the server to cancel
            # For now, just mark as cancelled
            self.active_executions.pop(execution_id, None)
            
            tool_def = self.tools[exec_context.tool_id]
            self.concurrent_counts[tool_def.tool_id] -= 1
            
            # Create cancellation result
            execution_time = 0.0
            if exec_context.started_at:
                execution_time = (datetime.utcnow() - exec_context.started_at).total_seconds() * 1000
            
            cancel_result = ExecutionResult(
                execution_id=execution_id,
                tool_id=exec_context.tool_id,
                status=ExecutionStatus.CANCELLED,
                execution_time_ms=execution_time,
                server_id=tool_def.server_id,
                retry_count=exec_context.retry_count,
                started_at=exec_context.started_at or datetime.utcnow()
            )
            
            self.execution_history.append(cancel_result)
            
            self.logger.info(f"Execution cancelled: {execution_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False
    
    def get_tool_definition(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get tool definition by ID."""
        return self.tools.get(tool_id)
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        server_id: Optional[str] = None,
        search_term: Optional[str] = None
    ) -> List[ToolDefinition]:
        """
        List available tools with optional filtering.
        
        Args:
            category: Filter by category
            server_id: Filter by server
            search_term: Search in name/description
            
        Returns:
            List of matching tools
        """
        tools = list(self.tools.values())
        
        # Filter by category
        if category:
            tools = [t for t in tools if t.category == category]
        
        # Filter by server
        if server_id:
            tools = [t for t in tools if t.server_id == server_id]
        
        # Search filter
        if search_term:
            search_lower = search_term.lower()
            tools = [
                t for t in tools
                if (search_lower in t.name.lower() or 
                    search_lower in t.description.lower())
            ]
        
        return tools
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an execution."""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            return {
                "execution_id": execution_id,
                "tool_id": context.tool_id,
                "status": ExecutionStatus.RUNNING.value,
                "started_at": context.started_at.isoformat() if context.started_at else None,
                "retry_count": context.retry_count
            }
        
        # Check execution history
        for result in reversed(self.execution_history):
            if result.execution_id == execution_id:
                return {
                    "execution_id": execution_id,
                    "tool_id": result.tool_id,
                    "status": result.status.value,
                    "started_at": result.started_at.isoformat(),
                    "completed_at": result.completed_at.isoformat(),
                    "execution_time_ms": result.execution_time_ms,
                    "retry_count": result.retry_count,
                    "success": result.status == ExecutionStatus.COMPLETED
                }
        
        return None
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive execution metrics."""
        if not self.execution_history:
            return {"total_executions": 0}
        
        recent_results = self.execution_history[-100:]  # Last 100 executions
        
        # Calculate metrics
        total_executions = len(recent_results)
        successful_executions = len([r for r in recent_results if r.status == ExecutionStatus.COMPLETED])
        failed_executions = len([r for r in recent_results if r.status == ExecutionStatus.FAILED])
        
        avg_execution_time = sum(r.execution_time_ms for r in recent_results) / total_executions
        
        # Tool usage stats
        tool_usage = {}
        for result in recent_results:
            tool_usage[result.tool_id] = tool_usage.get(result.tool_id, 0) + 1
        
        # Category usage stats
        category_usage = {}
        for tool_id, count in tool_usage.items():
            if tool_id in self.tools:
                category = self.tools[tool_id].category.value
                category_usage[category] = category_usage.get(category, 0) + count
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "average_execution_time_ms": avg_execution_time,
            "active_executions": len(self.active_executions),
            "total_tools": len(self.tools),
            "tool_usage": tool_usage,
            "category_usage": category_usage,
            "concurrent_limits": {
                tool_id: f"{self.concurrent_counts[tool_id]}/{self.tools[tool_id].concurrent_limit}"
                for tool_id in self.tools.keys()
            }
        }
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup of old execution data."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up old execution history
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.execution_history = [
                    result for result in self.execution_history
                    if result.completed_at > cutoff_time
                ]
                
                # Check for stuck executions
                for exec_id, context in list(self.active_executions.items()):
                    if (context.started_at and 
                        (datetime.utcnow() - context.started_at).total_seconds() > context.timeout):
                        await self.cancel_execution(exec_id)
                        self.logger.warning(f"Cancelled stuck execution: {exec_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the tool executor."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active executions
        for exec_id in list(self.active_executions.keys()):
            await self.cancel_execution(exec_id)
        
        self.logger.info("ToolExecutor shutdown complete") 