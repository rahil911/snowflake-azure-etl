"""
Base tool classes for the multi-agent data intelligence platform.

This module provides foundational classes for implementing MCP-compatible
tools that agents can use to perform specific operations.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, ConfigDict, validator


class ToolType(str, Enum):
    """Types of tools available in the system."""
    DATABASE = "database"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"
    FILE_SYSTEM = "file_system"
    WEB_API = "web_api"
    CUSTOM = "custom"


class ToolStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolError(Exception):
    """Base exception for tool-related errors."""
    
    def __init__(self, message: str, error_code: str = "TOOL_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class ToolResult(BaseModel):
    """Result of tool execution."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='allow'
    )
    
    # Execution metadata
    tool_id: str
    execution_id: str
    status: ToolStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Result data
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    result_text: Optional[str] = None
    
    # Error information
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('execution_time_ms', pre=True)
    def calculate_execution_time(cls, v, values):
        """Calculate execution time if not provided."""
        if v is None and values.get('started_at') and values.get('completed_at'):
            start = values['started_at']
            end = values['completed_at']
            return (end - start).total_seconds() * 1000
        return v
    
    def is_successful(self) -> bool:
        """Check if the tool execution was successful."""
        return self.success and self.status == ToolStatus.COMPLETED
    
    def get_result(self) -> Any:
        """Get the primary result data."""
        if self.result_data:
            return self.result_data
        return self.result_text


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    model_config = ConfigDict(extra='forbid')
    
    name: str
    type: str  # python type name or JSON schema type
    description: str
    required: bool = True
    default: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None  # validation constraints
    
    def validate_value(self, value: Any) -> bool:
        """Validate a parameter value against constraints."""
        # Basic type checking would go here
        # For now, just check if required parameter is provided
        if self.required and value is None:
            return False
        return True


class ToolDefinition(BaseModel):
    """Definition of a tool and its interface."""
    model_config = ConfigDict(extra='forbid')
    
    # Tool identity
    tool_id: str
    name: str
    description: str
    version: str = "1.0.0"
    tool_type: ToolType = ToolType.CUSTOM
    
    # Tool interface
    parameters: List[ToolParameter] = Field(default_factory=list)
    return_type: str = "Any"
    return_description: str = ""
    
    # Capabilities and requirements
    requires_auth: bool = False
    is_async: bool = True
    max_execution_time_seconds: int = 300
    max_concurrent_executions: int = 5
    
    # Resource requirements
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[float] = None
    
    # Tool metadata
    tags: List[str] = Field(default_factory=list)
    documentation_url: Optional[str] = None
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    
    def get_parameter(self, name: str) -> Optional[ToolParameter]:
        """Get parameter definition by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters against tool definition."""
        errors = []
        
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Required parameter '{param.name}' is missing")
            elif param.name in params and not param.validate_value(params[param.name]):
                errors.append(f"Parameter '{param.name}' validation failed")
        
        # Check for unknown parameters
        defined_params = {param.name for param in self.parameters}
        for param_name in params:
            if param_name not in defined_params:
                errors.append(f"Unknown parameter '{param_name}'")
        
        return errors


class BaseTool(ABC):
    """
    Abstract base class for all tools in the multi-agent system.
    
    Tools are discrete functions that agents can use to perform specific
    operations. This base class provides common functionality and ensures
    MCP protocol compatibility.
    """
    
    def __init__(self, tool_definition: ToolDefinition):
        self.definition = tool_definition
        self.tool_id = tool_definition.tool_id
        self.logger = logging.getLogger(f"Tool[{self.tool_id}]")
        
        # Execution tracking
        self.active_executions: Dict[str, ToolResult] = {}
        self.execution_history: List[ToolResult] = []
        self.total_executions = 0
        self.successful_executions = 0
        
        # Resource management
        self._execution_semaphore = asyncio.Semaphore(
            tool_definition.max_concurrent_executions
        )
        
        # State
        self._is_initialized = False
        self._shutdown_event = asyncio.Event()
    
    @property
    def is_initialized(self) -> bool:
        """Check if tool is initialized and ready to use."""
        return self._is_initialized
    
    @property
    def success_rate(self) -> float:
        """Calculate tool success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    async def initialize(self) -> None:
        """Initialize tool resources."""
        try:
            self.logger.info(f"Initializing tool {self.tool_id}")
            await self._on_initialize()
            self._is_initialized = True
            self.logger.info("Tool initialization completed")
        except Exception as e:
            self.logger.error(f"Tool initialization failed: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup tool resources."""
        try:
            self.logger.info("Starting tool cleanup")
            self._shutdown_event.set()
            
            # Cancel active executions
            for execution_id in list(self.active_executions.keys()):
                await self.cancel_execution(execution_id)
            
            await self._on_cleanup()
            self._is_initialized = False
            self.logger.info("Tool cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
            raise
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        execution_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Tool parameters
            execution_id: Optional execution ID
            timeout: Optional timeout in seconds
            
        Returns:
            Tool execution result
        """
        if not self._is_initialized:
            raise ToolError("Tool not initialized", "TOOL_NOT_INITIALIZED")
        
        execution_id = execution_id or str(uuid.uuid4())
        timeout = timeout or self.definition.max_execution_time_seconds
        
        # Validate parameters
        validation_errors = self.definition.validate_parameters(parameters)
        if validation_errors:
            error_msg = f"Parameter validation failed: {'; '.join(validation_errors)}"
            return ToolResult(
                tool_id=self.tool_id,
                execution_id=execution_id,
                status=ToolStatus.FAILED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                success=False,
                error_code="PARAMETER_VALIDATION_ERROR",
                error_message=error_msg
            )
        
        # Check if we can execute (concurrent limit)
        async with self._execution_semaphore:
            return await self._execute_with_tracking(execution_id, parameters, timeout)
    
    async def _execute_with_tracking(
        self,
        execution_id: str,
        parameters: Dict[str, Any],
        timeout: float
    ) -> ToolResult:
        """Execute tool with proper tracking and error handling."""
        started_at = datetime.utcnow()
        
        # Create initial result
        result = ToolResult(
            tool_id=self.tool_id,
            execution_id=execution_id,
            status=ToolStatus.PENDING,
            started_at=started_at,
            success=False
        )
        
        self.active_executions[execution_id] = result
        
        try:
            result.status = ToolStatus.RUNNING
            self.logger.debug(f"Starting execution {execution_id}")
            
            # Execute with timeout
            execution_result = await asyncio.wait_for(
                self._execute_implementation(parameters),
                timeout=timeout
            )
            
            # Update result with execution data
            result.status = ToolStatus.COMPLETED
            result.success = True
            result.completed_at = datetime.utcnow()
            
            if isinstance(execution_result, dict):
                result.result_data = execution_result
            else:
                result.result_text = str(execution_result)
            
            self.successful_executions += 1
            self.logger.debug(f"Execution {execution_id} completed successfully")
            
        except asyncio.TimeoutError:
            result.status = ToolStatus.FAILED
            result.error_code = "EXECUTION_TIMEOUT"
            result.error_message = f"Execution timed out after {timeout} seconds"
            result.completed_at = datetime.utcnow()
            self.logger.warning(f"Execution {execution_id} timed out")
            
        except ToolError as e:
            result.status = ToolStatus.FAILED
            result.error_code = e.error_code
            result.error_message = e.message
            result.error_details = e.details
            result.completed_at = datetime.utcnow()
            self.logger.error(f"Execution {execution_id} failed: {e}")
            
        except Exception as e:
            result.status = ToolStatus.FAILED
            result.error_code = "UNEXPECTED_ERROR"
            result.error_message = str(e)
            result.completed_at = datetime.utcnow()
            self.logger.error(f"Execution {execution_id} failed unexpectedly: {e}", exc_info=True)
        
        finally:
            # Update tracking
            self.total_executions += 1
            
            # Move to history and cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            self.execution_history.append(result)
            
            # Keep only last 1000 executions in history
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
        
        return result
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        if execution_id not in self.active_executions:
            return False
        
        try:
            result = self.active_executions[execution_id]
            result.status = ToolStatus.CANCELLED
            result.completed_at = datetime.utcnow()
            
            # Call implementation-specific cancellation
            await self._cancel_execution_implementation(execution_id)
            
            self.logger.info(f"Execution {execution_id} cancelled")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling execution {execution_id}: {e}")
            return False
    
    def get_execution_status(self, execution_id: str) -> Optional[ToolResult]:
        """Get status of an execution."""
        # Check active executions first
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        
        # Check history
        for result in reversed(self.execution_history):
            if result.execution_id == execution_id:
                return result
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive tool status."""
        return {
            'tool_id': self.tool_id,
            'name': self.definition.name,
            'type': self.definition.tool_type.value,
            'version': self.definition.version,
            'is_initialized': self._is_initialized,
            'active_executions': len(self.active_executions),
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate': self.success_rate,
            'max_concurrent_executions': self.definition.max_concurrent_executions,
            'recent_executions': [
                {
                    'execution_id': result.execution_id,
                    'status': result.status.value,
                    'success': result.success,
                    'started_at': result.started_at.isoformat(),
                    'execution_time_ms': result.execution_time_ms
                }
                for result in self.execution_history[-10:]
            ]
        }
    
    @asynccontextmanager
    async def execution_context(self, execution_id: str):
        """Context manager for tool executions."""
        try:
            yield self.active_executions.get(execution_id)
        finally:
            # Cleanup or logging could go here
            pass
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def _on_initialize(self) -> None:
        """Custom initialization logic for the tool."""
        pass
    
    @abstractmethod
    async def _on_cleanup(self) -> None:
        """Custom cleanup logic for the tool."""
        pass
    
    @abstractmethod
    async def _execute_implementation(self, parameters: Dict[str, Any]) -> Any:
        """Execute the actual tool functionality."""
        pass
    
    async def _cancel_execution_implementation(self, execution_id: str) -> None:
        """Cancel execution implementation (optional override)."""
        pass 