"""
Base classes and interfaces for the multi-agent data intelligence platform.

This module provides foundational classes that all agents and components
inherit from, ensuring consistency across the platform.
"""

from .agent_base import BaseAgent, AgentState, AgentCapabilities
from .tool_base import BaseTool, ToolResult, ToolError
from .connection_base import BaseConnection, ConnectionPool

__all__ = [
    'BaseAgent',
    'AgentState', 
    'AgentCapabilities',
    'BaseTool',
    'ToolResult',
    'ToolError',
    'BaseConnection',
    'ConnectionPool'
]

__version__ = '1.0.0' 