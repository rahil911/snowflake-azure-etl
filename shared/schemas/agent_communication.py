"""
Agent Communication Schema
==========================

Defines the message format and communication protocols for inter-agent messaging
using the model bus. All agents use these schemas for consistent communication.

Based on the multi-agent architecture defined in BIG_PICTURE.md and .cursorrules.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4
from pydantic import BaseModel, Field, validator


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""
    QUERY = "query"                    # User or agent query
    RESPONSE = "response"              # Agent response
    ERROR = "error"                    # Error notification
    STATUS = "status"                  # Status update
    TOOL_REQUEST = "tool_request"      # Request to use a tool
    TOOL_RESPONSE = "tool_response"    # Tool execution result
    HANDOFF = "handoff"                # Transfer to another agent
    CONTEXT_UPDATE = "context_update"  # Context/state update


class Priority(str, Enum):
    """Message priority levels for processing order."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AgentRole(str, Enum):
    """Defines the roles of agents in the system."""
    COORDINATOR = "coordinator"           # Main orchestrator
    DATA_INTELLIGENCE = "data_intelligence"  # Business intelligence
    ETL_AGENT = "etl_agent"              # ETL operations
    VISUALIZATION = "visualization"       # Chart generation
    USER = "user"                        # Human user
    SYSTEM = "system"                    # System notifications


class MessagePayload(BaseModel):
    """Base payload structure for all messages."""
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryPayload(MessagePayload):
    """Payload for query messages."""
    query_text: str = Field(..., description="The natural language query")
    query_type: str = Field(default="general", description="Type of query (data, etl, viz, etc)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    requires_tools: List[str] = Field(default_factory=list, description="Required tools")


class ResponsePayload(MessagePayload):
    """Payload for response messages."""
    response_text: str = Field(..., description="The response content")
    response_type: str = Field(default="text", description="Type of response")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    sources: List[str] = Field(default_factory=list, description="Data sources used")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Files, charts, etc")


class ErrorPayload(MessagePayload):
    """Payload for error messages."""
    error_code: str = Field(..., description="Error code identifier")
    error_message: str = Field(..., description="Human-readable error message")
    error_type: str = Field(default="general", description="Category of error")
    stack_trace: Optional[str] = Field(None, description="Technical stack trace")
    recovery_suggestions: List[str] = Field(default_factory=list, description="How to recover")


class ToolRequestPayload(MessagePayload):
    """Payload for tool execution requests."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    tool_params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")


class ToolResponsePayload(MessagePayload):
    """Payload for tool execution results."""
    tool_name: str = Field(..., description="Name of the executed tool")
    result: Any = Field(..., description="Tool execution result")
    execution_time: float = Field(..., description="Execution time in seconds")
    success: bool = Field(..., description="Whether execution was successful")
    error_message: Optional[str] = Field(None, description="Error if execution failed")


class AgentMessage(BaseModel):
    """
    Core message structure for inter-agent communication.
    
    This is the primary message format used by all agents in the system
    for communication through the model bus.
    """
    
    # Message identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique message ID")
    correlation_id: Optional[str] = Field(None, description="ID linking related messages")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")
    
    # Message routing
    type: MessageType = Field(..., description="Type of message")
    source_agent: str = Field(..., description="Agent sending the message")
    target_agent: Optional[str] = Field(None, description="Specific target agent (None for broadcast)")
    route_history: List[str] = Field(default_factory=list, description="Agents that have processed this message")
    
    # Message content
    payload: Union[QueryPayload, ResponsePayload, ErrorPayload, ToolRequestPayload, ToolResponsePayload, MessagePayload] = Field(
        ..., description="Message content based on type")
    
    # Message metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message creation time")
    priority: Priority = Field(default=Priority.NORMAL, description="Processing priority")
    expires_at: Optional[datetime] = Field(None, description="Message expiration time")
    
    # Processing metadata
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    processing_time: Optional[float] = Field(None, description="Time taken to process")
    
    @validator('payload', pre=True)
    def validate_payload_type(cls, v, values):
        """Ensure payload matches message type."""
        if 'type' not in values:
            return v
            
        message_type = values['type']
        if isinstance(v, dict):
            # Convert dict to appropriate payload type
            if message_type == MessageType.QUERY:
                return QueryPayload(**v)
            elif message_type == MessageType.RESPONSE:
                return ResponsePayload(**v)
            elif message_type == MessageType.ERROR:
                return ErrorPayload(**v)
            elif message_type == MessageType.TOOL_REQUEST:
                return ToolRequestPayload(**v)
            elif message_type == MessageType.TOOL_RESPONSE:
                return ToolResponsePayload(**v)
            else:
                return MessagePayload(**v)
        return v
    
    def add_to_route_history(self, agent_name: str) -> None:
        """Add an agent to the routing history."""
        if agent_name not in self.route_history:
            self.route_history.append(agent_name)
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1


class ConversationContext(BaseModel):
    """
    Conversation context shared between agents.
    Maintains state and history for multi-turn interactions.
    """
    
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = Field(None, description="Identifier for the user")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    # Conversation state
    messages: List[AgentMessage] = Field(default_factory=list, description="Message history")
    current_agent: Optional[str] = Field(None, description="Currently active agent")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Shared context")
    
    # Conversation metadata
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    
    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
    
    def get_recent_messages(self, count: int = 10) -> List[AgentMessage]:
        """Get the most recent messages."""
        return self.messages[-count:]
    
    def update_context(self, key: str, value: Any) -> None:
        """Update shared context data."""
        self.context_data[key] = value
        self.last_activity = datetime.utcnow()


class AgentCapability(BaseModel):
    """Describes what an agent can do for routing decisions."""
    
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="What this capability does")
    input_types: List[str] = Field(..., description="Types of input this capability accepts")
    output_types: List[str] = Field(..., description="Types of output this capability produces")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence to route here")
    priority: int = Field(default=5, description="Priority for capability selection")


class AgentDirectory(BaseModel):
    """Registry of available agents and their capabilities."""
    
    agents: Dict[str, List[AgentCapability]] = Field(default_factory=dict)
    
    def register_agent(self, agent_name: str, capabilities: List[AgentCapability]) -> None:
        """Register an agent and its capabilities."""
        self.agents[agent_name] = capabilities
    
    def find_capable_agents(self, required_capability: str) -> List[str]:
        """Find agents that have a specific capability."""
        capable_agents = []
        for agent_name, capabilities in self.agents.items():
            for capability in capabilities:
                if capability.name == required_capability:
                    capable_agents.append(agent_name)
        return capable_agents


# Message factory functions for easy creation
def create_query_message(
    source_agent: str,
    query_text: str,
    target_agent: Optional[str] = None,
    query_type: str = "general",
    context: Optional[Dict[str, Any]] = None
) -> AgentMessage:
    """Create a query message."""
    payload = QueryPayload(
        query_text=query_text,
        query_type=query_type,
        context=context or {}
    )
    return AgentMessage(
        type=MessageType.QUERY,
        source_agent=source_agent,
        target_agent=target_agent,
        payload=payload
    )


def create_response_message(
    source_agent: str,
    response_text: str,
    correlation_id: str,
    target_agent: Optional[str] = None,
    confidence: Optional[float] = None
) -> AgentMessage:
    """Create a response message."""
    payload = ResponsePayload(
        response_text=response_text,
        confidence=confidence
    )
    return AgentMessage(
        type=MessageType.RESPONSE,
        source_agent=source_agent,
        target_agent=target_agent,
        correlation_id=correlation_id,
        payload=payload
    )


def create_error_message(
    source_agent: str,
    error_code: str,
    error_message: str,
    correlation_id: Optional[str] = None,
    target_agent: Optional[str] = None
) -> AgentMessage:
    """Create an error message."""
    payload = ErrorPayload(
        error_code=error_code,
        error_message=error_message
    )
    return AgentMessage(
        type=MessageType.ERROR,
        source_agent=source_agent,
        target_agent=target_agent,
        correlation_id=correlation_id,
        payload=payload
    )


class Intent(BaseModel):
    """User intent analysis result."""
    
    intent_type: str = Field(..., description="Type of intent (data_query, insight, etc)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Intent parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    original_text: str = Field(..., description="Original user input")
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class EntityExtraction(BaseModel):
    """Entity extraction result."""
    
    entity_text: str = Field(..., description="Extracted entity text")
    entity_type: str = Field(..., description="Type of entity")
    start_position: int = Field(..., description="Start position in text")
    end_position: int = Field(..., description="End position in text")
    confidence: float = Field(..., description="Extraction confidence (0-1)")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    source_text: str = Field(..., description="Source text")
    extracted_at: datetime = Field(default_factory=datetime.utcnow) 