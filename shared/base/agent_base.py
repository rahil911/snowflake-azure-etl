"""
Base agent class for the multi-agent data intelligence platform.

This module provides the foundational BaseAgent class that all specialized
agents inherit from, ensuring consistent behavior and interfaces.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, ConfigDict
import google.genai as genai

from ..schemas.agent_communication import (
    AgentMessage, AgentRole, MessageType, ConversationContext,
    AgentCapability
)
from ..config.settings import get_settings


class AgentState(str, Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class AgentCapabilities(BaseModel):
    """Defines what an agent can do."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Core capabilities
    can_process_text: bool = True
    can_process_audio: bool = False
    can_process_images: bool = False
    can_access_database: bool = False
    can_generate_reports: bool = False
    can_visualize_data: bool = False
    
    # Tool capabilities
    available_tools: Set[str] = Field(default_factory=set)
    max_concurrent_requests: int = Field(default=5, ge=1, le=100)
    
    # Communication capabilities
    supported_message_types: Set[MessageType] = Field(
        default_factory=lambda: {MessageType.QUERY, MessageType.RESPONSE}
    )
    
    # Performance characteristics
    average_response_time_ms: Optional[float] = None
    max_context_length: int = Field(default=32000, ge=1000)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Provides common functionality including:
    - State management
    - Message handling
    - Conversation context
    - Google GenAI integration
    - Error handling and logging
    - Performance monitoring
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        role: AgentRole = AgentRole.COORDINATOR,
        capabilities: Optional[AgentCapabilities] = None,
        model_name: str = "gemini-2.0-flash-exp",
        **kwargs
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.role = role
        self.capabilities = capabilities or AgentCapabilities()
        self.model_name = model_name
        
        # State management
        self._state = AgentState.INITIALIZING
        self._state_history: List[tuple[AgentState, datetime]] = []
        self._update_state(AgentState.INITIALIZING)
        
        # Communication
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        
        # Configuration
        self.settings = get_settings()
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.agent_id[:8]}]")
        
        # GenAI client (will be initialized in startup)
        self._genai_client: Optional[genai.Client] = None
        
        # Shutdown event
        self._shutdown_event = asyncio.Event()
        
    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._state
        
    @property
    def is_healthy(self) -> bool:
        """Check if agent is in a healthy state."""
        return self._state in {AgentState.READY, AgentState.BUSY}
        
    @property
    def average_response_time(self) -> float:
        """Calculate average response time in milliseconds."""
        if self.request_count == 0:
            return 0.0
        return self.total_response_time / self.request_count
        
    def _update_state(self, new_state: AgentState) -> None:
        """Update agent state with timestamp tracking."""
        old_state = self._state
        self._state = new_state
        self._state_history.append((new_state, datetime.utcnow()))
        
        self.logger.info(f"State transition: {old_state} -> {new_state}")
        
        # Keep only last 100 state changes
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-100:]
    
    async def startup(self) -> None:
        """Initialize agent resources and prepare for operation."""
        try:
            self.logger.info(f"Starting up {self.role.value} agent {self.agent_id}")
            
            # Initialize GenAI client
            await self._initialize_genai_client()
            
            # Call custom initialization
            await self._on_startup()
            
            self._update_state(AgentState.READY)
            self.logger.info("Agent startup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Agent startup failed: {e}", exc_info=True)
            self._update_state(AgentState.ERROR)
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown agent resources."""
        try:
            self.logger.info("Starting agent shutdown")
            self._update_state(AgentState.SHUTTING_DOWN)
            
            # Signal shutdown to running tasks
            self._shutdown_event.set()
            
            # Close active conversations
            for conversation_id in list(self.active_conversations.keys()):
                await self._close_conversation(conversation_id)
            
            # Call custom cleanup
            await self._on_shutdown()
            
            self._update_state(AgentState.STOPPED)
            self.logger.info("Agent shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
            self._update_state(AgentState.ERROR)
            raise
    
    async def _initialize_genai_client(self) -> None:
        """Initialize Google GenAI client."""
        try:
            # Use API key from settings
            if not self.settings.GOOGLE_GENAI_API_KEY:
                raise ValueError("GOOGLE_GENAI_API_KEY not configured")
                
            self._genai_client = genai.Client(
                api_key=self.settings.GOOGLE_GENAI_API_KEY,
                http_options=genai.HTTPOptions(
                    timeout=30.0,
                    retry=genai.Retry(
                        initial=1.0,
                        multiplier=2.0,
                        maximum=60.0,
                        max_attempts=3
                    )
                )
            )
            
            self.logger.info("GenAI client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GenAI client: {e}")
            raise
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message and return response if needed.
        
        Args:
            message: The incoming message to process
            
        Returns:
            Response message if one is needed, None otherwise
        """
        if not self.is_healthy:
            error_msg = f"Agent {self.agent_id} is not healthy (state: {self.state})"
            self.logger.warning(error_msg)
            return AgentMessage.create_error_message(
                sender_id=self.agent_id,
                sender_role=self.role,
                recipient_id=message.sender_id,
                recipient_role=message.sender_role,
                error_type="AGENT_UNAVAILABLE",
                error_message=error_msg,
                correlation_id=message.correlation_id
            )
        
        start_time = datetime.utcnow()
        
        try:
            self._update_state(AgentState.BUSY)
            
            # Update conversation context
            await self._update_conversation_context(message)
            
            # Process the message
            response = await self._handle_message(message)
            
            # Track performance
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.request_count += 1
            self.total_response_time += response_time
            
            self.logger.debug(f"Processed message in {response_time:.2f}ms")
            
            return response
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing message: {e}", exc_info=True)
            
            return AgentMessage.create_error_message(
                sender_id=self.agent_id,
                sender_role=self.role,
                recipient_id=message.sender_id,
                recipient_role=message.sender_role,
                error_type="PROCESSING_ERROR",
                error_message=str(e),
                correlation_id=message.correlation_id
            )
        
        finally:
            if self.is_healthy:
                self._update_state(AgentState.READY)
    
    async def _update_conversation_context(self, message: AgentMessage) -> None:
        """Update or create conversation context for the message."""
        conversation_id = message.conversation_id
        
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                participants={message.sender_role, self.role}
            )
        
        context = self.active_conversations[conversation_id]
        context.message_count += 1
        context.last_activity = datetime.utcnow()
        context.message_history.append(message)
        
        # Keep only last 50 messages per conversation
        if len(context.message_history) > 50:
            context.message_history = context.message_history[-50:]
    
    async def _close_conversation(self, conversation_id: str) -> None:
        """Close and cleanup a conversation."""
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            self.logger.debug(
                f"Closing conversation {conversation_id} "
                f"({context.message_count} messages)"
            )
            del self.active_conversations[conversation_id]
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate response using Google GenAI."""
        if not self._genai_client:
            raise RuntimeError("GenAI client not initialized")
        
        try:
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuery: {prompt}"
            
            response = await self._genai_client.aio.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=genai.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_modalities=["TEXT"]
                )
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status information."""
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'state': self.state.value,
            'capabilities': self.capabilities.model_dump(),
            'performance': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(self.request_count, 1),
                'average_response_time_ms': self.average_response_time
            },
            'conversations': {
                'active_count': len(self.active_conversations),
                'conversation_ids': list(self.active_conversations.keys())
            },
            'state_history': [
                {'state': state.value, 'timestamp': timestamp.isoformat()}
                for state, timestamp in self._state_history[-10:]
            ]
        }
    
    @asynccontextmanager
    async def conversation_session(self, conversation_id: str):
        """Context manager for conversation sessions."""
        try:
            yield self.active_conversations.get(conversation_id)
        finally:
            # Cleanup if conversation is inactive
            if conversation_id in self.active_conversations:
                context = self.active_conversations[conversation_id]
                if (datetime.utcnow() - context.last_activity).total_seconds() > 300:
                    await self._close_conversation(conversation_id)
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def _on_startup(self) -> None:
        """Custom initialization logic for the agent."""
        pass
    
    @abstractmethod
    async def _on_shutdown(self) -> None:
        """Custom cleanup logic for the agent."""
        pass
    
    @abstractmethod
    async def _handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and return response."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> AgentCapabilities:
        """Return the agent's capabilities."""
        pass 