"""
Conversation Manager
===================

Manages conversation state and context for the coordinator agent.
Handles conversation history, turn management, and context preservation.

Features:
- Multi-turn conversation tracking
- Context window management
- Conversation history persistence
- Turn-based state management
- Memory optimization
- Session lifecycle management
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4

from shared.schemas.agent_communication import (
    AgentMessage, ConversationContext, MessageType
)
from shared.config.logging_config import setup_logging
from shared.utils.caching import get_cache_manager
from shared.utils.validation import validate_input, ValidationError


class ConversationTurn:
    """Represents a single conversation turn."""
    
    def __init__(
        self,
        turn_id: str,
        user_message: AgentMessage,
        assistant_response: Optional[AgentMessage] = None,
        timestamp: datetime = None
    ):
        self.turn_id = turn_id
        self.user_message = user_message
        self.assistant_response = assistant_response
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
        self.is_complete = assistant_response is not None
    
    def complete_turn(self, response: AgentMessage) -> None:
        """Mark turn as complete with assistant response."""
        self.assistant_response = response
        self.is_complete = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message.dict() if self.user_message else None,
            "assistant_response": self.assistant_response.dict() if self.assistant_response else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "is_complete": self.is_complete
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create turn from dictionary."""
        turn = cls(
            turn_id=data["turn_id"],
            user_message=AgentMessage.parse_obj(data["user_message"]) if data["user_message"] else None,
            assistant_response=AgentMessage.parse_obj(data["assistant_response"]) if data["assistant_response"] else None,
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        turn.metadata = data.get("metadata", {})
        turn.is_complete = data.get("is_complete", False)
        return turn


class ConversationSession:
    """Manages a single conversation session."""
    
    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        max_turns: int = 50,
        max_context_length: int = 8000
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns
        self.max_context_length = max_context_length
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_active = True
        self.metadata: Dict[str, Any] = {}
        self.context_data: Dict[str, Any] = {}
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a new turn to the session."""
        self.turns.append(turn)
        self.last_activity = datetime.utcnow()
        
        # Trim old turns if necessary
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def get_recent_turns(self, count: int = 10) -> List[ConversationTurn]:
        """Get the most recent turns."""
        return self.turns[-count:] if self.turns else []
    
    def get_context_summary(self) -> str:
        """Get a summary of the conversation context."""
        if not self.turns:
            return "No conversation history."
        
        recent_turns = self.get_recent_turns(5)
        summary_parts = []
        
        for turn in recent_turns:
            if turn.user_message and turn.user_message.payload:
                user_text = getattr(turn.user_message.payload, 'query_text', 'User message')
                summary_parts.append(f"User: {user_text[:100]}...")
            
            if turn.assistant_response and turn.assistant_response.payload:
                response_text = getattr(turn.assistant_response.payload, 'response_text', 'Assistant response')
                summary_parts.append(f"Assistant: {response_text[:100]}...")
        
        return "\n".join(summary_parts)
    
    def estimate_context_length(self) -> int:
        """Estimate the total context length."""
        total_length = 0
        for turn in self.turns:
            if turn.user_message and turn.user_message.payload:
                user_text = getattr(turn.user_message.payload, 'query_text', '')
                total_length += len(user_text)
            
            if turn.assistant_response and turn.assistant_response.payload:
                response_text = getattr(turn.assistant_response.payload, 'response_text', '')
                total_length += len(response_text)
        
        return total_length
    
    def needs_compression(self) -> bool:
        """Check if conversation needs context compression."""
        return self.estimate_context_length() > self.max_context_length
    
    def close_session(self) -> None:
        """Mark session as inactive."""
        self.is_active = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "max_turns": self.max_turns,
            "max_context_length": self.max_context_length,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata,
            "context_data": self.context_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Create session from dictionary."""
        session = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            max_turns=data.get("max_turns", 50),
            max_context_length=data.get("max_context_length", 8000)
        )
        
        session.turns = [ConversationTurn.from_dict(turn_data) for turn_data in data.get("turns", [])]
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.is_active = data.get("is_active", True)
        session.metadata = data.get("metadata", {})
        session.context_data = data.get("context_data", {})
        
        return session


class ConversationManager:
    """
    Manages conversation state and context for the coordinator agent.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = setup_logging("ConversationManager")
        self.cache_manager = get_cache_manager()
        
        # Active sessions
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # Configuration
        self.session_timeout = timedelta(hours=2)  # Auto-cleanup inactive sessions
        self.max_sessions = 1000
        self.compression_threshold = 0.8  # Compress when context is 80% full
        
        # Metrics
        self.total_sessions = 0
        self.total_turns = 0
        
        self.logger.info("ConversationManager initialized")
    
    async def initialize(self) -> None:
        """Initialize the conversation manager."""
        try:
            # Start cleanup task
            asyncio.create_task(self._cleanup_inactive_sessions())
            
            self.logger.info("ConversationManager initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ConversationManager: {str(e)}")
            raise
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_turns: int = 50,
        max_context_length: int = 8000
    ) -> ConversationSession:
        """Create a new conversation session."""
        try:
            if session_id is None:
                session_id = str(uuid4())
            
            # Check if session already exists
            if session_id in self.active_sessions:
                self.logger.warning(f"Session {session_id} already exists, returning existing session")
                return self.active_sessions[session_id]
            
            # Check session limit
            if len(self.active_sessions) >= self.max_sessions:
                self._cleanup_oldest_sessions(int(self.max_sessions * 0.1))  # Remove 10% oldest
            
            # Create new session
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                max_turns=max_turns,
                max_context_length=max_context_length
            )
            
            self.active_sessions[session_id] = session
            self.total_sessions += 1
            
            self.logger.info(f"Created new conversation session: {session_id}")
            return session
            
        except Exception as e:
            self.logger.error(f"Error creating session: {str(e)}")
            raise
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing conversation session."""
        return self.active_sessions.get(session_id)
    
    def get_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> ConversationSession:
        """Get existing session or create new one."""
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id, user_id, **kwargs)
        return session
    
    async def start_turn(
        self,
        session_id: str,
        user_message: AgentMessage
    ) -> ConversationTurn:
        """Start a new conversation turn."""
        try:
            session = self.get_session(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")
            
            # Create new turn
            turn_id = f"{session_id}_{len(session.turns)}"
            turn = ConversationTurn(
                turn_id=turn_id,
                user_message=user_message
            )
            
            # Check if context compression is needed
            if session.needs_compression():
                await self._compress_context(session)
            
            session.add_turn(turn)
            self.total_turns += 1
            
            self.logger.debug(f"Started new turn: {turn_id}")
            return turn
            
        except Exception as e:
            self.logger.error(f"Error starting turn: {str(e)}")
            raise
    
    async def complete_turn(
        self,
        session_id: str,
        turn_id: str,
        assistant_response: AgentMessage
    ) -> None:
        """Complete a conversation turn with assistant response."""
        try:
            session = self.get_session(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")
            
            # Find the turn
            turn = None
            for t in session.turns:
                if t.turn_id == turn_id:
                    turn = t
                    break
            
            if turn is None:
                raise ValueError(f"Turn {turn_id} not found in session {session_id}")
            
            turn.complete_turn(assistant_response)
            session.last_activity = datetime.utcnow()
            
            self.logger.debug(f"Completed turn: {turn_id}")
            
        except Exception as e:
            self.logger.error(f"Error completing turn: {str(e)}")
            raise
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a session."""
        try:
            session = self.get_session(session_id)
            if session is None:
                return None
            
            # Convert session to ConversationContext format
            messages = []
            for turn in session.turns:
                if turn.user_message:
                    messages.append(turn.user_message)
                if turn.assistant_response:
                    messages.append(turn.assistant_response)
            
            context = ConversationContext(
                conversation_id=session.session_id,
                user_id=session.user_id,
                session_id=session.session_id,
                messages=messages,
                context_data=session.context_data,
                started_at=session.created_at,
                last_activity=session.last_activity,
                is_active=session.is_active
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {str(e)}")
            return None
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the session."""
        try:
            session = self.get_session(session_id)
            if session is None:
                return {"error": "Session not found"}
            
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "turn_count": len(session.turns),
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "is_active": session.is_active,
                "context_length": session.estimate_context_length(),
                "context_summary": session.get_context_summary(),
                "needs_compression": session.needs_compression()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session summary: {str(e)}")
            return {"error": str(e)}
    
    async def close_session(self, session_id: str) -> None:
        """Close and optionally persist a conversation session."""
        try:
            session = self.get_session(session_id)
            if session is None:
                self.logger.warning(f"Attempted to close non-existent session: {session_id}")
                return
            
            session.close_session()
            
            # Optionally persist to cache for recovery
            if self.cache_manager and session.turns:
                await self.cache_manager.set(
                    f"conversation_session:{session_id}",
                    session.to_dict(),
                    ttl=3600  # Keep for 1 hour after closure
                )
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            self.logger.info(f"Closed conversation session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")
            raise
    
    async def _compress_context(self, session: ConversationSession) -> None:
        """Compress conversation context to manage memory."""
        try:
            if len(session.turns) <= 5:
                return  # Don't compress very short conversations
            
            # Keep the most recent turns and summarize older ones
            recent_turns = session.turns[-10:]  # Keep last 10 turns
            older_turns = session.turns[:-10]
            
            if older_turns:
                # Create summary of older turns
                summary_parts = []
                for turn in older_turns:
                    if turn.user_message and turn.user_message.payload:
                        user_text = getattr(turn.user_message.payload, 'query_text', '')
                        summary_parts.append(f"User asked about: {user_text[:50]}...")
                    
                    if turn.assistant_response and turn.assistant_response.payload:
                        response_text = getattr(turn.assistant_response.payload, 'response_text', '')
                        summary_parts.append(f"Assistant responded about: {response_text[:50]}...")
                
                # Store summary in session metadata
                session.metadata["conversation_summary"] = "\n".join(summary_parts)
                session.metadata["compressed_turns"] = len(older_turns)
                session.metadata["compressed_at"] = datetime.utcnow().isoformat()
            
            # Replace turns with recent ones only
            session.turns = recent_turns
            
            self.logger.info(f"Compressed context for session {session.session_id}")
            
        except Exception as e:
            self.logger.error(f"Error compressing context: {str(e)}")
    
    def _cleanup_oldest_sessions(self, count: int) -> None:
        """Remove the oldest inactive sessions."""
        try:
            # Sort by last activity
            sessions_by_activity = sorted(
                self.active_sessions.items(),
                key=lambda x: x[1].last_activity
            )
            
            removed_count = 0
            for session_id, session in sessions_by_activity:
                if not session.is_active and removed_count < count:
                    del self.active_sessions[session_id]
                    removed_count += 1
            
            self.logger.info(f"Cleaned up {removed_count} oldest sessions")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up sessions: {str(e)}")
    
    async def _cleanup_inactive_sessions(self) -> None:
        """Background task to cleanup inactive sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.utcnow() - self.session_timeout
                sessions_to_remove = []
                
                for session_id, session in self.active_sessions.items():
                    if session.last_activity < cutoff_time:
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    await self.close_session(session_id)
                
                if sessions_to_remove:
                    self.logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the conversation manager."""
        try:
            return {
                "status": "healthy",
                "active_sessions": len(self.active_sessions),
                "total_sessions": self.total_sessions,
                "total_turns": self.total_turns,
                "memory_usage": {
                    "sessions": len(self.active_sessions),
                    "max_sessions": self.max_sessions
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the conversation manager."""
        try:
            self.logger.info("Shutting down ConversationManager...")
            
            # Close all active sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.close_session(session_id)
            
            self.logger.info("ConversationManager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise 