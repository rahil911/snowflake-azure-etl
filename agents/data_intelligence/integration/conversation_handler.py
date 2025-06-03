"""
Conversation Handler for Data Intelligence Agent

This module maintains conversation context and manages multi-turn interactions,
ensuring continuity and context awareness across user sessions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from pydantic import BaseModel, Field
from shared.config.settings import Settings
from shared.utils.caching import get_cache_manager
from shared.utils.metrics import get_metrics_collector, track_performance

logger = logging.getLogger(__name__)

class MessageRole(Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationState(Enum):
    """Conversation state types."""
    ACTIVE = "active"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    EXPIRED = "expired"

class ContextType(Enum):
    """Types of conversation context."""
    QUERY_HISTORY = "query_history"
    DATA_CONTEXT = "data_context"
    BUSINESS_CONTEXT = "business_context"
    USER_PREFERENCES = "user_preferences"
    SESSION_METADATA = "session_metadata"

@dataclass
class ConversationMessage:
    """Individual message in conversation."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_references: List[str] = field(default_factory=list)

@dataclass
class ConversationContext:
    """Context information for conversation."""
    context_type: ContextType
    data: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    relevance_score: float = 1.0
    access_count: int = 0

@dataclass
class ConversationSession:
    """Complete conversation session."""
    session_id: str
    user_id: Optional[str]
    messages: List[ConversationMessage]
    contexts: Dict[str, ConversationContext]
    state: ConversationState
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConversationRequest(BaseModel):
    """Request for conversation handling."""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    message: str
    message_type: str = "query"
    context_hints: Dict[str, Any] = Field(default_factory=dict)
    preserve_context: bool = True
    max_context_length: int = 10
    
    class Config:
        validate_assignment = True

class ConversationResponse(BaseModel):
    """Response from conversation handling."""
    session_id: str
    response_message: str
    conversation_state: str
    context_used: List[str] = Field(default_factory=list)
    follow_up_suggestions: List[str] = Field(default_factory=list)
    clarification_needed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True

class ConversationHandler:
    """
    Conversation Handler for managing multi-turn conversations.
    
    Maintains conversation context, handles follow-up queries,
    and provides intelligent conversation flow management.
    """
    
    def __init__(self, settings: Settings):
        """Initialize conversation handler."""
        self.settings = settings
        self.cache = get_cache_manager()
        self.metrics = get_metrics_collector()
        
        # Configuration
        self.config = {
            'session_timeout': settings.CONVERSATION.get('session_timeout', 3600),  # 1 hour
            'max_sessions': settings.CONVERSATION.get('max_sessions', 1000),
            'context_retention': settings.CONVERSATION.get('context_retention', 86400),  # 24 hours
            'max_message_history': settings.CONVERSATION.get('max_message_history', 50),
            'auto_cleanup_interval': settings.CONVERSATION.get('auto_cleanup_interval', 300)  # 5 minutes
        }
        
        # Active sessions storage
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # Context management
        self._setup_context_rules()
        self._setup_clarification_patterns()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.info("Conversation Handler initialized")

    def _setup_context_rules(self):
        """Setup context retention and relevance rules."""
        self.context_rules = {
            ContextType.QUERY_HISTORY: {
                'retention_hours': 24,
                'max_items': 20,
                'relevance_decay': 0.9,  # Decay factor per hour
                'priority': 1.0
            },
            ContextType.DATA_CONTEXT: {
                'retention_hours': 12,
                'max_items': 10,
                'relevance_decay': 0.8,
                'priority': 0.9
            },
            ContextType.BUSINESS_CONTEXT: {
                'retention_hours': 48,
                'max_items': 15,
                'relevance_decay': 0.95,
                'priority': 0.8
            },
            ContextType.USER_PREFERENCES: {
                'retention_hours': 168,  # 1 week
                'max_items': 50,
                'relevance_decay': 0.99,
                'priority': 0.7
            }
        }

    def _setup_clarification_patterns(self):
        """Setup patterns that indicate need for clarification."""
        self.clarification_patterns = {
            'ambiguous_terms': ['which', 'what kind', 'what type', 'specific'],
            'incomplete_queries': ['more about', 'details on', 'expand', 'elaborate'],
            'conflicting_context': ['but', 'however', 'instead', 'different'],
            'missing_parameters': ['all', 'everything', 'any', 'some']
        }

    @track_performance(tags={"operation": "handle_conversation"})
    async def handle_conversation(self, request: ConversationRequest) -> ConversationResponse:
        """
        Handle conversation turn with context awareness.
        
        Args:
            request: Conversation request with message and context
            
        Returns:
            Conversation response with context-aware reply
        """
        try:
            # Get or create session
            session = await self._get_or_create_session(request.session_id, request.user_id)
            
            # Add user message to conversation
            user_message = ConversationMessage(
                id=str(uuid.uuid4()),
                role=MessageRole.USER,
                content=request.message,
                timestamp=datetime.now(),
                metadata={'message_type': request.message_type}
            )
            session.messages.append(user_message)
            
            # Update session activity
            session.last_activity = datetime.now()
            session.state = ConversationState.PROCESSING
            
            # Extract and update context
            await self._extract_and_update_context(session, request)
            
            # Analyze message for clarification needs
            clarification_needed = self._analyze_clarification_needs(request.message, session)
            
            # Get relevant context
            relevant_context = await self._get_relevant_context(session, request.message)
            
            # Generate context-aware response
            response_content = await self._generate_contextual_response(
                request.message, relevant_context, session, clarification_needed
            )
            
            # Add assistant response to conversation
            assistant_message = ConversationMessage(
                id=str(uuid.uuid4()),
                role=MessageRole.ASSISTANT,
                content=response_content,
                timestamp=datetime.now(),
                context_references=[ctx.context_type.value for ctx in relevant_context]
            )
            session.messages.append(assistant_message)
            
            # Update session state
            session.state = ConversationState.WAITING_FOR_CLARIFICATION if clarification_needed else ConversationState.ACTIVE
            
            # Generate follow-up suggestions
            follow_ups = await self._generate_follow_up_suggestions(session, relevant_context)
            
            # Trim conversation history if needed
            await self._trim_conversation_history(session)
            
            # Record metrics
            await self._record_conversation_metrics(session, request, len(relevant_context))
            
            return ConversationResponse(
                session_id=session.session_id,
                response_message=response_content,
                conversation_state=session.state.value,
                context_used=[ctx.context_type.value for ctx in relevant_context],
                follow_up_suggestions=follow_ups,
                clarification_needed=clarification_needed,
                metadata={
                    'message_count': len(session.messages),
                    'context_count': len(session.contexts),
                    'session_duration': (session.last_activity - session.created_at).total_seconds()
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling conversation: {str(e)}")
            return self._create_error_response(str(e), request.session_id)

    async def _get_or_create_session(self, session_id: Optional[str], user_id: Optional[str]) -> ConversationSession:
        """Get existing session or create new one."""
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            # Check if session is expired
            if self._is_session_expired(session):
                del self.active_sessions[session_id]
                session_id = None
        
        if not session_id or session_id not in self.active_sessions:
            # Create new session
            session_id = str(uuid.uuid4())
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                messages=[],
                contexts={},
                state=ConversationState.ACTIVE,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            self.active_sessions[session_id] = session
            
            # Clean up old sessions if limit reached
            if len(self.active_sessions) > self.config['max_sessions']:
                await self._cleanup_oldest_sessions()
        
        return self.active_sessions[session_id]

    def _is_session_expired(self, session: ConversationSession) -> bool:
        """Check if session has expired."""
        timeout = timedelta(seconds=self.config['session_timeout'])
        return datetime.now() - session.last_activity > timeout

    async def _extract_and_update_context(self, session: ConversationSession, request: ConversationRequest):
        """Extract and update conversation context."""
        # Extract query context
        if request.message:
            query_context = ConversationContext(
                context_type=ContextType.QUERY_HISTORY,
                data={
                    'query': request.message,
                    'timestamp': datetime.now().isoformat(),
                    'message_type': request.message_type
                },
                created_at=datetime.now()
            )
            session.contexts[f"query_{datetime.now().timestamp()}"] = query_context
        
        # Extract data context from hints
        if request.context_hints:
            for hint_type, hint_data in request.context_hints.items():
                if hint_type == 'data_context':
                    data_context = ConversationContext(
                        context_type=ContextType.DATA_CONTEXT,
                        data=hint_data,
                        created_at=datetime.now()
                    )
                    session.contexts[f"data_{datetime.now().timestamp()}"] = data_context
                
                elif hint_type == 'business_context':
                    business_context = ConversationContext(
                        context_type=ContextType.BUSINESS_CONTEXT,
                        data=hint_data,
                        created_at=datetime.now()
                    )
                    session.contexts[f"business_{datetime.now().timestamp()}"] = business_context
        
        # Clean up expired contexts
        await self._cleanup_expired_contexts(session)

    def _analyze_clarification_needs(self, message: str, session: ConversationSession) -> bool:
        """Analyze if message needs clarification."""
        message_lower = message.lower()
        
        # Check for ambiguous terms
        for pattern in self.clarification_patterns['ambiguous_terms']:
            if pattern in message_lower:
                return True
        
        # Check for incomplete queries
        for pattern in self.clarification_patterns['incomplete_queries']:
            if pattern in message_lower:
                return True
        
        # Check if query is too short and generic
        if len(message.split()) < 3 and any(word in message_lower for word in ['show', 'get', 'find', 'what']):
            return True
        
        # Check for conflicting context with previous messages
        if len(session.messages) > 1:
            recent_messages = [msg.content.lower() for msg in session.messages[-3:] if msg.role == MessageRole.USER]
            for recent_msg in recent_messages:
                for conflict_pattern in self.clarification_patterns['conflicting_context']:
                    if conflict_pattern in message_lower and conflict_pattern not in recent_msg:
                        return True
        
        return False

    async def _get_relevant_context(self, session: ConversationSession, current_message: str) -> List[ConversationContext]:
        """Get relevant context for current message."""
        relevant_contexts = []
        current_time = datetime.now()
        
        for context_id, context in session.contexts.items():
            # Check if context is expired
            rules = self.context_rules.get(context.context_type, {})
            retention_hours = rules.get('retention_hours', 24)
            
            if (current_time - context.created_at).total_seconds() > retention_hours * 3600:
                continue
            
            # Calculate relevance score
            relevance_score = self._calculate_context_relevance(context, current_message, current_time)
            
            if relevance_score > 0.1:  # Minimum relevance threshold
                context.relevance_score = relevance_score
                context.access_count += 1
                relevant_contexts.append(context)
        
        # Sort by relevance and priority
        relevant_contexts.sort(
            key=lambda ctx: (
                ctx.relevance_score * self.context_rules.get(ctx.context_type, {}).get('priority', 1.0)
            ),
            reverse=True
        )
        
        return relevant_contexts[:10]  # Limit to top 10 most relevant

    def _calculate_context_relevance(self, context: ConversationContext, current_message: str, current_time: datetime) -> float:
        """Calculate relevance score for context."""
        base_score = 1.0
        
        # Time decay
        hours_elapsed = (current_time - context.created_at).total_seconds() / 3600
        decay_factor = self.context_rules.get(context.context_type, {}).get('relevance_decay', 0.9)
        time_score = decay_factor ** hours_elapsed
        
        # Content similarity (simple keyword matching)
        content_score = self._calculate_content_similarity(context.data, current_message)
        
        # Access frequency boost
        access_boost = min(1.2, 1.0 + (context.access_count * 0.05))
        
        return base_score * time_score * content_score * access_boost

    def _calculate_content_similarity(self, context_data: Dict[str, Any], message: str) -> float:
        """Calculate content similarity between context and message."""
        message_words = set(message.lower().split())
        
        # Extract words from context data
        context_text = ""
        if isinstance(context_data, dict):
            for value in context_data.values():
                if isinstance(value, str):
                    context_text += " " + value
        elif isinstance(context_data, str):
            context_text = context_data
        
        context_words = set(context_text.lower().split())
        
        # Calculate Jaccard similarity
        if not context_words or not message_words:
            return 0.1
        
        intersection = len(message_words.intersection(context_words))
        union = len(message_words.union(context_words))
        
        return intersection / union if union > 0 else 0.1

    async def _generate_contextual_response(self, message: str, contexts: List[ConversationContext], session: ConversationSession, needs_clarification: bool) -> str:
        """Generate response using conversation context."""
        if needs_clarification:
            return await self._generate_clarification_request(message, contexts, session)
        
        # Build context-aware response
        response_parts = []
        
        # Acknowledge context if available
        if contexts:
            recent_queries = [ctx for ctx in contexts if ctx.context_type == ContextType.QUERY_HISTORY]
            if recent_queries and len(session.messages) > 2:
                response_parts.append("Based on our previous discussion,")
        
        # Add main response content
        response_parts.append("I'll help you analyze the data.")
        
        # Add context-specific insights
        data_contexts = [ctx for ctx in contexts if ctx.context_type == ContextType.DATA_CONTEXT]
        if data_contexts:
            response_parts.append("I'll use the data context from our conversation to provide relevant insights.")
        
        business_contexts = [ctx for ctx in contexts if ctx.context_type == ContextType.BUSINESS_CONTEXT]
        if business_contexts:
            response_parts.append("I'll consider the business context we discussed earlier.")
        
        return " ".join(response_parts)

    async def _generate_clarification_request(self, message: str, contexts: List[ConversationContext], session: ConversationSession) -> str:
        """Generate clarification request for ambiguous queries."""
        clarification_parts = ["I'd like to better understand your request."]
        
        # Analyze what needs clarification
        message_lower = message.lower()
        
        if any(term in message_lower for term in ['which', 'what kind', 'what type']):
            clarification_parts.append("Could you specify which particular aspect you're interested in?")
        
        elif len(message.split()) < 3:
            clarification_parts.append("Could you provide more details about what you'd like to analyze?")
        
        elif any(term in message_lower for term in ['all', 'everything', 'any']):
            clarification_parts.append("That's quite broad. Could you narrow down the scope?")
        
        # Suggest based on previous context
        recent_contexts = [ctx for ctx in contexts if ctx.context_type in [ContextType.QUERY_HISTORY, ContextType.DATA_CONTEXT]]
        if recent_contexts:
            clarification_parts.append("Are you looking for something similar to what we discussed earlier, or something different?")
        
        return " ".join(clarification_parts)

    async def _generate_follow_up_suggestions(self, session: ConversationSession, contexts: List[ConversationContext]) -> List[str]:
        """Generate follow-up suggestions based on conversation context."""
        suggestions = []
        
        # Analyze recent conversation patterns
        recent_messages = session.messages[-5:] if len(session.messages) >= 5 else session.messages
        user_messages = [msg for msg in recent_messages if msg.role == MessageRole.USER]
        
        if user_messages:
            last_query = user_messages[-1].content.lower()
            
            # Suggest drill-downs
            if 'summary' in last_query or 'overview' in last_query:
                suggestions.append("Would you like to see detailed analysis of specific metrics?")
            
            if 'sales' in last_query:
                suggestions.extend([
                    "Show me sales trends over time",
                    "Compare sales performance by region",
                    "Analyze customer segments"
                ])
            
            if 'performance' in last_query:
                suggestions.extend([
                    "Show top and bottom performers",
                    "Identify performance drivers",
                    "Compare with industry benchmarks"
                ])
        
        # Context-based suggestions
        data_contexts = [ctx for ctx in contexts if ctx.context_type == ContextType.DATA_CONTEXT]
        if data_contexts:
            suggestions.append("Analyze data quality and completeness")
        
        # Limit suggestions
        return suggestions[:3]

    async def _trim_conversation_history(self, session: ConversationSession):
        """Trim conversation history to maintain performance."""
        max_messages = self.config['max_message_history']
        
        if len(session.messages) > max_messages:
            # Keep system messages and recent messages
            system_messages = [msg for msg in session.messages if msg.role == MessageRole.SYSTEM]
            recent_messages = session.messages[-(max_messages - len(system_messages)):]
            session.messages = system_messages + recent_messages

    async def _cleanup_expired_contexts(self, session: ConversationSession):
        """Clean up expired contexts from session."""
        current_time = datetime.now()
        expired_contexts = []
        
        for context_id, context in session.contexts.items():
            rules = self.context_rules.get(context.context_type, {})
            retention_hours = rules.get('retention_hours', 24)
            
            if (current_time - context.created_at).total_seconds() > retention_hours * 3600:
                expired_contexts.append(context_id)
        
        for context_id in expired_contexts:
            del session.contexts[context_id]

    async def _cleanup_expired_sessions(self):
        """Periodic cleanup of expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.config['auto_cleanup_interval'])
                
                expired_sessions = []
                for session_id, session in self.active_sessions.items():
                    if self._is_session_expired(session):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
            except Exception as e:
                logger.error(f"Error during session cleanup: {str(e)}")

    async def _cleanup_oldest_sessions(self):
        """Clean up oldest sessions when limit is reached."""
        sessions_to_remove = len(self.active_sessions) - self.config['max_sessions'] + 10
        
        if sessions_to_remove > 0:
            # Sort sessions by last activity
            sorted_sessions = sorted(
                self.active_sessions.items(),
                key=lambda x: x[1].last_activity
            )
            
            for session_id, _ in sorted_sessions[:sessions_to_remove]:
                del self.active_sessions[session_id]
            
            logger.info(f"Cleaned up {sessions_to_remove} oldest sessions")

    def _create_error_response(self, error_message: str, session_id: Optional[str]) -> ConversationResponse:
        """Create error response for conversation handling failures."""
        return ConversationResponse(
            session_id=session_id or "error_session",
            response_message=f"I encountered an error while processing your request: {error_message}",
            conversation_state=ConversationState.ERROR.value,
            clarification_needed=False,
            metadata={'error': True, 'error_message': error_message}
        )

    async def _record_conversation_metrics(self, session: ConversationSession, request: ConversationRequest, context_count: int):
        """Record conversation metrics."""
        try:
            metrics_data = {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'message_count': len(session.messages),
                'context_count': context_count,
                'session_duration': (session.last_activity - session.created_at).total_seconds(),
                'message_type': request.message_type,
                'state': session.state.value
            }
            
            await self.metrics.record_event('conversation_turn', metrics_data)
        except Exception as e:
            logger.warning(f"Error recording metrics: {str(e)}")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'state': session.state.value,
            'message_count': len(session.messages),
            'context_count': len(session.contexts),
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'duration_seconds': (session.last_activity - session.created_at).total_seconds()
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of conversation handler."""
        return {
            'service': 'conversation_handler',
            'status': 'healthy',
            'active_sessions': len(self.active_sessions),
            'max_sessions': self.config['max_sessions'],
            'session_timeout': self.config['session_timeout'],
            'context_types': len(self.context_rules),
            'cache_enabled': self.cache is not None,
            'metrics_enabled': self.metrics is not None,
            'timestamp': datetime.now().isoformat()
        } 