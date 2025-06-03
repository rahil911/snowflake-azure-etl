"""
Coordinator Client for Data Intelligence Agent

This module provides integration with the coordinator agent, handling
communication and coordination between data intelligence and coordinator services.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from shared.config.settings import Settings
from shared.schemas.agent_communication import AgentMessage, MessageType
from shared.utils.model_bus import AgentBusInterface
from shared.utils.caching import get_cache_manager
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.validation import validate_input, ValidationError

logger = logging.getLogger(__name__)

class RequestType(Enum):
    """Types of requests to coordinator."""
    QUERY_PROCESSING = "query_processing"
    INSIGHT_DELIVERY = "insight_delivery"
    RECOMMENDATION_REQUEST = "recommendation_request"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    HEALTH_CHECK = "health_check"

class Priority(Enum):
    """Priority levels for coordinator requests."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class CoordinatorRequest:
    """Request to send to coordinator agent."""
    request_type: RequestType
    priority: Priority
    payload: Dict[str, Any]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoordinatorResponse:
    """Response from coordinator agent."""
    success: bool
    data: Dict[str, Any]
    message: str = ""
    error_code: Optional[str] = None
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class MessageProcessor(BaseModel):
    """Configuration for processing different message types."""
    handlers: Dict[str, str] = Field(default_factory=dict)
    filters: List[str] = Field(default_factory=list)
    transformers: List[str] = Field(default_factory=list)
    
    class Config:
        validate_assignment = True

class CoordinatorClient:
    """
    Client for communicating with the coordinator agent.
    
    Handles all communication between data intelligence agent and
    the coordinator, including query processing, insight delivery,
    and status updates.
    """
    
    def __init__(self, settings: Settings):
        """Initialize coordinator client."""
        self.settings = settings
        self.model_bus = AgentBusInterface(settings)
        self.cache = get_cache_manager()
        self.metrics = get_metrics_collector()
        
        # Configuration
        self.config = {
            'coordinator_agent_id': settings.COORDINATOR.get('agent_id', 'coordinator_agent'),
            'timeout': settings.COORDINATOR.get('timeout', 30),
            'retry_attempts': settings.COORDINATOR.get('retry_attempts', 3),
            'heartbeat_interval': settings.COORDINATOR.get('heartbeat_interval', 60),
            'cache_responses': settings.COORDINATOR.get('cache_responses', True),
            'cache_ttl': settings.COORDINATOR.get('cache_ttl', 300)
        }
        
        # Connection state
        self.connected = False
        self.last_heartbeat = None
        self.pending_requests = {}
        self.response_handlers = {}
        
        # Message processing
        self._setup_message_handlers()
        
        logger.info("Coordinator Client initialized")

    def _setup_message_handlers(self):
        """Setup message handlers for different coordinator responses."""
        self.message_handlers = {
            'query_response': self._handle_query_response,
            'insight_acknowledgment': self._handle_insight_acknowledgment,
            'recommendation_request': self._handle_recommendation_request,
            'status_request': self._handle_status_request,
            'error_notification': self._handle_error_notification,
            'health_check_response': self._handle_health_check_response
        }

    async def connect(self) -> bool:
        """Connect to coordinator agent."""
        try:
            # Register with model bus
            await self.model_bus.register_agent(
                agent_id="data_intelligence_agent",
                agent_type="data_intelligence",
                capabilities=["query_processing", "insight_generation", "data_analysis"]
            )
            
            # Send initial connection message
            connection_message = AgentMessage(
                message_type=MessageType.STATUS,
                source_agent="data_intelligence_agent",
                target_agent=self.config['coordinator_agent_id'],
                payload={
                    'action': 'connect',
                    'capabilities': [
                        'natural_language_queries',
                        'business_insights',
                        'pattern_detection',
                        'data_quality_analysis',
                        'recommendations'
                    ],
                    'status': 'ready'
                }
            )
            
            response = await self._send_message(connection_message)
            
            if response and response.success:
                self.connected = True
                self.last_heartbeat = datetime.now()
                logger.info("Successfully connected to coordinator agent")
                
                # Start heartbeat task
                asyncio.create_task(self._heartbeat_loop())
                
                return True
            else:
                logger.error("Failed to connect to coordinator agent")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to coordinator: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from coordinator agent."""
        try:
            if self.connected:
                # Send disconnect message
                disconnect_message = AgentMessage(
                    message_type=MessageType.STATUS,
                    source_agent="data_intelligence_agent",
                    target_agent=self.config['coordinator_agent_id'],
                    payload={'action': 'disconnect', 'status': 'disconnecting'}
                )
                
                await self._send_message(disconnect_message)
                
            self.connected = False
            await self.model_bus.unregister_agent("data_intelligence_agent")
            logger.info("Disconnected from coordinator agent")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")

    @track_performance(tags={"operation": "send_query_result"})
    async def send_query_result(self, query: str, result: Dict[str, Any], session_id: str, user_id: Optional[str] = None) -> CoordinatorResponse:
        """
        Send query processing result to coordinator.
        
        Args:
            query: Original user query
            result: Processing result data
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Response from coordinator
        """
        try:
            request = CoordinatorRequest(
                request_type=RequestType.QUERY_PROCESSING,
                priority=Priority.NORMAL,
                payload={
                    'query': query,
                    'result': result,
                    'processing_complete': True,
                    'timestamp': datetime.now().isoformat()
                },
                session_id=session_id,
                user_id=user_id
            )
            
            return await self._send_coordinator_request(request)
            
        except Exception as e:
            logger.error(f"Error sending query result: {str(e)}")
            return CoordinatorResponse(
                success=False,
                data={},
                message=f"Failed to send query result: {str(e)}",
                error_code="SEND_ERROR"
            )

    @track_performance(tags={"operation": "send_insights"})
    async def send_insights(self, insights: List[Dict[str, Any]], session_id: str, user_id: Optional[str] = None) -> CoordinatorResponse:
        """
        Send business insights to coordinator.
        
        Args:
            insights: List of generated insights
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Response from coordinator
        """
        try:
            request = CoordinatorRequest(
                request_type=RequestType.INSIGHT_DELIVERY,
                priority=Priority.HIGH,
                payload={
                    'insights': insights,
                    'insight_count': len(insights),
                    'generated_at': datetime.now().isoformat()
                },
                session_id=session_id,
                user_id=user_id
            )
            
            return await self._send_coordinator_request(request)
            
        except Exception as e:
            logger.error(f"Error sending insights: {str(e)}")
            return CoordinatorResponse(
                success=False,
                data={},
                message=f"Failed to send insights: {str(e)}",
                error_code="SEND_ERROR"
            )

    @track_performance(tags={"operation": "request_recommendations"})
    async def request_recommendations(self, context: Dict[str, Any], session_id: str, user_id: Optional[str] = None) -> CoordinatorResponse:
        """
        Request recommendation generation from coordinator.
        
        Args:
            context: Context for recommendation generation
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Response from coordinator
        """
        try:
            request = CoordinatorRequest(
                request_type=RequestType.RECOMMENDATION_REQUEST,
                priority=Priority.NORMAL,
                payload={
                    'context': context,
                    'request_type': 'recommendation_generation',
                    'timestamp': datetime.now().isoformat()
                },
                session_id=session_id,
                user_id=user_id
            )
            
            return await self._send_coordinator_request(request)
            
        except Exception as e:
            logger.error(f"Error requesting recommendations: {str(e)}")
            return CoordinatorResponse(
                success=False,
                data={},
                message=f"Failed to request recommendations: {str(e)}",
                error_code="REQUEST_ERROR"
            )

    async def send_status_update(self, status: str, details: Dict[str, Any] = None) -> CoordinatorResponse:
        """
        Send status update to coordinator.
        
        Args:
            status: Current status
            details: Additional status details
            
        Returns:
            Response from coordinator
        """
        try:
            request = CoordinatorRequest(
                request_type=RequestType.STATUS_UPDATE,
                priority=Priority.LOW,
                payload={
                    'status': status,
                    'details': details or {},
                    'agent_id': 'data_intelligence_agent',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return await self._send_coordinator_request(request)
            
        except Exception as e:
            logger.error(f"Error sending status update: {str(e)}")
            return CoordinatorResponse(
                success=False,
                data={},
                message=f"Failed to send status update: {str(e)}",
                error_code="STATUS_ERROR"
            )

    async def report_error(self, error: Exception, context: Dict[str, Any] = None) -> CoordinatorResponse:
        """
        Report error to coordinator.
        
        Args:
            error: Exception that occurred
            context: Error context information
            
        Returns:
            Response from coordinator
        """
        try:
            request = CoordinatorRequest(
                request_type=RequestType.ERROR_REPORT,
                priority=Priority.HIGH,
                payload={
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'context': context or {},
                    'agent_id': 'data_intelligence_agent',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return await self._send_coordinator_request(request)
            
        except Exception as e:
            logger.error(f"Error reporting error: {str(e)}")
            return CoordinatorResponse(
                success=False,
                data={},
                message=f"Failed to report error: {str(e)}",
                error_code="REPORT_ERROR"
            )

    async def _send_coordinator_request(self, request: CoordinatorRequest) -> CoordinatorResponse:
        """Send request to coordinator agent."""
        start_time = datetime.now()
        
        try:
            # Create agent message
            message = AgentMessage(
                message_type=MessageType.QUERY if request.request_type == RequestType.QUERY_PROCESSING else MessageType.STATUS,
                source_agent="data_intelligence_agent",
                target_agent=self.config['coordinator_agent_id'],
                payload=request.payload,
                metadata={
                    'request_type': request.request_type.value,
                    'priority': request.priority.value,
                    'session_id': request.session_id,
                    'user_id': request.user_id,
                    **request.metadata
                }
            )
            
            # Send message
            response = await self._send_message(message)
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            await self._record_request_metrics(request, response, response_time)
            
            if response and response.success:
                return CoordinatorResponse(
                    success=True,
                    data=response.data,
                    message=response.message,
                    response_time=response_time
                )
            else:
                error_msg = response.message if response else "No response received"
                return CoordinatorResponse(
                    success=False,
                    data={},
                    message=error_msg,
                    error_code="COORDINATOR_ERROR",
                    response_time=response_time
                )
                
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error sending coordinator request: {str(e)}")
            return CoordinatorResponse(
                success=False,
                data={},
                message=f"Request failed: {str(e)}",
                error_code="SEND_ERROR",
                response_time=response_time
            )

    async def _send_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Send message through model bus."""
        try:
            response = await self.model_bus.send_message(message)
            return response
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return None

    async def _heartbeat_loop(self):
        """Maintain heartbeat with coordinator."""
        while self.connected:
            try:
                await asyncio.sleep(self.config['heartbeat_interval'])
                
                if self.connected:
                    heartbeat_message = AgentMessage(
                        message_type=MessageType.STATUS,
                        source_agent="data_intelligence_agent",
                        target_agent=self.config['coordinator_agent_id'],
                        payload={'action': 'heartbeat', 'status': 'active'}
                    )
                    
                    response = await self._send_message(heartbeat_message)
                    
                    if response and response.success:
                        self.last_heartbeat = datetime.now()
                    else:
                        logger.warning("Heartbeat failed - coordinator may be unavailable")
                        
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")

    # Message handlers
    async def _handle_query_response(self, message: AgentMessage):
        """Handle query response from coordinator."""
        logger.info("Received query response from coordinator")
        # Process query response if needed

    async def _handle_insight_acknowledgment(self, message: AgentMessage):
        """Handle insight acknowledgment from coordinator."""
        logger.info("Insights acknowledged by coordinator")
        # Process acknowledgment if needed

    async def _handle_recommendation_request(self, message: AgentMessage):
        """Handle recommendation request from coordinator."""
        logger.info("Received recommendation request from coordinator")
        # Process recommendation request if needed

    async def _handle_status_request(self, message: AgentMessage):
        """Handle status request from coordinator."""
        logger.info("Received status request from coordinator")
        # Respond with current status
        status_response = AgentMessage(
            message_type=MessageType.RESPONSE,
            source_agent="data_intelligence_agent",
            target_agent=message.source_agent,
            payload={
                'status': 'active',
                'connected': self.connected,
                'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
                'pending_requests': len(self.pending_requests)
            }
        )
        await self._send_message(status_response)

    async def _handle_error_notification(self, message: AgentMessage):
        """Handle error notification from coordinator."""
        logger.warning(f"Received error notification from coordinator: {message.payload}")
        # Process error notification if needed

    async def _handle_health_check_response(self, message: AgentMessage):
        """Handle health check response from coordinator."""
        logger.info("Received health check response from coordinator")
        # Process health check response if needed

    async def _record_request_metrics(self, request: CoordinatorRequest, response: Optional[AgentMessage], response_time: float):
        """Record request metrics."""
        try:
            metrics_data = {
                'request_type': request.request_type.value,
                'priority': request.priority.value,
                'success': response.success if response else False,
                'response_time': response_time,
                'session_id': request.session_id,
                'user_id': request.user_id
            }
            
            await self.metrics.record_event('coordinator_request', metrics_data)
        except Exception as e:
            logger.warning(f"Error recording metrics: {str(e)}")

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            'connected': self.connected,
            'coordinator_agent_id': self.config['coordinator_agent_id'],
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'pending_requests': len(self.pending_requests),
            'config': {
                'timeout': self.config['timeout'],
                'retry_attempts': self.config['retry_attempts'],
                'heartbeat_interval': self.config['heartbeat_interval']
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of coordinator client."""
        connection_age = None
        if self.connected and self.last_heartbeat:
            connection_age = (datetime.now() - self.last_heartbeat).total_seconds()
        
        return {
            'service': 'coordinator_client',
            'status': 'healthy' if self.connected else 'disconnected',
            'connected': self.connected,
            'connection_age_seconds': connection_age,
            'coordinator_agent': self.config['coordinator_agent_id'],
            'model_bus_connected': self.model_bus is not None,
            'timestamp': datetime.now().isoformat()
        } 