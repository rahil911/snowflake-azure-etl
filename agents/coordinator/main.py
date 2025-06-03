#!/usr/bin/env python3
"""
Coordinator Agent - Main Entry Point
===================================

Main coordinator agent that serves as the primary interface for the multi-agent platform.
Extends BaseAgent from Session A and integrates with MCP servers from Session B.

Features:
- Gemini 2.0 Live API integration for real-time voice conversations
- Multi-agent orchestration and workflow management
- MCP server integration (Snowflake, Analytics)
- Natural language processing and intent classification
- Context-aware conversation management
- Enterprise-grade error handling and monitoring

This is the primary user-facing component that coordinates all other agents.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4

# Session A Foundation imports
from shared.base.agent_base import BaseAgent
from shared.schemas.agent_communication import (
    AgentMessage, MessageType, AgentRole, Priority,
    create_query_message, create_response_message, create_error_message,
    ConversationContext
)
from shared.config.settings import get_settings
from shared.config.logging_config import setup_logging
from shared.utils.metrics import get_metrics_collector
from shared.utils.model_bus import AgentBusInterface

# Coordinator components
from .conversation_manager import ConversationManager
from .intent_classifier import IntentClassifier
from .response_synthesizer import ResponseSynthesizer
from .gemini.live_api_client import GeminiLiveAPIClient
from .gemini.multimodal_processor import MultimodalProcessor
from .gemini.voice_synthesizer import VoiceSynthesizer
from .orchestration.workflow_manager import WorkflowManager
from .orchestration.agent_router import AgentRouter
from .orchestration.context_aggregator import ContextAggregator
from .mcp.server_connector import MCPServerConnector
from .mcp.tool_executor import ToolExecutor
from .mcp.result_processor import ResultProcessor


class CoordinatorAgent(BaseAgent):
    """
    Main Coordinator Agent for the multi-agent platform.
    
    Serves as the primary interface between users and the agent ecosystem,
    orchestrating conversations, routing requests, and managing responses.
    """
    
    def __init__(self, agent_id: str = None):
        # Initialize with coordinator role
        super().__init__(
            agent_id=agent_id or f"coordinator_{uuid4().hex[:8]}",
            agent_role=AgentRole.COORDINATOR,
            capabilities=[
                "conversation_management",
                "intent_classification", 
                "multi_agent_orchestration",
                "voice_interaction",
                "multimodal_processing",
                "tool_execution",
                "context_management"
            ]
        )
        
        # Initialize settings and logging
        self.settings = get_settings()
        self.logger = setup_logging(f"CoordinatorAgent.{self.agent_id}")
        self.metrics = get_metrics_collector()
        
        # Core conversation components
        self.conversation_manager: Optional[ConversationManager] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self.response_synthesizer: Optional[ResponseSynthesizer] = None
        
        # Gemini 2.0 components
        self.gemini_client: Optional[GeminiLiveAPIClient] = None
        self.multimodal_processor: Optional[MultimodalProcessor] = None
        self.voice_synthesizer: Optional[VoiceSynthesizer] = None
        
        # Orchestration components
        self.workflow_manager: Optional[WorkflowManager] = None
        self.agent_router: Optional[AgentRouter] = None
        self.context_aggregator: Optional[ContextAggregator] = None
        
        # MCP integration components
        self.mcp_connector: Optional[MCPServerConnector] = None
        self.tool_executor: Optional[ToolExecutor] = None
        self.result_processor: Optional[ResultProcessor] = None
        
        # Active sessions tracking
        self.active_sessions: Dict[str, ConversationContext] = {}
        
        # Performance tracking
        self.response_times: List[float] = []
        self.session_count: int = 0
        
        self.logger.info(f"Coordinator Agent {self.agent_id} initialized")
    
    async def initialize(self) -> None:
        """Initialize all coordinator components."""
        try:
            self.logger.info("Initializing Coordinator Agent components...")
            
            # Initialize parent
            await super().initialize()
            
            # Initialize core conversation components
            self.conversation_manager = ConversationManager(self.settings)
            await self.conversation_manager.initialize()
            
            self.intent_classifier = IntentClassifier(self.settings)
            await self.intent_classifier.initialize()
            
            self.response_synthesizer = ResponseSynthesizer(self.settings)
            await self.response_synthesizer.initialize()
            
            # Initialize Gemini 2.0 components
            self.gemini_client = GeminiLiveAPIClient(self.settings)
            await self.gemini_client.initialize()
            
            self.multimodal_processor = MultimodalProcessor(self.settings)
            await self.multimodal_processor.initialize()
            
            self.voice_synthesizer = VoiceSynthesizer(self.settings)
            await self.voice_synthesizer.initialize()
            
            # Initialize orchestration components
            self.workflow_manager = WorkflowManager(self.settings)
            await self.workflow_manager.initialize()
            
            self.agent_router = AgentRouter(self.settings)
            await self.agent_router.initialize()
            
            self.context_aggregator = ContextAggregator(self.settings)
            await self.context_aggregator.initialize()
            
            # Initialize MCP integration
            self.mcp_connector = MCPServerConnector(self.settings)
            await self.mcp_connector.initialize()
            
            self.tool_executor = ToolExecutor(
                mcp_connector=self.mcp_connector,
                settings=self.settings
            )
            await self.tool_executor.initialize()
            
            self.result_processor = ResultProcessor(self.settings)
            await self.result_processor.initialize()
            
            # Set status to ready
            self.status = "ready"
            self.logger.info("Coordinator Agent initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Coordinator Agent: {str(e)}")
            self.status = "error"
            raise
    
    async def handle_message(self, message: AgentMessage) -> AgentMessage:
        """
        Handle incoming messages from users or other agents.
        
        Args:
            message: The incoming message
            
        Returns:
            Response message
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Handling message: {message.type} from {message.source_agent}")
            
            # Track message in conversation context
            session_id = message.thread_id or str(uuid4())
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = ConversationContext(
                    conversation_id=session_id,
                    current_agent=self.agent_id
                )
            
            context = self.active_sessions[session_id]
            context.add_message(message)
            
            # Route based on message type
            if message.type == MessageType.QUERY:
                response = await self._handle_query(message, context)
            elif message.type == MessageType.TOOL_REQUEST:
                response = await self._handle_tool_request(message, context)
            elif message.type == MessageType.HANDOFF:
                response = await self._handle_handoff(message, context)
            else:
                response = create_error_message(
                    source_agent=self.agent_id,
                    error_code="UNSUPPORTED_MESSAGE_TYPE",
                    error_message=f"Unsupported message type: {message.type}",
                    correlation_id=message.id
                )
            
            # Track performance
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.response_times.append(response_time)
            
            # Metrics
            self.metrics.increment_counter("coordinator.messages_handled")
            self.metrics.record_histogram("coordinator.response_time", response_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            
            return create_error_message(
                source_agent=self.agent_id,
                error_code="MESSAGE_HANDLING_ERROR",
                error_message=str(e),
                correlation_id=message.id
            )
    
    async def _handle_query(self, message: AgentMessage, context: ConversationContext) -> AgentMessage:
        """Handle query messages from users."""
        try:
            # Extract query details
            query_payload = message.payload
            query_text = query_payload.query_text
            query_type = query_payload.query_type
            
            self.logger.info(f"Processing query: {query_text[:100]}...")
            
            # Classify intent
            intent_result = await self.intent_classifier.classify_intent(
                query_text, context
            )
            
            # Check if multimodal processing is needed
            if hasattr(query_payload, 'context') and query_payload.context.get('has_media'):
                multimodal_result = await self.multimodal_processor.process_input(
                    query_payload.context
                )
                intent_result.update(multimodal_result)
            
            # Determine if this requires other agents or tools
            if intent_result.get('requires_agents') or intent_result.get('requires_tools'):
                response = await self._orchestrate_complex_query(
                    query_text, intent_result, context
                )
            else:
                # Handle directly with Gemini
                response = await self._handle_direct_query(
                    query_text, intent_result, context
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling query: {str(e)}")
            raise
    
    async def _handle_direct_query(
        self, 
        query_text: str, 
        intent_result: Dict[str, Any], 
        context: ConversationContext
    ) -> AgentMessage:
        """Handle queries that can be answered directly by the coordinator."""
        try:
            # Use Gemini Live API for real-time response
            gemini_response = await self.gemini_client.generate_response(
                query_text=query_text,
                context=context,
                intent_info=intent_result
            )
            
            # Synthesize final response
            final_response = await self.response_synthesizer.synthesize_response(
                primary_response=gemini_response,
                context=context,
                intent_result=intent_result
            )
            
            return create_response_message(
                source_agent=self.agent_id,
                response_text=final_response,
                correlation_id=context.messages[-1].id,
                confidence=intent_result.get('confidence', 0.9)
            )
            
        except Exception as e:
            self.logger.error(f"Error in direct query handling: {str(e)}")
            raise
    
    async def _orchestrate_complex_query(
        self,
        query_text: str,
        intent_result: Dict[str, Any],
        context: ConversationContext
    ) -> AgentMessage:
        """Orchestrate complex queries requiring multiple agents or tools."""
        try:
            # Create workflow for the query
            workflow = await self.workflow_manager.create_workflow(
                query_text=query_text,
                intent_result=intent_result,
                context=context
            )
            
            # Execute workflow
            workflow_results = await self.workflow_manager.execute_workflow(
                workflow, context
            )
            
            # Aggregate results from multiple sources
            aggregated_context = await self.context_aggregator.aggregate_results(
                workflow_results, context
            )
            
            # Generate final response
            final_response = await self.response_synthesizer.synthesize_response(
                primary_response=aggregated_context.get('summary'),
                context=context,
                intent_result=intent_result,
                additional_data=aggregated_context
            )
            
            return create_response_message(
                source_agent=self.agent_id,
                response_text=final_response,
                correlation_id=context.messages[-1].id,
                confidence=aggregated_context.get('confidence', 0.8)
            )
            
        except Exception as e:
            self.logger.error(f"Error in complex query orchestration: {str(e)}")
            raise
    
    async def _handle_tool_request(self, message: AgentMessage, context: ConversationContext) -> AgentMessage:
        """Handle tool execution requests."""
        try:
            tool_payload = message.payload
            tool_name = tool_payload.tool_name
            tool_params = tool_payload.tool_params
            
            # Execute tool via MCP
            tool_result = await self.tool_executor.execute_tool(
                tool_name=tool_name,
                parameters=tool_params,
                context=context
            )
            
            # Process results
            processed_result = await self.result_processor.process_tool_result(
                tool_result, context
            )
            
            return create_response_message(
                source_agent=self.agent_id,
                response_text=processed_result.get('summary', 'Tool executed successfully'),
                correlation_id=message.id,
                confidence=processed_result.get('confidence', 1.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error handling tool request: {str(e)}")
            raise
    
    async def _handle_handoff(self, message: AgentMessage, context: ConversationContext) -> AgentMessage:
        """Handle handoff requests to other agents."""
        try:
            # Route to appropriate agent
            target_agent = await self.agent_router.route_message(message, context)
            
            if target_agent:
                # Forward message and await response
                response = await self.agent_router.forward_message(
                    message, target_agent, context
                )
                return response
            else:
                return create_error_message(
                    source_agent=self.agent_id,
                    error_code="NO_SUITABLE_AGENT",
                    error_message="No suitable agent found for handoff",
                    correlation_id=message.id
                )
                
        except Exception as e:
            self.logger.error(f"Error handling handoff: {str(e)}")
            raise
    
    async def start_voice_session(self, session_config: Dict[str, Any]) -> str:
        """Start a real-time voice conversation session."""
        try:
            session_id = str(uuid4())
            
            # Create conversation context
            context = ConversationContext(
                conversation_id=session_id,
                current_agent=self.agent_id
            )
            self.active_sessions[session_id] = context
            
            # Initialize Gemini Live session
            live_session = await self.gemini_client.start_live_session(
                session_config=session_config,
                context=context
            )
            
            self.session_count += 1
            self.logger.info(f"Started voice session: {session_id}")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error starting voice session: {str(e)}")
            raise
    
    async def process_voice_input(
        self, 
        session_id: str, 
        audio_data: bytes,
        mime_type: str = "audio/pcm;rate=16000"
    ) -> Dict[str, Any]:
        """Process real-time voice input."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"No active session found: {session_id}")
            
            context = self.active_sessions[session_id]
            
            # Process audio through Gemini Live API
            response = await self.gemini_client.process_audio_input(
                session_id=session_id,
                audio_data=audio_data,
                mime_type=mime_type,
                context=context
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing voice input: {str(e)}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the coordinator."""
        try:
            base_health = await super().get_health_status()
            
            # Add coordinator-specific health metrics
            coordinator_health = {
                "active_sessions": len(self.active_sessions),
                "total_sessions": self.session_count,
                "avg_response_time": sum(self.response_times[-100:]) / len(self.response_times[-100:]) if self.response_times else 0,
                "components": {
                    "conversation_manager": await self.conversation_manager.get_health() if self.conversation_manager else "not_initialized",
                    "intent_classifier": await self.intent_classifier.get_health() if self.intent_classifier else "not_initialized",
                    "gemini_client": await self.gemini_client.get_health() if self.gemini_client else "not_initialized",
                    "mcp_connector": await self.mcp_connector.get_health() if self.mcp_connector else "not_initialized",
                    "workflow_manager": await self.workflow_manager.get_health() if self.workflow_manager else "not_initialized"
                }
            }
            
            return {**base_health, **coordinator_health}
            
        except Exception as e:
            self.logger.error(f"Error getting health status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self) -> None:
        """Clean shutdown of all coordinator components."""
        try:
            self.logger.info("Shutting down Coordinator Agent...")
            
            # Close active sessions
            for session_id in list(self.active_sessions.keys()):
                try:
                    await self.gemini_client.close_session(session_id)
                except Exception as e:
                    self.logger.warning(f"Error closing session {session_id}: {str(e)}")
            
            # Shutdown components in reverse order
            if self.mcp_connector:
                await self.mcp_connector.shutdown()
            
            if self.gemini_client:
                await self.gemini_client.shutdown()
            
            if self.workflow_manager:
                await self.workflow_manager.shutdown()
            
            # Shutdown parent
            await super().shutdown()
            
            self.logger.info("Coordinator Agent shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise


async def main():
    """Main entry point for running the coordinator agent standalone."""
    coordinator = CoordinatorAgent()
    
    try:
        await coordinator.initialize()
        
        # Keep running until interrupted
        while coordinator.status == "ready":
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down coordinator agent...")
    except Exception as e:
        print(f"Error running coordinator agent: {str(e)}")
    finally:
        await coordinator.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 