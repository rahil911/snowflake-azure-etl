"""
Data Intelligence Agent - Main Entry Point

This module implements the core Data Intelligence Agent that provides business intelligence
capabilities including natural language to SQL conversion, data analysis, insights generation,
and business recommendations.

Features:
- Natural language query understanding
- SQL generation from business questions
- Data quality analysis and insights
- Business intelligence and recommendations
- Integration with Snowflake and Analytics MCP servers
- Context-aware conversation handling

Integration:
- Extends BaseAgent from Session A foundation
- Uses MCP servers from Session B (Snowflake, Analytics)
- Integrates with Coordinator from Session C patterns
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import traceback

# Session A Foundation imports
from shared.base.agent_base import BaseAgent
from shared.schemas.agent_communication import AgentMessage, ResponsePayload, MessageType
from shared.schemas.data_models import (
    DataQualityReport, ETLPipelineStatus, BusinessMetric,
    AnalysisResult, InsightRecommendation, QueryResult, QueryMetadata
)
from shared.config.settings import get_settings
from shared.config.logging_config import setup_logging
from shared.utils.model_bus import AgentBusInterface
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.caching import get_cache_manager, cache_result
from shared.utils.validation import validate_input, ValidationError
from shared.utils.retry import retry_with_backoff

# Data Intelligence components
from .nlp.query_generator import QueryGenerator
from .nlp.intent_analyzer import IntentAnalyzer
from .nlp.context_extractor import ContextExtractor
from .analytics.insight_extractor import InsightExtractor
from .analytics.pattern_detector import PatternDetector
from .analytics.recommendation_engine import RecommendationEngine
from .data.sql_executor import SQLExecutor
from .data.result_processor import ResultProcessor
from .data.quality_analyzer import QualityAnalyzer
from .integration.coordinator_client import CoordinatorClient
from .integration.response_formatter import ResponseFormatter
from .integration.conversation_handler import ConversationHandler


class DataIntelligenceAgent(BaseAgent):
    """
    Data Intelligence Agent for business intelligence and data analysis.
    
    This agent specializes in:
    - Converting natural language to SQL queries
    - Analyzing data quality and patterns
    - Generating business insights and recommendations
    - Providing context-aware responses to business questions
    """
    
    def __init__(self, agent_id: str = "data_intelligence_agent"):
        """Initialize the Data Intelligence Agent with all components."""
        super().__init__(
            agent_id=agent_id,
            agent_type="data_intelligence",
            description="Business intelligence and data analysis agent",
            version="1.0.0"
        )
        
        self.settings = get_settings()
        self.logger = setup_logging(f"agent.{agent_id}")
        self.metrics = get_metrics_collector()
        self.cache = get_cache_manager()
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.query_counter = self.metrics.counter("di_queries_total")
        self.success_counter = self.metrics.counter("di_queries_successful")
        self.error_counter = self.metrics.counter("di_queries_failed")
        self.response_timer = self.metrics.timer("di_response_time")
        
        self.logger.info(f"Data Intelligence Agent {agent_id} initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all data intelligence components."""
        try:
            # NLP Components
            self.query_generator = QueryGenerator()
            self.intent_analyzer = IntentAnalyzer()
            self.context_extractor = ContextExtractor()
            
            # Analytics Components
            self.insight_extractor = InsightExtractor()
            self.pattern_detector = PatternDetector()
            self.recommendation_engine = RecommendationEngine()
            
            # Data Components
            self.sql_executor = SQLExecutor()
            self.result_processor = ResultProcessor()
            self.quality_analyzer = QualityAnalyzer()
            
            # Integration Components
            self.coordinator_client = CoordinatorClient(self)
            self.response_formatter = ResponseFormatter()
            self.conversation_handler = ConversationHandler()
            
            self.logger.info("All data intelligence components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    @track_performance(tags={"operation": "process_message"})
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process incoming messages and generate intelligent responses.
        
        Args:
            message: Incoming agent message
            
        Returns:
            AgentResponse with analysis results and insights
        """
        self.query_counter.increment()
        
        try:
            start_time = datetime.now()
            
            # Extract content from message payload first
            if hasattr(message.payload, 'query_text'):
                content = message.payload.query_text
            else:
                content = str(message.payload.data.get('content', ''))
            
            # Validate input
            validate_input(content, max_length=10000)
            
            # Extract content from message payload
            if hasattr(message.payload, 'query_text'):
                content = message.payload.query_text
            else:
                content = str(message.payload.data.get('content', ''))
            
            # Extract context and intent
            context = await self.context_extractor.extract_context(
                content, 
                message.payload.data.get('context', {})
            )
            intent = await self.intent_analyzer.analyze_intent(
                content, 
                context
            )
            
            self.logger.info(
                f"Processing query with intent: {intent.intent_type}",
                extra={
                    "message_id": message.message_id,
                    "intent": intent.intent_type,
                    "confidence": intent.confidence
                }
            )
            
            # Route to appropriate handler based on intent
            if intent.intent_type == "data_query":
                response_data = await self._handle_data_query(message, context, intent)
            elif intent.intent_type == "data_quality":
                response_data = await self._handle_quality_analysis(message, context, intent)
            elif intent.intent_type == "business_insight":
                response_data = await self._handle_insight_request(message, context, intent)
            elif intent.intent_type == "pattern_analysis":
                response_data = await self._handle_pattern_analysis(message, context, intent)
            elif intent.intent_type == "recommendation":
                response_data = await self._handle_recommendation_request(message, context, intent)
            else:
                response_data = await self._handle_general_query(message, context, intent)
            
            # Format response
            formatted_response = await self.response_formatter.format_response(
                response_data, 
                intent, 
                context
            )
            
            # Update conversation context
            await self.conversation_handler.update_context(
                message.conversation_id if hasattr(message, 'conversation_id') else None,
                message,
                formatted_response
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.response_timer.record(execution_time)
            self.success_counter.increment()
            
            # Create response payload
            response_payload = ResponsePayload(
                response_text=formatted_response["content"],
                confidence=intent.confidence,
                sources=response_data.get("data_sources", []),
                data=formatted_response.get("data", {}),
                metadata={
                    "intent": intent.intent_type,
                    "execution_time": execution_time,
                    "query_complexity": response_data.get("complexity", "medium")
                }
            )
            
            return AgentMessage(
                type=MessageType.RESPONSE,
                source_agent=self.agent_id,
                target_agent=message.source_agent,
                correlation_id=message.id,
                payload=response_payload
            )
            
        except ValidationError as e:
            self.error_counter.increment(tags={"error_type": "validation"})
            self.logger.warning(f"Validation error: {str(e)}", extra={"message_id": message.message_id})
            
            # Create error response
            error_payload = ResponsePayload(
                response_text="I need more specific information to help you with that query.",
                response_type="error",
                metadata={"error_type": "validation", "error_message": str(e)}
            )
            
            return AgentMessage(
                type=MessageType.RESPONSE,
                source_agent=self.agent_id,
                target_agent=message.source_agent,
                correlation_id=message.id,
                payload=error_payload
            )
            
        except Exception as e:
            self.error_counter.increment(tags={"error_type": "internal"})
            self.logger.error(
                f"Error processing message: {str(e)}", 
                extra={
                    "message_id": message.message_id,
                    "traceback": traceback.format_exc()
                }
            )
            
            # Create general error response  
            error_payload = ResponsePayload(
                response_text="I encountered an error while processing your request. Please try again or rephrase your question.",
                response_type="error",
                metadata={"error_type": "internal", "error_message": str(e)}
            )
            
            return AgentMessage(
                type=MessageType.RESPONSE,
                source_agent=self.agent_id,
                target_agent=message.source_agent,
                correlation_id=message.id,
                payload=error_payload
            )
    
    @cache_result(ttl=300, cache_name="memory")
    async def _handle_data_query(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any], 
        intent: Any
    ) -> Dict[str, Any]:
        """Handle data query requests with SQL generation and execution."""
        
        # Generate SQL from natural language
        sql_query = await self.query_generator.generate_sql(
            message.content, 
            context, 
            intent.entities
        )
        
        # Execute query via MCP Snowflake server
        query_result = await self.sql_executor.execute_query(
            sql_query.sql,
            sql_query.parameters
        )
        
        # Process and analyze results
        processed_results = await self.result_processor.process_query_results(
            query_result,
            sql_query.metadata
        )
        
        # Generate insights from the data
        insights = await self.insight_extractor.extract_insights(
            processed_results,
            context,
            intent
        )
        
        return {
            "type": "data_query_response",
            "sql_query": sql_query.sql,
            "results": processed_results,
            "insights": insights,
            "data_sources": [sql_query.table_references],
            "complexity": sql_query.complexity,
            "execution_stats": query_result.get("execution_stats", {})
        }
    
    async def _handle_quality_analysis(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any], 
        intent: Any
    ) -> Dict[str, Any]:
        """Handle data quality analysis requests."""
        
        # Extract table/dataset references from the query
        target_tables = intent.entities.get("tables", [])
        if not target_tables:
            # Try to infer from context or use default analysis scope
            target_tables = context.get("recent_tables", ["FACT_SALESACTUAL"])
        
        quality_reports = []
        for table in target_tables:
            quality_report = await self.quality_analyzer.analyze_table_quality(table)
            quality_reports.append(quality_report)
        
        # Generate quality insights and recommendations
        quality_insights = await self.insight_extractor.extract_quality_insights(
            quality_reports
        )
        
        return {
            "type": "quality_analysis_response",
            "quality_reports": quality_reports,
            "insights": quality_insights,
            "data_sources": target_tables,
            "complexity": "medium"
        }
    
    async def _handle_insight_request(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any], 
        intent: Any
    ) -> Dict[str, Any]:
        """Handle business insight generation requests."""
        
        # Generate relevant query to get data for insights
        sql_query = await self.query_generator.generate_insight_query(
            message.content,
            context,
            intent.entities
        )
        
        # Execute query and get data
        query_result = await self.sql_executor.execute_query(
            sql_query.sql,
            sql_query.parameters
        )
        
        # Process results
        processed_results = await self.result_processor.process_query_results(
            query_result,
            sql_query.metadata
        )
        
        # Extract comprehensive insights
        insights = await self.insight_extractor.extract_comprehensive_insights(
            processed_results,
            context,
            intent
        )
        
        # Generate recommendations
        recommendations = await self.recommendation_engine.generate_recommendations(
            insights,
            context,
            intent
        )
        
        return {
            "type": "business_insight_response",
            "insights": insights,
            "recommendations": recommendations,
            "supporting_data": processed_results,
            "data_sources": [sql_query.table_references],
            "complexity": "high"
        }
    
    async def _handle_pattern_analysis(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any], 
        intent: Any
    ) -> Dict[str, Any]:
        """Handle pattern detection and trend analysis requests."""
        
        # Generate query for pattern analysis
        sql_query = await self.query_generator.generate_pattern_query(
            message.content,
            context,
            intent.entities
        )
        
        # Execute query
        query_result = await self.sql_executor.execute_query(
            sql_query.sql,
            sql_query.parameters
        )
        
        # Process results for pattern detection
        processed_results = await self.result_processor.process_time_series_results(
            query_result,
            sql_query.metadata
        )
        
        # Detect patterns and trends
        patterns = await self.pattern_detector.detect_patterns(
            processed_results,
            context,
            intent
        )
        
        # Generate insights from patterns
        pattern_insights = await self.insight_extractor.extract_pattern_insights(
            patterns,
            processed_results,
            context
        )
        
        return {
            "type": "pattern_analysis_response",
            "patterns": patterns,
            "insights": pattern_insights,
            "supporting_data": processed_results,
            "data_sources": [sql_query.table_references],
            "complexity": "high"
        }
    
    async def _handle_recommendation_request(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any], 
        intent: Any
    ) -> Dict[str, Any]:
        """Handle recommendation generation requests."""
        
        # Get relevant data for recommendations
        sql_query = await self.query_generator.generate_recommendation_query(
            message.content,
            context,
            intent.entities
        )
        
        query_result = await self.sql_executor.execute_query(
            sql_query.sql,
            sql_query.parameters
        )
        
        processed_results = await self.result_processor.process_query_results(
            query_result,
            sql_query.metadata
        )
        
        # Generate comprehensive recommendations
        recommendations = await self.recommendation_engine.generate_comprehensive_recommendations(
            processed_results,
            context,
            intent
        )
        
        return {
            "type": "recommendation_response",
            "recommendations": recommendations,
            "supporting_data": processed_results,
            "data_sources": [sql_query.table_references],
            "complexity": "high"
        }
    
    async def _handle_general_query(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any], 
        intent: Any
    ) -> Dict[str, Any]:
        """Handle general queries that don't fit specific categories."""
        
        # Try to determine best approach based on content
        if "data" in message.content.lower() or "table" in message.content.lower():
            return await self._handle_data_query(message, context, intent)
        elif "quality" in message.content.lower():
            return await self._handle_quality_analysis(message, context, intent)
        else:
            # Provide general assistance
            return {
                "type": "general_response",
                "content": "I can help you with data analysis, quality checks, business insights, and recommendations. Could you please be more specific about what you'd like to know?",
                "suggestions": [
                    "Ask about sales performance trends",
                    "Request data quality analysis",
                    "Get business insights and recommendations",
                    "Analyze customer or product patterns"
                ],
                "complexity": "low"
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status of the Data Intelligence Agent."""
        
        component_health = {}
        
        # Check each component
        components = [
            ("query_generator", self.query_generator),
            ("intent_analyzer", self.intent_analyzer),
            ("context_extractor", self.context_extractor),
            ("insight_extractor", self.insight_extractor),
            ("pattern_detector", self.pattern_detector),
            ("recommendation_engine", self.recommendation_engine),
            ("sql_executor", self.sql_executor),
            ("result_processor", self.result_processor),
            ("quality_analyzer", self.quality_analyzer)
        ]
        
        for name, component in components:
            try:
                if hasattr(component, 'health_check'):
                    health = await component.health_check()
                    component_health[name] = health
                else:
                    component_health[name] = {"status": "healthy", "details": "Component operational"}
            except Exception as e:
                component_health[name] = {"status": "unhealthy", "error": str(e)}
        
        overall_status = "healthy" if all(
            h["status"] == "healthy" for h in component_health.values()
        ) else "degraded"
        
        return {
            "agent_id": self.agent_id,
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": component_health,
            "metrics": {
                "total_queries": self.query_counter.value,
                "successful_queries": self.success_counter.value,
                "failed_queries": self.error_counter.value,
                "avg_response_time": self.response_timer.avg
            }
        }
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of the Data Intelligence Agent."""
        
        return {
            "agent_type": "data_intelligence",
            "version": "1.0.0",
            "capabilities": [
                "natural_language_to_sql",
                "data_quality_analysis",
                "business_insight_generation",
                "pattern_detection",
                "trend_analysis",
                "recommendation_generation",
                "context_aware_conversation"
            ],
            "supported_data_sources": [
                "snowflake_warehouse",
                "dimensional_model",
                "fact_tables",
                "dimension_tables"
            ],
            "supported_query_types": [
                "aggregation_queries",
                "time_series_analysis",
                "comparative_analysis",
                "drill_down_queries",
                "summary_statistics",
                "pattern_detection"
            ],
            "integration_points": [
                "snowflake_mcp_server",
                "analytics_mcp_server",
                "coordinator_agent",
                "shared_foundation"
            ]
        }


# Agent factory and startup functions
async def create_data_intelligence_agent(agent_id: str = "data_intelligence_agent") -> DataIntelligenceAgent:
    """Factory function to create and initialize a Data Intelligence Agent."""
    
    agent = DataIntelligenceAgent(agent_id)
    await agent.initialize()
    return agent


async def main():
    """Main function for running the Data Intelligence Agent standalone."""
    
    agent = await create_data_intelligence_agent()
    
    print(f"Data Intelligence Agent {agent.agent_id} started successfully")
    print("Agent Capabilities:")
    capabilities = await agent.get_capabilities()
    for cap in capabilities["capabilities"]:
        print(f"  - {cap}")
    
    # Keep the agent running
    try:
        while True:
            await asyncio.sleep(60)  # Check every minute
            health = await agent.get_health_status()
            if health["status"] != "healthy":
                print(f"Agent health check: {health['status']}")
                
    except KeyboardInterrupt:
        print("Shutting down Data Intelligence Agent...")
        await agent.shutdown()
        print("Agent shutdown complete")


if __name__ == "__main__":
    asyncio.run(main()) 