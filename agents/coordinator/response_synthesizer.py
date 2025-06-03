"""
Response Synthesizer
===================

Handles response generation and formatting for the coordinator agent.
Synthesizes responses from multiple agents and formats them for user consumption.

Features:
- Natural language response generation
- Multi-agent response aggregation
- Context-aware response formatting
- Different output formats (text, structured, multimodal)
- Response personalization
- Error message formatting
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from shared.schemas.agent_communication import AgentMessage, ResponsePayload
from shared.config.logging_config import setup_logging
from shared.utils.validation import validate_input


class ResponseFormat(Enum):
    """Response format types."""
    TEXT = "text"
    STRUCTURED = "structured"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    MULTIMODAL = "multimodal"


class ResponseTone(Enum):
    """Response tone types."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    CONCISE = "concise"
    DETAILED = "detailed"


class ResponseTemplate:
    """Template for response generation."""
    
    def __init__(
        self,
        template_id: str,
        name: str,
        format_type: ResponseFormat,
        template: str,
        variables: List[str] = None,
        tone: ResponseTone = ResponseTone.PROFESSIONAL
    ):
        self.template_id = template_id
        self.name = name
        self.format_type = format_type
        self.template = template
        self.variables = variables or []
        self.tone = tone


class ResponseContext:
    """Context for response generation."""
    
    def __init__(
        self,
        user_query: str,
        intent_classification: Dict[str, Any],
        agent_responses: List[AgentMessage] = None,
        tool_results: List[Dict[str, Any]] = None,
        conversation_context: Optional[str] = None,
        user_preferences: Dict[str, Any] = None
    ):
        self.user_query = user_query
        self.intent_classification = intent_classification
        self.agent_responses = agent_responses or []
        self.tool_results = tool_results or []
        self.conversation_context = conversation_context
        self.user_preferences = user_preferences or {}
        self.timestamp = datetime.utcnow()


class SynthesizedResponse:
    """Final synthesized response."""
    
    def __init__(
        self,
        content: str,
        format_type: ResponseFormat,
        confidence: float,
        sources: List[str] = None,
        metadata: Dict[str, Any] = None,
        follow_up_suggestions: List[str] = None
    ):
        self.content = content
        self.format_type = format_type
        self.confidence = confidence
        self.sources = sources or []
        self.metadata = metadata or {}
        self.follow_up_suggestions = follow_up_suggestions or []
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "format_type": self.format_type.value,
            "confidence": self.confidence,
            "sources": self.sources,
            "metadata": self.metadata,
            "follow_up_suggestions": self.follow_up_suggestions,
            "timestamp": self.timestamp.isoformat()
        }


class ResponseSynthesizer:
    """
    Handles response generation and formatting for the coordinator agent.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = setup_logging("ResponseSynthesizer")
        
        # Response templates
        self.templates: Dict[str, ResponseTemplate] = {}
        
        # Statistics
        self.responses_generated = 0
        self.templates_used = {}
        
        self.logger.info("ResponseSynthesizer initialized")
    
    async def initialize(self) -> None:
        """Initialize the response synthesizer."""
        try:
            # Load response templates
            self._load_response_templates()
            
            self.logger.info("ResponseSynthesizer initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ResponseSynthesizer: {str(e)}")
            raise
    
    def _load_response_templates(self) -> None:
        """Load predefined response templates."""
        try:
            # Data query response templates
            self.templates["data_query_success"] = ResponseTemplate(
                template_id="data_query_success",
                name="Data Query Success",
                format_type=ResponseFormat.STRUCTURED,
                template="I found {record_count} records based on your query. Here are the results:\n\n{results}\n\nWould you like me to analyze this data further or export it?",
                variables=["record_count", "results"],
                tone=ResponseTone.PROFESSIONAL
            )
            
            self.templates["data_query_empty"] = ResponseTemplate(
                template_id="data_query_empty",
                name="Data Query Empty",
                format_type=ResponseFormat.TEXT,
                template="I didn't find any records matching your query criteria. This could be because:\n- The specified conditions are too restrictive\n- The data doesn't exist in the current time range\n- There might be a typo in the query\n\nWould you like me to suggest alternative search criteria?",
                variables=[],
                tone=ResponseTone.HELPFUL
            )
            
            # Analytics response templates
            self.templates["analytics_summary"] = ResponseTemplate(
                template_id="analytics_summary",
                name="Analytics Summary",
                format_type=ResponseFormat.STRUCTURED,
                template="Here's the analysis of your data:\n\n**Key Statistics:**\n{statistics}\n\n**Insights:**\n{insights}\n\n**Recommendations:**\n{recommendations}",
                variables=["statistics", "insights", "recommendations"],
                tone=ResponseTone.PROFESSIONAL
            )
            
            self.templates["correlation_analysis"] = ResponseTemplate(
                template_id="correlation_analysis",
                name="Correlation Analysis",
                format_type=ResponseFormat.STRUCTURED,
                template="I've analyzed the correlations in your data:\n\n{correlation_summary}\n\n**Strong Correlations Found:**\n{strong_correlations}\n\nThese correlations suggest {interpretation}.",
                variables=["correlation_summary", "strong_correlations", "interpretation"],
                tone=ResponseTone.TECHNICAL
            )
            
            # Conversation templates
            self.templates["greeting"] = ResponseTemplate(
                template_id="greeting",
                name="Greeting",
                format_type=ResponseFormat.TEXT,
                template="Hello! I'm your data intelligence assistant. I can help you query databases, analyze data, generate reports, and provide insights. What would you like to explore today?",
                variables=[],
                tone=ResponseTone.FRIENDLY
            )
            
            self.templates["help"] = ResponseTemplate(
                template_id="help",
                name="Help",
                format_type=ResponseFormat.STRUCTURED,
                template="I can help you with:\n\n📊 **Data Queries**: Search and retrieve data from your databases\n🔍 **Analytics**: Statistical analysis, trends, and correlations\n📈 **Reporting**: Generate summaries and visualizations\n🎤 **Voice Commands**: Speak naturally or type your requests\n\nTry asking something like:\n- \"Show me sales data for last month\"\n- \"Analyze customer satisfaction trends\"\n- \"What are the key performance indicators?\"",
                variables=[],
                tone=ResponseTone.HELPFUL
            )
            
            # Error templates
            self.templates["error_general"] = ResponseTemplate(
                template_id="error_general",
                name="General Error",
                format_type=ResponseFormat.TEXT,
                template="I encountered an issue while processing your request: {error_message}\n\nPlease try rephrasing your question or contact support if the problem persists.",
                variables=["error_message"],
                tone=ResponseTone.APOLOGETIC
            )
            
            self.templates["error_no_data"] = ResponseTemplate(
                template_id="error_no_data",
                name="No Data Error",
                format_type=ResponseFormat.TEXT,
                template="I couldn't access the requested data. This might be due to:\n- Database connectivity issues\n- Insufficient permissions\n- Data not available for the specified time period\n\nPlease check your request and try again.",
                variables=[],
                tone=ResponseTone.HELPFUL
            )
            
            # Multi-agent response template
            self.templates["multi_agent_summary"] = ResponseTemplate(
                template_id="multi_agent_summary",
                name="Multi-Agent Summary",
                format_type=ResponseFormat.STRUCTURED,
                template="I've gathered information from multiple sources:\n\n{agent_summaries}\n\n**Combined Insights:**\n{combined_insights}",
                variables=["agent_summaries", "combined_insights"],
                tone=ResponseTone.COMPREHENSIVE
            )
            
            self.logger.info(f"Loaded {len(self.templates)} response templates")
            
        except Exception as e:
            self.logger.error(f"Error loading response templates: {str(e)}")
            raise
    
    async def synthesize_response(
        self,
        context: ResponseContext,
        preferred_format: ResponseFormat = ResponseFormat.TEXT,
        tone: ResponseTone = ResponseTone.PROFESSIONAL
    ) -> SynthesizedResponse:
        """
        Synthesize a response based on the given context.
        
        Args:
            context: Response context with query, intent, and agent responses
            preferred_format: Preferred response format
            tone: Preferred response tone
            
        Returns:
            SynthesizedResponse object
        """
        try:
            self.responses_generated += 1
            
            # Determine the appropriate response strategy
            intent_category = context.intent_classification.get("primary_intent", "unknown")
            
            # Check if we have agent responses to aggregate
            if context.agent_responses:
                response = await self._synthesize_multi_agent_response(context, preferred_format, tone)
            elif context.tool_results:
                response = await self._synthesize_tool_response(context, preferred_format, tone)
            else:
                response = await self._synthesize_direct_response(context, preferred_format, tone)
            
            # Add follow-up suggestions
            response.follow_up_suggestions = self._generate_follow_up_suggestions(context)
            
            # Track template usage
            template_used = response.metadata.get("template_used")
            if template_used:
                self.templates_used[template_used] = self.templates_used.get(template_used, 0) + 1
            
            self.logger.debug(f"Synthesized response with confidence: {response.confidence:.2f}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error synthesizing response: {str(e)}")
            
            # Return error response
            return self._create_error_response(str(e), preferred_format)
    
    async def _synthesize_multi_agent_response(
        self,
        context: ResponseContext,
        preferred_format: ResponseFormat,
        tone: ResponseTone
    ) -> SynthesizedResponse:
        """Synthesize response from multiple agent responses."""
        try:
            # Aggregate responses from different agents
            agent_summaries = []
            all_data = []
            confidence_scores = []
            
            for agent_response in context.agent_responses:
                agent_name = agent_response.source_agent
                response_data = agent_response.data
                
                # Extract meaningful content from each agent
                if isinstance(response_data, dict):
                    if "results" in response_data:
                        summary = f"**{agent_name}**: Found {len(response_data['results'])} results"
                        all_data.extend(response_data.get("results", []))
                    elif "analysis" in response_data:
                        summary = f"**{agent_name}**: {response_data['analysis']}"
                    elif "message" in response_data:
                        summary = f"**{agent_name}**: {response_data['message']}"
                    else:
                        summary = f"**{agent_name}**: Processed successfully"
                    
                    agent_summaries.append(summary)
                    confidence_scores.append(response_data.get("confidence", 0.8))
            
            # Generate combined insights
            combined_insights = self._generate_combined_insights(context.agent_responses)
            
            # Use multi-agent template
            template = self.templates.get("multi_agent_summary")
            if template:
                content = template.template.format(
                    agent_summaries="\n".join(agent_summaries),
                    combined_insights=combined_insights
                )
            else:
                content = f"Combined results from {len(context.agent_responses)} agents:\n\n" + "\n".join(agent_summaries)
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.7
            
            return SynthesizedResponse(
                content=content,
                format_type=preferred_format,
                confidence=overall_confidence,
                sources=[resp.source_agent for resp in context.agent_responses],
                metadata={
                    "template_used": "multi_agent_summary",
                    "agent_count": len(context.agent_responses),
                    "total_data_points": len(all_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error synthesizing multi-agent response: {str(e)}")
            return self._create_error_response(str(e), preferred_format)
    
    async def _synthesize_tool_response(
        self,
        context: ResponseContext,
        preferred_format: ResponseFormat,
        tone: ResponseTone
    ) -> SynthesizedResponse:
        """Synthesize response from tool execution results."""
        try:
            intent_category = context.intent_classification.get("primary_intent", "unknown")
            
            # Process tool results based on intent
            if intent_category == "data_query":
                return await self._synthesize_data_query_response(context, preferred_format)
            elif intent_category == "analytics":
                return await self._synthesize_analytics_response(context, preferred_format)
            else:
                return await self._synthesize_generic_tool_response(context, preferred_format)
                
        except Exception as e:
            self.logger.error(f"Error synthesizing tool response: {str(e)}")
            return self._create_error_response(str(e), preferred_format)
    
    async def _synthesize_data_query_response(
        self,
        context: ResponseContext,
        preferred_format: ResponseFormat
    ) -> SynthesizedResponse:
        """Synthesize response for data query results."""
        try:
            # Aggregate all tool results
            all_results = []
            for tool_result in context.tool_results:
                if "results" in tool_result:
                    all_results.extend(tool_result["results"])
            
            if not all_results:
                # No results found
                template = self.templates.get("data_query_empty")
                content = template.template if template else "No results found for your query."
                
                return SynthesizedResponse(
                    content=content,
                    format_type=preferred_format,
                    confidence=0.9,
                    metadata={"template_used": "data_query_empty"}
                )
            
            # Format results based on preferred format
            if preferred_format == ResponseFormat.STRUCTURED:
                formatted_results = self._format_structured_results(all_results)
            elif preferred_format == ResponseFormat.JSON:
                formatted_results = json.dumps(all_results, indent=2)
            else:
                formatted_results = self._format_text_results(all_results)
            
            # Use success template
            template = self.templates.get("data_query_success")
            if template:
                content = template.template.format(
                    record_count=len(all_results),
                    results=formatted_results
                )
            else:
                content = f"Found {len(all_results)} records:\n\n{formatted_results}"
            
            return SynthesizedResponse(
                content=content,
                format_type=preferred_format,
                confidence=0.9,
                sources=["data_query"],
                metadata={
                    "template_used": "data_query_success",
                    "record_count": len(all_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error synthesizing data query response: {str(e)}")
            return self._create_error_response(str(e), preferred_format)
    
    async def _synthesize_analytics_response(
        self,
        context: ResponseContext,
        preferred_format: ResponseFormat
    ) -> SynthesizedResponse:
        """Synthesize response for analytics results."""
        try:
            # Extract analytics data
            statistics = {}
            insights = []
            recommendations = []
            
            for tool_result in context.tool_results:
                if "statistics" in tool_result:
                    statistics.update(tool_result["statistics"])
                if "insights" in tool_result:
                    insights.extend(tool_result["insights"])
                if "recommendations" in tool_result:
                    recommendations.extend(tool_result["recommendations"])
            
            # Format statistics
            stats_text = self._format_statistics(statistics)
            insights_text = "\n".join([f"• {insight}" for insight in insights])
            recommendations_text = "\n".join([f"• {rec}" for rec in recommendations])
            
            # Use analytics template
            template = self.templates.get("analytics_summary")
            if template:
                content = template.template.format(
                    statistics=stats_text,
                    insights=insights_text or "No specific insights identified.",
                    recommendations=recommendations_text or "No specific recommendations at this time."
                )
            else:
                content = f"Analytics Results:\n\nStatistics:\n{stats_text}\n\nInsights:\n{insights_text}"
            
            return SynthesizedResponse(
                content=content,
                format_type=preferred_format,
                confidence=0.85,
                sources=["analytics"],
                metadata={
                    "template_used": "analytics_summary",
                    "stat_count": len(statistics),
                    "insight_count": len(insights)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error synthesizing analytics response: {str(e)}")
            return self._create_error_response(str(e), preferred_format)
    
    async def _synthesize_generic_tool_response(
        self,
        context: ResponseContext,
        preferred_format: ResponseFormat
    ) -> SynthesizedResponse:
        """Synthesize response for generic tool results."""
        try:
            # Create a simple summary of tool results
            summary_parts = []
            for tool_result in context.tool_results:
                tool_name = tool_result.get("tool_name", "Unknown Tool")
                status = tool_result.get("status", "completed")
                message = tool_result.get("message", "Operation completed successfully")
                
                summary_parts.append(f"**{tool_name}**: {message}")
            
            content = "Tool execution results:\n\n" + "\n".join(summary_parts)
            
            return SynthesizedResponse(
                content=content,
                format_type=preferred_format,
                confidence=0.8,
                sources=["tools"],
                metadata={
                    "template_used": "generic_tool",
                    "tool_count": len(context.tool_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error synthesizing generic tool response: {str(e)}")
            return self._create_error_response(str(e), preferred_format)
    
    async def _synthesize_direct_response(
        self,
        context: ResponseContext,
        preferred_format: ResponseFormat,
        tone: ResponseTone
    ) -> SynthesizedResponse:
        """Synthesize direct response without agent/tool results."""
        try:
            intent_category = context.intent_classification.get("primary_intent", "unknown")
            
            # Handle different intent categories
            if intent_category == "conversation":
                sub_intent = context.intent_classification.get("sub_intents", [])
                if "greeting" in sub_intent:
                    template = self.templates.get("greeting")
                elif "help" in sub_intent:
                    template = self.templates.get("help")
                else:
                    template = None
                
                if template:
                    content = template.template
                    confidence = 0.9
                else:
                    content = "I'm here to help with your data questions. What would you like to know?"
                    confidence = 0.7
            
            elif intent_category == "help":
                template = self.templates.get("help")
                content = template.template if template else "I can help you with data queries, analytics, and reporting. What do you need assistance with?"
                confidence = 0.9
            
            else:
                # Unknown or complex intent
                content = f"I understand you're asking about {context.user_query}. Could you provide more specific details about what you'd like me to help you with?"
                confidence = 0.6
            
            return SynthesizedResponse(
                content=content,
                format_type=preferred_format,
                confidence=confidence,
                metadata={
                    "template_used": intent_category,
                    "direct_response": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error synthesizing direct response: {str(e)}")
            return self._create_error_response(str(e), preferred_format)
    
    def _generate_combined_insights(self, agent_responses: List[AgentMessage]) -> str:
        """Generate combined insights from multiple agent responses."""
        try:
            insights = []
            
            # Look for patterns across agent responses
            data_sources = set()
            common_themes = []
            
            for response in agent_responses:
                if hasattr(response, 'data') and isinstance(response.data, dict):
                    # Extract data sources
                    if "source" in response.data:
                        data_sources.add(response.data["source"])
                    
                    # Look for insights or patterns
                    if "insights" in response.data:
                        common_themes.extend(response.data["insights"])
            
            if data_sources:
                insights.append(f"Data analyzed from {len(data_sources)} different sources")
            
            if common_themes:
                insights.append(f"Common patterns identified across {len(common_themes)} areas")
            
            if not insights:
                insights.append("Multiple data processing operations completed successfully")
            
            return "\n".join([f"• {insight}" for insight in insights])
            
        except Exception as e:
            self.logger.error(f"Error generating combined insights: {str(e)}")
            return "Combined analysis completed successfully"
    
    def _format_structured_results(self, results: List[Dict[str, Any]]) -> str:
        """Format results in a structured text format."""
        try:
            if not results:
                return "No data to display"
            
            # Get column names from first result
            if isinstance(results[0], dict):
                columns = list(results[0].keys())
                
                # Create table format
                lines = []
                
                # Header
                header = " | ".join(columns)
                lines.append(header)
                lines.append("-" * len(header))
                
                # Data rows (limit to first 10 for readability)
                for i, row in enumerate(results[:10]):
                    row_data = " | ".join(str(row.get(col, "")) for col in columns)
                    lines.append(row_data)
                
                if len(results) > 10:
                    lines.append(f"... and {len(results) - 10} more rows")
                
                return "\n".join(lines)
            else:
                return "\n".join([str(result) for result in results[:10]])
                
        except Exception as e:
            self.logger.error(f"Error formatting structured results: {str(e)}")
            return str(results)
    
    def _format_text_results(self, results: List[Dict[str, Any]]) -> str:
        """Format results in a simple text format."""
        try:
            if not results:
                return "No data found"
            
            # Simple text format for each result
            formatted = []
            for i, result in enumerate(results[:5]):  # Limit to 5 for text format
                if isinstance(result, dict):
                    result_text = ", ".join([f"{k}: {v}" for k, v in result.items()])
                    formatted.append(f"{i+1}. {result_text}")
                else:
                    formatted.append(f"{i+1}. {result}")
            
            if len(results) > 5:
                formatted.append(f"... and {len(results) - 5} more results")
            
            return "\n".join(formatted)
            
        except Exception as e:
            self.logger.error(f"Error formatting text results: {str(e)}")
            return str(results)
    
    def _format_statistics(self, statistics: Dict[str, Any]) -> str:
        """Format statistics for display."""
        try:
            if not statistics:
                return "No statistics available"
            
            formatted = []
            for key, value in statistics.items():
                if isinstance(value, float):
                    formatted.append(f"• {key}: {value:.2f}")
                else:
                    formatted.append(f"• {key}: {value}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            self.logger.error(f"Error formatting statistics: {str(e)}")
            return str(statistics)
    
    def _generate_follow_up_suggestions(self, context: ResponseContext) -> List[str]:
        """Generate follow-up suggestions based on context."""
        try:
            suggestions = []
            intent_category = context.intent_classification.get("primary_intent", "unknown")
            
            if intent_category == "data_query":
                suggestions.extend([
                    "Would you like me to analyze this data?",
                    "Should I export these results?",
                    "Do you want to see related data?"
                ])
            
            elif intent_category == "analytics":
                suggestions.extend([
                    "Would you like deeper analysis?",
                    "Should I create a visualization?",
                    "Do you want to explore correlations?"
                ])
            
            elif intent_category == "conversation":
                suggestions.extend([
                    "What data would you like to explore?",
                    "Can I help you with any analysis?",
                    "Would you like to see our capabilities?"
                ])
            
            return suggestions[:3]  # Return max 3 suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating follow-up suggestions: {str(e)}")
            return []
    
    def _create_error_response(
        self,
        error_message: str,
        preferred_format: ResponseFormat
    ) -> SynthesizedResponse:
        """Create an error response."""
        try:
            template = self.templates.get("error_general")
            if template:
                content = template.template.format(error_message=error_message)
            else:
                content = f"I encountered an error: {error_message}"
            
            return SynthesizedResponse(
                content=content,
                format_type=preferred_format,
                confidence=0.9,  # High confidence in error message
                metadata={
                    "template_used": "error_general",
                    "error": True
                }
            )
            
        except Exception as e:
            # Fallback error response
            return SynthesizedResponse(
                content="I'm sorry, I encountered an unexpected error while processing your request.",
                format_type=ResponseFormat.TEXT,
                confidence=0.5,
                metadata={"error": True, "fallback": True}
            )
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get response synthesis statistics."""
        return {
            "responses_generated": self.responses_generated,
            "templates_loaded": len(self.templates),
            "templates_used": self.templates_used,
            "most_used_template": max(self.templates_used.items(), key=lambda x: x[1])[0] if self.templates_used else None
        }
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the response synthesizer."""
        try:
            return {
                "status": "healthy",
                "templates_loaded": len(self.templates),
                "responses_generated": self.responses_generated
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the response synthesizer."""
        try:
            self.logger.info("Shutting down ResponseSynthesizer...")
            
            # Log final statistics
            stats = self.get_synthesis_stats()
            self.logger.info(f"Final synthesis stats: {stats}")
            
            self.logger.info("ResponseSynthesizer shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise 