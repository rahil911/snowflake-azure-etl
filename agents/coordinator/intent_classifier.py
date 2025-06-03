"""
Intent Classifier
================

Natural language intent classification for determining user goals,
required agents, and tools needed to fulfill requests.

Features:
- Intent classification using Gemini 2.0
- Multi-domain intent recognition
- Confidence scoring
- Context-aware classification
- Tool and agent routing decisions
- Fallback handling
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum

from shared.schemas.agent_communication import ConversationContext
from shared.config.logging_config import setup_logging
from shared.utils.caching import get_cache_manager
from shared.utils.validation import validate_input

# Intent categories
class IntentCategory(Enum):
    """High-level intent categories."""
    DATA_QUERY = "data_query"
    ANALYTICS = "analytics"
    REPORTING = "reporting"
    CONVERSATION = "conversation"
    SYSTEM_CONTROL = "system_control"
    HELP = "help"
    TOOL_EXECUTION = "tool_execution"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class IntentClassificationResult:
    """Result of intent classification."""
    
    def __init__(
        self,
        primary_intent: IntentCategory,
        confidence: float,
        sub_intents: List[str] = None,
        required_agents: List[str] = None,
        required_tools: List[str] = None,
        context_requirements: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        self.primary_intent = primary_intent
        self.confidence = confidence
        self.sub_intents = sub_intents or []
        self.required_agents = required_agents or []
        self.required_tools = required_tools or []
        self.context_requirements = context_requirements or {}
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_intent": self.primary_intent.value,
            "confidence": self.confidence,
            "sub_intents": self.sub_intents,
            "required_agents": self.required_agents,
            "required_tools": self.required_tools,
            "context_requirements": self.context_requirements,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "requires_agents": len(self.required_agents) > 0,
            "requires_tools": len(self.required_tools) > 0
        }


class IntentPattern:
    """Pattern for matching intents."""
    
    def __init__(
        self,
        pattern_id: str,
        category: IntentCategory,
        keywords: List[str],
        phrases: List[str] = None,
        required_agents: List[str] = None,
        required_tools: List[str] = None,
        confidence_threshold: float = 0.6
    ):
        self.pattern_id = pattern_id
        self.category = category
        self.keywords = [kw.lower() for kw in keywords]
        self.phrases = [p.lower() for p in phrases] if phrases else []
        self.required_agents = required_agents or []
        self.required_tools = required_tools or []
        self.confidence_threshold = confidence_threshold


class IntentClassifier:
    """
    Natural language intent classifier for the coordinator agent.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = setup_logging("IntentClassifier")
        self.cache_manager = get_cache_manager()
        
        # Intent patterns
        self.intent_patterns: List[IntentPattern] = []
        
        # Statistics
        self.classification_count = 0
        self.cache_hits = 0
        
        self.logger.info("IntentClassifier initialized")
    
    async def initialize(self) -> None:
        """Initialize the intent classifier."""
        try:
            # Load intent patterns
            self._load_intent_patterns()
            
            self.logger.info("IntentClassifier initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IntentClassifier: {str(e)}")
            raise
    
    def _load_intent_patterns(self) -> None:
        """Load predefined intent patterns."""
        try:
            # Data query patterns
            self.intent_patterns.extend([
                IntentPattern(
                    pattern_id="sql_query",
                    category=IntentCategory.DATA_QUERY,
                    keywords=["select", "query", "database", "table", "sql", "data", "fetch", "retrieve"],
                    phrases=["show me data", "get data from", "query the database", "find records"],
                    required_tools=["execute_query"],
                    required_agents=["etl_agent"]
                ),
                IntentPattern(
                    pattern_id="schema_inspection",
                    category=IntentCategory.DATA_QUERY,
                    keywords=["schema", "structure", "tables", "columns", "metadata"],
                    phrases=["show schema", "describe table", "what tables", "database structure"],
                    required_tools=["get_schema"]
                ),
                IntentPattern(
                    pattern_id="data_quality",
                    category=IntentCategory.DATA_QUERY,
                    keywords=["quality", "validation", "check", "integrity", "completeness"],
                    phrases=["check data quality", "validate data", "data integrity"],
                    required_tools=["check_data_quality"]
                ),
            ])
            
            # Analytics patterns
            self.intent_patterns.extend([
                IntentPattern(
                    pattern_id="statistical_analysis",
                    category=IntentCategory.ANALYTICS,
                    keywords=["statistics", "mean", "median", "average", "correlation", "analysis"],
                    phrases=["analyze data", "statistical analysis", "calculate statistics"],
                    required_tools=["calculate_statistics", "correlation_analysis"],
                    required_agents=["data_intelligence"]
                ),
                IntentPattern(
                    pattern_id="trend_analysis",
                    category=IntentCategory.ANALYTICS,
                    keywords=["trend", "pattern", "forecast", "prediction", "time series"],
                    phrases=["show trends", "analyze patterns", "predict future"],
                    required_tools=["trend_analysis", "forecast_values"],
                    required_agents=["data_intelligence"]
                ),
                IntentPattern(
                    pattern_id="outlier_detection",
                    category=IntentCategory.ANALYTICS,
                    keywords=["outlier", "anomaly", "unusual", "detection", "abnormal"],
                    phrases=["find outliers", "detect anomalies", "unusual patterns"],
                    required_tools=["outlier_detection"],
                    required_agents=["data_intelligence"]
                ),
            ])
            
            # Reporting patterns
            self.intent_patterns.extend([
                IntentPattern(
                    pattern_id="generate_report",
                    category=IntentCategory.REPORTING,
                    keywords=["report", "summary", "dashboard", "visualization"],
                    phrases=["generate report", "create summary", "build dashboard"],
                    required_agents=["visualization"]
                ),
                IntentPattern(
                    pattern_id="export_data",
                    category=IntentCategory.REPORTING,
                    keywords=["export", "download", "save", "csv", "excel"],
                    phrases=["export data", "download report", "save as"],
                    required_tools=["export_data"]
                ),
            ])
            
            # Conversation patterns
            self.intent_patterns.extend([
                IntentPattern(
                    pattern_id="greeting",
                    category=IntentCategory.CONVERSATION,
                    keywords=["hello", "hi", "hey", "greetings"],
                    phrases=["hello", "hi there", "good morning", "good afternoon"]
                ),
                IntentPattern(
                    pattern_id="gratitude",
                    category=IntentCategory.CONVERSATION,
                    keywords=["thank", "thanks", "appreciate"],
                    phrases=["thank you", "thanks", "appreciate it"]
                ),
                IntentPattern(
                    pattern_id="farewell",
                    category=IntentCategory.CONVERSATION,
                    keywords=["bye", "goodbye", "farewell", "exit"],
                    phrases=["goodbye", "see you later", "bye"]
                ),
            ])
            
            # System control patterns
            self.intent_patterns.extend([
                IntentPattern(
                    pattern_id="system_status",
                    category=IntentCategory.SYSTEM_CONTROL,
                    keywords=["status", "health", "system", "performance"],
                    phrases=["system status", "health check", "how is the system"]
                ),
                IntentPattern(
                    pattern_id="configuration",
                    category=IntentCategory.SYSTEM_CONTROL,
                    keywords=["config", "configuration", "settings", "setup"],
                    phrases=["change settings", "configure system", "update config"]
                ),
            ])
            
            # Help patterns
            self.intent_patterns.extend([
                IntentPattern(
                    pattern_id="help_request",
                    category=IntentCategory.HELP,
                    keywords=["help", "assist", "support", "guide", "how"],
                    phrases=["can you help", "I need help", "how do I", "what can you do"]
                ),
                IntentPattern(
                    pattern_id="capabilities",
                    category=IntentCategory.HELP,
                    keywords=["capabilities", "features", "what", "can", "do"],
                    phrases=["what can you do", "your capabilities", "available features"]
                ),
            ])
            
            # Multimodal patterns
            self.intent_patterns.extend([
                IntentPattern(
                    pattern_id="image_analysis",
                    category=IntentCategory.MULTIMODAL,
                    keywords=["image", "picture", "photo", "visual", "analyze"],
                    phrases=["analyze image", "describe picture", "what's in the image"]
                ),
                IntentPattern(
                    pattern_id="audio_processing",
                    category=IntentCategory.MULTIMODAL,
                    keywords=["audio", "voice", "sound", "listen", "transcribe"],
                    phrases=["process audio", "transcribe speech", "analyze voice"]
                ),
                IntentPattern(
                    pattern_id="video_analysis",
                    category=IntentCategory.MULTIMODAL,
                    keywords=["video", "movie", "clip", "analyze"],
                    phrases=["analyze video", "describe video", "what's in the video"]
                ),
            ])
            
            self.logger.info(f"Loaded {len(self.intent_patterns)} intent patterns")
            
        except Exception as e:
            self.logger.error(f"Error loading intent patterns: {str(e)}")
            raise
    
    async def classify_intent(
        self,
        query_text: str,
        context: Optional[ConversationContext] = None
    ) -> IntentClassificationResult:
        """
        Classify the intent of a user query.
        
        Args:
            query_text: The user's query text
            context: Optional conversation context
            
        Returns:
            IntentClassificationResult with classification details
        """
        try:
            self.classification_count += 1
            
            # Check cache first
            cache_key = f"intent_classification:{hash(query_text)}"
            if self.cache_manager:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    self.cache_hits += 1
                    return IntentClassificationResult(**cached_result)
            
            # Normalize query
            query_lower = query_text.lower().strip()
            
            # Pattern-based classification
            pattern_scores = self._score_patterns(query_lower, context)
            
            # If we have good pattern matches, use them
            if pattern_scores and pattern_scores[0][1] > 0.7:
                result = self._create_result_from_pattern(pattern_scores[0][0], pattern_scores[0][1])
            else:
                # Use AI-based classification for complex queries
                result = await self._ai_classify_intent(query_text, context)
            
            # Enhance with context if available
            if context:
                result = self._enhance_with_context(result, context)
            
            # Cache result
            if self.cache_manager and result.confidence > 0.6:
                await self.cache_manager.set(cache_key, result.to_dict(), ttl=3600)
            
            self.logger.debug(f"Classified intent: {result.primary_intent.value} (confidence: {result.confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying intent: {str(e)}")
            
            # Return unknown intent as fallback
            return IntentClassificationResult(
                primary_intent=IntentCategory.UNKNOWN,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _score_patterns(
        self,
        query_lower: str,
        context: Optional[ConversationContext] = None
    ) -> List[Tuple[IntentPattern, float]]:
        """Score intent patterns against the query."""
        try:
            scores = []
            
            for pattern in self.intent_patterns:
                score = 0.0
                
                # Score keywords
                keyword_matches = sum(1 for kw in pattern.keywords if kw in query_lower)
                if keyword_matches > 0:
                    score += (keyword_matches / len(pattern.keywords)) * 0.6
                
                # Score phrases
                phrase_matches = sum(1 for phrase in pattern.phrases if phrase in query_lower)
                if phrase_matches > 0:
                    score += (phrase_matches / len(pattern.phrases)) * 0.8
                
                # Bonus for exact phrase matches
                for phrase in pattern.phrases:
                    if phrase in query_lower:
                        score += 0.3
                        break
                
                # Context bonus
                if context and self._has_relevant_context(pattern, context):
                    score += 0.2
                
                if score > pattern.confidence_threshold:
                    scores.append((pattern, min(score, 1.0)))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores
            
        except Exception as e:
            self.logger.error(f"Error scoring patterns: {str(e)}")
            return []
    
    def _has_relevant_context(
        self,
        pattern: IntentPattern,
        context: ConversationContext
    ) -> bool:
        """Check if context is relevant to the pattern."""
        try:
            # Look at recent messages for context clues
            recent_messages = context.messages[-3:] if context.messages else []
            
            for message in recent_messages:
                if hasattr(message.payload, 'query_text'):
                    msg_text = message.payload.query_text.lower()
                    
                    # Check for related keywords
                    for keyword in pattern.keywords:
                        if keyword in msg_text:
                            return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking context relevance: {str(e)}")
            return False
    
    def _create_result_from_pattern(
        self,
        pattern: IntentPattern,
        confidence: float
    ) -> IntentClassificationResult:
        """Create classification result from a pattern match."""
        return IntentClassificationResult(
            primary_intent=pattern.category,
            confidence=confidence,
            sub_intents=[pattern.pattern_id],
            required_agents=pattern.required_agents,
            required_tools=pattern.required_tools,
            metadata={
                "pattern_id": pattern.pattern_id,
                "classification_method": "pattern_based"
            }
        )
    
    async def _ai_classify_intent(
        self,
        query_text: str,
        context: Optional[ConversationContext] = None
    ) -> IntentClassificationResult:
        """Use AI to classify complex intents."""
        try:
            # This would typically use a more sophisticated LLM call
            # For now, implementing a simplified approach
            
            query_lower = query_text.lower()
            
            # Check for data-related terms
            data_terms = ["data", "query", "select", "database", "table", "records"]
            if any(term in query_lower for term in data_terms):
                return IntentClassificationResult(
                    primary_intent=IntentCategory.DATA_QUERY,
                    confidence=0.8,
                    required_agents=["etl_agent"],
                    required_tools=["execute_query"],
                    metadata={"classification_method": "ai_based"}
                )
            
            # Check for analytics terms
            analytics_terms = ["analyze", "statistics", "trend", "correlation", "pattern"]
            if any(term in query_lower for term in analytics_terms):
                return IntentClassificationResult(
                    primary_intent=IntentCategory.ANALYTICS,
                    confidence=0.8,
                    required_agents=["data_intelligence"],
                    required_tools=["calculate_statistics"],
                    metadata={"classification_method": "ai_based"}
                )
            
            # Check for conversational terms
            conversation_terms = ["hello", "hi", "thank", "help", "how"]
            if any(term in query_lower for term in conversation_terms):
                return IntentClassificationResult(
                    primary_intent=IntentCategory.CONVERSATION,
                    confidence=0.7,
                    metadata={"classification_method": "ai_based"}
                )
            
            # Default to unknown
            return IntentClassificationResult(
                primary_intent=IntentCategory.UNKNOWN,
                confidence=0.3,
                metadata={"classification_method": "ai_based"}
            )
            
        except Exception as e:
            self.logger.error(f"Error in AI classification: {str(e)}")
            return IntentClassificationResult(
                primary_intent=IntentCategory.UNKNOWN,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _enhance_with_context(
        self,
        result: IntentClassificationResult,
        context: ConversationContext
    ) -> IntentClassificationResult:
        """Enhance classification result with context information."""
        try:
            # Check if previous queries suggest certain agents or tools
            recent_messages = context.messages[-5:] if context.messages else []
            
            context_agents = set()
            context_tools = set()
            
            for message in recent_messages:
                if hasattr(message.payload, 'query_text'):
                    msg_text = message.payload.query_text.lower()
                    
                    # Infer likely agents/tools from context
                    if any(term in msg_text for term in ["data", "query", "database"]):
                        context_agents.add("etl_agent")
                        context_tools.add("execute_query")
                    
                    if any(term in msg_text for term in ["analyze", "statistics"]):
                        context_agents.add("data_intelligence")
                        context_tools.add("calculate_statistics")
            
            # Add context-suggested agents/tools if not already present
            if context_agents and not result.required_agents:
                result.required_agents = list(context_agents)
                result.confidence = min(result.confidence + 0.1, 1.0)
            
            if context_tools and not result.required_tools:
                result.required_tools = list(context_tools)
                result.confidence = min(result.confidence + 0.1, 1.0)
            
            result.metadata["context_enhanced"] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error enhancing with context: {str(e)}")
            return result
    
    def get_intent_suggestions(self, partial_query: str) -> List[Dict[str, Any]]:
        """Get intent suggestions for partial queries."""
        try:
            suggestions = []
            partial_lower = partial_query.lower()
            
            for pattern in self.intent_patterns:
                # Check if any keywords match the partial query
                matching_keywords = [kw for kw in pattern.keywords if kw.startswith(partial_lower)]
                matching_phrases = [phrase for phrase in pattern.phrases if partial_lower in phrase]
                
                if matching_keywords or matching_phrases:
                    suggestions.append({
                        "category": pattern.category.value,
                        "pattern_id": pattern.pattern_id,
                        "matching_keywords": matching_keywords,
                        "matching_phrases": matching_phrases,
                        "confidence": len(matching_keywords + matching_phrases) / (len(pattern.keywords) + len(pattern.phrases))
                    })
            
            # Sort by confidence
            suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting intent suggestions: {str(e)}")
            return []
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        return {
            "total_classifications": self.classification_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / self.classification_count if self.classification_count > 0 else 0,
            "pattern_count": len(self.intent_patterns),
            "categories": [category.value for category in IntentCategory]
        }
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the intent classifier."""
        try:
            return {
                "status": "healthy",
                "pattern_count": len(self.intent_patterns),
                "classification_count": self.classification_count,
                "cache_hit_rate": self.cache_hits / self.classification_count if self.classification_count > 0 else 0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the intent classifier."""
        try:
            self.logger.info("Shutting down IntentClassifier...")
            
            # Log final statistics
            stats = self.get_classification_stats()
            self.logger.info(f"Final classification stats: {stats}")
            
            self.logger.info("IntentClassifier shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise 