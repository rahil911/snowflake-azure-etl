"""
Context aggregation system for multi-agent responses.

This module provides the ContextAggregator class that combines and synthesizes
responses from multiple agents, handles conflicts, and provides unified results.
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ConfigDict
import google.genai as genai

from shared.schemas.agent_communication import (
    AgentMessage, AgentRole, MessageType, ConversationContext
)
from shared.config.logging_config import setup_logging
from shared.utils.metrics import get_metrics_collector


class CoordinatorError(Exception):
    """Base exception for coordinator errors."""
    pass


class AggregationStrategy(str, Enum):
    """Response aggregation strategies."""
    MERGE = "merge"              # Combine all responses
    CONSENSUS = "consensus"      # Use majority consensus
    PRIORITY = "priority"        # Use highest priority response
    WEIGHTED = "weighted"        # Weight responses by agent reliability
    CONFLICT_RESOLUTION = "conflict_resolution"  # Resolve conflicts intelligently
    BEST_SCORE = "best_score"    # Use response with highest confidence score


class ConflictResolution(str, Enum):
    """Conflict resolution methods."""
    MAJORITY_VOTE = "majority_vote"
    EXPERT_OVERRIDE = "expert_override"
    CONFIDENCE_BASED = "confidence_based"
    USER_CHOICE = "user_choice"
    HYBRID_MERGE = "hybrid_merge"


class ResponseConfidence(str, Enum):
    """Response confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class AgentResponse:
    """Structured agent response with metadata."""
    agent_id: str
    role: AgentRole
    content: Any
    confidence: float = 0.0
    confidence_level: ResponseConfidence = ResponseConfidence.UNKNOWN
    response_time_ms: float = 0.0
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    priority: int = 0
    reliability_score: float = 1.0
    
    # Timestamps
    received_at: datetime = field(default_factory=datetime.utcnow)
    
    # Source message
    source_message: Optional[AgentMessage] = None


class AggregationResult(BaseModel):
    """Result of response aggregation."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    aggregated_content: Any = Field(..., description="Final aggregated content")
    strategy_used: AggregationStrategy = Field(..., description="Aggregation strategy used")
    
    # Source information
    source_agents: List[str] = Field(..., description="Agents that contributed")
    total_responses: int = Field(..., description="Total number of responses")
    
    # Confidence and quality
    overall_confidence: float = Field(..., description="Overall confidence (0-1)")
    consensus_level: float = Field(..., description="Level of consensus (0-1)")
    
    # Conflicts and resolution
    conflicts_detected: bool = Field(default=False)
    conflict_resolution_method: Optional[ConflictResolution] = None
    resolved_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance metrics
    aggregation_time_ms: float = Field(..., description="Time taken to aggregate")
    
    # Metadata
    aggregation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Alternative responses (for user choice)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)


class ConflictDetector:
    """Detects conflicts between agent responses."""
    
    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)
    
    def detect_conflicts(self, responses: List[AgentResponse]) -> List[Dict[str, Any]]:
        """
        Detect conflicts between responses.
        
        Args:
            responses: List of agent responses
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group responses by type/topic
        content_groups = self._group_responses_by_content(responses)
        
        for topic, group_responses in content_groups.items():
            if len(group_responses) <= 1:
                continue
            
            # Check for numerical conflicts
            numeric_conflicts = self._detect_numeric_conflicts(group_responses, topic)
            conflicts.extend(numeric_conflicts)
            
            # Check for categorical conflicts
            categorical_conflicts = self._detect_categorical_conflicts(group_responses, topic)
            conflicts.extend(categorical_conflicts)
            
            # Check for semantic conflicts
            semantic_conflicts = self._detect_semantic_conflicts(group_responses, topic)
            conflicts.extend(semantic_conflicts)
        
        return conflicts
    
    def _group_responses_by_content(self, responses: List[AgentResponse]) -> Dict[str, List[AgentResponse]]:
        """Group responses by content type or topic."""
        groups = defaultdict(list)
        
        for response in responses:
            # Simple grouping by content type
            if isinstance(response.content, dict):
                for key in response.content.keys():
                    groups[key].append(response)
            else:
                groups["default"].append(response)
        
        return dict(groups)
    
    def _detect_numeric_conflicts(
        self, 
        responses: List[AgentResponse], 
        topic: str
    ) -> List[Dict[str, Any]]:
        """Detect conflicts in numeric values."""
        conflicts = []
        numeric_values = []
        
        for response in responses:
            if isinstance(response.content, dict) and topic in response.content:
                value = response.content[topic]
                if isinstance(value, (int, float)):
                    numeric_values.append((response, value))
        
        if len(numeric_values) < 2:
            return conflicts
        
        # Check for significant differences
        values = [v[1] for v in numeric_values]
        mean_value = sum(values) / len(values)
        max_deviation = max(abs(v - mean_value) for v in values)
        
        # If deviation is more than 20% of mean, consider it a conflict
        if max_deviation > 0.2 * abs(mean_value):
            conflicts.append({
                "type": "numeric_conflict",
                "topic": topic,
                "responses": [
                    {
                        "agent_id": resp.agent_id,
                        "value": value,
                        "confidence": resp.confidence
                    }
                    for resp, value in numeric_values
                ],
                "mean_value": mean_value,
                "max_deviation": max_deviation
            })
        
        return conflicts
    
    def _detect_categorical_conflicts(
        self, 
        responses: List[AgentResponse], 
        topic: str
    ) -> List[Dict[str, Any]]:
        """Detect conflicts in categorical values."""
        conflicts = []
        categorical_values = []
        
        for response in responses:
            if isinstance(response.content, dict) and topic in response.content:
                value = response.content[topic]
                if isinstance(value, str):
                    categorical_values.append((response, value.lower().strip()))
        
        if len(categorical_values) < 2:
            return conflicts
        
        # Count occurrences
        value_counts = Counter(v[1] for v in categorical_values)
        
        # If no clear majority (no value appears in >50% of responses)
        if len(value_counts) > 1 and max(value_counts.values()) <= len(categorical_values) / 2:
            conflicts.append({
                "type": "categorical_conflict",
                "topic": topic,
                "responses": [
                    {
                        "agent_id": resp.agent_id,
                        "value": value,
                        "confidence": resp.confidence
                    }
                    for resp, value in categorical_values
                ],
                "value_distribution": dict(value_counts)
            })
        
        return conflicts
    
    def _detect_semantic_conflicts(
        self, 
        responses: List[AgentResponse], 
        topic: str
    ) -> List[Dict[str, Any]]:
        """Detect semantic conflicts in text responses."""
        # Simplified semantic conflict detection
        # In production, this would use embeddings and semantic similarity
        conflicts = []
        
        text_responses = []
        for response in responses:
            if isinstance(response.content, dict) and topic in response.content:
                value = response.content[topic]
                if isinstance(value, str) and len(value) > 10:  # Only for substantial text
                    text_responses.append((response, value))
        
        if len(text_responses) < 2:
            return conflicts
        
        # Simple heuristic: if responses are very different in length or keywords
        lengths = [len(text) for _, text in text_responses]
        avg_length = sum(lengths) / len(lengths)
        
        # If there's significant length variation
        if max(lengths) > 2 * min(lengths) and avg_length > 50:
            conflicts.append({
                "type": "semantic_conflict",
                "topic": topic,
                "responses": [
                    {
                        "agent_id": resp.agent_id,
                        "length": len(text),
                        "confidence": resp.confidence
                    }
                    for resp, text in text_responses
                ],
                "length_variance": max(lengths) - min(lengths)
            })
        
        return conflicts


class ContextAggregator:
    """
    Multi-agent response aggregation system.
    
    Features:
    - Multiple aggregation strategies
    - Conflict detection and resolution
    - Confidence-based weighting
    - Quality scoring
    - Alternative response tracking
    """
    
    def __init__(
        self,
        default_strategy: AggregationStrategy = AggregationStrategy.CONSENSUS,
        conflict_resolution: ConflictResolution = ConflictResolution.CONFIDENCE_BASED,
        min_consensus_threshold: float = 0.6,
        metrics_manager: Optional[Any] = None
    ):
        self.logger = setup_logging(self.__class__.__name__)
        self.default_strategy = default_strategy
        self.conflict_resolution = conflict_resolution
        self.min_consensus_threshold = min_consensus_threshold
        self.metrics = metrics_manager or get_metrics_collector()
        
        # Components
        self.conflict_detector = ConflictDetector()
        
        # Agent reliability tracking
        self.agent_reliability: Dict[str, float] = defaultdict(lambda: 1.0)
        self.agent_performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Aggregation history
        self.aggregation_history: List[AggregationResult] = []
        
        self.logger.info("ContextAggregator initialized")
    
    async def aggregate_responses(
        self,
        responses: List[AgentMessage],
        strategy: Optional[AggregationStrategy] = None,
        context: Optional[ConversationContext] = None
    ) -> AggregationResult:
        """
        Aggregate multiple agent responses into a unified result.
        
        Args:
            responses: List of agent responses to aggregate
            strategy: Aggregation strategy to use
            context: Conversation context for informed aggregation
            
        Returns:
            Aggregated result with metadata
        """
        start_time = time.time()
        
        try:
            if not responses:
                raise ValueError("No responses provided for aggregation")
            
            # Convert to structured responses
            structured_responses = await self._prepare_responses(responses)
            
            # Use provided strategy or default
            agg_strategy = strategy or self.default_strategy
            
            # Detect conflicts
            conflicts = self.conflict_detector.detect_conflicts(structured_responses)
            
            # Aggregate based on strategy
            aggregated_content = await self._execute_aggregation(
                structured_responses, agg_strategy, conflicts
            )
            
            # Calculate confidence and consensus
            overall_confidence = self._calculate_overall_confidence(structured_responses)
            consensus_level = self._calculate_consensus_level(structured_responses, conflicts)
            
            # Resolve conflicts if needed
            conflict_resolution_method = None
            resolved_conflicts = []
            
            if conflicts:
                conflict_resolution_method = self.conflict_resolution
                resolved_conflicts = await self._resolve_conflicts(
                    conflicts, structured_responses, agg_strategy
                )
            
            # Generate alternatives
            alternatives = await self._generate_alternatives(structured_responses)
            
            # Create result
            result = AggregationResult(
                aggregated_content=aggregated_content,
                strategy_used=agg_strategy,
                source_agents=[r.agent_id for r in structured_responses],
                total_responses=len(structured_responses),
                overall_confidence=overall_confidence,
                consensus_level=consensus_level,
                conflicts_detected=len(conflicts) > 0,
                conflict_resolution_method=conflict_resolution_method,
                resolved_conflicts=resolved_conflicts,
                aggregation_time_ms=(time.time() - start_time) * 1000,
                alternatives=alternatives
            )
            
            # Store in history
            self.aggregation_history.append(result)
            
            # Update agent reliability
            await self._update_agent_reliability(structured_responses, result)
            
            self.logger.info(
                f"Aggregated {len(responses)} responses using {agg_strategy.value} strategy"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Response aggregation failed: {e}")
            raise CoordinatorError(f"Failed to aggregate responses: {e}")
    
    async def _prepare_responses(self, messages: List[AgentMessage]) -> List[AgentResponse]:
        """Convert agent messages to structured responses."""
        structured_responses = []
        
        for message in messages:
            # Calculate confidence from message metadata
            confidence = self._extract_confidence(message)
            
            # Determine confidence level
            confidence_level = self._classify_confidence(confidence)
            
            # Get agent reliability
            reliability = self.agent_reliability[message.sender_role.value]
            
            response = AgentResponse(
                agent_id=message.sender_role.value,
                role=message.sender_role,
                content=message.content,
                confidence=confidence,
                confidence_level=confidence_level,
                response_time_ms=getattr(message, 'response_time_ms', 0.0),
                reliability_score=reliability,
                source_message=message
            )
            
            structured_responses.append(response)
        
        return structured_responses
    
    def _extract_confidence(self, message: AgentMessage) -> float:
        """Extract confidence score from message."""
        if isinstance(message.content, dict):
            # Look for confidence in various fields
            confidence_fields = ['confidence', 'certainty', 'score', 'probability']
            for field in confidence_fields:
                if field in message.content:
                    value = message.content[field]
                    if isinstance(value, (int, float)) and 0 <= value <= 1:
                        return float(value)
        
        # Default confidence based on message type
        if message.message_type == MessageType.ERROR:
            return 0.1
        elif message.message_type == MessageType.WARNING:
            return 0.5
        else:
            return 0.8
    
    def _classify_confidence(self, confidence: float) -> ResponseConfidence:
        """Classify numeric confidence into levels."""
        if confidence >= 0.8:
            return ResponseConfidence.HIGH
        elif confidence >= 0.5:
            return ResponseConfidence.MEDIUM
        elif confidence >= 0.2:
            return ResponseConfidence.LOW
        else:
            return ResponseConfidence.UNKNOWN
    
    async def _execute_aggregation(
        self,
        responses: List[AgentResponse],
        strategy: AggregationStrategy,
        conflicts: List[Dict[str, Any]]
    ) -> Any:
        """Execute the aggregation strategy."""
        if strategy == AggregationStrategy.MERGE:
            return await self._merge_responses(responses)
        
        elif strategy == AggregationStrategy.CONSENSUS:
            return await self._consensus_aggregation(responses)
        
        elif strategy == AggregationStrategy.PRIORITY:
            return await self._priority_aggregation(responses)
        
        elif strategy == AggregationStrategy.WEIGHTED:
            return await self._weighted_aggregation(responses)
        
        elif strategy == AggregationStrategy.CONFLICT_RESOLUTION:
            return await self._conflict_resolution_aggregation(responses, conflicts)
        
        elif strategy == AggregationStrategy.BEST_SCORE:
            return await self._best_score_aggregation(responses)
        
        else:
            # Default to merge
            return await self._merge_responses(responses)
    
    async def _merge_responses(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Merge all responses into a combined result."""
        merged = {
            "type": "merged_response",
            "responses": [],
            "summary": "Combined response from multiple agents"
        }
        
        for response in responses:
            merged["responses"].append({
                "agent": response.agent_id,
                "role": response.role.value,
                "content": response.content,
                "confidence": response.confidence
            })
        
        return merged
    
    async def _consensus_aggregation(self, responses: List[AgentResponse]) -> Any:
        """Use majority consensus for aggregation."""
        # Group similar responses
        content_groups = defaultdict(list)
        
        for response in responses:
            # Simple grouping by string representation
            content_key = str(response.content)
            content_groups[content_key].append(response)
        
        # Find the group with the highest combined confidence
        best_group = None
        best_score = 0
        
        for content_key, group in content_groups.items():
            # Score is combination of count and confidence
            count_score = len(group) / len(responses)
            confidence_score = sum(r.confidence for r in group) / len(group)
            combined_score = count_score * 0.6 + confidence_score * 0.4
            
            if combined_score > best_score:
                best_score = combined_score
                best_group = group
        
        if best_group:
            # Return the highest confidence response from the best group
            return max(best_group, key=lambda r: r.confidence).content
        
        # Fallback to first response
        return responses[0].content
    
    async def _priority_aggregation(self, responses: List[AgentResponse]) -> Any:
        """Use highest priority response."""
        # Sort by priority, then by confidence
        sorted_responses = sorted(
            responses,
            key=lambda r: (r.priority, r.confidence),
            reverse=True
        )
        
        return sorted_responses[0].content
    
    async def _weighted_aggregation(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Use weighted aggregation based on reliability and confidence."""
        weighted_content = {
            "type": "weighted_response",
            "primary_response": None,
            "weighted_score": 0.0,
            "contributing_agents": []
        }
        
        best_response = None
        best_score = 0
        
        for response in responses:
            # Calculate weighted score
            weight = response.reliability_score * response.confidence
            
            weighted_content["contributing_agents"].append({
                "agent": response.agent_id,
                "weight": weight,
                "content": response.content
            })
            
            if weight > best_score:
                best_score = weight
                best_response = response
        
        if best_response:
            weighted_content["primary_response"] = best_response.content
            weighted_content["weighted_score"] = best_score
        
        return weighted_content
    
    async def _conflict_resolution_aggregation(
        self,
        responses: List[AgentResponse],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate with explicit conflict resolution."""
        if not conflicts:
            # No conflicts, use consensus
            return await self._consensus_aggregation(responses)
        
        resolved_content = {
            "type": "conflict_resolved_response",
            "base_response": None,
            "conflicts_resolved": len(conflicts),
            "resolution_method": self.conflict_resolution.value
        }
        
        # Start with highest confidence response
        base_response = max(responses, key=lambda r: r.confidence)
        resolved_content["base_response"] = base_response.content
        
        # Apply conflict resolution
        for conflict in conflicts:
            resolution = await self._resolve_single_conflict(conflict, responses)
            if resolution:
                # Update the base response with resolved values
                if isinstance(resolved_content["base_response"], dict):
                    resolved_content["base_response"].update(resolution)
        
        return resolved_content
    
    async def _best_score_aggregation(self, responses: List[AgentResponse]) -> Any:
        """Use response with the highest confidence score."""
        best_response = max(responses, key=lambda r: r.confidence)
        return best_response.content
    
    async def _resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        responses: List[AgentResponse],
        strategy: AggregationStrategy
    ) -> List[Dict[str, Any]]:
        """Resolve detected conflicts."""
        resolved = []
        
        for conflict in conflicts:
            resolution = await self._resolve_single_conflict(conflict, responses)
            if resolution:
                resolved.append({
                    "conflict": conflict,
                    "resolution": resolution,
                    "method": self.conflict_resolution.value
                })
        
        return resolved
    
    async def _resolve_single_conflict(
        self,
        conflict: Dict[str, Any],
        responses: List[AgentResponse]
    ) -> Optional[Dict[str, Any]]:
        """Resolve a single conflict."""
        if self.conflict_resolution == ConflictResolution.CONFIDENCE_BASED:
            # Use response with highest confidence
            conflict_responses = conflict.get("responses", [])
            if conflict_responses:
                best = max(conflict_responses, key=lambda r: r.get("confidence", 0))
                return {conflict["topic"]: best.get("value")}
        
        elif self.conflict_resolution == ConflictResolution.MAJORITY_VOTE:
            # Use majority value
            if conflict["type"] == "categorical_conflict":
                distribution = conflict.get("value_distribution", {})
                if distribution:
                    majority_value = max(distribution, key=distribution.get)
                    return {conflict["topic"]: majority_value}
        
        elif self.conflict_resolution == ConflictResolution.EXPERT_OVERRIDE:
            # Use response from most reliable agent
            conflict_responses = conflict.get("responses", [])
            if conflict_responses:
                agent_ids = [r.get("agent_id") for r in conflict_responses]
                most_reliable_agent = max(agent_ids, key=lambda a: self.agent_reliability.get(a, 0))
                
                for response in conflict_responses:
                    if response.get("agent_id") == most_reliable_agent:
                        return {conflict["topic"]: response.get("value")}
        
        return None
    
    def _calculate_overall_confidence(self, responses: List[AgentResponse]) -> float:
        """Calculate overall confidence across all responses."""
        if not responses:
            return 0.0
        
        # Weighted average based on reliability
        total_weight = sum(r.reliability_score for r in responses)
        if total_weight == 0:
            return 0.0
        
        weighted_confidence = sum(
            r.confidence * r.reliability_score for r in responses
        ) / total_weight
        
        return min(1.0, max(0.0, weighted_confidence))
    
    def _calculate_consensus_level(
        self,
        responses: List[AgentResponse],
        conflicts: List[Dict[str, Any]]
    ) -> float:
        """Calculate level of consensus among responses."""
        if not responses:
            return 0.0
        
        if len(responses) == 1:
            return 1.0
        
        # Consensus is inversely related to conflicts
        conflict_penalty = len(conflicts) / len(responses)
        base_consensus = 1.0 - min(conflict_penalty, 0.8)
        
        # Adjust based on confidence spread
        confidences = [r.confidence for r in responses]
        confidence_spread = max(confidences) - min(confidences)
        spread_penalty = confidence_spread * 0.2
        
        consensus = max(0.0, base_consensus - spread_penalty)
        return consensus
    
    async def _generate_alternatives(self, responses: List[AgentResponse]) -> List[Dict[str, Any]]:
        """Generate alternative response options."""
        alternatives = []
        
        # Sort responses by confidence
        sorted_responses = sorted(responses, key=lambda r: r.confidence, reverse=True)
        
        for i, response in enumerate(sorted_responses[:3]):  # Top 3 alternatives
            alternatives.append({
                "rank": i + 1,
                "agent": response.agent_id,
                "content": response.content,
                "confidence": response.confidence,
                "rationale": f"High confidence response from {response.role.value} agent"
            })
        
        return alternatives
    
    async def _update_agent_reliability(
        self,
        responses: List[AgentResponse],
        result: AggregationResult
    ) -> None:
        """Update agent reliability based on aggregation results."""
        # Simple reliability update based on consensus level
        consensus_bonus = result.consensus_level * 0.1
        
        for response in responses:
            agent_id = response.agent_id
            
            # Reward agents that contributed to high-consensus results
            if result.consensus_level > self.min_consensus_threshold:
                self.agent_reliability[agent_id] += consensus_bonus
            
            # Penalize agents that had low confidence in high-consensus results
            if (result.consensus_level > 0.8 and 
                response.confidence < 0.3):
                self.agent_reliability[agent_id] -= 0.05
            
            # Keep reliability in bounds
            self.agent_reliability[agent_id] = max(0.1, min(2.0, self.agent_reliability[agent_id]))
            
            # Track performance history
            self.agent_performance_history[agent_id].append(response.confidence)
            if len(self.agent_performance_history[agent_id]) > 100:
                self.agent_performance_history[agent_id] = self.agent_performance_history[agent_id][-100:]
    
    def get_aggregation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive aggregation metrics."""
        if not self.aggregation_history:
            return {"total_aggregations": 0}
        
        recent_results = self.aggregation_history[-100:]  # Last 100 aggregations
        
        return {
            "total_aggregations": len(self.aggregation_history),
            "average_confidence": sum(r.overall_confidence for r in recent_results) / len(recent_results),
            "average_consensus": sum(r.consensus_level for r in recent_results) / len(recent_results),
            "conflict_rate": sum(1 for r in recent_results if r.conflicts_detected) / len(recent_results),
            "average_aggregation_time_ms": sum(r.aggregation_time_ms for r in recent_results) / len(recent_results),
            "strategy_usage": Counter(r.strategy_used for r in recent_results),
            "agent_reliability": dict(self.agent_reliability)
        }
    
    def get_agent_performance(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific agent."""
        if agent_id not in self.agent_performance_history:
            return None
        
        history = self.agent_performance_history[agent_id]
        if not history:
            return None
        
        return {
            "agent_id": agent_id,
            "reliability_score": self.agent_reliability[agent_id],
            "confidence_stats": {
                "average": sum(history) / len(history),
                "min": min(history),
                "max": max(history),
                "recent_trend": history[-10:] if len(history) >= 10 else history
            },
            "total_responses": len(history)
        } 