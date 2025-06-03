"""
Agent routing system for the coordinator agent.

This module provides intelligent routing of messages to appropriate agents
based on capability matching, load balancing, and performance metrics.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict, deque

from pydantic import BaseModel, Field, ConfigDict
import google.genai as genai

from shared.schemas.agent_communication import (
    AgentMessage, AgentRole, MessageType, ConversationContext,
    AgentCapability
)
from shared.config.logging_config import setup_logging
# CoordinatorError will be defined locally
from shared.utils.metrics import get_metrics_collector


class CoordinatorError(Exception):
    """Base exception for coordinator errors."""
    pass


class RoutingStrategy(str, Enum):
    """Message routing strategies."""
    CAPABILITY_MATCH = "capability_match"
    LOAD_BALANCED = "load_balanced"
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"
    PERFORMANCE_BASED = "performance_based"
    STICKY_SESSION = "sticky_session"


class AgentHealth(str, Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class AgentEndpoint(BaseModel):
    """Agent endpoint information."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    agent_id: str = Field(..., description="Unique agent identifier")
    role: AgentRole = Field(..., description="Agent role")
    capabilities: Set[AgentCapability] = Field(..., description="Agent capabilities")
    
    # Connection info
    endpoint_url: Optional[str] = Field(default=None, description="Agent endpoint URL")
    priority: int = Field(default=0, description="Routing priority (higher = preferred)")
    weight: float = Field(default=1.0, description="Load balancing weight")
    
    # Health and performance
    health_status: AgentHealth = Field(default=AgentHealth.HEALTHY)
    last_health_check: datetime = Field(default_factory=datetime.utcnow)
    response_time_ms: float = Field(default=0.0, description="Average response time")
    success_rate: float = Field(default=1.0, description="Success rate (0-1)")
    
    # Load tracking
    active_requests: int = Field(default=0, description="Current active requests")
    max_concurrent: int = Field(default=10, description="Max concurrent requests")
    
    # Circuit breaker
    circuit_state: CircuitState = Field(default=CircuitState.CLOSED)
    failure_count: int = Field(default=0)
    last_failure_time: Optional[datetime] = None
    
    # Timestamps
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None


class RoutingRule(BaseModel):
    """Message routing rule."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    priority: int = Field(default=0, description="Rule priority (higher = first)")
    
    # Matching criteria
    message_types: Set[MessageType] = Field(default_factory=set)
    sender_roles: Set[AgentRole] = Field(default_factory=set)
    target_roles: Set[AgentRole] = Field(default_factory=set)
    content_patterns: List[str] = Field(default_factory=list)
    required_capabilities: Set[AgentCapability] = Field(default_factory=set)
    
    # Routing config
    strategy: RoutingStrategy = Field(default=RoutingStrategy.CAPABILITY_MATCH)
    target_agents: Set[str] = Field(default_factory=set)
    fallback_agents: Set[str] = Field(default_factory=set)
    
    # Constraints
    max_retries: int = Field(default=3)
    timeout_seconds: int = Field(default=30)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    enabled: bool = Field(default=True)


class RoutingMetrics(BaseModel):
    """Routing performance metrics."""
    total_messages: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    average_routing_time_ms: float = 0.0
    
    # Strategy usage
    strategy_usage: Dict[str, int] = Field(default_factory=dict)
    
    # Agent usage
    agent_usage: Dict[str, int] = Field(default_factory=dict)
    
    # Error tracking
    error_counts: Dict[str, int] = Field(default_factory=dict)


class AgentRouter:
    """
    Intelligent message routing system.
    
    Features:
    - Multiple routing strategies
    - Agent health monitoring
    - Load balancing
    - Circuit breaker pattern
    - Performance tracking
    - Fallback mechanisms
    """
    
    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH,
        health_check_interval: int = 30,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60,
        metrics_manager: Optional[Any] = None
    ):
        self.logger = setup_logging(self.__class__.__name__)
        self.default_strategy = default_strategy
        self.health_check_interval = health_check_interval
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.metrics = metrics_manager or get_metrics_collector()
        
        # Agent management
        self.agents: Dict[str, AgentEndpoint] = {}
        self.agents_by_role: Dict[AgentRole, List[str]] = defaultdict(list)
        self.agents_by_capability: Dict[AgentCapability, List[str]] = defaultdict(list)
        
        # Routing rules
        self.routing_rules: List[RoutingRule] = []
        
        # Load balancing state
        self.round_robin_counters: Dict[AgentRole, int] = defaultdict(int)
        self.sticky_sessions: Dict[str, str] = {}  # conversation_id -> agent_id
        
        # Performance tracking
        self.routing_metrics = RoutingMetrics()
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Health monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.start_health_monitoring()
        
        self.logger.info("AgentRouter initialized")
    
    def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.health_check_task and not self.health_check_task.done():
            return
        
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await self._check_agent_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    async def register_agent(self, agent: AgentEndpoint) -> bool:
        """
        Register a new agent endpoint.
        
        Args:
            agent: Agent endpoint to register
            
        Returns:
            True if registration successful
        """
        try:
            # Store agent
            self.agents[agent.agent_id] = agent
            
            # Update indexes
            self.agents_by_role[agent.role].append(agent.agent_id)
            for capability in agent.capabilities:
                self.agents_by_capability[capability].append(agent.agent_id)
            
            self.logger.info(f"Agent registered: {agent.agent_id} ({agent.role.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent endpoint.
        
        Args:
            agent_id: Agent to unregister
            
        Returns:
            True if unregistration successful
        """
        try:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            
            # Remove from indexes
            self.agents_by_role[agent.role].remove(agent_id)
            for capability in agent.capabilities:
                self.agents_by_capability[capability].remove(agent_id)
            
            # Remove from agents
            del self.agents[agent_id]
            
            # Clean up sessions
            sessions_to_remove = [
                conv_id for conv_id, a_id in self.sticky_sessions.items()
                if a_id == agent_id
            ]
            for conv_id in sessions_to_remove:
                del self.sticky_sessions[conv_id]
            
            self.logger.info(f"Agent unregistered: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add a new routing rule."""
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        self.logger.info(f"Routing rule added: {rule.rule_id}")
    
    def remove_routing_rule(self, rule_id: str) -> bool:
        """Remove a routing rule."""
        for i, rule in enumerate(self.routing_rules):
            if rule.rule_id == rule_id:
                del self.routing_rules[i]
                self.logger.info(f"Routing rule removed: {rule_id}")
                return True
        return False
    
    async def route_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Route a message to an appropriate agent.
        
        Args:
            message: Message to route
            
        Returns:
            Response from the agent, or None if routing failed
        """
        start_time = time.time()
        
        try:
            self.routing_metrics.total_messages += 1
            
            # Find matching routing rule
            rule = self._find_matching_rule(message)
            strategy = rule.strategy if rule else self.default_strategy
            
            # Get candidate agents
            candidates = await self._get_candidate_agents(message, rule)
            
            if not candidates:
                self.logger.warning(f"No candidate agents found for message {message.message_id}")
                self.routing_metrics.failed_routes += 1
                return None
            
            # Select target agent
            target_agent_id = await self._select_agent(candidates, strategy, message)
            
            if not target_agent_id:
                self.logger.warning(f"No target agent selected for message {message.message_id}")
                self.routing_metrics.failed_routes += 1
                return None
            
            # Route message
            response = await self._send_to_agent(target_agent_id, message, rule)
            
            # Update metrics
            routing_time = (time.time() - start_time) * 1000
            self._update_routing_metrics(strategy.value, target_agent_id, routing_time, response is not None)
            
            if response:
                self.routing_metrics.successful_routes += 1
            else:
                self.routing_metrics.failed_routes += 1
            
            return response
            
        except Exception as e:
            self.logger.error(f"Message routing failed: {e}")
            self.routing_metrics.failed_routes += 1
            return None
    
    def _find_matching_rule(self, message: AgentMessage) -> Optional[RoutingRule]:
        """Find the first matching routing rule."""
        for rule in self.routing_rules:
            if not rule.enabled:
                continue
            
            # Check message type
            if rule.message_types and message.message_type not in rule.message_types:
                continue
            
            # Check sender role
            if rule.sender_roles and message.sender_role not in rule.sender_roles:
                continue
            
            # Check target role
            if rule.target_roles and message.target_role not in rule.target_roles:
                continue
            
            # Check content patterns (simplified)
            if rule.content_patterns:
                content_str = str(message.content).lower()
                if not any(pattern.lower() in content_str for pattern in rule.content_patterns):
                    continue
            
            return rule
        
        return None
    
    async def _get_candidate_agents(
        self, 
        message: AgentMessage, 
        rule: Optional[RoutingRule]
    ) -> List[str]:
        """Get candidate agents for message routing."""
        candidates = set()
        
        if rule and rule.target_agents:
            # Use specific agents from rule
            candidates.update(rule.target_agents)
        else:
            # Find by role
            if message.target_role:
                candidates.update(self.agents_by_role.get(message.target_role, []))
            
            # Find by capabilities
            if rule and rule.required_capabilities:
                for capability in rule.required_capabilities:
                    candidates.update(self.agents_by_capability.get(capability, []))
        
        # Filter healthy agents
        healthy_candidates = []
        for agent_id in candidates:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if (agent.health_status != AgentHealth.OFFLINE and
                    agent.circuit_state != CircuitState.OPEN and
                    agent.active_requests < agent.max_concurrent):
                    healthy_candidates.append(agent_id)
        
        # Add fallback agents if no healthy candidates
        if not healthy_candidates and rule and rule.fallback_agents:
            for agent_id in rule.fallback_agents:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    if agent.health_status != AgentHealth.OFFLINE:
                        healthy_candidates.append(agent_id)
        
        return healthy_candidates
    
    async def _select_agent(
        self, 
        candidates: List[str], 
        strategy: RoutingStrategy, 
        message: AgentMessage
    ) -> Optional[str]:
        """Select target agent based on strategy."""
        if not candidates:
            return None
        
        if strategy == RoutingStrategy.STICKY_SESSION:
            # Check for existing session
            if message.conversation_id in self.sticky_sessions:
                session_agent = self.sticky_sessions[message.conversation_id]
                if session_agent in candidates:
                    return session_agent
            
            # Fall back to capability match
            strategy = RoutingStrategy.CAPABILITY_MATCH
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(candidates, message.target_role)
        
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return self._select_load_balanced(candidates)
        
        elif strategy == RoutingStrategy.PRIORITY_BASED:
            return self._select_priority_based(candidates)
        
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            return self._select_performance_based(candidates)
        
        else:  # CAPABILITY_MATCH
            return self._select_capability_match(candidates, message)
    
    def _select_round_robin(self, candidates: List[str], role: Optional[AgentRole]) -> str:
        """Select agent using round-robin strategy."""
        if not role:
            return candidates[0]
        
        counter = self.round_robin_counters[role]
        selected = candidates[counter % len(candidates)]
        self.round_robin_counters[role] = (counter + 1) % len(candidates)
        
        return selected
    
    def _select_load_balanced(self, candidates: List[str]) -> str:
        """Select agent with lowest current load."""
        best_agent = None
        lowest_load = float('inf')
        
        for agent_id in candidates:
            agent = self.agents[agent_id]
            # Calculate load as ratio of active requests to max concurrent
            load_ratio = agent.active_requests / agent.max_concurrent if agent.max_concurrent > 0 else 0
            weighted_load = load_ratio / agent.weight
            
            if weighted_load < lowest_load:
                lowest_load = weighted_load
                best_agent = agent_id
        
        return best_agent or candidates[0]
    
    def _select_priority_based(self, candidates: List[str]) -> str:
        """Select agent with highest priority."""
        best_agent = None
        highest_priority = -1
        
        for agent_id in candidates:
            agent = self.agents[agent_id]
            if agent.priority > highest_priority:
                highest_priority = agent.priority
                best_agent = agent_id
        
        return best_agent or candidates[0]
    
    def _select_performance_based(self, candidates: List[str]) -> str:
        """Select agent with best performance metrics."""
        best_agent = None
        best_score = -1
        
        for agent_id in candidates:
            agent = self.agents[agent_id]
            # Combine success rate and response time for scoring
            response_penalty = min(agent.response_time_ms / 1000, 1.0)  # Cap at 1 second
            score = agent.success_rate * (1 - response_penalty)
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent or candidates[0]
    
    def _select_capability_match(self, candidates: List[str], message: AgentMessage) -> str:
        """Select agent with best capability match."""
        # For now, just return first candidate
        # In production, this would analyze message content and match capabilities
        return candidates[0]
    
    async def _send_to_agent(
        self, 
        agent_id: str, 
        message: AgentMessage, 
        rule: Optional[RoutingRule]
    ) -> Optional[AgentMessage]:
        """Send message to specific agent."""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        start_time = time.time()
        
        try:
            # Update agent state
            agent.active_requests += 1
            agent.last_used = datetime.utcnow()
            
            # Set sticky session if needed
            if message.conversation_id:
                self.sticky_sessions[message.conversation_id] = agent_id
            
            # Simulate agent call (in real implementation, this would be HTTP/gRPC/message queue)
            timeout = rule.timeout_seconds if rule else 30
            response = await self._simulate_agent_call(agent_id, message, timeout)
            
            # Update performance metrics
            response_time = (time.time() - start_time) * 1000
            self._update_agent_performance(agent_id, response_time, response is not None)
            
            # Update circuit breaker
            if response:
                self._record_success(agent_id)
            else:
                self._record_failure(agent_id)
            
            return response
            
        finally:
            agent.active_requests = max(0, agent.active_requests - 1)
    
    async def _simulate_agent_call(
        self, 
        agent_id: str, 
        message: AgentMessage, 
        timeout: int
    ) -> Optional[AgentMessage]:
        """Simulate agent call (placeholder for actual implementation)."""
        try:
            # Simulate processing delay
            await asyncio.sleep(0.1)
            
            # Return mock response
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=message.conversation_id,
                sender_role=self.agents[agent_id].role,
                target_role=message.sender_role,
                message_type=MessageType.RESPONSE,
                content={"response": f"Processed by {agent_id}", "original_content": message.content},
                context=message.context
            )
            
        except Exception as e:
            self.logger.error(f"Agent call failed: {e}")
            return None
    
    def _update_agent_performance(self, agent_id: str, response_time: float, success: bool) -> None:
        """Update agent performance metrics."""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # Update response time (exponential moving average)
        alpha = 0.1
        agent.response_time_ms = (1 - alpha) * agent.response_time_ms + alpha * response_time
        
        # Update success rate (exponential moving average)
        success_value = 1.0 if success else 0.0
        agent.success_rate = (1 - alpha) * agent.success_rate + alpha * success_value
        
        # Store response time for detailed tracking
        self.response_times[agent_id].append(response_time)
    
    def _record_success(self, agent_id: str) -> None:
        """Record successful request for circuit breaker."""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        if agent.circuit_state == CircuitState.HALF_OPEN:
            # Success in half-open state closes the circuit
            agent.circuit_state = CircuitState.CLOSED
            agent.failure_count = 0
            self.logger.info(f"Circuit closed for agent {agent_id}")
        elif agent.circuit_state == CircuitState.CLOSED:
            # Reset failure count on success
            agent.failure_count = 0
    
    def _record_failure(self, agent_id: str) -> None:
        """Record failed request for circuit breaker."""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        agent.failure_count += 1
        agent.last_failure_time = datetime.utcnow()
        
        if agent.circuit_state == CircuitState.CLOSED:
            if agent.failure_count >= self.circuit_breaker_threshold:
                agent.circuit_state = CircuitState.OPEN
                self.logger.warning(f"Circuit opened for agent {agent_id} after {agent.failure_count} failures")
        
        elif agent.circuit_state == CircuitState.HALF_OPEN:
            # Failure in half-open state reopens the circuit
            agent.circuit_state = CircuitState.OPEN
            self.logger.warning(f"Circuit reopened for agent {agent_id}")
    
    async def _check_agent_health(self) -> None:
        """Check health of all registered agents."""
        for agent_id, agent in self.agents.items():
            try:
                # Check circuit breaker state
                if (agent.circuit_state == CircuitState.OPEN and
                    agent.last_failure_time and
                    (datetime.utcnow() - agent.last_failure_time).seconds >= self.circuit_breaker_timeout):
                    # Try to transition to half-open
                    agent.circuit_state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit transitioned to half-open for agent {agent_id}")
                
                # Update health status based on metrics
                if agent.success_rate >= 0.9 and agent.response_time_ms < 1000:
                    agent.health_status = AgentHealth.HEALTHY
                elif agent.success_rate >= 0.7 and agent.response_time_ms < 2000:
                    agent.health_status = AgentHealth.DEGRADED
                else:
                    agent.health_status = AgentHealth.UNHEALTHY
                
                agent.last_health_check = datetime.utcnow()
                
            except Exception as e:
                self.logger.error(f"Health check failed for agent {agent_id}: {e}")
                agent.health_status = AgentHealth.OFFLINE
    
    def _update_routing_metrics(
        self, 
        strategy: str, 
        agent_id: str, 
        routing_time: float, 
        success: bool
    ) -> None:
        """Update routing performance metrics."""
        # Update strategy usage
        self.routing_metrics.strategy_usage[strategy] = (
            self.routing_metrics.strategy_usage.get(strategy, 0) + 1
        )
        
        # Update agent usage
        self.routing_metrics.agent_usage[agent_id] = (
            self.routing_metrics.agent_usage.get(agent_id, 0) + 1
        )
        
        # Update average routing time
        total = self.routing_metrics.total_messages
        current_avg = self.routing_metrics.average_routing_time_ms
        self.routing_metrics.average_routing_time_ms = (
            (current_avg * (total - 1) + routing_time) / total
        )
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent."""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        
        return {
            "agent_id": agent.agent_id,
            "role": agent.role.value,
            "capabilities": [cap.value for cap in agent.capabilities],
            "health_status": agent.health_status.value,
            "circuit_state": agent.circuit_state.value,
            "performance": {
                "response_time_ms": agent.response_time_ms,
                "success_rate": agent.success_rate,
                "active_requests": agent.active_requests,
                "max_concurrent": agent.max_concurrent
            },
            "last_health_check": agent.last_health_check.isoformat(),
            "last_used": agent.last_used.isoformat() if agent.last_used else None
        }
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing metrics."""
        return {
            "total_agents": len(self.agents),
            "healthy_agents": len([a for a in self.agents.values() if a.health_status == AgentHealth.HEALTHY]),
            "routing_metrics": self.routing_metrics.dict(),
            "agents_by_role": {role.value: len(agents) for role, agents in self.agents_by_role.items()},
            "circuit_breaker_stats": {
                "open_circuits": len([a for a in self.agents.values() if a.circuit_state == CircuitState.OPEN]),
                "half_open_circuits": len([a for a in self.agents.values() if a.circuit_state == CircuitState.HALF_OPEN])
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown the router and cleanup resources."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("AgentRouter shutdown complete") 