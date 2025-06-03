"""
Inter-agent communication bus for the multi-agent data intelligence platform.

This module provides a comprehensive message bus system for communication
between agents, including routing, filtering, subscriptions, and event handling.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union, Coroutine
from dataclasses import dataclass, field
from collections import defaultdict, deque

from pydantic import BaseModel, Field, ConfigDict

from ..schemas.agent_communication import AgentMessage, MessageType, Priority
from ..utils.metrics import get_metrics_collector, Timer


class MessageStatus(str, Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"


class BusEventType(str, Enum):
    """Bus event types."""
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_DELIVERED = "message_delivered"
    MESSAGE_FAILED = "message_failed"
    AGENT_CONNECTED = "agent_connected"
    AGENT_DISCONNECTED = "agent_disconnected"
    SUBSCRIPTION_ADDED = "subscription_added"
    SUBSCRIPTION_REMOVED = "subscription_removed"


@dataclass
class MessageEnvelope:
    """Message envelope with routing and metadata."""
    message: AgentMessage
    sender_id: str
    recipient_id: Optional[str] = None
    routing_key: Optional[str] = None
    ttl_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: MessageStatus = MessageStatus.PENDING
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries and not self.is_expired


class MessageFilter:
    """Filter for message subscriptions."""
    
    def __init__(
        self,
        message_types: Optional[List[MessageType]] = None,
        sender_patterns: Optional[List[str]] = None,
        content_filters: Optional[Dict[str, Any]] = None,
        routing_patterns: Optional[List[str]] = None
    ):
        self.message_types = message_types or []
        self.sender_patterns = sender_patterns or []
        self.content_filters = content_filters or {}
        self.routing_patterns = routing_patterns or []
    
    def matches(self, envelope: MessageEnvelope) -> bool:
        """Check if message envelope matches filter criteria."""
        # Check message type
        if self.message_types and envelope.message.type not in self.message_types:
            return False
        
        # Check sender patterns
        if self.sender_patterns:
            matches_sender = any(
                self._pattern_matches(pattern, envelope.sender_id)
                for pattern in self.sender_patterns
            )
            if not matches_sender:
                return False
        
        # Check routing patterns
        if self.routing_patterns and envelope.routing_key:
            matches_routing = any(
                self._pattern_matches(pattern, envelope.routing_key)
                for pattern in self.routing_patterns
            )
            if not matches_routing:
                return False
        
        # Check content filters
        for key, expected_value in self.content_filters.items():
            if key not in envelope.message.content:
                return False
            if envelope.message.content[key] != expected_value:
                return False
        
        return True
    
    def _pattern_matches(self, pattern: str, value: str) -> bool:
        """Check if pattern matches value (supports wildcards)."""
        import fnmatch
        return fnmatch.fnmatch(value, pattern)


@dataclass
class Subscription:
    """Message subscription."""
    subscriber_id: str
    handler: Callable[[MessageEnvelope], Union[None, Coroutine[Any, Any, None]]]
    filter: MessageFilter
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    delivery_count: int = 0
    last_delivery: Optional[datetime] = None


class BusEvent(BaseModel):
    """Bus event for monitoring and logging."""
    model_config = ConfigDict(extra='forbid')
    
    event_type: BusEventType
    agent_id: str
    message_id: Optional[str] = None
    subscription_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageBus:
    """Central message bus for inter-agent communication."""
    
    def __init__(self, max_message_history: int = 1000):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_message_history = max_message_history
        self.metrics = get_metrics_collector()
        
        # Agent registry
        self._connected_agents: Set[str] = set()
        
        # Subscriptions
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._subscription_lookup: Dict[str, Subscription] = {}
        
        # Message handling
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._message_history: deque = deque(maxlen=max_message_history)
        self._pending_messages: Dict[str, MessageEnvelope] = {}
        
        # Event handling
        self._event_handlers: List[Callable[[BusEvent], None]] = []
        self._events: deque = deque(maxlen=1000)
        
        # Processing tasks
        self._processor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self._message_counter = self.metrics.counter("message_bus_messages_total", "Total messages processed")
        self._delivery_timer = self.metrics.timer("message_bus_delivery_duration", "Message delivery time")
        self._error_counter = self.metrics.counter("message_bus_errors_total", "Total bus errors")
    
    async def start(self) -> None:
        """Start the message bus."""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._message_processor())
        self._cleanup_task = asyncio.create_task(self._cleanup_processor())
        
        self.logger.info("Message bus started")
        await self._emit_event(BusEvent(
            event_type=BusEventType.MESSAGE_SENT,
            agent_id="system",
            metadata={"action": "bus_started"}
        ))
    
    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Message bus stopped")
    
    async def connect_agent(self, agent_id: str) -> None:
        """Connect an agent to the bus."""
        self._connected_agents.add(agent_id)
        self.logger.info(f"Agent {agent_id} connected to message bus")
        
        await self._emit_event(BusEvent(
            event_type=BusEventType.AGENT_CONNECTED,
            agent_id=agent_id
        ))
    
    async def disconnect_agent(self, agent_id: str) -> None:
        """Disconnect an agent from the bus."""
        self._connected_agents.discard(agent_id)
        
        # Remove agent's subscriptions
        for subscription_list in self._subscriptions.values():
            subscription_list[:] = [s for s in subscription_list if s.subscriber_id != agent_id]
        
        # Clean up subscription lookup
        to_remove = [sid for sid, sub in self._subscription_lookup.items() if sub.subscriber_id == agent_id]
        for sid in to_remove:
            del self._subscription_lookup[sid]
        
        self.logger.info(f"Agent {agent_id} disconnected from message bus")
        
        await self._emit_event(BusEvent(
            event_type=BusEventType.AGENT_DISCONNECTED,
            agent_id=agent_id
        ))
    
    async def send_message(
        self,
        message: AgentMessage,
        sender_id: str,
        recipient_id: Optional[str] = None,
        routing_key: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None
    ) -> str:
        """Send a message through the bus."""
        envelope = MessageEnvelope(
            message=message,
            sender_id=sender_id,
            recipient_id=recipient_id,
            routing_key=routing_key,
            ttl_seconds=ttl_seconds,
            correlation_id=correlation_id,
            reply_to=reply_to
        )
        
        # Generate message ID if not present
        if not envelope.message.id:
            envelope.message.id = str(uuid.uuid4())
        
        # Add to pending messages
        self._pending_messages[envelope.message.id] = envelope
        
        # Queue for processing
        await self._message_queue.put(envelope)
        
        # Emit event
        await self._emit_event(BusEvent(
            event_type=BusEventType.MESSAGE_SENT,
            agent_id=sender_id,
            message_id=envelope.message.id,
            metadata={
                "recipient_id": recipient_id,
                "routing_key": routing_key,
                "message_type": envelope.message.type.value
            }
        ))
        
        self._message_counter.increment()
        self.logger.debug(f"Message {envelope.message.id} queued for delivery")
        
        return envelope.message.id
    
    async def subscribe(
        self,
        subscriber_id: str,
        handler: Callable[[MessageEnvelope], Union[None, Coroutine[Any, Any, None]]],
        message_filter: Optional[MessageFilter] = None
    ) -> str:
        """Subscribe to messages with optional filtering."""
        if message_filter is None:
            message_filter = MessageFilter()  # Match all messages
        
        subscription = Subscription(
            subscriber_id=subscriber_id,
            handler=handler,
            filter=message_filter
        )
        
        # Add to subscriptions
        routing_key = "*"  # Default routing key for global subscriptions
        self._subscriptions[routing_key].append(subscription)
        self._subscription_lookup[subscription.subscription_id] = subscription
        
        await self._emit_event(BusEvent(
            event_type=BusEventType.SUBSCRIPTION_ADDED,
            agent_id=subscriber_id,
            subscription_id=subscription.subscription_id
        ))
        
        self.logger.info(f"Agent {subscriber_id} subscribed with ID {subscription.subscription_id}")
        return subscription.subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from messages."""
        if subscription_id not in self._subscription_lookup:
            return False
        
        subscription = self._subscription_lookup[subscription_id]
        
        # Remove from subscriptions
        for subscription_list in self._subscriptions.values():
            subscription_list[:] = [s for s in subscription_list if s.subscription_id != subscription_id]
        
        # Remove from lookup
        del self._subscription_lookup[subscription_id]
        
        await self._emit_event(BusEvent(
            event_type=BusEventType.SUBSCRIPTION_REMOVED,
            agent_id=subscription.subscriber_id,
            subscription_id=subscription_id
        ))
        
        self.logger.info(f"Subscription {subscription_id} removed")
        return True
    
    async def subscribe_to_routing_key(
        self,
        subscriber_id: str,
        routing_key: str,
        handler: Callable[[MessageEnvelope], Union[None, Coroutine[Any, Any, None]]],
        message_filter: Optional[MessageFilter] = None
    ) -> str:
        """Subscribe to messages with specific routing key."""
        if message_filter is None:
            message_filter = MessageFilter()
        
        # Add routing pattern to filter
        if not message_filter.routing_patterns:
            message_filter.routing_patterns = [routing_key]
        elif routing_key not in message_filter.routing_patterns:
            message_filter.routing_patterns.append(routing_key)
        
        subscription = Subscription(
            subscriber_id=subscriber_id,
            handler=handler,
            filter=message_filter
        )
        
        self._subscriptions[routing_key].append(subscription)
        self._subscription_lookup[subscription.subscription_id] = subscription
        
        self.logger.info(f"Agent {subscriber_id} subscribed to routing key '{routing_key}'")
        return subscription.subscription_id
    
    async def _message_processor(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                # Wait for message with timeout
                envelope = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self._process_message(envelope)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in message processor: {e}")
                self._error_counter.increment()
    
    async def _process_message(self, envelope: MessageEnvelope) -> None:
        """Process a single message."""
        try:
            # Check if message has expired
            if envelope.is_expired:
                envelope.status = MessageStatus.EXPIRED
                self.logger.warning(f"Message {envelope.message.id} expired")
                return
            
            envelope.status = MessageStatus.PROCESSING
            delivered = False
            
            with self._delivery_timer.time_context():
                # Find matching subscriptions
                matching_subscriptions = self._find_matching_subscriptions(envelope)
                
                if matching_subscriptions:
                    # Deliver to all matching subscribers
                    delivery_tasks = []
                    for subscription in matching_subscriptions:
                        task = asyncio.create_task(
                            self._deliver_to_subscription(envelope, subscription)
                        )
                        delivery_tasks.append(task)
                    
                    # Wait for all deliveries
                    results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
                    
                    # Check if any delivery succeeded
                    for result in results:
                        if result is not True and isinstance(result, Exception):
                            self.logger.error(f"Delivery error: {result}")
                        elif result is True:
                            delivered = True
                
                # Update status
                if delivered:
                    envelope.status = MessageStatus.DELIVERED
                    await self._emit_event(BusEvent(
                        event_type=BusEventType.MESSAGE_DELIVERED,
                        agent_id=envelope.sender_id,
                        message_id=envelope.message.id
                    ))
                else:
                    envelope.status = MessageStatus.FAILED
                    await self._emit_event(BusEvent(
                        event_type=BusEventType.MESSAGE_FAILED,
                        agent_id=envelope.sender_id,
                        message_id=envelope.message.id
                    ))
            
            # Add to history
            self._message_history.append(envelope)
            
            # Remove from pending if delivered or max retries reached
            if delivered or not envelope.can_retry:
                self._pending_messages.pop(envelope.message.id, None)
            
        except Exception as e:
            self.logger.error(f"Error processing message {envelope.message.id}: {e}")
            envelope.status = MessageStatus.FAILED
            self._error_counter.increment()
    
    def _find_matching_subscriptions(self, envelope: MessageEnvelope) -> List[Subscription]:
        """Find subscriptions that match the message envelope."""
        matching = []
        
        # Check direct routing key matches
        if envelope.routing_key and envelope.routing_key in self._subscriptions:
            for subscription in self._subscriptions[envelope.routing_key]:
                if subscription.active and subscription.filter.matches(envelope):
                    matching.append(subscription)
        
        # Check wildcard subscriptions
        for subscription in self._subscriptions.get("*", []):
            if subscription.active and subscription.filter.matches(envelope):
                matching.append(subscription)
        
        # Direct recipient matching
        if envelope.recipient_id:
            for subscription_list in self._subscriptions.values():
                for subscription in subscription_list:
                    if (subscription.active and 
                        subscription.subscriber_id == envelope.recipient_id and
                        subscription.filter.matches(envelope)):
                        matching.append(subscription)
        
        return matching
    
    async def _deliver_to_subscription(self, envelope: MessageEnvelope, subscription: Subscription) -> bool:
        """Deliver message to a specific subscription."""
        try:
            # Update subscription metrics
            subscription.delivery_count += 1
            subscription.last_delivery = datetime.utcnow()
            
            # Call handler
            if asyncio.iscoroutinefunction(subscription.handler):
                await subscription.handler(envelope)
            else:
                subscription.handler(envelope)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error delivering to subscription {subscription.subscription_id}: {e}")
            return False
    
    async def _cleanup_processor(self) -> None:
        """Clean up expired messages and subscriptions."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Run cleanup every 30 seconds
                await self._cleanup_expired_messages()
            except Exception as e:
                self.logger.error(f"Error in cleanup processor: {e}")
    
    async def _cleanup_expired_messages(self) -> None:
        """Remove expired messages from pending queue."""
        expired_ids = []
        
        for msg_id, envelope in self._pending_messages.items():
            if envelope.is_expired:
                expired_ids.append(msg_id)
        
        for msg_id in expired_ids:
            envelope = self._pending_messages.pop(msg_id)
            envelope.status = MessageStatus.EXPIRED
            self.logger.debug(f"Cleaned up expired message {msg_id}")
    
    async def _emit_event(self, event: BusEvent) -> None:
        """Emit a bus event."""
        self._events.append(event)
        
        # Call event handlers
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
    
    def add_event_handler(self, handler: Callable[[BusEvent], Union[None, Coroutine[Any, Any, None]]]) -> None:
        """Add an event handler."""
        self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[BusEvent], Union[None, Coroutine[Any, Any, None]]]) -> None:
        """Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
    
    def get_connected_agents(self) -> Set[str]:
        """Get list of connected agents."""
        return self._connected_agents.copy()
    
    def get_subscription_count(self, agent_id: Optional[str] = None) -> int:
        """Get subscription count for an agent or total."""
        if agent_id:
            return sum(
                1 for sub in self._subscription_lookup.values()
                if sub.subscriber_id == agent_id and sub.active
            )
        return len([sub for sub in self._subscription_lookup.values() if sub.active])
    
    def get_pending_message_count(self) -> int:
        """Get number of pending messages."""
        return len(self._pending_messages)
    
    def get_message_history(self, limit: Optional[int] = None) -> List[MessageEnvelope]:
        """Get message history."""
        history = list(self._message_history)
        if limit:
            history = history[-limit:]
        return history
    
    def get_recent_events(self, limit: Optional[int] = None) -> List[BusEvent]:
        """Get recent bus events."""
        events = list(self._events)
        if limit:
            events = events[-limit:]
        return events
    
    def get_bus_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bus statistics."""
        return {
            'connected_agents': len(self._connected_agents),
            'total_subscriptions': len(self._subscription_lookup),
            'active_subscriptions': sum(1 for s in self._subscription_lookup.values() if s.active),
            'pending_messages': len(self._pending_messages),
            'message_history_size': len(self._message_history),
            'recent_events': len(self._events),
            'is_running': self._running,
            'queue_size': self._message_queue.qsize()
        }


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get global message bus instance."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus


# Convenience functions

async def send_message(
    message: AgentMessage,
    sender_id: str,
    recipient_id: Optional[str] = None,
    routing_key: Optional[str] = None,
    **kwargs
) -> str:
    """Send a message using the global bus."""
    bus = get_message_bus()
    return await bus.send_message(message, sender_id, recipient_id, routing_key, **kwargs)


async def subscribe(
    subscriber_id: str,
    handler: Callable[[MessageEnvelope], Union[None, Coroutine[Any, Any, None]]],
    message_filter: Optional[MessageFilter] = None
) -> str:
    """Subscribe to messages using the global bus."""
    bus = get_message_bus()
    return await bus.subscribe(subscriber_id, handler, message_filter)


async def subscribe_to_routing_key(
    subscriber_id: str,
    routing_key: str,
    handler: Callable[[MessageEnvelope], Union[None, Coroutine[Any, Any, None]]],
    message_filter: Optional[MessageFilter] = None
) -> str:
    """Subscribe to routing key using the global bus."""
    bus = get_message_bus()
    return await bus.subscribe_to_routing_key(subscriber_id, routing_key, handler, message_filter)


# Agent helper functions

class AgentBusInterface:
    """Interface for agents to interact with the message bus."""
    
    def __init__(self, agent_id: str, bus: Optional[MessageBus] = None):
        self.agent_id = agent_id
        self.bus = bus or get_message_bus()
        self.subscriptions: List[str] = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{agent_id}")
    
    async def connect(self) -> None:
        """Connect agent to the bus."""
        await self.bus.connect_agent(self.agent_id)
    
    async def disconnect(self) -> None:
        """Disconnect agent from the bus."""
        # Unsubscribe from all subscriptions
        for sub_id in self.subscriptions[:]:
            await self.unsubscribe(sub_id)
        
        await self.bus.disconnect_agent(self.agent_id)
    
    async def send(
        self,
        message: AgentMessage,
        recipient_id: Optional[str] = None,
        routing_key: Optional[str] = None,
        **kwargs
    ) -> str:
        """Send a message."""
        return await self.bus.send_message(
            message, self.agent_id, recipient_id, routing_key, **kwargs
        )
    
    async def subscribe(
        self,
        handler: Callable[[MessageEnvelope], Union[None, Coroutine[Any, Any, None]]],
        message_filter: Optional[MessageFilter] = None
    ) -> str:
        """Subscribe to messages."""
        sub_id = await self.bus.subscribe(self.agent_id, handler, message_filter)
        self.subscriptions.append(sub_id)
        return sub_id
    
    async def subscribe_to_routing_key(
        self,
        routing_key: str,
        handler: Callable[[MessageEnvelope], Union[None, Coroutine[Any, Any, None]]],
        message_filter: Optional[MessageFilter] = None
    ) -> str:
        """Subscribe to specific routing key."""
        sub_id = await self.bus.subscribe_to_routing_key(
            self.agent_id, routing_key, handler, message_filter
        )
        self.subscriptions.append(sub_id)
        return sub_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from messages."""
        success = await self.bus.unsubscribe(subscription_id)
        if success and subscription_id in self.subscriptions:
            self.subscriptions.remove(subscription_id)
        return success
    
    async def send_request(self, message: AgentMessage, recipient_id: str, timeout: float = 30.0) -> Optional[AgentMessage]:
        """Send a request and wait for response."""
        correlation_id = str(uuid.uuid4())
        reply_queue = f"reply.{self.agent_id}.{correlation_id}"
        
        # Set up reply handler
        response_message = None
        response_event = asyncio.Event()
        
        async def reply_handler(envelope: MessageEnvelope):
            nonlocal response_message
            if envelope.message.correlation_id == correlation_id:
                response_message = envelope.message
                response_event.set()
        
        # Subscribe to replies
        sub_id = await self.subscribe_to_routing_key(reply_queue, reply_handler)
        
        try:
            # Send request
            await self.send(
                message,
                recipient_id=recipient_id,
                correlation_id=correlation_id,
                reply_to=reply_queue
            )
            
            # Wait for response
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
            return response_message
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Request timeout for correlation_id {correlation_id}")
            return None
        finally:
            # Clean up subscription
            await self.unsubscribe(sub_id)
    
    async def send_reply(self, original_envelope: MessageEnvelope, reply_message: AgentMessage) -> str:
        """Send a reply to a message."""
        if not original_envelope.reply_to:
            raise ValueError("Original message has no reply_to address")
        
        reply_message.correlation_id = original_envelope.correlation_id
        
        return await self.send(
            reply_message,
            routing_key=original_envelope.reply_to
        ) 