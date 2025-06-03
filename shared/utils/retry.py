"""
Retry utilities for the multi-agent data intelligence platform.

This module provides sophisticated retry mechanisms with exponential backoff,
circuit breakers, and configurable retry strategies for reliable operations.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ConfigDict


class RetryStrategy(str, Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"


class RetryableError(Exception):
    """Base exception for retryable errors."""
    pass


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, attempt_count: int, last_error: Exception, retry_history: List[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.attempt_count = attempt_count
        self.last_error = last_error
        self.retry_history = retry_history or []
        self.timestamp = datetime.utcnow()


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    exception: Optional[Exception] = None
    delay_seconds: float = 0.0
    success: bool = False
    
    @property
    def duration_ms(self) -> float:
        """Get attempt duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    model_config = ConfigDict(extra='forbid')
    
    # Basic retry settings
    max_attempts: int = Field(default=3, ge=1, le=20)
    strategy: RetryStrategy = Field(default=RetryStrategy.EXPONENTIAL_BACKOFF)
    
    # Delay configuration
    initial_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=3600.0)
    multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    jitter: bool = Field(default=True, description="Add random jitter to delays")
    
    # Exception handling
    retryable_exceptions: List[str] = Field(
        default_factory=lambda: ['ConnectionError', 'TimeoutError', 'RetryableError']
    )
    non_retryable_exceptions: List[str] = Field(
        default_factory=lambda: ['ValueError', 'TypeError', 'KeyError']
    )
    
    # Circuit breaker settings
    enable_circuit_breaker: bool = Field(default=False)
    failure_threshold: int = Field(default=5, ge=1, le=20)
    recovery_timeout: float = Field(default=60.0, ge=10.0, le=3600.0)
    
    # Logging and monitoring
    log_attempts: bool = Field(default=True)
    include_stack_trace: bool = Field(default=False)
    
    def is_retryable_exception(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry."""
        exception_name = type(exception).__name__
        
        # Check non-retryable first (takes precedence)
        if exception_name in self.non_retryable_exceptions:
            return False
        
        # Check if it's explicitly retryable
        if exception_name in self.retryable_exceptions:
            return True
        
        # Check if it's a subclass of RetryableError
        return isinstance(exception, RetryableError)
    
    def calculate_delay(self, attempt_number: int) -> float:
        """Calculate delay for a given attempt number."""
        if self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.initial_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.initial_delay * (self.multiplier ** (attempt_number - 1))
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.initial_delay * attempt_number
        elif self.strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = self.initial_delay * (self.multiplier ** (attempt_number - 1))
            delay = base_delay * (0.5 + random.random() * 0.5)  # 50-100% of base delay
        else:
            delay = self.initial_delay
        
        # Apply jitter if enabled
        if self.jitter and self.strategy != RetryStrategy.JITTERED_BACKOFF:
            jitter_amount = delay * 0.1 * random.random()  # Up to 10% jitter
            delay += jitter_amount
        
        # Ensure delay doesn't exceed maximum
        return min(delay, self.max_delay)


class CircuitBreaker:
    """Circuit breaker implementation for retry operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit breaker state."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)):
                self.state = "half-open"
                self.logger.info("Circuit breaker moving to half-open state")
                return True
            return False
        elif self.state == "half-open":
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
            self.logger.info("Circuit breaker closed after successful recovery")
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != "open":
                self.state = "open"
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RetryManager:
    """Manages retry operations with configuration and monitoring."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout
        ) if config.enable_circuit_breaker else None
        
        # Metrics
        self.total_attempts = 0
        self.total_successes = 0
        self.total_failures = 0
        self.retry_history: List[RetryAttempt] = []
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a function with retry logic."""
        if asyncio.iscoroutinefunction(func):
            return await self._execute_async_with_retry(func, *args, **kwargs)
        else:
            return await self._execute_sync_with_retry(func, *args, **kwargs)
    
    async def _execute_async_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute async function with retry logic."""
        attempts = []
        last_exception = None
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.can_execute():
                raise RetryError(
                    "Circuit breaker is open",
                    attempt_count=attempt_num - 1,
                    last_error=last_exception or Exception("Circuit breaker open"),
                    retry_history=[attempt.__dict__ for attempt in attempts]
                )
            
            attempt = RetryAttempt(
                attempt_number=attempt_num,
                start_time=datetime.utcnow()
            )
            
            try:
                if self.config.log_attempts:
                    self.logger.debug(f"Attempt {attempt_num}/{self.config.max_attempts}")
                
                result = await func(*args, **kwargs)
                
                # Success
                attempt.end_time = datetime.utcnow()
                attempt.success = True
                attempts.append(attempt)
                
                self.total_attempts += attempt_num
                self.total_successes += 1
                self.retry_history.extend(attempts)
                
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                if self.config.log_attempts and attempt_num > 1:
                    self.logger.info(f"Operation succeeded on attempt {attempt_num}")
                
                return result
                
            except Exception as e:
                attempt.end_time = datetime.utcnow()
                attempt.exception = e
                attempts.append(attempt)
                last_exception = e
                
                # Check if this exception should trigger a retry
                if not self.config.is_retryable_exception(e):
                    if self.config.log_attempts:
                        self.logger.error(f"Non-retryable exception: {type(e).__name__}: {str(e)}")
                    raise e
                
                # Log the attempt failure
                if self.config.log_attempts:
                    log_level = logging.WARNING if attempt_num == self.config.max_attempts else logging.DEBUG
                    self.logger.log(
                        log_level,
                        f"Attempt {attempt_num} failed: {type(e).__name__}: {str(e)}"
                    )
                
                # If this was the last attempt, don't wait
                if attempt_num >= self.config.max_attempts:
                    break
                
                # Calculate and apply delay
                delay = self.config.calculate_delay(attempt_num)
                attempt.delay_seconds = delay
                
                if self.config.log_attempts:
                    self.logger.debug(f"Waiting {delay:.2f} seconds before retry")
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        self.total_attempts += self.config.max_attempts
        self.total_failures += 1
        self.retry_history.extend(attempts)
        
        if self.circuit_breaker:
            self.circuit_breaker.record_failure()
        
        raise RetryError(
            f"All {self.config.max_attempts} retry attempts failed",
            attempt_count=self.config.max_attempts,
            last_error=last_exception,
            retry_history=[attempt.__dict__ for attempt in attempts]
        )
    
    async def _execute_sync_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute sync function with retry logic (run in executor)."""
        import concurrent.futures
        
        def sync_wrapper():
            return func(*args, **kwargs)
        
        # Run sync function in thread pool
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            
            async def async_func():
                return await loop.run_in_executor(executor, sync_wrapper)
            
            return await self._execute_async_with_retry(async_func)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        success_rate = self.total_successes / max(self.total_successes + self.total_failures, 1)
        
        return {
            'total_attempts': self.total_attempts,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'success_rate': success_rate,
            'config': self.config.model_dump(),
            'circuit_breaker': {
                'enabled': self.circuit_breaker is not None,
                'state': self.circuit_breaker.state if self.circuit_breaker else None,
                'failure_count': self.circuit_breaker.failure_count if self.circuit_breaker else 0
            } if self.circuit_breaker else None,
            'recent_attempts': [
                attempt.__dict__ for attempt in self.retry_history[-10:]
            ]
        }


# Decorator functions

def retry_with_backoff(
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    **config_kwargs
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        strategy: Retry strategy to use
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        multiplier: Backoff multiplier
        retryable_exceptions: List of exception types that should trigger retry
        **config_kwargs: Additional configuration options
    
    Usage:
        @retry_with_backoff(max_attempts=5, initial_delay=0.5)
        async def unreliable_api_call():
            # This will be retried up to 5 times with exponential backoff
            pass
    """
    # Convert exception types to names for config
    exception_names = []
    if retryable_exceptions:
        exception_names = [exc.__name__ for exc in retryable_exceptions]
    else:
        exception_names = ['ConnectionError', 'TimeoutError', 'RetryableError']
    
    config = RetryConfig(
        max_attempts=max_attempts,
        strategy=strategy,
        initial_delay=initial_delay,
        max_delay=max_delay,
        multiplier=multiplier,
        retryable_exceptions=exception_names,
        **config_kwargs
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = RetryManager(config)
            return await manager.execute_with_retry(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            manager = RetryManager(config)
            return loop.run_until_complete(manager.execute_with_retry(func, *args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def exponential_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    jitter: bool = True
):
    """
    Simple exponential backoff decorator.
    
    Usage:
        @exponential_backoff(max_attempts=5)
        def flaky_operation():
            pass
    """
    return retry_with_backoff(
        max_attempts=max_attempts,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay=initial_delay,
        max_delay=max_delay,
        multiplier=multiplier,
        jitter=jitter
    )


def retry_on_exception(*exception_types, max_attempts: int = 3):
    """
    Retry only on specific exception types.
    
    Usage:
        @retry_on_exception(ConnectionError, TimeoutError, max_attempts=5)
        def network_operation():
            pass
    """
    return retry_with_backoff(
        max_attempts=max_attempts,
        retryable_exceptions=list(exception_types)
    )


# Utility functions

async def with_retry(
    func: Callable,
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> Any:
    """
    Execute a function with retry logic.
    
    Args:
        func: Function to execute
        config: Retry configuration (uses defaults if not provided)
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Function result
    
    Usage:
        result = await with_retry(api_call, config, param1, param2)
    """
    if config is None:
        config = RetryConfig()
    
    manager = RetryManager(config)
    return await manager.execute_with_retry(func, *args, **kwargs)


def create_retry_config(
    max_attempts: int = 3,
    strategy: str = "exponential_backoff",
    initial_delay: float = 1.0,
    **kwargs
) -> RetryConfig:
    """
    Create a retry configuration with sensible defaults.
    
    Args:
        max_attempts: Maximum retry attempts
        strategy: Retry strategy name
        initial_delay: Initial delay in seconds
        **kwargs: Additional configuration options
    
    Returns:
        RetryConfig instance
    """
    return RetryConfig(
        max_attempts=max_attempts,
        strategy=RetryStrategy(strategy),
        initial_delay=initial_delay,
        **kwargs
    )


class RetryableConnectionError(RetryableError):
    """Retryable connection error."""
    pass


class RetryableTimeoutError(RetryableError):
    """Retryable timeout error."""
    pass


class RetryableServiceError(RetryableError):
    """Retryable service error (e.g., 503 Service Unavailable)."""
    pass


# Pre-configured retry decorators for common scenarios

database_retry = retry_with_backoff(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=30.0,
    retryable_exceptions=[RetryableConnectionError, RetryableTimeoutError]
)

api_retry = retry_with_backoff(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=60.0,
    retryable_exceptions=[RetryableConnectionError, RetryableServiceError]
)

file_operation_retry = retry_with_backoff(
    max_attempts=3,
    initial_delay=0.1,
    max_delay=5.0,
    strategy=RetryStrategy.LINEAR_BACKOFF
) 