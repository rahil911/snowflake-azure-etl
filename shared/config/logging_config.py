"""
Logging configuration for the multi-agent data intelligence platform.

This module provides centralized logging setup with support for file rotation,
structured logging, and environment-specific configurations.
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from .environment import get_environment, Environment


class LoggingConfig(BaseModel):
    """Logging configuration model."""
    model_config = ConfigDict(extra='forbid')
    
    # Basic settings
    level: str = Field(default="INFO", description="Root logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        description="Log message format"
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for log messages"
    )
    
    # File logging
    enable_file_logging: bool = Field(default=True)
    log_file_path: str = Field(default="logs/agent_platform.log")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    backup_count: int = Field(default=5, ge=1, le=50)
    
    # Console logging
    enable_console_logging: bool = Field(default=True)
    console_level: str = Field(default="INFO")
    
    # Structured logging
    enable_json_logging: bool = Field(default=False)
    include_trace_id: bool = Field(default=True)
    include_hostname: bool = Field(default=False)
    include_process_info: bool = Field(default=False)
    
    # Component-specific logging levels
    component_levels: Dict[str, str] = Field(default_factory=dict)
    
    # Filtering
    suppress_noisy_loggers: bool = Field(default=True)
    noisy_loggers: list[str] = Field(
        default_factory=lambda: [
            'urllib3.connectionpool',
            'requests.packages.urllib3',
            'snowflake.connector',
            'botocore',
            'boto3'
        ]
    )
    
    def get_effective_level(self, logger_name: str = None) -> str:
        """Get effective log level for a logger."""
        if logger_name and logger_name in self.component_levels:
            return self.component_levels[logger_name]
        return self.level


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, config: LoggingConfig):
        super().__init__()
        self.config = config
        self.hostname = None
        self.process_id = None
        
        if config.include_hostname:
            import socket
            self.hostname = socket.gethostname()
        
        if config.include_process_info:
            import os
            self.process_id = os.getpid()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add stack info if present
        if record.stack_info:
            log_entry['stack_info'] = record.stack_info
        
        # Add trace ID if available
        if self.config.include_trace_id and hasattr(record, 'trace_id'):
            log_entry['trace_id'] = record.trace_id
        
        # Add hostname if configured
        if self.hostname:
            log_entry['hostname'] = self.hostname
        
        # Add process info if configured
        if self.process_id:
            log_entry['process_id'] = self.process_id
            log_entry['thread_id'] = record.thread
            log_entry['thread_name'] = record.threadName
        
        # Add any additional fields from record
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'exc_info', 'exc_text', 
                          'stack_info', 'getMessage'}:
                if not key.startswith('_'):
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class TraceIDFilter(logging.Filter):
    """Filter to add trace ID to log records."""
    
    def __init__(self):
        super().__init__()
        self._trace_id_context = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add trace ID to log record if available."""
        # Try to get trace ID from various sources
        trace_id = self._get_trace_id()
        if trace_id:
            record.trace_id = trace_id
        return True
    
    def _get_trace_id(self) -> Optional[str]:
        """Get trace ID from context or generate one."""
        # Try to get from threading local storage
        try:
            import threading
            thread_local = getattr(threading.current_thread(), '_trace_id', None)
            if thread_local:
                return thread_local
        except:
            pass
        
        # Try to get from contextvars (Python 3.7+)
        try:
            import contextvars
            trace_var = contextvars.ContextVar('trace_id', default=None)
            trace_id = trace_var.get()
            if trace_id:
                return trace_id
        except:
            pass
        
        return None
    
    def set_trace_id(self, trace_id: str):
        """Set trace ID for current context."""
        try:
            import threading
            threading.current_thread()._trace_id = trace_id
        except:
            pass
        
        try:
            import contextvars
            trace_var = contextvars.ContextVar('trace_id', default=None)
            trace_var.set(trace_id)
        except:
            pass


def setup_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        config: Optional logging configuration. If not provided, will use
                environment-appropriate defaults.
    
    Returns:
        Root logger instance
    """
    # Use environment-specific defaults if no config provided
    if config is None:
        config = get_default_logging_config()
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Set root level
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Setup file logging
    if config.enable_file_logging:
        _setup_file_logging(root_logger, config)
    
    # Setup console logging
    if config.enable_console_logging:
        _setup_console_logging(root_logger, config)
    
    # Setup component-specific log levels
    _setup_component_levels(config)
    
    # Suppress noisy loggers
    if config.suppress_noisy_loggers:
        _suppress_noisy_loggers(config.noisy_loggers)
    
    # Add trace ID filter if enabled
    if config.include_trace_id:
        trace_filter = TraceIDFilter()
        root_logger.addFilter(trace_filter)
    
    root_logger.info(f"Logging configured for environment: {get_environment().value}")
    return root_logger


def _setup_file_logging(logger: logging.Logger, config: LoggingConfig) -> None:
    """Setup file logging with rotation."""
    # Ensure log directory exists
    log_path = Path(config.log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=config.log_file_path,
        maxBytes=config.max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
        backupCount=config.backup_count,
        encoding='utf-8'
    )
    
    # Set formatter
    if config.enable_json_logging:
        formatter = StructuredFormatter(config)
    else:
        formatter = logging.Formatter(
            fmt=config.format,
            datefmt=config.date_format
        )
    
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, config.level.upper()))
    
    logger.addHandler(file_handler)


def _setup_console_logging(logger: logging.Logger, config: LoggingConfig) -> None:
    """Setup console logging."""
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Use simple format for console in development, structured in production
    env = get_environment()
    if env.is_production and config.enable_json_logging:
        formatter = StructuredFormatter(config)
    else:
        # Simpler format for console
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(
            fmt=console_format,
            datefmt=config.date_format
        )
    
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, config.console_level.upper()))
    
    logger.addHandler(console_handler)


def _setup_component_levels(config: LoggingConfig) -> None:
    """Setup component-specific logging levels."""
    for logger_name, level in config.component_levels.items():
        component_logger = logging.getLogger(logger_name)
        component_logger.setLevel(getattr(logging, level.upper()))


def _suppress_noisy_loggers(noisy_loggers: list[str]) -> None:
    """Suppress noisy third-party loggers."""
    for logger_name in noisy_loggers:
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.setLevel(logging.WARNING)


def get_default_logging_config() -> LoggingConfig:
    """Get default logging configuration based on environment."""
    env = get_environment()
    
    # Base configuration
    config_dict = {
        'level': env.log_level,
        'enable_console_logging': True,
        'enable_file_logging': True,
        'suppress_noisy_loggers': True
    }
    
    # Environment-specific overrides
    if env.is_development:
        config_dict.update({
            'console_level': 'DEBUG',
            'enable_json_logging': False,
            'include_trace_id': True,
            'include_process_info': False
        })
    elif env.is_testing:
        config_dict.update({
            'console_level': 'WARNING',  # Reduce noise in tests
            'enable_file_logging': False,  # No file logging in tests
            'enable_json_logging': False,
            'include_trace_id': False
        })
    elif env.is_production:
        config_dict.update({
            'console_level': 'WARNING',
            'enable_json_logging': True,
            'include_trace_id': True,
            'include_hostname': True,
            'include_process_info': True,
            'max_file_size_mb': 500,
            'backup_count': 10
        })
    
    return LoggingConfig(**config_dict)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    This is a convenience function that ensures logging is properly configured.
    """
    # Ensure logging is configured
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)


def configure_logger_for_agent(agent_name: str, agent_id: str = None) -> logging.Logger:
    """
    Configure a logger specifically for an agent.
    
    Args:
        agent_name: Name of the agent (e.g., 'coordinator', 'data_intelligence')
        agent_id: Optional agent ID for more specific logging
    
    Returns:
        Configured logger instance
    """
    logger_name = f"agents.{agent_name}"
    if agent_id:
        logger_name += f".{agent_id[:8]}"  # Use first 8 chars of ID
    
    logger = get_logger(logger_name)
    
    # Add agent-specific context
    logger = logging.LoggerAdapter(logger, {
        'agent_name': agent_name,
        'agent_id': agent_id
    })
    
    return logger


def configure_logger_for_tool(tool_name: str, tool_id: str = None) -> logging.Logger:
    """
    Configure a logger specifically for a tool.
    
    Args:
        tool_name: Name of the tool (e.g., 'snowflake_query', 'analytics')
        tool_id: Optional tool ID for more specific logging
    
    Returns:
        Configured logger instance
    """
    logger_name = f"tools.{tool_name}"
    if tool_id:
        logger_name += f".{tool_id[:8]}"
    
    logger = get_logger(logger_name)
    
    # Add tool-specific context
    logger = logging.LoggerAdapter(logger, {
        'tool_name': tool_name,
        'tool_id': tool_id
    })
    
    return logger


def set_trace_id(trace_id: str) -> None:
    """Set trace ID for current execution context."""
    # Get trace filter and set trace ID
    root_logger = logging.getLogger()
    for filter_obj in root_logger.filters:
        if isinstance(filter_obj, TraceIDFilter):
            filter_obj.set_trace_id(trace_id)
            break


def get_logging_status() -> Dict[str, Any]:
    """Get current logging configuration status."""
    root_logger = logging.getLogger()
    
    return {
        'root_level': logging.getLevelName(root_logger.level),
        'handlers': [
            {
                'type': type(handler).__name__,
                'level': logging.getLevelName(handler.level),
                'formatter': type(handler.formatter).__name__ if handler.formatter else None
            }
            for handler in root_logger.handlers
        ],
        'filters': [type(filter_obj).__name__ for filter_obj in root_logger.filters],
        'environment': get_environment().value
    } 