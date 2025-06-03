"""
Configuration management for the multi-agent data intelligence platform.

This module provides centralized configuration management with support for
environment variables, file-based config, and validation.
"""

from .settings import Settings, get_settings, DatabaseConfig, GenAIConfig
from .logging_config import setup_logging, LoggingConfig
from .secrets_manager import SecretsManager, SecretValue
from .environment import Environment, get_environment

__all__ = [
    'Settings',
    'get_settings',
    'DatabaseConfig',
    'GenAIConfig',
    'setup_logging',
    'LoggingConfig',
    'SecretsManager',
    'SecretValue',
    'Environment',
    'get_environment'
]

__version__ = '1.0.0' 