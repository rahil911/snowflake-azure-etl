"""
Environment management for the multi-agent data intelligence platform.

This module provides environment detection and configuration management
for different deployment environments (development, testing, production).
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Dict, Any, Optional


class Environment(str, Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    @property
    def is_development(self) -> bool:
        """Check if this is development environment."""
        return self == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if this is testing environment."""
        return self == Environment.TESTING
    
    @property
    def is_staging(self) -> bool:
        """Check if this is staging environment."""
        return self == Environment.STAGING
    
    @property
    def is_production(self) -> bool:
        """Check if this is production environment."""
        return self == Environment.PRODUCTION
    
    @property
    def is_debug_enabled(self) -> bool:
        """Check if debug mode should be enabled."""
        return self in {Environment.DEVELOPMENT, Environment.TESTING}
    
    @property
    def is_metrics_enabled(self) -> bool:
        """Check if detailed metrics should be collected."""
        return self in {Environment.STAGING, Environment.PRODUCTION}
    
    @property
    def log_level(self) -> str:
        """Get default log level for environment."""
        if self.is_production:
            return "WARNING"
        elif self.is_staging:
            return "INFO"
        else:
            return "DEBUG"
    
    @property
    def default_config(self) -> Dict[str, Any]:
        """Get default configuration for environment."""
        base_config = {
            'debug': self.is_debug_enabled,
            'log_level': self.log_level,
            'metrics_enabled': self.is_metrics_enabled,
            'detailed_logging': not self.is_production
        }
        
        # Environment-specific configurations
        if self.is_development:
            base_config.update({
                'enable_hot_reload': True,
                'enable_profiling': True,
                'strict_validation': False,
                'cache_ttl_seconds': 60,
                'connection_pool_size': 2
            })
        elif self.is_testing:
            base_config.update({
                'enable_hot_reload': False,
                'enable_profiling': False,
                'strict_validation': True,
                'cache_ttl_seconds': 30,
                'connection_pool_size': 1,
                'mock_external_services': True
            })
        elif self.is_staging:
            base_config.update({
                'enable_hot_reload': False,
                'enable_profiling': False,
                'strict_validation': True,
                'cache_ttl_seconds': 300,
                'connection_pool_size': 5,
                'enable_performance_monitoring': True
            })
        elif self.is_production:
            base_config.update({
                'enable_hot_reload': False,
                'enable_profiling': False,
                'strict_validation': True,
                'cache_ttl_seconds': 3600,
                'connection_pool_size': 10,
                'enable_performance_monitoring': True,
                'enable_alerting': True
            })
        
        return base_config


class EnvironmentConfig:
    """Configuration manager for environment-specific settings."""
    
    def __init__(self, environment: Environment):
        self.environment = environment
        self._config = environment.default_config
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config.update(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('debug', False)
    
    @property
    def log_level(self) -> str:
        """Get log level."""
        return self.get('log_level', 'INFO')
    
    @property
    def metrics_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self.get('metrics_enabled', False)
    
    @property
    def connection_pool_size(self) -> int:
        """Get connection pool size."""
        return self.get('connection_pool_size', 5)
    
    @property
    def cache_ttl_seconds(self) -> int:
        """Get cache TTL in seconds."""
        return self.get('cache_ttl_seconds', 300)


@lru_cache()
def get_environment() -> Environment:
    """
    Detect and return current environment.
    
    Uses the following precedence:
    1. ENVIRONMENT environment variable
    2. APP_ENV environment variable  
    3. NODE_ENV environment variable (for compatibility)
    4. Defaults to DEVELOPMENT
    """
    env_vars = ['ENVIRONMENT', 'APP_ENV', 'NODE_ENV']
    
    for env_var in env_vars:
        env_value = os.getenv(env_var)
        if env_value:
            try:
                return Environment(env_value.lower())
            except ValueError:
                # Invalid environment value, continue to next
                continue
    
    # Default to development
    return Environment.DEVELOPMENT


@lru_cache()
def get_environment_config() -> EnvironmentConfig:
    """Get environment configuration."""
    return EnvironmentConfig(get_environment())


def is_development() -> bool:
    """Check if running in development environment."""
    return get_environment().is_development


def is_testing() -> bool:
    """Check if running in testing environment."""
    return get_environment().is_testing


def is_staging() -> bool:
    """Check if running in staging environment."""
    return get_environment().is_staging


def is_production() -> bool:
    """Check if running in production environment."""
    return get_environment().is_production


def get_debug_mode() -> bool:
    """Get debug mode setting for current environment."""
    return get_environment().is_debug_enabled


def get_log_level() -> str:
    """Get log level for current environment."""
    return get_environment().log_level


def override_environment(environment: Environment) -> None:
    """
    Override detected environment (useful for testing).
    
    This clears the LRU cache to force re-detection.
    """
    # Clear caches to force re-detection
    get_environment.cache_clear()
    get_environment_config.cache_clear()
    
    # Set environment variable
    os.environ['ENVIRONMENT'] = environment.value


def reset_environment() -> None:
    """Reset environment detection to auto-detect."""
    # Remove override
    if 'ENVIRONMENT' in os.environ:
        del os.environ['ENVIRONMENT']
    
    # Clear caches
    get_environment.cache_clear()
    get_environment_config.cache_clear()


def get_environment_info() -> Dict[str, Any]:
    """Get comprehensive environment information."""
    env = get_environment()
    config = get_environment_config()
    
    return {
        'environment': env.value,
        'is_development': env.is_development,
        'is_testing': env.is_testing,
        'is_staging': env.is_staging,
        'is_production': env.is_production,
        'debug_enabled': env.is_debug_enabled,
        'metrics_enabled': env.is_metrics_enabled,
        'log_level': env.log_level,
        'config': config.to_dict(),
        'env_vars': {
            'ENVIRONMENT': os.getenv('ENVIRONMENT'),
            'APP_ENV': os.getenv('APP_ENV'),
            'NODE_ENV': os.getenv('NODE_ENV')
        }
    } 