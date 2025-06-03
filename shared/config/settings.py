"""
Centralized settings management for the multi-agent data intelligence platform.

This module provides comprehensive configuration management building on the
existing rahil/config.py patterns while adding enterprise features.
"""

import os
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .environment import Environment, get_environment


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Connection parameters
    host: str = Field(..., description="Database host")
    port: int = Field(default=443, ge=1, le=65535, description="Database port")
    database: str = Field(..., description="Database name")
    schema: str = Field(default="PUBLIC", description="Default schema")
    warehouse: str = Field(..., description="Snowflake warehouse")
    
    # Authentication
    username: str = Field(..., description="Database username")
    password: SecretStr = Field(..., description="Database password")
    account: str = Field(..., description="Snowflake account identifier")
    
    # Connection behavior
    role: str = Field(default_factory=lambda: os.getenv('APP_SNOWFLAKE_ROLE', 'APPLICATION_ROLE'), description="Database role")
    autocommit: bool = Field(default=True, description="Enable autocommit")
    client_session_keep_alive: bool = Field(default=True, description="Keep session alive")
    
    # Pool configuration
    max_connections: int = Field(default=10, ge=1, le=100)
    min_connections: int = Field(default=1, ge=0, le=50)
    connection_timeout_seconds: int = Field(default=30, ge=5, le=300)
    query_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    
    @validator('min_connections')
    def validate_min_connections(cls, v, values):
        if 'max_connections' in values and v > values['max_connections']:
            raise ValueError('min_connections cannot be greater than max_connections')
        return v
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for Snowflake connector."""
        params = {
            'account': self.account,
            'user': self.username,
            'password': self.password.get_secret_value(),
            'warehouse': self.warehouse,
            'database': self.database,
            'schema': self.schema,
            'autocommit': self.autocommit,
            'client_session_keep_alive': self.client_session_keep_alive,
            'login_timeout': self.connection_timeout_seconds,
            'network_timeout': self.query_timeout_seconds
        }
        
        if self.role:
            params['role'] = self.role
            
        return params


class GenAIConfig(BaseModel):
    """Google GenAI configuration."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # API configuration
    api_key: SecretStr = Field(..., description="Google GenAI API key")
    
    # Model settings
    default_model: str = Field(default="gemini-2.0-flash-exp", description="Default model to use")
    fallback_model: str = Field(default="gemini-1.5-pro", description="Fallback model")
    
    # Generation parameters
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=1000, ge=1, le=8192)
    default_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    default_top_k: int = Field(default=40, ge=1, le=100)
    
    # Request configuration
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, ge=1, le=1000)
    tokens_per_minute: int = Field(default=10000, ge=100, le=100000)
    
    # Audio configuration (for future Gemini 2.0 features)
    enable_audio: bool = Field(default=False, description="Enable audio processing")
    audio_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    audio_encoding: str = Field(default="LINEAR16", description="Audio encoding format")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    model_config = ConfigDict(extra='forbid')
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
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


class AgentConfig(BaseModel):
    """Agent-specific configuration."""
    model_config = ConfigDict(extra='forbid')
    
    # Agent discovery
    agent_registry_url: Optional[str] = Field(default=None)
    discovery_interval_seconds: int = Field(default=300, ge=30, le=3600)
    
    # Communication
    message_timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_message_retries: int = Field(default=3, ge=0, le=10)
    heartbeat_interval_seconds: int = Field(default=60, ge=10, le=600)
    
    # Performance
    max_concurrent_conversations: int = Field(default=10, ge=1, le=100)
    conversation_timeout_minutes: int = Field(default=30, ge=5, le=480)
    memory_limit_mb: Optional[int] = Field(default=None, ge=100, le=8192)
    
    # Health monitoring
    health_check_interval_seconds: int = Field(default=60, ge=10, le=600)
    health_check_timeout_seconds: int = Field(default=10, ge=1, le=60)


class Settings(BaseSettings):
    """
    Main application settings.
    
    Inherits from BaseSettings to automatically load from environment
    variables and .env files, extending the patterns from rahil/config.py.
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',
        extra='allow',
        case_sensitive=False
    )
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # Database configuration (backward compatible with rahil/config.py)
    SNOWFLAKE_ACCOUNT: str = Field(..., env='SNOWFLAKE_ACCOUNT')
    SNOWFLAKE_USERNAME: str = Field(..., env='SNOWFLAKE_USERNAME')  
    SNOWFLAKE_PASSWORD: SecretStr = Field(..., env='SNOWFLAKE_PASSWORD')
    SNOWFLAKE_DATABASE: str = Field(..., env='SNOWFLAKE_DATABASE')
    SNOWFLAKE_WAREHOUSE: str = Field(..., env='SNOWFLAKE_WAREHOUSE')
    SNOWFLAKE_SCHEMA: str = Field(default='PUBLIC', env='SNOWFLAKE_SCHEMA')
    
    # Google GenAI configuration
    GOOGLE_GENAI_API_KEY: SecretStr = Field(..., env='GOOGLE_GENAI_API_KEY')
    
    # Entity list (backward compatible with rahil/config.py)
    ENTITIES_LIST: str = Field(
        default="customers,products,stores,sales",
        env='ENTITIES_LIST'
    )

    # MCP Server Configurations
    MCP_SNOWFLAKE_SERVER_ENDPOINT: str = Field(default="http://localhost:8001", env="MCP_SNOWFLAKE_SERVER_ENDPOINT")
    MCP_ANALYTICS_SERVER_ENDPOINT: str = Field(default="http://localhost:8002", env="MCP_ANALYTICS_SERVER_ENDPOINT")
    MCP_SNOWFLAKE_API_KEY: Optional[SecretStr] = Field(default=None, env="MCP_SNOWFLAKE_API_KEY")
    MCP_SERVER_DEFAULT_TIMEOUT_SECONDS: int = Field(default=30, env="MCP_SERVER_DEFAULT_TIMEOUT_SECONDS")
    MCP_SERVER_MAX_CONCURRENT_REQUESTS: int = Field(default=10, env="MCP_SERVER_MAX_CONCURRENT_REQUESTS")
    
    # Component configurations
    database: Optional[DatabaseConfig] = None
    genai: Optional[GenAIConfig] = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    
    # Additional paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_path: Path = Field(default_factory=lambda: Path("data"))
    logs_path: Path = Field(default_factory=lambda: Path("logs"))
    config_path: Path = Field(default_factory=lambda: Path("config"))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Auto-create component configurations if not provided
        if self.database is None:
            self.database = DatabaseConfig(
                host=f"{self.SNOWFLAKE_ACCOUNT}.snowflakecomputing.com",
                database=self.SNOWFLAKE_DATABASE,
                warehouse=self.SNOWFLAKE_WAREHOUSE,
                schema=self.SNOWFLAKE_SCHEMA,
                username=self.SNOWFLAKE_USERNAME,
                password=self.SNOWFLAKE_PASSWORD,
                account=self.SNOWFLAKE_ACCOUNT
                # role is now handled by DatabaseConfig default_factory
            )
        
        if self.genai is None:
            self.genai = GenAIConfig(
                api_key=self.GOOGLE_GENAI_API_KEY
            )
    
    @validator('environment', pre=True)
    def parse_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @property
    def entities_list(self) -> List[str]:
        """Parse entities list from string (backward compatible)."""
        return [entity.strip() for entity in self.ENTITIES_LIST.split(',') if entity.strip()]
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING
    
    def get_database_connection_params(self) -> Dict[str, Any]:
        """Get database connection parameters (backward compatible)."""
        return self.database.get_connection_params()
    
    def get_genai_client_config(self) -> Dict[str, Any]:
        """Get GenAI client configuration."""
        return {
            'api_key': self.genai.api_key.get_secret_value(),
            'timeout': self.genai.timeout_seconds,
            'max_retries': self.genai.max_retries,
            'retry_delay': self.genai.retry_delay_seconds
        }
    
    def validate_required_credentials(self) -> List[str]:
        """
        Validate required credentials are present.
        
        Returns list of missing credentials (backward compatible with rahil/config.py).
        """
        missing = []
        
        # Check database credentials
        if not self.SNOWFLAKE_ACCOUNT:
            missing.append('SNOWFLAKE_ACCOUNT')
        if not self.SNOWFLAKE_USERNAME:
            missing.append('SNOWFLAKE_USERNAME')
        if not self.SNOWFLAKE_PASSWORD.get_secret_value():
            missing.append('SNOWFLAKE_PASSWORD')
        if not self.SNOWFLAKE_DATABASE:
            missing.append('SNOWFLAKE_DATABASE')
        if not self.SNOWFLAKE_WAREHOUSE:
            missing.append('SNOWFLAKE_WAREHOUSE')
        
        # Check GenAI credentials
        if not self.GOOGLE_GENAI_API_KEY.get_secret_value():
            missing.append('GOOGLE_GENAI_API_KEY')
        
        return missing
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        for path in [self.data_path, self.logs_path, self.config_path]:
            path.mkdir(parents=True, exist_ok=True)
    

        

# Global settings instance
_settings: Optional[Settings] = None


@lru_cache()
def get_settings() -> Settings:
    """
    Get global settings instance.
    
    Uses LRU cache to ensure singleton behavior and improve performance.
    """
    global _settings
    
    if _settings is None:
        _settings = Settings()
        
        # Validate required credentials
        missing_creds = _settings.validate_required_credentials()
        if missing_creds:
            logger = logging.getLogger(__name__)
            logger.error(f"Missing required credentials: {', '.join(missing_creds)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_creds)}")
        
        # Ensure directories exist
        _settings.ensure_directories()
        
        logger = logging.getLogger(__name__)
        logger.info(f"Settings loaded for environment: {_settings.environment.value}")
    
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment/files.
    
    Useful for testing or when configuration changes at runtime.
    """
    global _settings
    _settings = None
    get_settings.cache_clear()
    return get_settings() 