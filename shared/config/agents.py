"""
Agent Configuration Management
==============================

Configuration management specifically for individual agents in the multi-agent system.
Each agent can have specialized configurations while inheriting from base settings.

Based on requirements in AGENT_DEPLOYMENT_GUIDE.md and BIG_PICTURE.md.
"""

from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, validator
from .settings import BaseSettings


class AgentType(str, Enum):
    """Types of agents in the system."""
    COORDINATOR = "coordinator"
    DATA_INTELLIGENCE = "data_intelligence"
    ETL_AGENT = "etl_agent"
    VISUALIZATION = "visualization"
    CUSTOM = "custom"


class AgentCapability(str, Enum):
    """Standard agent capabilities."""
    TEXT_PROCESSING = "text_processing"
    AUDIO_PROCESSING = "audio_processing"
    IMAGE_PROCESSING = "image_processing"
    DATABASE_ACCESS = "database_access"
    WEB_API_ACCESS = "web_api_access"
    FILE_OPERATIONS = "file_operations"
    ML_INFERENCE = "ml_inference"
    DATA_VISUALIZATION = "data_visualization"
    NATURAL_LANGUAGE_GENERATION = "natural_language_generation"
    SQL_GENERATION = "sql_generation"
    REPORT_GENERATION = "report_generation"


class AgentConfig(BaseModel):
    """Configuration for a single agent."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='allow'
    )
    
    # Agent identity
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: AgentType = Field(..., description="Type of agent")
    name: str = Field(..., description="Human-readable agent name")
    description: str = Field(default="", description="Agent description")
    version: str = Field(default="1.0.0", description="Agent version")
    
    # Capabilities
    capabilities: Set[AgentCapability] = Field(
        default_factory=set, description="Agent capabilities"
    )
    
    # Model configuration
    model_name: str = Field(default="gemini-2.0-flash-exp", description="LLM model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=2048, ge=1, le=32000, description="Maximum tokens")
    
    # Communication settings
    max_concurrent_conversations: int = Field(default=5, ge=1, le=100)
    message_queue_size: int = Field(default=1000, ge=10, le=10000)
    response_timeout_seconds: int = Field(default=30, ge=5, le=300)
    
    # Tool access
    allowed_tools: Set[str] = Field(default_factory=set, description="Allowed MCP tools")
    tool_timeout_seconds: int = Field(default=60, ge=5, le=600)
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=300, ge=10, le=3600)
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    
    # Specific agent configurations
    agent_specific_config: Dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific configuration parameters"
    )
    
    @validator('agent_id')
    def validate_agent_id(cls, v):
        """Validate agent ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Agent ID must contain only alphanumeric characters, hyphens, and underscores")
        return v.lower()


class CoordinatorAgentConfig(AgentConfig):
    """Configuration specific to the Coordinator Agent."""
    
    agent_type: AgentType = Field(default=AgentType.COORDINATOR, description="Agent type")
    
    # Voice/Audio settings for Gemini 2.0 Live API
    enable_voice: bool = Field(default=True, description="Enable voice interactions")
    voice_language: str = Field(default="en-US", description="Voice language code")
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Voice speed multiplier")
    
    # Agent routing configuration
    routing_strategy: str = Field(default="capability_based", description="Agent routing strategy")
    max_agent_hops: int = Field(default=5, ge=1, le=10, description="Maximum agent routing hops")
    
    # Session management
    session_timeout_minutes: int = Field(default=30, ge=5, le=240)
    max_active_sessions: int = Field(default=100, ge=1, le=1000)
    
    def __init__(self, **kwargs):
        # Set default capabilities for coordinator
        if 'capabilities' not in kwargs:
            kwargs['capabilities'] = {
                AgentCapability.TEXT_PROCESSING,
                AgentCapability.AUDIO_PROCESSING,
                AgentCapability.NATURAL_LANGUAGE_GENERATION
            }
        super().__init__(**kwargs)


class DataIntelligenceAgentConfig(AgentConfig):
    """Configuration specific to the Data Intelligence Agent."""
    
    agent_type: AgentType = Field(default=AgentType.DATA_INTELLIGENCE, description="Agent type")
    
    # SQL generation settings
    sql_dialect: str = Field(default="snowflake", description="SQL dialect for generation")
    max_query_complexity: int = Field(default=5, ge=1, le=10, description="Maximum query complexity")
    enable_query_optimization: bool = Field(default=True, description="Enable query optimization")
    
    # Data analysis settings
    max_result_rows: int = Field(default=10000, ge=100, le=1000000)
    enable_statistical_analysis: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Business intelligence
    enable_insights_generation: bool = Field(default=True)
    insight_categories: List[str] = Field(
        default_factory=lambda: ["trends", "anomalies", "patterns", "recommendations"]
    )
    
    def __init__(self, **kwargs):
        # Set default capabilities for data intelligence
        if 'capabilities' not in kwargs:
            kwargs['capabilities'] = {
                AgentCapability.TEXT_PROCESSING,
                AgentCapability.DATABASE_ACCESS,
                AgentCapability.SQL_GENERATION,
                AgentCapability.ML_INFERENCE,
                AgentCapability.REPORT_GENERATION
            }
        
        # Set default allowed tools
        if 'allowed_tools' not in kwargs:
            kwargs['allowed_tools'] = {
                "snowflake_query",
                "data_analysis",
                "statistical_analysis",
                "pattern_detection"
            }
        
        super().__init__(**kwargs)


class ETLAgentConfig(AgentConfig):
    """Configuration specific to the ETL Agent."""
    
    agent_type: AgentType = Field(default=AgentType.ETL_AGENT, description="Agent type")
    
    # Pipeline monitoring
    monitoring_interval_seconds: int = Field(default=60, ge=10, le=3600)
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "error_rate": 0.05,
            "processing_time_multiplier": 2.0,
            "memory_usage_percent": 80.0
        }
    )
    
    # Data quality settings
    enable_data_quality_checks: bool = Field(default=True)
    quality_check_sample_size: int = Field(default=1000, ge=100, le=100000)
    min_quality_score: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Integration with existing ETL
    rahil_pipeline_path: str = Field(default="rahil/", description="Path to existing ETL code")
    enable_legacy_monitoring: bool = Field(default=True)
    
    def __init__(self, **kwargs):
        # Set default capabilities for ETL agent
        if 'capabilities' not in kwargs:
            kwargs['capabilities'] = {
                AgentCapability.TEXT_PROCESSING,
                AgentCapability.DATABASE_ACCESS,
                AgentCapability.FILE_OPERATIONS,
                AgentCapability.ML_INFERENCE
            }
        
        # Set default allowed tools
        if 'allowed_tools' not in kwargs:
            kwargs['allowed_tools'] = {
                "snowflake_query",
                "data_quality_check",
                "pipeline_status",
                "performance_monitor"
            }
        
        super().__init__(**kwargs)


class VisualizationAgentConfig(AgentConfig):
    """Configuration specific to the Visualization Agent (future)."""
    
    agent_type: AgentType = Field(default=AgentType.VISUALIZATION, description="Agent type")
    
    # Chart generation settings
    default_chart_library: str = Field(default="plotly", description="Default charting library")
    max_data_points: int = Field(default=50000, ge=100, le=1000000)
    enable_interactive_charts: bool = Field(default=True)
    
    # Export settings
    supported_formats: List[str] = Field(
        default_factory=lambda: ["png", "svg", "pdf", "html"]
    )
    default_resolution: str = Field(default="1920x1080", description="Default image resolution")
    
    def __init__(self, **kwargs):
        # Set default capabilities for visualization agent
        if 'capabilities' not in kwargs:
            kwargs['capabilities'] = {
                AgentCapability.TEXT_PROCESSING,
                AgentCapability.DATA_VISUALIZATION,
                AgentCapability.FILE_OPERATIONS
            }
        
        # Set default allowed tools
        if 'allowed_tools' not in kwargs:
            kwargs['allowed_tools'] = {
                "chart_generator",
                "image_export",
                "data_formatter"
            }
        
        super().__init__(**kwargs)


class AgentRegistry(BaseModel):
    """Registry of all agent configurations."""
    
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    
    def register_agent(self, config: AgentConfig) -> None:
        """Register an agent configuration."""
        self.agents[config.agent_id] = config
    
    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentConfig]:
        """Get all agents of a specific type."""
        return [
            config for config in self.agents.values()
            if config.agent_type == agent_type
        ]
    
    def get_agents_with_capability(self, capability: AgentCapability) -> List[AgentConfig]:
        """Get all agents with a specific capability."""
        return [
            config for config in self.agents.values()
            if capability in config.capabilities
        ]
    
    def get_agents_with_tool(self, tool_name: str) -> List[AgentConfig]:
        """Get all agents that can use a specific tool."""
        return [
            config for config in self.agents.values()
            if tool_name in config.allowed_tools
        ]


class AgentSettings(BaseSettings):
    """Global agent system settings."""
    
    # System-wide agent settings
    DEFAULT_AGENT_TIMEOUT: int = Field(default=30, env='DEFAULT_AGENT_TIMEOUT')
    MAX_CONCURRENT_AGENTS: int = Field(default=10, env='MAX_CONCURRENT_AGENTS')
    AGENT_HEALTH_CHECK_INTERVAL: int = Field(default=60, env='AGENT_HEALTH_CHECK_INTERVAL')
    
    # Model bus settings
    MESSAGE_BUS_TYPE: str = Field(default="local", env='MESSAGE_BUS_TYPE')
    MESSAGE_RETENTION_HOURS: int = Field(default=24, env='MESSAGE_RETENTION_HOURS')
    
    # Performance settings
    ENABLE_AGENT_METRICS: bool = Field(default=True, env='ENABLE_AGENT_METRICS')
    ENABLE_AGENT_CACHING: bool = Field(default=True, env='ENABLE_AGENT_CACHING')
    
    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"


# Global registry instance
_agent_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _agent_registry


def create_default_agent_configs() -> Dict[str, AgentConfig]:
    """Create default configurations for all standard agents."""
    return {
        "coordinator": CoordinatorAgentConfig(
            agent_id="coordinator",
            name="Coordinator Agent",
            description="Main orchestrator for user interactions and agent coordination"
        ),
        "data_intelligence": DataIntelligenceAgentConfig(
            agent_id="data_intelligence",
            name="Data Intelligence Agent",
            description="Business intelligence and data analysis specialist"
        ),
        "etl_agent": ETLAgentConfig(
            agent_id="etl_agent",
            name="ETL Agent",
            description="ETL pipeline monitoring and data quality specialist"
        ),
        "visualization": VisualizationAgentConfig(
            agent_id="visualization",
            name="Visualization Agent",
            description="Data visualization and chart generation specialist"
        )
    }


def initialize_agent_registry() -> None:
    """Initialize the agent registry with default configurations."""
    default_configs = create_default_agent_configs()
    for config in default_configs.values():
        _agent_registry.register_agent(config)


# Initialize with defaults
initialize_agent_registry() 