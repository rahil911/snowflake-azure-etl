#!/usr/bin/env python3
"""
Analytics MCP Server - Main Entry Point
========================================

FastAPI-based MCP server providing statistical analysis and data transformation
capabilities for the multi-agent platform. Complements the Snowflake server
with analytical computations and ML functions.

Uses Session A foundation for metrics, caching, and error handling.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Session A Foundation imports
from shared.schemas.mcp_protocol import (
    MCPServerInfo, MCPToolSchema, MCPToolRequest, MCPToolResponse,
    MCPHealthCheck, MCPStatus, create_server_info, create_tool_response
)
from shared.config.settings import get_settings
from shared.config.logging_config import setup_logging
from shared.utils.metrics import get_metrics_collector

# Local server components
from .statistical_functions import StatisticalAnalyzer
from .data_transformer import DataTransformer
from .ml_functions import MLAnalyzer


class AnalyticsMCPServer:
    """
    Main Analytics MCP Server class implementing FastAPI-based analytical tools.
    
    Provides statistical analysis, data transformation, and basic ML capabilities
    as MCP-compatible tools for the multi-agent platform.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8002):
        self.host = host
        self.port = port
        self.server_id = f"analytics_server_{port}"
        
        # Initialize logging
        self.logger = setup_logging("AnalyticsMCPServer")
        
        # Initialize settings and metrics
        self.settings = get_settings()
        self.metrics = get_metrics_collector()
        
        # Server state
        self.status = MCPStatus.INITIALIZING
        self.started_at = datetime.utcnow()
        
        # Initialize components
        self.statistical_analyzer: Optional[StatisticalAnalyzer] = None
        self.data_transformer: Optional[DataTransformer] = None
        self.ml_analyzer: Optional[MLAnalyzer] = None
        
        # Available tools registry
        self.tools: Dict[str, MCPToolSchema] = {}
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Analytics MCP Server",
            description="Model Context Protocol server for statistical analysis and data transformation",
            version="1.0.0",
            lifespan=self.lifespan
        )
        
        self._setup_middleware()
        self._setup_routes()
        self._register_tools()
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan handler for startup and shutdown."""
        # Startup
        await self.startup()
        yield
        # Shutdown
        await self.shutdown()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup FastAPI routes for MCP protocol endpoints."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            health = await self.get_health_status()
            return health.dict()
        
        @self.app.get("/server-info")
        async def get_server_info():
            """Get server information."""
            info = await self.get_server_info()
            return info.dict()
        
        @self.app.get("/tools")
        async def list_tools():
            """List available tools."""
            return {
                "tools": [tool.dict() for tool in self.tools.values()]
            }
        
        @self.app.get("/tools/{tool_name}")
        async def get_tool_schema(tool_name: str):
            """Get schema for specific tool."""
            if tool_name not in self.tools:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            return self.tools[tool_name].dict()
        
        @self.app.post("/tools/{tool_name}/execute")
        async def execute_tool(
            tool_name: str,
            request: MCPToolRequest,
            background_tasks: BackgroundTasks
        ):
            """Execute a tool."""
            if tool_name not in self.tools:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            
            try:
                response = await self._execute_tool(tool_name, request)
                return response.dict()
            except Exception as e:
                self.logger.error(f"Tool execution failed: {str(e)}")
                error_response = create_tool_response(
                    request_id=request.request_id,
                    tool_name=tool_name,
                    success=False,
                    error_message=str(e)
                )
                return error_response.dict()
    
    def _register_tools(self):
        """Register available analytics tools with their schemas."""
        
        # Statistical Analysis tools
        self.tools["calculate_statistics"] = MCPToolSchema(
            name="calculate_statistics",
            description="Calculate descriptive statistics for numerical data",
            parameters=[
                {"name": "data", "type": "array", "description": "Array of numerical values", "required": True},
                {"name": "statistics", "type": "array", "description": "List of statistics to calculate", "required": False, "default": ["mean", "median", "std"]},
                {"name": "confidence_level", "type": "number", "description": "Confidence level for intervals", "required": False, "default": 0.95}
            ],
            category="statistics",
            tags=["statistics", "descriptive", "analysis"],
            estimated_runtime_ms=100,
            max_timeout_ms=5000
        )
        
        self.tools["correlation_analysis"] = MCPToolSchema(
            name="correlation_analysis",
            description="Perform correlation analysis between variables",
            parameters=[
                {"name": "data", "type": "object", "description": "Dictionary of variable arrays", "required": True},
                {"name": "method", "type": "string", "description": "Correlation method (pearson, spearman, kendall)", "required": False, "default": "pearson"}
            ],
            category="statistics",
            tags=["correlation", "relationship", "analysis"],
            estimated_runtime_ms=200,
            max_timeout_ms=10000
        )
        
        # Data Transformation tools
        self.tools["transform_data"] = MCPToolSchema(
            name="transform_data",
            description="Apply data transformations (normalize, standardize, etc.)",
            parameters=[
                {"name": "data", "type": "array", "description": "Data to transform", "required": True},
                {"name": "transformation", "type": "string", "description": "Type of transformation", "required": True},
                {"name": "parameters", "type": "object", "description": "Transformation parameters", "required": False}
            ],
            category="transformation",
            tags=["transformation", "normalization", "scaling"],
            estimated_runtime_ms=150,
            max_timeout_ms=8000
        )
        
        self.tools["aggregate_data"] = MCPToolSchema(
            name="aggregate_data",
            description="Aggregate data by groups with various functions",
            parameters=[
                {"name": "data", "type": "array", "description": "Data to aggregate", "required": True},
                {"name": "group_by", "type": "string", "description": "Column to group by", "required": True},
                {"name": "aggregations", "type": "object", "description": "Aggregation functions by column", "required": True}
            ],
            category="transformation",
            tags=["aggregation", "grouping", "summary"],
            estimated_runtime_ms=300,
            max_timeout_ms=15000
        )
        
        # Machine Learning tools
        self.tools["cluster_analysis"] = MCPToolSchema(
            name="cluster_analysis",
            description="Perform clustering analysis on data",
            parameters=[
                {"name": "data", "type": "array", "description": "Data for clustering", "required": True},
                {"name": "n_clusters", "type": "integer", "description": "Number of clusters", "required": False, "default": 3},
                {"name": "algorithm", "type": "string", "description": "Clustering algorithm", "required": False, "default": "kmeans"}
            ],
            category="machine_learning",
            tags=["clustering", "unsupervised", "analysis"],
            estimated_runtime_ms=1000,
            max_timeout_ms=30000
        )
        
        self.tools["trend_analysis"] = MCPToolSchema(
            name="trend_analysis",
            description="Analyze trends in time series data",
            parameters=[
                {"name": "data", "type": "array", "description": "Time series data", "required": True},
                {"name": "timestamps", "type": "array", "description": "Timestamp array", "required": True},
                {"name": "period", "type": "string", "description": "Analysis period", "required": False, "default": "monthly"}
            ],
            category="time_series",
            tags=["trend", "time_series", "analysis"],
            estimated_runtime_ms=500,
            max_timeout_ms=20000
        )
        
        self.tools["outlier_detection"] = MCPToolSchema(
            name="outlier_detection",
            description="Detect outliers in numerical data",
            parameters=[
                {"name": "data", "type": "array", "description": "Numerical data", "required": True},
                {"name": "method", "type": "string", "description": "Detection method (iqr, zscore, isolation)", "required": False, "default": "iqr"},
                {"name": "threshold", "type": "number", "description": "Outlier threshold", "required": False, "default": 1.5}
            ],
            category="analysis",
            tags=["outliers", "anomaly", "detection"],
            estimated_runtime_ms=200,
            max_timeout_ms=10000
        )
    
    async def startup(self):
        """Initialize server components."""
        try:
            self.logger.info("Starting Analytics MCP Server...")
            
            # Initialize components
            self.statistical_analyzer = StatisticalAnalyzer()
            await self.statistical_analyzer.initialize()
            
            self.data_transformer = DataTransformer()
            await self.data_transformer.initialize()
            
            self.ml_analyzer = MLAnalyzer()
            await self.ml_analyzer.initialize()
            
            self.status = MCPStatus.READY
            self.logger.info(f"Analytics MCP Server started on {self.host}:{self.port}")
            
            # Record startup metrics
            self.metrics.counter("analytics_server.startup.total").increment()
            
        except Exception as e:
            self.status = MCPStatus.ERROR
            self.logger.error(f"Failed to start server: {str(e)}")
            raise
    
    async def shutdown(self):
        """Cleanup server components."""
        try:
            self.logger.info("Shutting down Analytics MCP Server...")
            
            # Cleanup components
            if self.ml_analyzer:
                await self.ml_analyzer.cleanup()
            if self.data_transformer:
                await self.data_transformer.cleanup()
            if self.statistical_analyzer:
                await self.statistical_analyzer.cleanup()
            
            self.status = MCPStatus.MAINTENANCE
            self.logger.info("Analytics MCP Server shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    async def _execute_tool(self, tool_name: str, request: MCPToolRequest) -> MCPToolResponse:
        """Execute a tool and return response."""
        start_time = datetime.utcnow()
        
        try:
            # Track tool execution
            self.metrics.counter(f"analytics_server.tool.{tool_name}.executions").increment()
            
            with self.metrics.timer(f"analytics_server.tool.{tool_name}.execution_time"):
                # Route to appropriate component
                if tool_name in ["calculate_statistics", "correlation_analysis"]:
                    result = await self._execute_statistical_tool(tool_name, request.parameters)
                elif tool_name in ["transform_data", "aggregate_data"]:
                    result = await self._execute_transformation_tool(tool_name, request.parameters)
                elif tool_name in ["cluster_analysis", "trend_analysis", "outlier_detection"]:
                    result = await self._execute_ml_tool(tool_name, request.parameters)
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.metrics.counter(f"analytics_server.tool.{tool_name}.success").increment()
            
            return create_tool_response(
                request_id=request.request_id,
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.metrics.counter(f"analytics_server.tool.{tool_name}.errors").increment()
            
            return create_tool_response(
                request_id=request.request_id,
                tool_name=tool_name,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    async def _execute_statistical_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute statistical analysis tools."""
        if tool_name == "calculate_statistics":
            return await self.statistical_analyzer.calculate_statistics(
                data=parameters.get("data"),
                statistics=parameters.get("statistics", ["mean", "median", "std"]),
                confidence_level=parameters.get("confidence_level", 0.95)
            )
        elif tool_name == "correlation_analysis":
            return await self.statistical_analyzer.correlation_analysis(
                data=parameters.get("data"),
                method=parameters.get("method", "pearson")
            )
        else:
            raise ValueError(f"Unknown statistical tool: {tool_name}")
    
    async def _execute_transformation_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute data transformation tools."""
        if tool_name == "transform_data":
            return await self.data_transformer.transform_data(
                data=parameters.get("data"),
                transformation=parameters.get("transformation"),
                parameters=parameters.get("parameters", {})
            )
        elif tool_name == "aggregate_data":
            return await self.data_transformer.aggregate_data(
                data=parameters.get("data"),
                group_by=parameters.get("group_by"),
                aggregations=parameters.get("aggregations")
            )
        else:
            raise ValueError(f"Unknown transformation tool: {tool_name}")
    
    async def _execute_ml_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute machine learning tools."""
        if tool_name == "cluster_analysis":
            return await self.ml_analyzer.cluster_analysis(
                data=parameters.get("data"),
                n_clusters=parameters.get("n_clusters", 3),
                algorithm=parameters.get("algorithm", "kmeans")
            )
        elif tool_name == "trend_analysis":
            return await self.ml_analyzer.trend_analysis(
                data=parameters.get("data"),
                timestamps=parameters.get("timestamps"),
                period=parameters.get("period", "monthly")
            )
        elif tool_name == "outlier_detection":
            return await self.ml_analyzer.outlier_detection(
                data=parameters.get("data"),
                method=parameters.get("method", "iqr"),
                threshold=parameters.get("threshold", 1.5)
            )
        else:
            raise ValueError(f"Unknown ML tool: {tool_name}")
    
    async def get_server_info(self) -> MCPServerInfo:
        """Get current server information."""
        uptime = (datetime.utcnow() - self.started_at).total_seconds()
        
        return create_server_info(
            server_id=self.server_id,
            name="Analytics MCP Server",
            port=self.port,
            description="Statistical analysis and data transformation tools",
            version="1.0.0"
        ).copy(update={
            "available_tools": list(self.tools.keys()),
            "status": self.status,
            "uptime_seconds": uptime
        })
    
    async def get_health_status(self) -> MCPHealthCheck:
        """Get current health status."""
        uptime = (datetime.utcnow() - self.started_at).total_seconds()
        
        # Check component health
        components_healthy = all([
            self.statistical_analyzer is not None,
            self.data_transformer is not None,
            self.ml_analyzer is not None
        ])
        
        overall_status = MCPStatus.READY if components_healthy else MCPStatus.ERROR
        
        return MCPHealthCheck(
            server_id=self.server_id,
            status=overall_status,
            uptime_seconds=uptime,
            active_requests=0,  # Would track active requests
            total_requests=0,   # Would track total requests
            error_rate=0.0,     # Would calculate error rate
            average_response_time_ms=0.0,  # Would track response times
            health_data={
                "components_healthy": components_healthy,
                "tools_available": len(self.tools),
                "statistical_analyzer": self.statistical_analyzer is not None,
                "data_transformer": self.data_transformer is not None,
                "ml_analyzer": self.ml_analyzer is not None
            }
        )


def create_app(host: str = "localhost", port: int = 8002) -> FastAPI:
    """Create FastAPI application."""
    server = AnalyticsMCPServer(host, port)
    return server.app


async def run_server(host: str = "localhost", port: int = 8002):
    """Run the MCP server."""
    server = AnalyticsMCPServer(host, port)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(server.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the server
    config = uvicorn.Config(
        server.app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
    
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analytics MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    
    args = parser.parse_args()
    
    asyncio.run(run_server(args.host, args.port)) 