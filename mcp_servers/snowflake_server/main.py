#!/usr/bin/env python3
"""
Snowflake MCP Server - Main Entry Point
========================================

FastAPI-based MCP server providing database operations for the multi-agent platform.
Implements tools for SQL execution, schema inspection, and data quality checks.

Uses Session A foundation from shared/ directory for connection management,
error handling, and MCP protocol compliance.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Session A Foundation imports
from shared.base.connection_base import BaseConnection, ConnectionStatus
from shared.schemas.mcp_protocol import (
    MCPServerInfo, MCPToolSchema, MCPToolRequest, MCPToolResponse,
    MCPHealthCheck, MCPStatus, create_server_info, create_tool_response
)
from shared.config.settings import get_settings
from shared.config.logging_config import setup_logging
from shared.utils.metrics import get_metrics_collector

# Local server components
from .connection_manager import SnowflakeConnectionManager
from .query_executor import SnowflakeQueryExecutor
from .schema_inspector import SnowflakeSchemaInspector
from .data_quality import SnowflakeDataQuality


class SnowflakeMCPServer:
    """
    Main Snowflake MCP Server class implementing FastAPI-based tool server.
    
    Provides database operations as MCP-compatible tools for the multi-agent platform.
    Integrates with existing Snowflake patterns from rahil/ directory.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.server_id = f"snowflake_server_{port}"
        
        # Initialize logging
        self.logger = setup_logging("SnowflakeMCPServer")
        
        # Initialize settings and metrics
        self.settings = get_settings()
        self.metrics = get_metrics_collector()
        
        # Server state
        self.status = MCPStatus.INITIALIZING
        self.started_at = datetime.utcnow()
        
        # Initialize components
        self.connection_manager: Optional[SnowflakeConnectionManager] = None
        self.query_executor: Optional[SnowflakeQueryExecutor] = None
        self.schema_inspector: Optional[SnowflakeSchemaInspector] = None
        self.data_quality: Optional[SnowflakeDataQuality] = None
        
        # Available tools registry
        self.tools: Dict[str, MCPToolSchema] = {}
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Snowflake MCP Server",
            description="Model Context Protocol server for Snowflake database operations",
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
        """Register available tools with their schemas."""
        
        # Execute Query tool
        self.tools["execute_query"] = MCPToolSchema(
            name="execute_query",
            description="Execute SQL query against Snowflake database",
            parameters=[
                {"name": "query", "type": "string", "description": "SQL query to execute", "required": True},
                {"name": "parameters", "type": "object", "description": "Query parameters", "required": False},
                {"name": "limit", "type": "integer", "description": "Maximum rows to return", "required": False, "default": 1000},
                {"name": "timeout", "type": "integer", "description": "Query timeout in seconds", "required": False, "default": 60}
            ],
            category="database",
            tags=["sql", "query", "database"],
            estimated_runtime_ms=5000,
            max_timeout_ms=300000
        )
        
        # Get Schema tool
        self.tools["get_schema"] = MCPToolSchema(
            name="get_schema",
            description="Get database schema information",
            parameters=[
                {"name": "database", "type": "string", "description": "Database name", "required": False},
                {"name": "schema", "type": "string", "description": "Schema name", "required": False},
                {"name": "table", "type": "string", "description": "Table name", "required": False},
                {"name": "include_columns", "type": "boolean", "description": "Include column details", "required": False, "default": True}
            ],
            category="database",
            tags=["schema", "metadata", "database"],
            estimated_runtime_ms=2000,
            max_timeout_ms=30000
        )
        
        # Check Data Quality tool
        self.tools["check_data_quality"] = MCPToolSchema(
            name="check_data_quality",
            description="Perform data quality checks on tables",
            parameters=[
                {"name": "table_name", "type": "string", "description": "Table to check", "required": True},
                {"name": "check_types", "type": "array", "description": "Types of checks to perform", "required": False, "default": ["completeness", "uniqueness"]},
                {"name": "sample_size", "type": "integer", "description": "Sample size for checks", "required": False, "default": 10000}
            ],
            category="data_quality",
            tags=["quality", "validation", "data"],
            estimated_runtime_ms=10000,
            max_timeout_ms=120000
        )
        
        # Get Table Stats tool
        self.tools["get_table_stats"] = MCPToolSchema(
            name="get_table_stats",
            description="Get table statistics and metadata",
            parameters=[
                {"name": "table_name", "type": "string", "description": "Table name", "required": True},
                {"name": "include_column_stats", "type": "boolean", "description": "Include column statistics", "required": False, "default": False}
            ],
            category="analytics",
            tags=["statistics", "metadata", "table"],
            estimated_runtime_ms=3000,
            max_timeout_ms=60000
        )
        
        # Monitor ETL tool
        self.tools["monitor_etl"] = MCPToolSchema(
            name="monitor_etl",
            description="Monitor ETL pipeline status and performance",
            parameters=[
                {"name": "pipeline_id", "type": "string", "description": "Pipeline identifier", "required": False},
                {"name": "time_range_hours", "type": "integer", "description": "Time range to check in hours", "required": False, "default": 24}
            ],
            category="monitoring",
            tags=["etl", "pipeline", "monitoring"],
            estimated_runtime_ms=5000,
            max_timeout_ms=30000
        )
    
    async def startup(self):
        """Initialize server components."""
        try:
            self.logger.info("Starting Snowflake MCP Server...")
            
            # Initialize connection manager
            self.connection_manager = SnowflakeConnectionManager(self.settings)
            await self.connection_manager.initialize()
            
            # Initialize components
            self.query_executor = SnowflakeQueryExecutor(self.connection_manager)
            await self.query_executor.initialize()
            
            self.schema_inspector = SnowflakeSchemaInspector(self.connection_manager)
            await self.schema_inspector.initialize()
            
            self.data_quality = SnowflakeDataQuality(self.connection_manager)
            await self.data_quality.initialize()
            
            self.status = MCPStatus.READY
            self.logger.info(f"Snowflake MCP Server started on {self.host}:{self.port}")
            
            # Record startup metrics
            self.metrics.counter("snowflake_server.startup.total").increment()
            
        except Exception as e:
            self.status = MCPStatus.ERROR
            self.logger.error(f"Failed to start server: {str(e)}")
            raise
    
    async def shutdown(self):
        """Cleanup server components."""
        try:
            self.logger.info("Shutting down Snowflake MCP Server...")
            
            # Cleanup components
            if self.data_quality:
                await self.data_quality.cleanup()
            if self.schema_inspector:
                await self.schema_inspector.cleanup()
            if self.query_executor:
                await self.query_executor.cleanup()
            if self.connection_manager:
                await self.connection_manager.cleanup()
            
            self.status = MCPStatus.MAINTENANCE
            self.logger.info("Snowflake MCP Server shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    async def _execute_tool(self, tool_name: str, request: MCPToolRequest) -> MCPToolResponse:
        """Execute a tool and return response."""
        start_time = datetime.utcnow()
        
        try:
            # Track tool execution
            self.metrics.counter(f"snowflake_server.tool.{tool_name}.executions").increment()
            
            with self.metrics.timer(f"snowflake_server.tool.{tool_name}.execution_time"):
                if tool_name == "execute_query":
                    result = await self.query_executor.execute_query(
                        request.parameters.get("query"),
                        request.parameters.get("parameters"),
                        request.parameters.get("limit", 1000),
                        request.parameters.get("timeout", 60)
                    )
                elif tool_name == "get_schema":
                    result = await self.schema_inspector.get_schema(
                        request.parameters.get("database"),
                        request.parameters.get("schema"),
                        request.parameters.get("table"),
                        request.parameters.get("include_columns", True)
                    )
                elif tool_name == "check_data_quality":
                    result = await self.data_quality.check_quality(
                        request.parameters.get("table_name"),
                        request.parameters.get("check_types", ["completeness", "uniqueness"]),
                        request.parameters.get("sample_size", 10000)
                    )
                elif tool_name == "get_table_stats":
                    result = await self.schema_inspector.get_table_stats(
                        request.parameters.get("table_name"),
                        request.parameters.get("include_column_stats", False)
                    )
                elif tool_name == "monitor_etl":
                    result = await self._monitor_etl_pipeline(
                        request.parameters.get("pipeline_id"),
                        request.parameters.get("time_range_hours", 24)
                    )
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.metrics.counter(f"snowflake_server.tool.{tool_name}.success").increment()
            
            return create_tool_response(
                request_id=request.request_id,
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.metrics.counter(f"snowflake_server.tool.{tool_name}.errors").increment()
            
            return create_tool_response(
                request_id=request.request_id,
                tool_name=tool_name,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    async def _monitor_etl_pipeline(self, pipeline_id: Optional[str], time_range_hours: int) -> Dict[str, Any]:
        """Monitor ETL pipeline status by returning table statistics for the given pipeline table."""
        if not pipeline_id:
            raise ValueError("pipeline_id is required for monitoring ETL pipeline")
        # Retrieve statistics for the pipeline table via schema_inspector
        stats = await self.schema_inspector.get_table_stats(pipeline_id, include_column_stats=False)
        return {
            "pipeline_id": pipeline_id,
            "table_stats": stats,
            "time_range_hours": time_range_hours
        }
    
    async def get_server_info(self) -> MCPServerInfo:
        """Get current server information."""
        uptime = (datetime.utcnow() - self.started_at).total_seconds()
        
        return create_server_info(
            server_id=self.server_id,
            name="Snowflake MCP Server",
            port=self.port,
            description="Database operations and analytics for Snowflake",
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
        connection_healthy = (
            self.connection_manager and 
            await self.connection_manager.is_healthy()
        )
        
        overall_status = MCPStatus.READY if connection_healthy else MCPStatus.ERROR
        
        return MCPHealthCheck(
            server_id=self.server_id,
            status=overall_status,
            uptime_seconds=uptime,
            active_requests=0,  # Would track active requests
            total_requests=0,   # Would track total requests
            error_rate=0.0,     # Would calculate error rate
            average_response_time_ms=0.0,  # Would track response times
            health_data={
                "connection_healthy": connection_healthy,
                "tools_available": len(self.tools),
                "components_initialized": all([
                    self.connection_manager is not None,
                    self.query_executor is not None,
                    self.schema_inspector is not None,
                    self.data_quality is not None
                ])
            }
        )


def create_app(host: str = "localhost", port: int = 8001) -> FastAPI:
    """Create FastAPI application."""
    server = SnowflakeMCPServer(host, port)
    return server.app


async def run_server(host: str = "localhost", port: int = 8001):
    """Run the MCP server."""
    server = SnowflakeMCPServer(host, port)
    
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
    
    parser = argparse.ArgumentParser(description="Snowflake MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    
    args = parser.parse_args()
    
    asyncio.run(run_server(args.host, args.port)) 