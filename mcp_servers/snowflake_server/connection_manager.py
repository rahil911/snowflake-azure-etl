#!/usr/bin/env python3
"""
Snowflake Connection Manager
============================

Enhanced Snowflake connection management with pooling, health monitoring,
and integration with existing rahil/ patterns. Extends Session A foundation
classes for enterprise-grade connection management.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import snowflake.connector
import snowflake.connector.errors as sf_errors
from snowflake.connector import DictCursor

# Session A Foundation imports
from shared.base.connection_base import BaseConnection, ConnectionConfig, ConnectionPool, PoolConfig, ConnectionStatus
from shared.config.settings import get_settings
from shared.utils.retry import retry_with_backoff
from shared.utils.validation import validate_sql_query, sanitize_input
from shared.utils.metrics import get_metrics_collector
from shared.utils.caching import cache_result

# Removed direct import of rahil_config

_global_snowflake_manager: Optional["SnowflakeConnectionManager"] = None
_manager_lock = asyncio.Lock() # For async singleton initialization


class SnowflakeConnectionConfig(ConnectionConfig):
    """Snowflake-specific connection configuration."""
    
    def __init__(self, **kwargs):
        settings = get_settings()
        db_config = settings.database # Get DatabaseConfig instance

        super().__init__(
            connection_id=kwargs.get("connection_id", "snowflake_default"),
            connection_type="database",
            name="Snowflake Database",
            description="Snowflake data warehouse connection",
            host=db_config.host,
            username=db_config.username,
            password=db_config.password.get_secret_value(), # Extract plain string from SecretStr
            database=db_config.database,
            **kwargs
        )
        
        # Snowflake-specific parameters, now sourced from db_config
        self.account = db_config.account
        self.warehouse = db_config.warehouse
        self.role = db_config.role # Already correctly sourced
        self.schema = db_config.schema
        
        # Override health check query for Snowflake
        self.health_check_query = "SELECT CURRENT_VERSION()"
    
    def get_connection_string(self) -> str:
        """Build Snowflake connection string."""
        return f"snowflake://{self.username}@{self.account}/{self.database}/{self.schema}"


class SnowflakeConnection(BaseConnection):
    """
    Snowflake-specific connection implementation extending BaseConnection.
    
    Integrates with existing rahil/connection.py patterns while providing
    enterprise-grade features from Session A foundation.
    """
    
    def __init__(self, config: SnowflakeConnectionConfig):
        super().__init__(config)
        self.sf_config = config
        self.metrics = get_metrics_collector()
        
        # Snowflake-specific state
        self._cursor = None
        self._session_id = None
    
    async def _connect_implementation(self) -> snowflake.connector.SnowflakeConnection:
        """Establish Snowflake connection."""
        try:
            self.logger.info(f"Connecting to Snowflake account: {self.sf_config.account}")
            
            # Create connection using patterns from rahil/connection.py
            connection = snowflake.connector.connect(
                user=self.sf_config.username,
                password=self.sf_config.password,
                account=self.sf_config.account,
                warehouse=self.sf_config.warehouse,
                role=self.sf_config.role,
                database=self.sf_config.database,
                schema=self.sf_config.schema,
                # Additional connection parameters for better performance
                client_session_keep_alive=True,
                application="MCP_Snowflake_Server",
                autocommit=True,
                timeout=self.config.connection_timeout_seconds
            )
            
            # Test connection and get session info
            cursor = connection.cursor()
            cursor.execute("SELECT CURRENT_VERSION(), CURRENT_SESSION()")
            version, session_id = cursor.fetchone()
            
            self._session_id = session_id
            self.logger.info(f"Connected to Snowflake version: {version}, Session: {session_id}")
            
            cursor.close()
            
            # Record connection metrics
            self.metrics.counter("snowflake.connections.established").increment()
            
            return connection
            
        except sf_errors.OperationalError as e:
            self.logger.error(f"Snowflake operational error: {str(e)}")
            self.metrics.counter("snowflake.connections.operational_errors").increment()
            raise
        except sf_errors.DatabaseError as e:
            self.logger.error(f"Snowflake database error: {str(e)}")
            self.metrics.counter("snowflake.connections.database_errors").increment()
            raise
        except Exception as e:
            self.logger.error(f"Unexpected Snowflake connection error: {str(e)}")
            self.metrics.counter("snowflake.connections.unknown_errors").increment()
            raise
    
    async def _disconnect_implementation(self) -> None:
        """Close Snowflake connection."""
        try:
            if self._cursor:
                self._cursor.close()
                self._cursor = None
            
            if self._connection:
                self._connection.close()
                self.logger.info(f"Disconnected from Snowflake session: {self._session_id}")
                self._session_id = None
                
            self.metrics.counter("snowflake.connections.closed").increment()
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from Snowflake: {str(e)}")
            self.metrics.counter("snowflake.connections.disconnect_errors").increment()
            raise
    
    async def _execute_query_implementation(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute SQL query against Snowflake."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Snowflake")
        
        # Validate and sanitize query
        if not validate_sql_query(query):
            raise ValueError("Invalid or potentially unsafe SQL query")
        
        sanitized_query = sanitize_input(query)
        
        try:
            # Create cursor with dictionary results
            cursor = self._connection.cursor(DictCursor)
            
            start_time = datetime.utcnow()
            
            # Execute query with parameters
            if parameters:
                cursor.execute(sanitized_query, parameters)
            else:
                cursor.execute(sanitized_query)
            
            # Fetch results
            results = cursor.fetchall()
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update metrics
            self.metrics.counter("snowflake.queries.executed").increment()
            self.metrics.histogram("snowflake.queries.execution_time_ms").update(execution_time)
            
            # Log query execution
            self.logger.debug(f"Query executed in {execution_time:.2f}ms, returned {len(results)} rows")
            
            cursor.close()
            return results
            
        except sf_errors.ProgrammingError as e:
            self.logger.error(f"Snowflake programming error: {str(e)}")
            self.metrics.counter("snowflake.queries.programming_errors").increment()
            raise
        except sf_errors.OperationalError as e:
            self.logger.error(f"Snowflake operational error: {str(e)}")
            self.metrics.counter("snowflake.queries.operational_errors").increment()
            raise
        except Exception as e:
            self.logger.error(f"Unexpected query error: {str(e)}")
            self.metrics.counter("snowflake.queries.unknown_errors").increment()
            raise
    
    async def _health_check_implementation(self) -> bool:
        """Perform Snowflake health check."""
        try:
            cursor = self._connection.cursor()
            cursor.execute(self.config.health_check_query)
            version = cursor.fetchone()[0]
            cursor.close()
            
            self.logger.debug(f"Health check passed, Snowflake version: {version}")
            self.metrics.counter("snowflake.health_checks.success").increment()
            return True
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {str(e)}")
            self.metrics.counter("snowflake.health_checks.failed").increment()
            return False
    
    async def execute_streaming_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000
    ) -> Any:
        """Execute query with streaming results for large datasets."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Snowflake")
        
        # Validate query
        if not validate_sql_query(query):
            raise ValueError("Invalid or potentially unsafe SQL query")
        
        sanitized_query = sanitize_input(query)
        
        try:
            cursor = self._connection.cursor(DictCursor)
            
            if parameters:
                cursor.execute(sanitized_query, parameters)
            else:
                cursor.execute(sanitized_query)
            
            # Stream results in chunks
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                yield rows
            
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Streaming query error: {str(e)}")
            self.metrics.counter("snowflake.streaming_queries.errors").increment()
            raise
    
    @cache_result(ttl=300, cache_name="schema_cache")
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get cached table information."""
        query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            comment
        FROM information_schema.columns 
        WHERE table_name = %s
        ORDER BY ordinal_position
        """
        
        results = await self.execute_query(query, {"table_name": table_name.upper()})
        return {"table_name": table_name, "columns": results}
    
    async def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        query = """
        SELECT 
            CURRENT_SESSION() as session_id,
            CURRENT_USER() as user_name,
            CURRENT_ROLE() as role_name,
            CURRENT_WAREHOUSE() as warehouse,
            CURRENT_DATABASE() as database_name,
            CURRENT_SCHEMA() as schema_name,
            CURRENT_VERSION() as version
        """
        
        results = await self.execute_query(query)
        return results[0] if results else {}


class SnowflakeConnectionManager:
    """
    High-level Snowflake connection manager with pooling and monitoring.
    
    Provides enterprise-grade connection management for the Snowflake MCP server.
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.logger = logging.getLogger("SnowflakeConnectionManager")
        self.metrics = get_metrics_collector()
        
        # Connection configuration
        self.config = SnowflakeConnectionConfig(
            connection_id="snowflake_mcp_pool",
            max_retries=3,
            retry_delay_seconds=2.0,
            connection_timeout_seconds=30.0,
            query_timeout_seconds=300.0,
            health_check_interval_seconds=300
        )
        
        # Pool configuration
        self.pool_config = PoolConfig(
            min_size=2,
            max_size=10,
            max_idle_time_seconds=3600,
            acquire_timeout_seconds=30.0
        )
        
        # Connection pool
        self.pool: Optional[ConnectionPool] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize connection manager and pool."""
        if self._initialized:
            return
        
        try:
            self.logger.info("Initializing Snowflake connection manager...")
            
            # Create connection factory
            def connection_factory() -> SnowflakeConnection:
                return SnowflakeConnection(self.config)
            
            # Initialize connection pool
            self.pool = ConnectionPool(connection_factory, self.pool_config)
            await self.pool.initialize()
            
            self._initialized = True
            self.logger.info("Snowflake connection manager initialized successfully")
            
            # Record initialization
            self.metrics.counter("snowflake.manager.initialized").increment()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection manager: {str(e)}")
            self.metrics.counter("snowflake.manager.initialization_errors").increment()
            raise
    
    async def cleanup(self) -> None:
        """Cleanup connection manager and pool."""
        if not self._initialized:
            return
        
        try:
            self.logger.info("Cleaning up Snowflake connection manager...")
            
            if self.pool:
                await self.pool.cleanup()
                self.pool = None
            
            self._initialized = False
            self.logger.info("Snowflake connection manager cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    @asynccontextmanager
    async def get_connection(self, timeout: Optional[float] = None):
        """Get connection from pool with context management."""
        if not self._initialized:
            raise RuntimeError("Connection manager not initialized")
        
        async with self.pool.acquire(timeout=timeout) as connection:
            yield connection
    
    @retry_with_backoff(max_attempts=3, multiplier=2.0)
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """Execute query using pooled connection with retry logic."""
        async with self.get_connection(timeout=timeout) as connection:
            return await connection.execute_query(query, parameters, timeout)
    
    async def execute_streaming_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        timeout: Optional[float] = None
    ) -> Any:
        """Execute streaming query using pooled connection."""
        async with self.get_connection(timeout=timeout) as connection:
            async for chunk in connection.execute_streaming_query(query, parameters, chunk_size):
                yield chunk
    
    async def is_healthy(self) -> bool:
        """Check if connection manager is healthy."""
        if not self._initialized or not self.pool:
            return False
        
        try:
            # Test with a simple connection acquisition
            async with self.get_connection(timeout=5.0) as connection:
                return await connection.health_check()
        except Exception:
            return False
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status."""
        if not self.pool:
            return {"status": "not_initialized"}
        
        return self.pool.get_status()
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get connection configuration information."""
        return {
            "account": self.config.account,
            "warehouse": self.config.warehouse,
            "database": self.config.database,
            "schema": self.config.schema,
            "pool_config": {
                "min_size": self.pool_config.min_size,
                "max_size": self.pool_config.max_size,
                "max_idle_time_seconds": self.pool_config.max_idle_time_seconds
            },
            "initialized": self._initialized
        }


# Convenience function for backward compatibility with rahil/ patterns
async def get_managed_snowflake_connection():
    """
    Get a managed Snowflake connection compatible with existing rahil/ patterns.
    
    This provides a bridge between the new connection manager and existing code.
    Ensures that only one instance of SnowflakeConnectionManager (and thus one
    connection pool) is used throughout the application when accessed via this function.
    """
    global _global_snowflake_manager
    if _global_snowflake_manager is None:
        async with _manager_lock:
            if _global_snowflake_manager is None: # Double-check after acquiring lock
                _global_snowflake_manager = SnowflakeConnectionManager()
                await _global_snowflake_manager.initialize()
    return _global_snowflake_manager