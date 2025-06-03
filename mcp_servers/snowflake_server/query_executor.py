#!/usr/bin/env python3
"""
Snowflake Query Executor
=========================

Advanced SQL query execution with streaming, caching, validation, and
performance monitoring. Integrates with Session A foundation utilities
for enterprise-grade query handling.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from enum import Enum

# Session A Foundation imports
from shared.utils.validation import validate_sql_query, sanitize_input, check_data_quality
from shared.utils.caching import cache_result, cache_query_result
from shared.utils.retry import retry_with_backoff
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.data_processing import DataProcessor

from .connection_manager import SnowflakeConnectionManager


class QueryType(str, Enum):
    """Types of SQL queries."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    DDL = "DDL"
    SHOW = "SHOW"
    DESCRIBE = "DESCRIBE"
    EXPLAIN = "EXPLAIN"
    UNKNOWN = "UNKNOWN"


class QueryResult:
    """Query execution result with metadata."""
    
    def __init__(
        self,
        data: Any,
        query: str,
        execution_time_ms: float,
        row_count: int = 0,
        query_type: QueryType = QueryType.UNKNOWN,
        cached: bool = False
    ):
        self.data = data
        self.query = query
        self.execution_time_ms = execution_time_ms
        self.row_count = row_count
        self.query_type = query_type
        self.cached = cached
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "data": self.data,
            "query": self.query,
            "execution_time_ms": self.execution_time_ms,
            "row_count": self.row_count,
            "query_type": self.query_type.value,
            "cached": self.cached,
            "timestamp": self.timestamp.isoformat()
        }


class SnowflakeQueryExecutor:
    """
    Advanced Snowflake query executor with caching, streaming, and monitoring.
    
    Provides enterprise-grade query execution capabilities for the MCP server.
    """
    
    def __init__(self, connection_manager: SnowflakeConnectionManager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger("SnowflakeQueryExecutor")
        self.metrics = get_metrics_collector()
        self.data_processor = DataProcessor()
        
        # Query execution state
        self._active_queries: Dict[str, asyncio.Task] = {}
        self._query_history: List[QueryResult] = []
        self._max_history_size = 1000
        
        # Performance tracking
        self._total_queries = 0
        self._successful_queries = 0
        self._failed_queries = 0
        self._total_execution_time = 0.0
    
    async def initialize(self) -> None:
        """Initialize query executor."""
        self.logger.info("Initializing Snowflake query executor...")
        # Additional initialization if needed
        self.logger.info("Snowflake query executor initialized")
    
    async def cleanup(self) -> None:
        """Cleanup query executor."""
        self.logger.info("Cleaning up Snowflake query executor...")
        
        # Cancel active queries
        for query_id, task in self._active_queries.items():
            if not task.done():
                self.logger.warning(f"Cancelling active query: {query_id}")
                task.cancel()
        
        self._active_queries.clear()
        self.logger.info("Snowflake query executor cleanup complete")
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect SQL query type."""
        query_upper = query.strip().upper()
        
        if query_upper.startswith('SELECT'):
            return QueryType.SELECT
        elif query_upper.startswith('INSERT'):
            return QueryType.INSERT
        elif query_upper.startswith('UPDATE'):
            return QueryType.UPDATE
        elif query_upper.startswith('DELETE'):
            return QueryType.DELETE
        elif query_upper.startswith(('CREATE', 'DROP', 'ALTER')):
            return QueryType.DDL
        elif query_upper.startswith('SHOW'):
            return QueryType.SHOW
        elif query_upper.startswith(('DESCRIBE', 'DESC')):
            return QueryType.DESCRIBE
        elif query_upper.startswith('EXPLAIN'):
            return QueryType.EXPLAIN
        else:
            return QueryType.UNKNOWN
    
    @track_performance(tags={"component": "query_executor"})
    @retry_with_backoff(max_attempts=2, multiplier=1.5)
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute SQL query with validation, caching, and monitoring.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters for prepared statements
            limit: Maximum number of rows to return
            timeout: Query timeout in seconds
            
        Returns:
            Query result with metadata
        """
        start_time = time.time()
        query_id = f"query_{int(start_time * 1000)}"
        query_type = self._detect_query_type(query)
        
        try:
            # Validate query
            if not validate_sql_query(query):
                raise ValueError("Invalid or potentially unsafe SQL query")
            
            # Sanitize query
            sanitized_query = sanitize_input(query)
            
            # Apply limit if specified
            if limit and query_type == QueryType.SELECT:
                if "LIMIT" not in sanitized_query.upper():
                    sanitized_query = f"{sanitized_query} LIMIT {limit}"
            
            self.logger.info(f"Executing {query_type.value} query (ID: {query_id})")
            
            data: Any
            was_cached = False # Default, as decorator does not signal cache status to caller

            if query_type == QueryType.SELECT:
                # SELECT queries go through the cached internal method
                query_task = asyncio.create_task(
                    self._execute_query_internal(sanitized_query, parameters, timeout)
                )
                self._active_queries[query_id] = query_task
                try:
                    data = await query_task
                    # Note: was_cached remains False here. The decorator handles caching transparently.
                    # If the decorator itself logged cache hits/misses, that would be separate.
                finally:
                    self._active_queries.pop(query_id, None)
            else:
                # Non-SELECT queries bypass the caching layer
                query_task = asyncio.create_task(
                    self.connection_manager.execute_query(sanitized_query, parameters, timeout)
                )
                self._active_queries[query_id] = query_task
                try:
                    data = await query_task
                finally:
                    self._active_queries.pop(query_id, None)

            execution_time = (time.time() - start_time) * 1000
            row_count = len(data) if isinstance(data, list) else 0
            
            # Update metrics; was_cached is False as we don't get this info from the decorator here
            self._update_metrics(query_type, execution_time, was_cached)
            
            # Create result
            result = QueryResult(
                data=data,
                query=query,
                execution_time_ms=execution_time,
                row_count=row_count,
                query_type=query_type,
                cached=was_cached # Will be False. Comment can explain cache is transparent.
                                 # If SELECT, it might have been from cache, but we don't flag it here.
            )
            
            self._add_to_history(result)
            
            self.logger.info(
                f"Query completed (ID: {query_id}): {row_count} rows, "
                f"{execution_time:.2f}ms"
            )
            
            return result.to_dict()
            
        except asyncio.CancelledError:
            self.logger.warning(f"Query cancelled (ID: {query_id})")
            self.metrics.counter("snowflake.queries.cancelled").increment()
            raise
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Query failed (ID: {query_id}): {str(e)}")
            
            self._failed_queries += 1
            self.metrics.counter("snowflake.queries.failed").increment()
            self.metrics.counter(f"snowflake.queries.{query_type.value.lower()}.failed").increment()
            
            raise
    
    @cache_query_result(ttl=300, cache_name="query_cache")
    async def _execute_query_internal(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Any: # Returns just data; decorator handles caching transparently
        """
        Internal query execution for SELECT queries, decorated with @cache_query_result.
        The decorator handles the caching mechanism. This method only fetches data if not cached.
        """
        return await self.connection_manager.execute_query(query, parameters, timeout)


    async def execute_streaming_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        timeout: Optional[float] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute query with streaming results for large datasets.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters
            chunk_size: Number of rows per chunk
            timeout: Query timeout in seconds
            
        Yields:
            Chunks of query results
        """
        start_time = time.time()
        query_id = f"stream_{int(start_time * 1000)}"
        query_type = self._detect_query_type(query)
        
        if query_type != QueryType.SELECT:
            raise ValueError("Streaming is only supported for SELECT queries")
        
        try:
            # Validate and sanitize query
            if not validate_sql_query(query):
                raise ValueError("Invalid or potentially unsafe SQL query")
            
            sanitized_query = sanitize_input(query)
            
            self.logger.info(f"Starting streaming query (ID: {query_id})")
            
            total_rows = 0
            chunk_count = 0
            
            async for chunk in self.connection_manager.execute_streaming_query(
                sanitized_query, parameters, chunk_size, timeout
            ):
                chunk_count += 1
                total_rows += len(chunk)
                
                yield {
                    "chunk_id": chunk_count,
                    "data": chunk,
                    "chunk_size": len(chunk),
                    "total_rows_so_far": total_rows
                }
            
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Streaming query completed (ID: {query_id}): {total_rows} rows "
                f"in {chunk_count} chunks, {execution_time:.2f}ms"
            )
            
            # Update metrics
            self.metrics.counter("snowflake.streaming_queries.completed").increment()
            self.metrics.histogram("snowflake.streaming_queries.execution_time_ms").update(execution_time)
            self.metrics.histogram("snowflake.streaming_queries.total_rows").update(total_rows)
            
        except Exception as e:
            self.logger.error(f"Streaming query failed (ID: {query_id}): {str(e)}")
            self.metrics.counter("snowflake.streaming_queries.failed").increment()
            raise
    
    async def explain_query(self, query: str) -> Dict[str, Any]:
        """Get query execution plan."""
        explain_query = f"EXPLAIN {query}"
        
        result = await self.execute_query(explain_query)
        
        return {
            "original_query": query,
            "execution_plan": result["data"],
            "estimated_cost": self._extract_cost_from_plan(result["data"]) if result["data"] else None
        }
    
    def _extract_cost_from_plan(self, plan_data: Any) -> Optional[float]:
        """Extract estimated cost from execution plan."""
        # This would parse the Snowflake execution plan to extract cost information
        # Implementation depends on Snowflake's plan format
        return None
    
    async def get_query_statistics(
        self,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get query execution statistics."""
        return {
            "total_queries": self._total_queries,
            "successful_queries": self._successful_queries,
            "failed_queries": self._failed_queries,
            "success_rate": self._successful_queries / max(self._total_queries, 1),
            "average_execution_time_ms": self._total_execution_time / max(self._total_queries, 1),
            "active_queries": len(self._active_queries),
            "query_history_size": len(self._query_history),
            "query_types": self._get_query_type_distribution()
        }
    
    def _get_query_type_distribution(self) -> Dict[str, int]:
        """Get distribution of query types from history."""
        distribution = {}
        for result in self._query_history:
            query_type = result.query_type.value
            distribution[query_type] = distribution.get(query_type, 0) + 1
        return distribution
    
    def _update_metrics(
        self,
        query_type: QueryType,
        execution_time: float,
        cached: bool
    ) -> None:
        """Update query execution metrics."""
        self._total_queries += 1
        self._successful_queries += 1
        self._total_execution_time += execution_time
        
        # Update global metrics
        self.metrics.counter("snowflake.queries.total").increment()
        self.metrics.counter(f"snowflake.queries.{query_type.value.lower()}.total").increment()
        self.metrics.histogram("snowflake.queries.execution_time_ms").update(execution_time)
        
        if cached:
            self.metrics.counter("snowflake.queries.cache_hits").increment()
        else:
            self.metrics.counter("snowflake.queries.cache_misses").increment()
    
    def _add_to_history(self, result: QueryResult) -> None:
        """Add query result to history."""
        self._query_history.append(result)
        
        # Trim history if too large
        if len(self._query_history) > self._max_history_size:
            self._query_history = self._query_history[-self._max_history_size:]
    
    async def cancel_query(self, query_id: str) -> bool:
        """Cancel an active query."""
        if query_id in self._active_queries:
            task = self._active_queries[query_id]
            if not task.done():
                task.cancel()
                self.logger.info(f"Cancelled query: {query_id}")
                return True
        return False
    
    async def get_active_queries(self) -> List[Dict[str, Any]]:
        """Get list of currently active queries."""
        active = []
        
        for query_id, task in self._active_queries.items():
            if not task.done():
                active.append({
                    "query_id": query_id,
                    "status": "running",
                    "started_at": task.get_name() if hasattr(task, 'get_name') else None
                })
        
        return active
    
    async def validate_query_syntax(self, query: str) -> Dict[str, Any]:
        """Validate query syntax without execution."""
        try:
            # Use EXPLAIN to validate syntax
            explain_query = f"EXPLAIN {query}"
            
            # This will raise an exception if syntax is invalid
            await self.connection_manager.execute_query(explain_query)
            
            return {
                "valid": True,
                "query": query,
                "message": "Query syntax is valid"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "query": query,
                "error": str(e),
                "message": "Query syntax validation failed"
            } 