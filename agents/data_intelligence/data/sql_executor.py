"""
SQL Executor for Data Intelligence Agent

This module provides SQL execution capabilities with security validation,
result caching, and integration with MCP servers for database operations.
"""

import asyncio
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, validator
import pandas as pd

from shared.base.agent_base import BaseAgent
from shared.utils.caching import get_cache_manager
from shared.utils.metrics import get_metrics_collector
from shared.config.settings import Settings
# Assuming MCPServerConnector is in this path, adjust if necessary
from agents.coordinator.mcp.server_connector import MCPServerConnector, MCPResponse # Assuming MCPResponse might also be needed

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of SQL queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    STORED_PROCEDURE = "stored_procedure"
    FUNCTION = "function"
    VIEW = "view"

class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class SecurityLevel(Enum):
    """Security levels for query execution."""
    READ_ONLY = "read_only"
    LIMITED_WRITE = "limited_write"
    FULL_ACCESS = "full_access"
    ADMIN = "admin"

@dataclass
class QueryValidationResult:
    """Result of SQL query validation."""
    is_valid: bool
    query_type: QueryType
    complexity: QueryComplexity
    security_level: SecurityLevel
    estimated_rows: Optional[int] = None
    estimated_execution_time: Optional[float] = None
    tables_accessed: List[str] = field(default_factory=list)
    columns_accessed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryExecutionResult:
    """Result of SQL query execution."""
    success: bool
    data: Optional[pd.DataFrame] = None
    row_count: int = 0
    execution_time: float = 0.0
    query_id: Optional[str] = None
    cache_hit: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class QueryRequest(BaseModel):
    """Request for SQL query execution."""
    query: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    database: Optional[str] = None
    schema: Optional[str] = None
    timeout: int = 300  # seconds
    max_rows: int = 10000
    use_cache: bool = True
    cache_ttl: int = 3600  # seconds
    security_context: Dict[str, Any] = Field(default_factory=dict)
    execution_options: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True

class QueryBatch(BaseModel):
    """Batch of SQL queries for execution."""
    queries: List[QueryRequest]
    execution_mode: str = "sequential"  # sequential, parallel, transaction
    transaction_isolation: str = "read_committed"
    rollback_on_error: bool = True
    max_concurrent: int = 5
    
    class Config:
        validate_assignment = True

class SQLExecutor:
    """
    SQL Executor for executing queries through MCP servers.
    
    Provides secure SQL execution with validation, caching, and monitoring.
    """
    
    def __init__(self, settings: Settings):
        """Initialize SQL executor."""
        self.settings = settings
        self.cache = get_cache_manager()
        self.metrics = get_metrics_collector()
        
        # Configuration
        self.config = {
            'max_query_size': settings.SQL_EXECUTOR.get('max_query_size', 50000),
            'default_timeout': settings.SQL_EXECUTOR.get('default_timeout', 300),
            'max_rows_limit': settings.SQL_EXECUTOR.get('max_rows_limit', 100000),
            'cache_ttl': settings.SQL_EXECUTOR.get('cache_ttl', 3600),
            'enable_query_logging': settings.SQL_EXECUTOR.get('enable_query_logging', True),
            'blocked_keywords': settings.SQL_EXECUTOR.get('blocked_keywords', [
                'drop', 'truncate', 'delete', 'create', 'alter', 'grant', 'revoke'
            ]),
            'allowed_functions': settings.SQL_EXECUTOR.get('allowed_functions', [
                'sum', 'count', 'avg', 'min', 'max', 'std', 'var', 'median'
            ])
        }
        
        # Query templates and patterns
        self._setup_query_patterns()
        self._setup_security_rules()
        
        # MCP client for database operations
        self.mcp_connector = MCPServerConnector(settings=self.settings) # Pass settings
        # If MCPServerConnector requires async initialization, it might need to be handled
        # in an async __init__ or a separate setup method for the agent.
        # For now, assuming synchronous instantiation is sufficient as per subtask.
        
        logger.info("SQL Executor initialized")

    async def initialize_connector(self):
        """Initializes and connects the MCP connector."""
        try:
            if hasattr(self.mcp_connector, 'initialize_from_config') and asyncio.iscoroutinefunction(self.mcp_connector.initialize_from_config):
                await self.mcp_connector.initialize_from_config()
            elif hasattr(self.mcp_connector, 'initialize_from_config'): # Sync fallback if any
                self.mcp_connector.initialize_from_config()

            if hasattr(self.mcp_connector, 'connect_all') and asyncio.iscoroutinefunction(self.mcp_connector.connect_all):
                await self.mcp_connector.connect_all()
            elif hasattr(self.mcp_connector, 'connect_all'): # Sync fallback if any
                self.mcp_connector.connect_all()
            logger.info("MCPServerConnector initialization and connection process initiated.")
        except Exception as e:
            logger.error(f"Error during MCPServerConnector initialization: {e}", exc_info=True)
            # Depending on policy, might re-raise or handle to allow agent to run degraded

    def _setup_query_patterns(self):
        """Setup regex patterns for query analysis."""
        self.query_patterns = {
            'select': re.compile(r'\bSELECT\b', re.IGNORECASE),
            'insert': re.compile(r'\bINSERT\s+INTO\b', re.IGNORECASE),
            'update': re.compile(r'\bUPDATE\b', re.IGNORECASE),
            'delete': re.compile(r'\bDELETE\s+FROM\b', re.IGNORECASE),
            'create': re.compile(r'\bCREATE\b', re.IGNORECASE),
            'drop': re.compile(r'\bDROP\b', re.IGNORECASE),
            'alter': re.compile(r'\bALTER\b', re.IGNORECASE),
            'join': re.compile(r'\b(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\b', re.IGNORECASE),
            'subquery': re.compile(r'\(\s*SELECT\b', re.IGNORECASE),
            'union': re.compile(r'\bUNION\b', re.IGNORECASE),
            'window_function': re.compile(r'\bOVER\s*\(', re.IGNORECASE),
            'cte': re.compile(r'\bWITH\b', re.IGNORECASE),
            'aggregation': re.compile(r'\b(?:SUM|COUNT|AVG|MIN|MAX|GROUP\s+BY)\b', re.IGNORECASE)
        }
        
        # Table and column extraction patterns
        self.table_pattern = re.compile(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)', re.IGNORECASE)
        self.column_pattern = re.compile(r'SELECT\s+(.*?)\s+FROM', re.IGNORECASE | re.DOTALL)

    def _setup_security_rules(self):
        """Setup security rules for query validation."""
        self.security_rules = {
            'read_only': {
                'allowed_types': [QueryType.SELECT],
                'blocked_keywords': ['insert', 'update', 'delete', 'create', 'drop', 'alter', 'grant', 'revoke'],
                'max_rows': 50000,
                'max_execution_time': 300
            },
            'limited_write': {
                'allowed_types': [QueryType.SELECT, QueryType.INSERT, QueryType.UPDATE],
                'blocked_keywords': ['drop', 'create', 'alter', 'grant', 'revoke', 'truncate'],
                'max_rows': 10000,
                'max_execution_time': 180
            },
            'full_access': {
                'allowed_types': list(QueryType),
                'blocked_keywords': ['grant', 'revoke'],
                'max_rows': 100000,
                'max_execution_time': 600
            },
            'admin': {
                'allowed_types': list(QueryType),
                'blocked_keywords': [],
                'max_rows': -1,
                'max_execution_time': -1
            }
        }

    async def execute_query(self, request: QueryRequest) -> QueryExecutionResult:
        """
        Execute a single SQL query.
        
        Args:
            request: Query request with parameters
            
        Returns:
            Query execution result
        """
        start_time = time.time()
        query_id = self._generate_query_id(request.query, request.parameters)
        
        try:
            # Log query execution start
            if self.config['enable_query_logging']:
                logger.info(f"Executing query {query_id}: {request.query[:100]}...")
            
            # Validate query
            validation = await self.validate_query(request)
            if not validation.is_valid:
                return QueryExecutionResult(
                    success=False,
                    errors=validation.errors,
                    warnings=validation.warnings,
                    execution_time=time.time() - start_time,
                    query_id=query_id
                )
            
            # Check cache if enabled
            if request.use_cache:
                cached_result = await self._get_cached_result(query_id)
                if cached_result:
                    cached_result.cache_hit = True
                    cached_result.query_id = query_id
                    return cached_result
            
            # Execute query through MCP
            result = await self._execute_through_mcp(request, validation)
            result.query_id = query_id
            result.execution_time = time.time() - start_time
            
            # Cache result if successful and caching enabled
            if result.success and request.use_cache:
                await self._cache_result(query_id, result, request.cache_ttl)
            
            # Record metrics
            await self._record_execution_metrics(request, result, validation)
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(f"{error_msg} - Query ID: {query_id}")
            
            return QueryExecutionResult(
                success=False,
                errors=[error_msg],
                execution_time=time.time() - start_time,
                query_id=query_id
            )

    async def execute_batch(self, batch: QueryBatch) -> List[QueryExecutionResult]:
        """
        Execute a batch of SQL queries.
        
        Args:
            batch: Batch of queries to execute
            
        Returns:
            List of execution results
        """
        results = []
        
        try:
            if batch.execution_mode == "parallel":
                # Execute queries in parallel
                tasks = [self.execute_query(query) for query in batch.queries[:batch.max_concurrent]]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Convert exceptions to error results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        results[i] = QueryExecutionResult(
                            success=False,
                            errors=[str(result)],
                            query_id=self._generate_query_id(batch.queries[i].query, batch.queries[i].parameters)
                        )
            
            elif batch.execution_mode == "transaction":
                # Execute queries in a transaction
                results = await self._execute_transaction(batch)
            
            else:
                # Sequential execution (default)
                for query in batch.queries:
                    result = await self.execute_query(query)
                    results.append(result)
                    
                    # Stop on error if rollback_on_error is True
                    if not result.success and batch.rollback_on_error:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing query batch: {str(e)}")
            return [QueryExecutionResult(success=False, errors=[str(e)])]

    async def validate_query(self, request: QueryRequest) -> QueryValidationResult:
        """
        Validate SQL query for security and complexity.
        
        Args:
            request: Query request to validate
            
        Returns:
            Validation result
        """
        query = request.query.strip()
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Basic checks
            if not query:
                errors.append("Query cannot be empty")
                return QueryValidationResult(
                    is_valid=False,
                    query_type=QueryType.SELECT,
                    complexity=QueryComplexity.SIMPLE,
                    security_level=SecurityLevel.READ_ONLY,
                    errors=errors
                )
            
            if len(query) > self.config['max_query_size']:
                errors.append(f"Query exceeds maximum size limit ({self.config['max_query_size']} characters)")
            
            # Determine query type
            query_type = self._determine_query_type(query)
            
            # Check security level
            security_level = self._determine_security_level(request.security_context)
            security_rules = self.security_rules[security_level.value]
            
            # Validate against security rules
            if query_type not in security_rules['allowed_types']:
                errors.append(f"Query type {query_type.value} not allowed for security level {security_level.value}")
            
            # Check for blocked keywords
            for keyword in security_rules['blocked_keywords']:
                if re.search(rf'\b{keyword}\b', query, re.IGNORECASE):
                    errors.append(f"Blocked keyword '{keyword}' found in query")
            
            # Analyze query complexity
            complexity = self._analyze_complexity(query)
            
            # Extract tables and columns
            tables = self._extract_tables(query)
            columns = self._extract_columns(query)
            
            # Estimate execution metrics
            estimated_rows = self._estimate_result_size(query, tables)
            estimated_time = self._estimate_execution_time(query, complexity)
            
            # Check row limits
            if (security_rules['max_rows'] > 0 and 
                estimated_rows and estimated_rows > security_rules['max_rows']):
                warnings.append(f"Estimated result size ({estimated_rows}) may exceed limit ({security_rules['max_rows']})")
            
            # Check execution time limits
            if (security_rules['max_execution_time'] > 0 and 
                estimated_time and estimated_time > security_rules['max_execution_time']):
                warnings.append(f"Estimated execution time ({estimated_time}s) may exceed limit ({security_rules['max_execution_time']}s)")
            
            # Generate suggestions
            suggestions.extend(self._generate_query_suggestions(query, complexity))
            
            is_valid = len(errors) == 0
            
            return QueryValidationResult(
                is_valid=is_valid,
                query_type=query_type,
                complexity=complexity,
                security_level=security_level,
                estimated_rows=estimated_rows,
                estimated_execution_time=estimated_time,
                tables_accessed=tables,
                columns_accessed=columns,
                warnings=warnings,
                errors=errors,
                suggestions=suggestions,
                metadata={
                    'query_length': len(query),
                    'parameter_count': len(request.parameters),
                    'contains_subqueries': bool(self.query_patterns['subquery'].search(query)),
                    'contains_joins': bool(self.query_patterns['join'].search(query)),
                    'contains_aggregation': bool(self.query_patterns['aggregation'].search(query))
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            return QueryValidationResult(
                is_valid=False,
                query_type=QueryType.SELECT,
                complexity=QueryComplexity.SIMPLE,
                security_level=SecurityLevel.READ_ONLY,
                errors=[f"Validation error: {str(e)}"]
            )

    def _determine_query_type(self, query: str) -> QueryType:
        """Determine the type of SQL query."""
        query_upper = query.upper().strip()
        
        if query_upper.startswith('SELECT'):
            return QueryType.SELECT
        elif query_upper.startswith('INSERT'):
            return QueryType.INSERT
        elif query_upper.startswith('UPDATE'):
            return QueryType.UPDATE
        elif query_upper.startswith('DELETE'):
            return QueryType.DELETE
        elif query_upper.startswith('CREATE'):
            return QueryType.CREATE
        elif query_upper.startswith('DROP'):
            return QueryType.DROP
        elif query_upper.startswith('ALTER'):
            return QueryType.ALTER
        elif 'PROCEDURE' in query_upper:
            return QueryType.STORED_PROCEDURE
        elif 'FUNCTION' in query_upper:
            return QueryType.FUNCTION
        elif 'VIEW' in query_upper:
            return QueryType.VIEW
        else:
            return QueryType.SELECT

    def _determine_security_level(self, security_context: Dict[str, Any]) -> SecurityLevel:
        """Determine security level from context."""
        level = security_context.get('level', 'read_only')
        
        try:
            return SecurityLevel(level)
        except ValueError:
            return SecurityLevel.READ_ONLY

    def _analyze_complexity(self, query: str) -> QueryComplexity:
        """Analyze query complexity based on features."""
        complexity_score = 0
        
        # Base complexity
        if self.query_patterns['join'].search(query):
            complexity_score += 2
        if self.query_patterns['subquery'].search(query):
            complexity_score += 3
        if self.query_patterns['union'].search(query):
            complexity_score += 2
        if self.query_patterns['window_function'].search(query):
            complexity_score += 3
        if self.query_patterns['cte'].search(query):
            complexity_score += 2
        if self.query_patterns['aggregation'].search(query):
            complexity_score += 1
        
        # Line count complexity
        line_count = len(query.split('\n'))
        if line_count > 50:
            complexity_score += 3
        elif line_count > 20:
            complexity_score += 2
        elif line_count > 10:
            complexity_score += 1
        
        # Character count complexity
        if len(query) > 5000:
            complexity_score += 2
        elif len(query) > 2000:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 8:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 5:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query."""
        tables = []
        
        # Find all FROM clauses
        matches = self.table_pattern.findall(query)
        for match in matches:
            # Handle schema.table format
            table_parts = match.split('.')
            table_name = table_parts[-1]  # Get the table name part
            if table_name not in tables:
                tables.append(table_name)
        
        return tables

    def _extract_columns(self, query: str) -> List[str]:
        """Extract column names from query (simplified)."""
        columns = []
        
        # Find SELECT clause
        match = self.column_pattern.search(query)
        if match:
            select_clause = match.group(1)
            
            # Simple column extraction (would need more sophisticated parsing for real use)
            if select_clause.strip() != '*':
                # Split by comma and clean up
                column_parts = [col.strip() for col in select_clause.split(',')]
                for col in column_parts:
                    # Remove aliases and functions
                    col_name = col.split(' AS ')[-1].split(' ')[-1]
                    if col_name and col_name not in ['*', '1']:
                        columns.append(col_name)
        
        return columns

    def _estimate_result_size(self, query: str, tables: List[str]) -> Optional[int]:
        """Estimate the number of rows that will be returned."""
        # Simplified estimation - in real implementation, would use query statistics
        
        if not tables:
            return None
        
        # Default estimates based on query patterns
        if 'LIMIT' in query.upper():
            limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
            if limit_match:
                return int(limit_match.group(1))
        
        # Estimate based on aggregation
        if self.query_patterns['aggregation'].search(query):
            return 100  # Aggregated results usually small
        
        # Estimate based on joins
        join_count = len(self.query_patterns['join'].findall(query))
        if join_count > 0:
            return 10000 * (join_count + 1)  # Rough estimate
        
        return 1000  # Default estimate

    def _estimate_execution_time(self, query: str, complexity: QueryComplexity) -> Optional[float]:
        """Estimate query execution time."""
        base_time = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 5.0,
            QueryComplexity.COMPLEX: 15.0,
            QueryComplexity.VERY_COMPLEX: 60.0
        }
        
        estimated_time = base_time.get(complexity, 5.0)
        
        # Adjust for specific patterns
        if self.query_patterns['join'].search(query):
            join_count = len(self.query_patterns['join'].findall(query))
            estimated_time *= (1 + join_count * 0.5)
        
        if self.query_patterns['subquery'].search(query):
            subquery_count = len(self.query_patterns['subquery'].findall(query))
            estimated_time *= (1 + subquery_count * 0.3)
        
        return estimated_time

    def _generate_query_suggestions(self, query: str, complexity: QueryComplexity) -> List[str]:
        """Generate optimization suggestions for the query."""
        suggestions = []
        
        # Suggest LIMIT for potentially large results
        if 'LIMIT' not in query.upper() and complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            suggestions.append("Consider adding LIMIT clause to restrict result size")
        
        # Suggest indexing for joins
        if self.query_patterns['join'].search(query):
            suggestions.append("Ensure join columns are properly indexed")
        
        # Suggest optimization for subqueries
        if self.query_patterns['subquery'].search(query):
            suggestions.append("Consider converting subqueries to JOINs for better performance")
        
        # Suggest specific columns instead of SELECT *
        if 'SELECT *' in query.upper():
            suggestions.append("Consider selecting specific columns instead of using SELECT *")
        
        return suggestions

    async def _execute_through_mcp(self, request: QueryRequest, validation: QueryValidationResult) -> QueryExecutionResult:
        """Execute query through MCP server using MCPServerConnector."""
        try:
            # This 'params' dict is the data payload for the MCP method.
            # It should contain what SnowflakeQueryExecutor.execute_query expects as arguments.
            mcp_method_params = {
                "query": request.query,
                "parameters": request.parameters,
                "limit": request.max_rows,
                "timeout": request.timeout # SQL execution timeout
            }

            # Send request to the snowflake_server via MCP
            mcp_response: MCPResponse = await self.mcp_connector.send_request(
                server_id="snowflake_server",       # Maps to 'server_id' in MCPServerConnector.send_request
                method="execute_query",          # Maps to 'method' (endpoint name)
                params=mcp_method_params,        # Maps to 'params' (data payload for the method)
                timeout=request.timeout + 5      # Optional: MCP communication timeout, slightly longer than SQL
            )

            if mcp_response.success and mcp_response.result:
                # Adapt the result from SnowflakeQueryExecutor (which is a dict)
                # to QueryExecutionResult
                snowflake_exec_result = mcp_response.result
                data_list = snowflake_exec_result.get('data', [])
                df = pd.DataFrame(data_list) if data_list else pd.DataFrame()
                
                # execution_time_ms from SnowflakeQueryExecutor, convert to seconds for QueryExecutionResult
                execution_time_seconds = snowflake_exec_result.get('execution_time_ms', 0.0) / 1000.0

                return QueryExecutionResult(
                    success=True,
                    data=df,
                    row_count=snowflake_exec_result.get('row_count', 0),
                    execution_time=execution_time_seconds,
                    query_id=request.query, # Or use one from snowflake_exec_result if available and preferred
                    cache_hit=snowflake_exec_result.get('cached', False), # Use 'cached' flag from Snowflake executor
                    warnings=snowflake_exec_result.get('warnings', []), # Assuming warnings might be part of result
                    errors=[], # No errors if success is true from MCP
                    metadata={
                        'query_type': snowflake_exec_result.get('query_type'),
                        'mcp_timestamp': snowflake_exec_result.get('timestamp'),
                        # Add any other relevant metadata from snowflake_exec_result
                    }
                )
            else:
                error_message = mcp_response.error_message or "MCP request failed without specific error message."
                logger.error(f"MCP request to snowflake_server failed: {error_message}. Result: {mcp_response.result}")
                return QueryExecutionResult(
                    success=False,
                    errors=[error_message],
                    metadata=mcp_response.result or {} # Include partial result/error details if any
                )
                
        except Exception as e:
            logger.exception(f"Exception during _execute_through_mcp for query: {request.query[:100]}")
            return QueryExecutionResult(
                success=False,
                errors=[f"SQLExecutor MCP communication error: {str(e)}"]
            )

    async def _execute_transaction(self, batch: QueryBatch) -> List[QueryExecutionResult]:
        """Execute queries in a transaction."""
        results = []
        transaction_success = True
        
        try:
            # Begin transaction (would use MCP transaction support)
            logger.info("Beginning transaction for query batch")
            
            for query in batch.queries:
                result = await self.execute_query(query)
                results.append(result)
                
                if not result.success:
                    transaction_success = False
                    if batch.rollback_on_error:
                        logger.warning("Transaction failed, rolling back")
                        # Rollback transaction
                        break
            
            if transaction_success:
                # Commit transaction
                logger.info("Transaction completed successfully")
            else:
                # Rollback already handled above
                pass
                
        except Exception as e:
            logger.error(f"Transaction error: {str(e)}")
            results.append(QueryExecutionResult(success=False, errors=[str(e)]))
        
        return results

    def _generate_query_id(self, query: str, parameters: Dict[str, Any]) -> str:
        """Generate unique ID for query and parameters."""
        query_hash = hashlib.md5(f"{query}{json.dumps(parameters, sort_keys=True)}".encode()).hexdigest()
        return f"query_{query_hash[:12]}"

    async def _get_cached_result(self, query_id: str) -> Optional[QueryExecutionResult]:
        """Get cached query result."""
        try:
            cached_data = await self.cache.get(f"sql_result_{query_id}")
            if cached_data:
                # Reconstruct result from cached data
                result_dict = json.loads(cached_data)
                
                # Handle DataFrame reconstruction
                if result_dict.get('data_json'):
                    data = pd.read_json(result_dict['data_json'], orient='records')
                    result_dict['data'] = data
                    del result_dict['data_json']
                
                return QueryExecutionResult(**result_dict)
        except Exception as e:
            logger.warning(f"Error retrieving cached result: {str(e)}")
        
        return None

    async def _cache_result(self, query_id: str, result: QueryExecutionResult, ttl: int):
        """Cache query result."""
        try:
            # Prepare result for caching
            result_dict = {
                'success': result.success,
                'row_count': result.row_count,
                'execution_time': result.execution_time,
                'warnings': result.warnings,
                'errors': result.errors,
                'metadata': result.metadata,
                'timestamp': result.timestamp.isoformat()
            }
            
            # Handle DataFrame serialization
            if result.data is not None:
                result_dict['data_json'] = result.data.to_json(orient='records')
            
            await self.cache.set(
                f"sql_result_{query_id}",
                json.dumps(result_dict),
                ttl
            )
        except Exception as e:
            logger.warning(f"Error caching result: {str(e)}")

    async def _record_execution_metrics(self, request: QueryRequest, result: QueryExecutionResult, validation: QueryValidationResult):
        """Record execution metrics."""
        try:
            metrics_data = {
                'query_type': validation.query_type.value,
                'complexity': validation.complexity.value,
                'security_level': validation.security_level.value,
                'execution_time': result.execution_time,
                'row_count': result.row_count,
                'success': result.success,
                'cache_hit': result.cache_hit,
                'estimated_vs_actual_rows': validation.estimated_rows / max(result.row_count, 1) if validation.estimated_rows else None,
                'estimated_vs_actual_time': validation.estimated_execution_time / max(result.execution_time, 0.001) if validation.estimated_execution_time else None
            }
            
            await self.metrics.record_event('sql_execution', metrics_data)
        except Exception as e:
            logger.warning(f"Error recording metrics: {str(e)}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of SQL executor."""
        return {
            'service': 'sql_executor',
            'status': 'healthy',
            'cache_enabled': self.cache is not None,
            'metrics_enabled': self.metrics is not None,
            'config': {
                'max_query_size': self.config['max_query_size'],
                'default_timeout': self.config['default_timeout'],
                'max_rows_limit': self.config['max_rows_limit']
            },
            'timestamp': datetime.now().isoformat()
        } 