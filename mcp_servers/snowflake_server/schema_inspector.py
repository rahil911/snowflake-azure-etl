#!/usr/bin/env python3
"""
Snowflake Schema Inspector
==========================

Database schema introspection and metadata tools for the Snowflake MCP server.
Provides cached access to table structures, column information, and database statistics.

Integrates with Session A foundation for caching, validation, and performance monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# Session A Foundation imports
from shared.utils.caching import cache_result, cache_analytics
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.validation import sanitize_input
from shared.utils.data_processing import DataProcessor

from .connection_manager import SnowflakeConnectionManager


@dataclass
class TableInfo:
    """Table metadata information."""
    database_name: str
    schema_name: str
    table_name: str
    table_type: str
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None
    last_altered: Optional[datetime] = None
    comment: Optional[str] = None


@dataclass
class ColumnInfo:
    """Column metadata information."""
    column_name: str
    data_type: str
    is_nullable: bool
    column_default: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    ordinal_position: int = 0
    character_maximum_length: Optional[int] = None
    numeric_precision: Optional[int] = None
    numeric_scale: Optional[int] = None
    comment: Optional[str] = None


class SnowflakeSchemaInspector:
    """
    Snowflake schema inspection with caching and performance monitoring.
    
    Provides tools for database schema introspection, metadata queries,
    and statistical analysis of database structures.
    """
    
    def __init__(self, connection_manager: SnowflakeConnectionManager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger("SnowflakeSchemaInspector")
        self.metrics = get_metrics_collector()
        self.data_processor = DataProcessor()
        
        # State
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize schema inspector."""
        if self._initialized:
            return
        
        self.logger.info("Initializing Snowflake schema inspector...")
        self._initialized = True
        self.logger.info("Snowflake schema inspector initialized")
    
    async def cleanup(self) -> None:
        """Cleanup schema inspector."""
        self.logger.info("Cleaning up Snowflake schema inspector...")
        self._initialized = False
        self.logger.info("Snowflake schema inspector cleanup complete")
    
    @cache_result(ttl=900, cache_name="schema_cache")  # 15 minutes cache
    @track_performance(tags={"component": "schema_inspector", "operation": "get_schema"})
    async def get_schema(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        include_columns: bool = True
    ) -> Dict[str, Any]:
        """
        Get database schema information with optional filtering.
        
        Args:
            database: Database name filter
            schema: Schema name filter  
            table: Table name filter
            include_columns: Whether to include column details
            
        Returns:
            Schema information dictionary
        """
        try:
            self.logger.info(f"Getting schema info: db={database}, schema={schema}, table={table}")
            
            result = {
                "databases": [],
                "schemas": [],
                "tables": [],
                "columns": [] if include_columns else None,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "filters": {
                        "database": database,
                        "schema": schema,
                        "table": table
                    }
                }
            }
            
            # Get databases
            if not database:
                result["databases"] = await self._get_databases()
            elif database:
                # Verify database exists
                db_exists = await self._verify_database_exists(database)
                if db_exists:
                    result["databases"] = [{"database_name": database}]
            
            # Get schemas
            if not schema:
                result["schemas"] = await self._get_schemas(database)
            elif schema:
                schema_exists = await self._verify_schema_exists(database, schema)
                if schema_exists:
                    result["schemas"] = [{"schema_name": schema, "database_name": database}]
            
            # Get tables
            result["tables"] = await self._get_tables(database, schema, table)
            
            # Get columns if requested
            if include_columns:
                result["columns"] = await self._get_columns(database, schema, table)
            
            # Add summary statistics
            result["summary"] = {
                "total_databases": len(result["databases"]),
                "total_schemas": len(result["schemas"]),
                "total_tables": len(result["tables"]),
                "total_columns": len(result["columns"]) if include_columns else None
            }
            
            self.metrics.counter("snowflake.schema.inspections").increment()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Schema inspection failed: {str(e)}")
            self.metrics.counter("snowflake.schema.inspection_errors").increment()
            raise
    
    async def _get_databases(self) -> List[Dict[str, Any]]:
        """Get list of available databases."""
        query = """
        SELECT 
            database_name,
            database_owner,
            comment,
            created,
            last_altered
        FROM information_schema.databases 
        ORDER BY database_name
        """
        
        result = await self.connection_manager.execute_query(query)
        return result or []
    
    async def _get_schemas(self, database: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of schemas in database(s)."""
        if database:
            query = """
            SELECT 
                catalog_name as database_name,
                schema_name,
                schema_owner,
                comment,
                created,
                last_altered
            FROM information_schema.schemata 
            WHERE catalog_name = %s
            ORDER BY schema_name
            """
            params = {"catalog_name": database.upper()}
        else:
            query = """
            SELECT 
                catalog_name as database_name,
                schema_name,
                schema_owner,
                comment,
                created,
                last_altered
            FROM information_schema.schemata 
            ORDER BY catalog_name, schema_name
            """
            params = None
        
        result = await self.connection_manager.execute_query(query, params)
        return result or []
    
    async def _get_tables(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get list of tables with optional filtering."""
        where_conditions = []
        params = {}
        
        if database:
            where_conditions.append("table_catalog = %(database)s")
            params["database"] = database.upper()
        
        if schema:
            where_conditions.append("table_schema = %(schema)s")
            params["schema"] = schema.upper()
        
        if table:
            where_conditions.append("table_name = %(table)s")
            params["table"] = table.upper()
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        query = f"""
        SELECT 
            table_catalog as database_name,
            table_schema as schema_name,
            table_name,
            table_type,
            row_count,
            bytes,
            created,
            last_altered,
            comment
        FROM information_schema.tables 
        {where_clause}
        ORDER BY table_catalog, table_schema, table_name
        """
        
        result = await self.connection_manager.execute_query(query, params)
        return result or []
    
    async def _get_columns(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get column information with optional filtering."""
        where_conditions = []
        params = {}
        
        if database:
            where_conditions.append("table_catalog = %(database)s")
            params["database"] = database.upper()
        
        if schema:
            where_conditions.append("table_schema = %(schema)s")
            params["schema"] = schema.upper()
        
        if table:
            where_conditions.append("table_name = %(table)s")
            params["table"] = table.upper()
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        query = f"""
        SELECT 
            table_catalog as database_name,
            table_schema as schema_name,
            table_name,
            column_name,
            ordinal_position,
            column_default,
            is_nullable,
            data_type,
            character_maximum_length,
            character_octet_length,
            numeric_precision,
            numeric_precision_radix,
            numeric_scale,
            datetime_precision,
            character_set_name,
            collation_name,
            comment
        FROM information_schema.columns 
        {where_clause}
        ORDER BY table_catalog, table_schema, table_name, ordinal_position
        """
        
        result = await self.connection_manager.execute_query(query, params)
        return result or []
    
    async def _verify_database_exists(self, database: str) -> bool:
        """Verify if database exists."""
        query = "SELECT COUNT(*) as count FROM information_schema.databases WHERE database_name = %s"
        result = await self.connection_manager.execute_query(query, {"database_name": database.upper()})
        return result and result[0].get("COUNT", 0) > 0
    
    async def _verify_schema_exists(self, database: Optional[str], schema: str) -> bool:
        """Verify if schema exists."""
        if database:
            query = """
            SELECT COUNT(*) as count FROM information_schema.schemata 
            WHERE catalog_name = %s AND schema_name = %s
            """
            params = {"catalog_name": database.upper(), "schema_name": schema.upper()}
        else:
            query = "SELECT COUNT(*) as count FROM information_schema.schemata WHERE schema_name = %s"
            params = {"schema_name": schema.upper()}
        
        result = await self.connection_manager.execute_query(query, params)
        return result and result[0].get("COUNT", 0) > 0
    
    @cache_analytics(ttl=1800, cache_name="table_stats")  # 30 minutes cache
    @track_performance(tags={"component": "schema_inspector", "operation": "get_table_stats"})
    async def get_table_stats(
        self,
        table_name: str,
        include_column_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive table statistics.
        
        Args:
            table_name: Name of the table (can include schema)
            include_column_stats: Whether to include column-level statistics
            
        Returns:
            Table statistics dictionary
        """
        try:
            self.logger.info(f"Getting table stats for: {table_name}")
            
            # Parse table name (handle schema.table format)
            table_parts = table_name.split('.')
            if len(table_parts) == 2:
                schema_name, table_name_only = table_parts
            elif len(table_parts) == 3:
                _, schema_name, table_name_only = table_parts
            else:
                schema_name = None
                table_name_only = table_name
            
            # Get basic table info
            table_info = await self._get_table_basic_stats(table_name_only, schema_name)
            
            # Get column information
            columns = await self._get_table_columns(table_name_only, schema_name)
            
            result = {
                "table_info": table_info,
                "column_count": len(columns),
                "columns": columns,
                "column_stats": None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Get column statistics if requested
            if include_column_stats and columns:
                result["column_stats"] = await self._get_column_statistics(
                    table_name_only, schema_name, columns
                )
            
            self.metrics.counter("snowflake.schema.table_stats").increment()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get table stats for {table_name}: {str(e)}")
            self.metrics.counter("snowflake.schema.table_stats_errors").increment()
            raise
    
    async def _get_table_basic_stats(
        self,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get basic table statistics."""
        where_conditions = ["table_name = %(table_name)s"]
        params = {"table_name": table_name.upper()}
        
        if schema_name:
            where_conditions.append("table_schema = %(schema_name)s")
            params["schema_name"] = schema_name.upper()
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        SELECT 
            table_catalog as database_name,
            table_schema as schema_name,
            table_name,
            table_type,
            row_count,
            bytes,
            created,
            last_altered,
            comment
        FROM information_schema.tables 
        WHERE {where_clause}
        LIMIT 1
        """
        
        result = await self.connection_manager.execute_query(query, params)
        return result[0] if result else {}
    
    async def _get_table_columns(
        self,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get table column information."""
        where_conditions = ["table_name = %(table_name)s"]
        params = {"table_name": table_name.upper()}
        
        if schema_name:
            where_conditions.append("table_schema = %(schema_name)s")
            params["schema_name"] = schema_name.upper()
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        SELECT 
            column_name,
            ordinal_position,
            column_default,
            is_nullable,
            data_type,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            comment
        FROM information_schema.columns 
        WHERE {where_clause}
        ORDER BY ordinal_position
        """
        
        result = await self.connection_manager.execute_query(query, params)
        return result or []
    
    async def _get_column_statistics(
        self,
        table_name: str,
        schema_name: Optional[str],
        columns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get basic column-level statistics including data type and nullability."""
        stats = {}
        
        for column in columns:
            column_name = column.get("COLUMN_NAME")
            data_type = column.get("DATA_TYPE", "").upper()
            
            # Create basic stats structure
            column_stats = {
                "data_type": data_type,
                "is_nullable": column.get("IS_NULLABLE") == "YES",
                "character_length": column.get("CHARACTER_MAXIMUM_LENGTH"),
                "numeric_precision": column.get("NUMERIC_PRECISION"),
                "numeric_scale": column.get("NUMERIC_SCALE")
            }
            
            # Could add more detailed statistics here like:
            # - NULL count
            # - Distinct value count
            # - Min/Max values
            # - Value distribution
            
            stats[column_name] = column_stats
        
        return stats
    
    @cache_result(ttl=3600, cache_name="database_summary")  # 1 hour cache
    async def get_database_summary(self) -> Dict[str, Any]:
        """Get comprehensive database summary statistics."""
        try:
            # Get database counts
            databases = await self._get_databases()
            schemas = await self._get_schemas()
            tables = await self._get_tables()
            
            # Calculate storage usage
            total_bytes = sum(
                table.get("BYTES", 0) or 0 
                for table in tables 
                if table.get("BYTES") is not None
            )
            
            total_rows = sum(
                table.get("ROW_COUNT", 0) or 0 
                for table in tables 
                if table.get("ROW_COUNT") is not None
            )
            
            # Group by table types
            table_types = {}
            for table in tables:
                table_type = table.get("TABLE_TYPE", "UNKNOWN")
                table_types[table_type] = table_types.get(table_type, 0) + 1
            
            return {
                "summary": {
                    "total_databases": len(databases),
                    "total_schemas": len(schemas),
                    "total_tables": len(tables),
                    "total_rows": total_rows,
                    "total_size_bytes": total_bytes,
                    "total_size_mb": round(total_bytes / (1024 * 1024), 2) if total_bytes else 0
                },
                "table_types": table_types,
                "largest_tables": sorted(
                    [t for t in tables if t.get("BYTES")],
                    key=lambda x: x.get("BYTES", 0),
                    reverse=True
                )[:10],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get database summary: {str(e)}")
            raise 