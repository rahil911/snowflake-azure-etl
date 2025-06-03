"""
Natural Language to SQL Query Generator

This module converts natural language business questions into SQL queries using:
- Database schema knowledge from private_ddl/
- Business context and entity extraction
- Template-based query generation
- SQL security validation and optimization

Features:
- Schema-aware query generation
- Entity recognition and mapping
- Query complexity analysis
- Parameter binding for security
- Query optimization suggestions
"""

import re
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# Session A Foundation imports
from shared.config.logging_config import setup_logging
from shared.utils.validation import validate_sql_query, sanitize_input
from shared.utils.caching import cache_result
from shared.utils.metrics import track_performance, get_metrics_collector


class QueryComplexity(Enum):
    """Query complexity levels for resource planning."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class SQLQuery:
    """Generated SQL query with metadata."""
    sql: str
    parameters: Dict[str, Any]
    table_references: List[str]
    complexity: QueryComplexity
    confidence: float
    metadata: Dict[str, Any]
    optimization_suggestions: List[str]


@dataclass
class EntityMapping:
    """Entity mapping from natural language to database entities."""
    entity_type: str  # table, column, value, metric
    natural_name: str
    database_name: str
    table: Optional[str] = None
    confidence: float = 1.0


class QueryGenerator:
    """
    Natural Language to SQL Query Generator.
    
    Converts business questions into optimized SQL queries using schema knowledge
    and business context understanding.
    """
    
    def __init__(self):
        """Initialize the Query Generator with schema knowledge."""
        self.logger = setup_logging("nlp.query_generator")
        self.metrics = get_metrics_collector()
        
        # Performance tracking
        self.generation_counter = self.metrics.counter("sql_queries_generated")
        self.success_counter = self.metrics.counter("sql_queries_successful") 
        self.error_counter = self.metrics.counter("sql_queries_failed")
        self.generation_timer = self.metrics.timer("sql_generation_time")
        
        # Initialize schema knowledge
        self._load_schema_knowledge()
        self._load_query_templates()
        self._load_business_mappings()
        
        self.logger.info("Query Generator initialized successfully")
    
    def _load_schema_knowledge(self) -> None:
        """Load database schema knowledge for query generation."""
        
        # Dimensional Model Schema (from BIG_PICTURE.md analysis)
        self.schema = {
            "fact_tables": {
                "FACT_SALESACTUAL": {
                    "columns": [
                        "SALES_ID", "CUSTOMER_KEY", "PRODUCT_KEY", "STORE_KEY", 
                        "DATE_KEY", "CHANNEL_KEY", "QUANTITY_SOLD", "UNIT_PRICE",
                        "TOTAL_SALES_AMOUNT", "COST_OF_GOODS_SOLD", "GROSS_PROFIT",
                        "DISCOUNT_AMOUNT", "TAX_AMOUNT", "CREATED_DATE", "UPDATED_DATE"
                    ],
                    "metrics": ["QUANTITY_SOLD", "TOTAL_SALES_AMOUNT", "GROSS_PROFIT"],
                    "dimensions": ["CUSTOMER_KEY", "PRODUCT_KEY", "STORE_KEY", "DATE_KEY", "CHANNEL_KEY"]
                }
            },
            "dimension_tables": {
                "DIM_CUSTOMER": {
                    "columns": [
                        "CUSTOMER_KEY", "CUSTOMER_ID", "CUSTOMER_NAME", "EMAIL", 
                        "PHONE", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE",
                        "CUSTOMER_TYPE", "REGISTRATION_DATE", "LAST_ORDER_DATE"
                    ],
                    "business_key": "CUSTOMER_ID",
                    "attributes": ["CUSTOMER_NAME", "CITY", "STATE", "COUNTRY", "CUSTOMER_TYPE"]
                },
                "DIM_PRODUCT": {
                    "columns": [
                        "PRODUCT_KEY", "PRODUCT_ID", "PRODUCT_NAME", "CATEGORY", 
                        "SUBCATEGORY", "BRAND", "UNIT_PRICE", "COST", "PRODUCT_STATUS",
                        "LAUNCH_DATE", "DISCONTINUE_DATE"
                    ],
                    "business_key": "PRODUCT_ID",
                    "hierarchy": ["CATEGORY", "SUBCATEGORY", "BRAND", "PRODUCT_NAME"],
                    "attributes": ["CATEGORY", "SUBCATEGORY", "BRAND", "PRODUCT_STATUS"]
                },
                "DIM_STORE": {
                    "columns": [
                        "STORE_KEY", "STORE_ID", "STORE_NAME", "ADDRESS", "CITY", 
                        "STATE", "COUNTRY", "ZIP_CODE", "STORE_TYPE", "STORE_SIZE",
                        "OPENING_DATE", "MANAGER_NAME"
                    ],
                    "business_key": "STORE_ID",
                    "attributes": ["STORE_NAME", "CITY", "STATE", "COUNTRY", "STORE_TYPE"]
                },
                "DIM_DATE": {
                    "columns": [
                        "DATE_KEY", "FULL_DATE", "DAY_OF_WEEK", "DAY_NAME", "MONTH",
                        "MONTH_NAME", "QUARTER", "YEAR", "FISCAL_QUARTER", "FISCAL_YEAR",
                        "IS_WEEKEND", "IS_HOLIDAY", "HOLIDAY_NAME"
                    ],
                    "business_key": "FULL_DATE",
                    "time_attributes": ["DAY_NAME", "MONTH_NAME", "QUARTER", "YEAR"]
                },
                "DIM_CHANNEL": {
                    "columns": [
                        "CHANNEL_KEY", "CHANNEL_ID", "CHANNEL_NAME", "CHANNEL_TYPE",
                        "CHANNEL_DESCRIPTION", "IS_ACTIVE"
                    ],
                    "business_key": "CHANNEL_ID",
                    "attributes": ["CHANNEL_NAME", "CHANNEL_TYPE"]
                }
            },
            "staging_tables": {
                "STAGING_CUSTOMER": {"source": "customer_data"},
                "STAGING_PRODUCT": {"source": "product_catalog"},
                "STAGING_SALESHEADER": {"source": "sales_transactions"},
                "STAGING_SALESDETAIL": {"source": "sales_line_items"},
                "STAGING_STORE": {"source": "store_master"},
                "STAGING_CHANNEL": {"source": "channel_master"}
            }
        }
        
        # Create reverse mappings for entity recognition
        self.column_mappings = {}
        self.table_mappings = {}
        
        for table_type, tables in self.schema.items():
            for table_name, table_info in tables.items():
                self.table_mappings[table_name.lower()] = table_name
                if "columns" in table_info:
                    for col in table_info["columns"]:
                        key = f"{table_name}.{col}".lower()
                        self.column_mappings[key] = (table_name, col)
    
    def _load_query_templates(self) -> None:
        """Load SQL query templates for different question types."""
        
        self.query_templates = {
            "sales_performance": {
                "base_query": """
                    SELECT 
                        {time_dimension},
                        {group_by_dimensions},
                        SUM(f.TOTAL_SALES_AMOUNT) as total_sales,
                        SUM(f.QUANTITY_SOLD) as total_quantity,
                        SUM(f.GROSS_PROFIT) as total_profit,
                        COUNT(DISTINCT f.SALES_ID) as transaction_count
                    FROM FACT_SALESACTUAL f
                    {joins}
                    {where_clause}
                    GROUP BY {group_by_clause}
                    {having_clause}
                    ORDER BY {order_by_clause}
                    {limit_clause}
                """,
                "required_joins": ["DIM_DATE d ON f.DATE_KEY = d.DATE_KEY"],
                "complexity": QueryComplexity.MEDIUM
            },
            "customer_analysis": {
                "base_query": """
                    SELECT 
                        c.CUSTOMER_NAME,
                        c.CITY,
                        c.STATE,
                        {customer_metrics}
                    FROM DIM_CUSTOMER c
                    INNER JOIN FACT_SALESACTUAL f ON c.CUSTOMER_KEY = f.CUSTOMER_KEY
                    {additional_joins}
                    {where_clause}
                    GROUP BY c.CUSTOMER_KEY, c.CUSTOMER_NAME, c.CITY, c.STATE
                    {having_clause}
                    ORDER BY {order_by_clause}
                    {limit_clause}
                """,
                "default_metrics": [
                    "SUM(f.TOTAL_SALES_AMOUNT) as customer_total_sales",
                    "COUNT(DISTINCT f.SALES_ID) as customer_transaction_count",
                    "AVG(f.TOTAL_SALES_AMOUNT) as customer_avg_order_value"
                ],
                "complexity": QueryComplexity.MEDIUM
            },
            "product_analysis": {
                "base_query": """
                    SELECT 
                        p.CATEGORY,
                        p.SUBCATEGORY,
                        p.BRAND,
                        p.PRODUCT_NAME,
                        {product_metrics}
                    FROM DIM_PRODUCT p
                    INNER JOIN FACT_SALESACTUAL f ON p.PRODUCT_KEY = f.PRODUCT_KEY
                    {additional_joins}
                    {where_clause}
                    GROUP BY p.PRODUCT_KEY, p.CATEGORY, p.SUBCATEGORY, p.BRAND, p.PRODUCT_NAME
                    {having_clause}
                    ORDER BY {order_by_clause}
                    {limit_clause}
                """,
                "default_metrics": [
                    "SUM(f.TOTAL_SALES_AMOUNT) as product_total_sales",
                    "SUM(f.QUANTITY_SOLD) as product_total_quantity",
                    "SUM(f.GROSS_PROFIT) as product_total_profit"
                ],
                "complexity": QueryComplexity.MEDIUM
            },
            "time_series": {
                "base_query": """
                    SELECT 
                        {time_grouping},
                        {metrics}
                    FROM FACT_SALESACTUAL f
                    INNER JOIN DIM_DATE d ON f.DATE_KEY = d.DATE_KEY
                    {additional_joins}
                    {where_clause}
                    GROUP BY {time_grouping}
                    ORDER BY {time_grouping}
                """,
                "time_groupings": {
                    "daily": "d.FULL_DATE",
                    "weekly": "YEAR(d.FULL_DATE), WEEK(d.FULL_DATE)",
                    "monthly": "d.YEAR, d.MONTH",
                    "quarterly": "d.YEAR, d.QUARTER",
                    "yearly": "d.YEAR"
                },
                "complexity": QueryComplexity.HIGH
            },
            "comparison": {
                "base_query": """
                    SELECT 
                        {comparison_dimension},
                        {metrics},
                        {comparison_metrics}
                    FROM FACT_SALESACTUAL f
                    {joins}
                    {where_clause}
                    GROUP BY {comparison_dimension}
                    ORDER BY {order_by_clause}
                """,
                "complexity": QueryComplexity.HIGH
            }
        }
    
    def _load_business_mappings(self) -> None:
        """Load mappings from business terms to database entities."""
        
        self.business_mappings = {
            # Sales metrics
            "sales": ["TOTAL_SALES_AMOUNT", "sales_amount", "revenue"],
            "revenue": ["TOTAL_SALES_AMOUNT"],
            "profit": ["GROSS_PROFIT"],
            "quantity": ["QUANTITY_SOLD"],
            "units": ["QUANTITY_SOLD"],
            "volume": ["QUANTITY_SOLD"],
            "transactions": ["count of SALES_ID"],
            "orders": ["count of SALES_ID"],
            
            # Time periods
            "this year": f"d.YEAR = {datetime.now().year}",
            "last year": f"d.YEAR = {datetime.now().year - 1}",
            "this month": f"d.YEAR = {datetime.now().year} AND d.MONTH = {datetime.now().month}",
            "last month": f"d.YEAR = {datetime.now().year} AND d.MONTH = {datetime.now().month - 1}",
            "this quarter": f"d.YEAR = {datetime.now().year} AND d.QUARTER = {(datetime.now().month-1)//3 + 1}",
            "ytd": f"d.YEAR = {datetime.now().year} AND d.FULL_DATE <= CURRENT_DATE",
            
            # Customer attributes
            "customer": "DIM_CUSTOMER",
            "customers": "DIM_CUSTOMER", 
            "client": "DIM_CUSTOMER",
            "clients": "DIM_CUSTOMER",
            
            # Product attributes
            "product": "DIM_PRODUCT",
            "products": "DIM_PRODUCT",
            "item": "DIM_PRODUCT",
            "items": "DIM_PRODUCT",
            "category": "p.CATEGORY",
            "categories": "p.CATEGORY",
            "brand": "p.BRAND",
            "brands": "p.BRAND",
            
            # Geographic attributes
            "store": "DIM_STORE",
            "stores": "DIM_STORE",
            "location": "DIM_STORE",
            "locations": "DIM_STORE",
            "city": "s.CITY",
            "cities": "s.CITY",
            "state": "s.STATE",
            "states": "s.STATE",
            "region": "s.STATE",
            "regions": "s.STATE",
            
            # Channel attributes
            "channel": "DIM_CHANNEL",
            "channels": "DIM_CHANNEL"
        }
    
    @track_performance(tags={"operation": "generate_sql"})
    async def generate_sql(
        self, 
        natural_language: str, 
        context: Dict[str, Any], 
        entities: Dict[str, Any]
    ) -> SQLQuery:
        """
        Generate SQL query from natural language input.
        
        Args:
            natural_language: Business question in natural language
            context: Conversation and business context
            entities: Extracted entities from intent analysis
            
        Returns:
            SQLQuery object with generated SQL and metadata
        """
        self.generation_counter.increment()
        
        try:
            # Sanitize input
            natural_language = sanitize_input(natural_language)
            
            # Analyze query requirements
            query_analysis = await self._analyze_query_requirements(
                natural_language, context, entities
            )
            
            # Select appropriate template
            template = self._select_query_template(query_analysis)
            
            # Generate SQL from template
            sql_query = await self._generate_from_template(
                template, query_analysis, natural_language
            )
            
            # Validate and optimize SQL
            validated_sql = await self._validate_and_optimize_sql(sql_query)
            
            self.success_counter.increment()
            
            self.logger.info(
                f"Generated SQL query successfully",
                extra={
                    "query_type": query_analysis["query_type"],
                    "complexity": validated_sql.complexity.value,
                    "confidence": validated_sql.confidence,
                    "tables": validated_sql.table_references
                }
            )
            
            return validated_sql
            
        except Exception as e:
            self.error_counter.increment()
            self.logger.error(f"SQL generation failed: {str(e)}")
            raise
    
    async def _analyze_query_requirements(
        self, 
        natural_language: str, 
        context: Dict[str, Any], 
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze natural language to determine query requirements."""
        
        text_lower = natural_language.lower()
        
        # Determine query type
        query_type = "general"
        if any(word in text_lower for word in ["sales", "revenue", "profit", "performance"]):
            query_type = "sales_performance"
        elif any(word in text_lower for word in ["customer", "client", "buyer"]):
            query_type = "customer_analysis"
        elif any(word in text_lower for word in ["product", "item", "category", "brand"]):
            query_type = "product_analysis"
        elif any(word in text_lower for word in ["trend", "over time", "monthly", "yearly", "daily"]):
            query_type = "time_series"
        elif any(word in text_lower for word in ["compare", "vs", "versus", "against"]):
            query_type = "comparison"
        
        # Extract metrics requested
        requested_metrics = []
        for business_term, db_fields in self.business_mappings.items():
            if business_term in text_lower:
                if isinstance(db_fields, list):
                    requested_metrics.extend(db_fields)
                else:
                    requested_metrics.append(db_fields)
        
        # Extract time filters
        time_filters = []
        for time_term, condition in self.business_mappings.items():
            if time_term in text_lower and "d." in str(condition):
                time_filters.append(condition)
        
        # Extract dimensional filters
        dimension_filters = {}
        if "entities" in entities:
            for entity in entities["entities"]:
                if entity["type"] in ["product", "customer", "store", "channel"]:
                    dimension_filters[entity["type"]] = entity["value"]
        
        # Determine grouping requirements
        grouping_dimensions = []
        if "by category" in text_lower or "by product category" in text_lower:
            grouping_dimensions.append("p.CATEGORY")
        if "by brand" in text_lower:
            grouping_dimensions.append("p.BRAND")
        if "by store" in text_lower or "by location" in text_lower:
            grouping_dimensions.append("s.STORE_NAME")
        if "by customer" in text_lower:
            grouping_dimensions.append("c.CUSTOMER_NAME")
        if "by month" in text_lower:
            grouping_dimensions.append("d.YEAR, d.MONTH")
        if "by quarter" in text_lower:
            grouping_dimensions.append("d.YEAR, d.QUARTER")
        if "by year" in text_lower:
            grouping_dimensions.append("d.YEAR")
        
        # Determine ordering requirements
        order_by = "total_sales DESC"  # Default
        if "highest" in text_lower or "top" in text_lower:
            order_by = "total_sales DESC"
        elif "lowest" in text_lower or "bottom" in text_lower:
            order_by = "total_sales ASC"
        elif "alphabetical" in text_lower:
            order_by = "1 ASC"  # First column
        
        # Determine limit requirements
        limit = None
        limit_match = re.search(r'top (\d+)', text_lower)
        if limit_match:
            limit = int(limit_match.group(1))
        elif "limit" in text_lower:
            limit_match = re.search(r'limit (\d+)', text_lower)
            if limit_match:
                limit = int(limit_match.group(1))
        
        return {
            "query_type": query_type,
            "requested_metrics": requested_metrics,
            "time_filters": time_filters,
            "dimension_filters": dimension_filters,
            "grouping_dimensions": grouping_dimensions,
            "order_by": order_by,
            "limit": limit,
            "requires_joins": self._determine_required_joins(
                query_type, grouping_dimensions, dimension_filters
            )
        }
    
    def _determine_required_joins(
        self, 
        query_type: str, 
        grouping_dimensions: List[str], 
        dimension_filters: Dict[str, Any]
    ) -> List[str]:
        """Determine which table joins are required for the query."""
        
        required_joins = ["INNER JOIN DIM_DATE d ON f.DATE_KEY = d.DATE_KEY"]
        
        # Check if customer dimension is needed
        if (query_type == "customer_analysis" or 
            any("c." in dim for dim in grouping_dimensions) or
            "customer" in dimension_filters):
            required_joins.append("INNER JOIN DIM_CUSTOMER c ON f.CUSTOMER_KEY = c.CUSTOMER_KEY")
        
        # Check if product dimension is needed
        if (query_type == "product_analysis" or 
            any("p." in dim for dim in grouping_dimensions) or
            "product" in dimension_filters):
            required_joins.append("INNER JOIN DIM_PRODUCT p ON f.PRODUCT_KEY = p.PRODUCT_KEY")
        
        # Check if store dimension is needed
        if (any("s." in dim for dim in grouping_dimensions) or
            "store" in dimension_filters):
            required_joins.append("INNER JOIN DIM_STORE s ON f.STORE_KEY = s.STORE_KEY")
        
        # Check if channel dimension is needed
        if "channel" in dimension_filters:
            required_joins.append("INNER JOIN DIM_CHANNEL ch ON f.CHANNEL_KEY = ch.CHANNEL_KEY")
        
        return required_joins
    
    def _select_query_template(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select the most appropriate query template."""
        
        query_type = query_analysis["query_type"]
        
        if query_type in self.query_templates:
            return self.query_templates[query_type]
        else:
            # Default to sales performance template
            return self.query_templates["sales_performance"]
    
    async def _generate_from_template(
        self,
        template: Dict[str, Any],
        analysis: Dict[str, Any],
        original_query: str
    ) -> SQLQuery:
        """Generate SQL query from template and analysis."""
        
        base_query = template["base_query"]
        
        # Build components
        metrics = self._build_metrics_clause(analysis)
        joins = " ".join(analysis["requires_joins"])
        where_clause = self._build_where_clause(analysis)
        group_by = self._build_group_by_clause(analysis)
        having_clause = self._build_having_clause(analysis)
        order_by = analysis.get("order_by", "total_sales DESC")
        limit_clause = f"LIMIT {analysis['limit']}" if analysis.get("limit") else ""
        
        # Handle template-specific substitutions
        if analysis["query_type"] == "sales_performance":
            substitutions = {
                "time_dimension": "d.YEAR, d.MONTH",
                "group_by_dimensions": ", ".join(analysis["grouping_dimensions"]) or "d.YEAR, d.MONTH",
                "joins": joins,
                "where_clause": where_clause,
                "group_by_clause": group_by or "d.YEAR, d.MONTH",
                "having_clause": having_clause,
                "order_by_clause": order_by,
                "limit_clause": limit_clause
            }
        elif analysis["query_type"] == "customer_analysis":
            substitutions = {
                "customer_metrics": ", ".join(template["default_metrics"]),
                "additional_joins": joins.replace("INNER JOIN DIM_CUSTOMER c ON f.CUSTOMER_KEY = c.CUSTOMER_KEY", ""),
                "where_clause": where_clause,
                "having_clause": having_clause,
                "order_by_clause": order_by,
                "limit_clause": limit_clause
            }
        elif analysis["query_type"] == "product_analysis":
            substitutions = {
                "product_metrics": ", ".join(template["default_metrics"]),
                "additional_joins": joins.replace("INNER JOIN DIM_PRODUCT p ON f.PRODUCT_KEY = p.PRODUCT_KEY", ""),
                "where_clause": where_clause,
                "having_clause": having_clause,
                "order_by_clause": order_by,
                "limit_clause": limit_clause
            }
        else:
            # Generic substitution
            substitutions = {
                "metrics": metrics,
                "joins": joins,
                "where_clause": where_clause,
                "group_by_clause": group_by,
                "having_clause": having_clause,
                "order_by_clause": order_by,
                "limit_clause": limit_clause
            }
        
        # Apply substitutions
        sql = base_query
        for placeholder, value in substitutions.items():
            sql = sql.replace(f"{{{placeholder}}}", str(value))
        
        # Clean up SQL
        sql = re.sub(r'\s+', ' ', sql)  # Remove extra whitespace
        sql = sql.strip()
        
        # Extract table references
        table_references = self._extract_table_references(sql)
        
        # Determine complexity
        complexity = self._determine_complexity(sql, analysis)
        
        # Calculate confidence
        confidence = self._calculate_confidence(analysis, original_query)
        
        return SQLQuery(
            sql=sql,
            parameters={},  # Parameters would be used for parameterized queries
            table_references=table_references,
            complexity=complexity,
            confidence=confidence,
            metadata={
                "query_type": analysis["query_type"],
                "requested_metrics": analysis["requested_metrics"],
                "grouping_dimensions": analysis["grouping_dimensions"],
                "original_query": original_query
            },
            optimization_suggestions=[]
        )
    
    def _build_metrics_clause(self, analysis: Dict[str, Any]) -> str:
        """Build the metrics/SELECT clause."""
        
        default_metrics = [
            "SUM(f.TOTAL_SALES_AMOUNT) as total_sales",
            "SUM(f.QUANTITY_SOLD) as total_quantity",
            "SUM(f.GROSS_PROFIT) as total_profit"
        ]
        
        if analysis["requested_metrics"]:
            metrics = []
            for metric in analysis["requested_metrics"]:
                if "count of" in metric.lower():
                    field = metric.replace("count of ", "").strip()
                    metrics.append(f"COUNT(DISTINCT f.{field}) as {field.lower()}_count")
                elif metric in ["TOTAL_SALES_AMOUNT", "QUANTITY_SOLD", "GROSS_PROFIT"]:
                    metrics.append(f"SUM(f.{metric}) as {metric.lower()}")
                else:
                    metrics.append(metric)
            return ", ".join(metrics)
        else:
            return ", ".join(default_metrics)
    
    def _build_where_clause(self, analysis: Dict[str, Any]) -> str:
        """Build the WHERE clause."""
        
        conditions = []
        
        # Add time filters
        if analysis["time_filters"]:
            conditions.extend(analysis["time_filters"])
        
        # Add dimension filters
        for dim_type, dim_value in analysis["dimension_filters"].items():
            if dim_type == "product":
                conditions.append(f"p.PRODUCT_NAME ILIKE '%{dim_value}%'")
            elif dim_type == "customer":
                conditions.append(f"c.CUSTOMER_NAME ILIKE '%{dim_value}%'")
            elif dim_type == "store":
                conditions.append(f"s.STORE_NAME ILIKE '%{dim_value}%'")
            elif dim_type == "channel":
                conditions.append(f"ch.CHANNEL_NAME ILIKE '%{dim_value}%'")
        
        if conditions:
            return "WHERE " + " AND ".join(conditions)
        else:
            return ""
    
    def _build_group_by_clause(self, analysis: Dict[str, Any]) -> str:
        """Build the GROUP BY clause."""
        
        if analysis["grouping_dimensions"]:
            return ", ".join(analysis["grouping_dimensions"])
        else:
            return ""
    
    def _build_having_clause(self, analysis: Dict[str, Any]) -> str:
        """Build the HAVING clause."""
        
        # Could add logic for aggregate filtering
        return ""
    
    def _extract_table_references(self, sql: str) -> List[str]:
        """Extract table references from SQL query."""
        
        tables = []
        sql_upper = sql.upper()
        
        # Look for table names in the schema
        for table_type, table_dict in self.schema.items():
            for table_name in table_dict.keys():
                if table_name in sql_upper:
                    tables.append(table_name)
        
        return list(set(tables))
    
    def _determine_complexity(self, sql: str, analysis: Dict[str, Any]) -> QueryComplexity:
        """Determine query complexity based on structure."""
        
        complexity_score = 0
        
        # Count joins
        join_count = sql.upper().count("JOIN")
        complexity_score += join_count * 10
        
        # Count subqueries
        subquery_count = sql.count("(SELECT")
        complexity_score += subquery_count * 20
        
        # Count aggregations
        agg_count = (sql.upper().count("SUM(") + sql.upper().count("COUNT(") + 
                    sql.upper().count("AVG(") + sql.upper().count("MAX(") + 
                    sql.upper().count("MIN("))
        complexity_score += agg_count * 5
        
        # Check for window functions
        if "OVER(" in sql.upper():
            complexity_score += 25
        
        # Check for HAVING clause
        if "HAVING" in sql.upper():
            complexity_score += 10
        
        if complexity_score <= 20:
            return QueryComplexity.LOW
        elif complexity_score <= 50:
            return QueryComplexity.MEDIUM
        elif complexity_score <= 80:
            return QueryComplexity.HIGH
        else:
            return QueryComplexity.VERY_HIGH
    
    def _calculate_confidence(self, analysis: Dict[str, Any], original_query: str) -> float:
        """Calculate confidence score for the generated query."""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we found specific entities
        if analysis["requested_metrics"]:
            confidence += 0.2
        
        if analysis["dimension_filters"]:
            confidence += 0.1
        
        if analysis["grouping_dimensions"]:
            confidence += 0.1
        
        # Increase confidence if query type is well-matched
        if analysis["query_type"] != "general":
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _validate_and_optimize_sql(self, sql_query: SQLQuery) -> SQLQuery:
        """Validate SQL syntax and add optimization suggestions."""
        
        try:
            # Use Session A validation utilities
            is_valid, errors = validate_sql_query(sql_query.sql)
            
            if not is_valid:
                self.logger.warning(f"Generated SQL has validation errors: {errors}")
                # Could attempt to fix common issues here
            
            # Add optimization suggestions
            suggestions = []
            
            # Check for missing indexes
            if "ORDER BY" in sql_query.sql.upper() and sql_query.complexity in [QueryComplexity.HIGH, QueryComplexity.VERY_HIGH]:
                suggestions.append("Consider adding indexes on ORDER BY columns for better performance")
            
            # Check for large table scans
            if "FACT_SALESACTUAL" in sql_query.table_references and not any("WHERE" in clause for clause in [sql_query.sql]):
                suggestions.append("Consider adding date filters to limit fact table scan")
            
            # Update optimization suggestions
            sql_query.optimization_suggestions = suggestions
            
            return sql_query
            
        except Exception as e:
            self.logger.error(f"SQL validation failed: {str(e)}")
            return sql_query
    
    # Additional specialized query generation methods
    
    async def generate_insight_query(
        self, 
        natural_language: str, 
        context: Dict[str, Any], 
        entities: Dict[str, Any]
    ) -> SQLQuery:
        """Generate query specifically for insight extraction."""
        
        # Add insight-specific logic
        insight_context = {**context, "query_purpose": "insight_generation"}
        return await self.generate_sql(natural_language, insight_context, entities)
    
    async def generate_pattern_query(
        self, 
        natural_language: str, 
        context: Dict[str, Any], 
        entities: Dict[str, Any]
    ) -> SQLQuery:
        """Generate query for pattern detection and trend analysis."""
        
        # Force time series analysis for pattern detection
        pattern_context = {**context, "force_time_series": True}
        return await self.generate_sql(natural_language, pattern_context, entities)
    
    async def generate_recommendation_query(
        self, 
        natural_language: str, 
        context: Dict[str, Any], 
        entities: Dict[str, Any]
    ) -> SQLQuery:
        """Generate query for recommendation generation."""
        
        # Add recommendation-specific context
        rec_context = {**context, "query_purpose": "recommendation_generation"}
        return await self.generate_sql(natural_language, rec_context, entities)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the Query Generator."""
        
        try:
            # Test basic functionality
            test_query = "What were our total sales last month?"
            test_context = {}
            test_entities = {"entities": []}
            
            sql_result = await self.generate_sql(test_query, test_context, test_entities)
            
            return {
                "status": "healthy",
                "details": "Query generation operational",
                "test_query_generated": bool(sql_result.sql),
                "schema_tables_loaded": len(self.schema),
                "templates_loaded": len(self.query_templates)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Query generation failed health check"
            } 