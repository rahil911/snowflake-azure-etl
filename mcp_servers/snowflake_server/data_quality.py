#!/usr/bin/env python3
"""
Snowflake Data Quality Checker
===============================

Data quality validation and assessment tools for the Snowflake MCP server.
Provides completeness checks, uniqueness validation, pattern matching,
and overall data quality scoring.

Integrates with Session A foundation for validation, caching, and metrics.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set
from enum import Enum
from dataclasses import dataclass

# Session A Foundation imports
from shared.utils.validation import check_data_quality, validate_sql_query
from shared.utils.caching import cache_analytics
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.data_processing import DataProcessor

from .connection_manager import SnowflakeConnectionManager


class QualityCheckType(str, Enum):
    """Types of data quality checks."""
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    INTEGRITY = "integrity"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"


@dataclass
class QualityCheckResult:
    """Result of a data quality check."""
    check_type: QualityCheckType
    table_name: str
    column_name: Optional[str]
    passed: bool
    score: float  # 0.0 to 1.0
    total_records: int
    failed_records: int
    details: Dict[str, Any]
    timestamp: datetime


class SnowflakeDataQuality:
    """
    Comprehensive data quality assessment for Snowflake tables.
    
    Provides various data quality checks with scoring and detailed reporting.
    """
    
    def __init__(self, connection_manager: SnowflakeConnectionManager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger("SnowflakeDataQuality")
        self.metrics = get_metrics_collector()
        self.data_processor = DataProcessor()
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityCheckType.COMPLETENESS: 0.95,  # 95% non-null
            QualityCheckType.UNIQUENESS: 0.95,    # 95% unique values
            QualityCheckType.VALIDITY: 0.90,      # 90% valid format
            QualityCheckType.CONSISTENCY: 0.95,   # 95% consistent
            QualityCheckType.DUPLICATES: 0.05     # Max 5% duplicates
        }
        
        # State
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize data quality checker."""
        if self._initialized:
            return
        
        self.logger.info("Initializing Snowflake data quality checker...")
        self._initialized = True
        self.logger.info("Snowflake data quality checker initialized")
    
    async def cleanup(self) -> None:
        """Cleanup data quality checker."""
        self.logger.info("Cleaning up Snowflake data quality checker...")
        self._initialized = False
        self.logger.info("Snowflake data quality checker cleanup complete")
    
    @track_performance(tags={"component": "data_quality", "operation": "check_quality"})
    async def check_quality(
        self,
        table_name: str,
        check_types: List[str],
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks on a table.
        
        Args:
            table_name: Name of the table to check
            check_types: List of quality check types to perform
            sample_size: Number of rows to sample (None for full table)
            
        Returns:
            Comprehensive quality assessment report
        """
        try:
            self.logger.info(f"Starting data quality checks for table: {table_name}")
            
            # Convert string check types to enums
            check_enums = []
            for check_type in check_types:
                try:
                    check_enums.append(QualityCheckType(check_type.lower()))
                except ValueError:
                    self.logger.warning(f"Unknown check type: {check_type}")
            
            # Get table metadata
            table_info = await self._get_table_metadata(table_name)
            if not table_info:
                raise ValueError(f"Table {table_name} not found or not accessible")
            
            # Perform checks
            check_results = []
            for check_type in check_enums:
                try:
                    result = await self._perform_quality_check(
                        table_name, check_type, sample_size
                    )
                    check_results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed {check_type.value} check: {str(e)}")
                    # Add failed check result
                    check_results.append(QualityCheckResult(
                        check_type=check_type,
                        table_name=table_name,
                        column_name=None,
                        passed=False,
                        score=0.0,
                        total_records=0,
                        failed_records=0,
                        details={"error": str(e)},
                        timestamp=datetime.utcnow()
                    ))
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(check_results)
            
            # Create comprehensive report
            report = {
                "table_name": table_name,
                "table_info": table_info,
                "sample_size": sample_size,
                "overall_score": overall_score,
                "quality_grade": self._get_quality_grade(overall_score),
                "checks_performed": len(check_results),
                "checks_passed": sum(1 for r in check_results if r.passed),
                "individual_checks": [
                    {
                        "check_type": r.check_type.value,
                        "column_name": r.column_name,
                        "passed": r.passed,
                        "score": r.score,
                        "total_records": r.total_records,
                        "failed_records": r.failed_records,
                        "details": r.details,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in check_results
                ],
                "recommendations": self._generate_recommendations(check_results),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update metrics
            self.metrics.counter("snowflake.data_quality.assessments").increment()
            self.metrics.histogram("snowflake.data_quality.overall_score").update(overall_score)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Data quality assessment failed for {table_name}: {str(e)}")
            self.metrics.counter("snowflake.data_quality.assessment_errors").increment()
            raise
    
    async def _get_table_metadata(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get basic table metadata."""
        # Parse table name
        table_parts = table_name.split('.')
        if len(table_parts) == 2:
            schema_name, table_name_only = table_parts
            where_clause = "table_schema = %(schema)s AND table_name = %(table)s"
            params = {"schema": schema_name.upper(), "table": table_name_only.upper()}
        else:
            where_clause = "table_name = %(table)s"
            params = {"table": table_name.upper()}
        
        query = f"""
        SELECT 
            table_catalog as database_name,
            table_schema as schema_name,
            table_name,
            table_type,
            row_count,
            bytes
        FROM information_schema.tables 
        WHERE {where_clause}
        LIMIT 1
        """
        
        result = await self.connection_manager.execute_query(query, params)
        return result[0] if result else None
    
    async def _perform_quality_check(
        self,
        table_name: str,
        check_type: QualityCheckType,
        sample_size: Optional[int] = None
    ) -> QualityCheckResult:
        """Perform a specific quality check."""
        
        if check_type == QualityCheckType.COMPLETENESS:
            return await self._check_completeness(table_name, sample_size)
        elif check_type == QualityCheckType.UNIQUENESS:
            return await self._check_uniqueness(table_name, sample_size)
        elif check_type == QualityCheckType.DUPLICATES:
            return await self._check_duplicates(table_name, sample_size)
        elif check_type == QualityCheckType.VALIDITY:
            return await self._check_validity(table_name, sample_size)
        elif check_type == QualityCheckType.CONSISTENCY:
            return await self._check_consistency(table_name, sample_size)
        else:
            raise ValueError(f"Unsupported check type: {check_type}")
    
    async def _check_completeness(
        self,
        table_name: str,
        sample_size: Optional[int] = None
    ) -> QualityCheckResult:
        """Check data completeness (non-null values)."""
        
        # Get table columns
        columns = await self._get_table_columns(table_name)
        
        sample_clause = f"TABLESAMPLE ({sample_size} ROWS)" if sample_size else ""
        
        # Build query to check null counts for each column
        null_checks = []
        for column in columns:
            col_name = column["COLUMN_NAME"]
            null_checks.append(f"SUM(CASE WHEN {col_name} IS NULL THEN 1 ELSE 0 END) as {col_name}_nulls")
        
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            {', '.join(null_checks)}
        FROM {table_name} {sample_clause}
        """
        
        result = await self.connection_manager.execute_query(query)
        
        if not result:
            raise ValueError("Failed to execute completeness check query")
        
        row = result[0]
        total_rows = row["TOTAL_ROWS"]
        
        # Calculate completeness scores for each column
        column_scores = {}
        total_nulls = 0
        
        for column in columns:
            col_name = column["COLUMN_NAME"]
            null_count = row.get(f"{col_name}_NULLS", 0)
            completeness = (total_rows - null_count) / total_rows if total_rows > 0 else 0
            column_scores[col_name] = {
                "completeness": completeness,
                "null_count": null_count,
                "non_null_count": total_rows - null_count
            }
            total_nulls += null_count
        
        # Overall completeness score
        total_cells = total_rows * len(columns)
        overall_score = (total_cells - total_nulls) / total_cells if total_cells > 0 else 0
        
        threshold = self.quality_thresholds[QualityCheckType.COMPLETENESS]
        passed = overall_score >= threshold
        
        return QualityCheckResult(
            check_type=QualityCheckType.COMPLETENESS,
            table_name=table_name,
            column_name=None,
            passed=passed,
            score=overall_score,
            total_records=total_rows,
            failed_records=total_nulls,
            details={
                "column_scores": column_scores,
                "threshold": threshold,
                "total_cells": total_cells,
                "null_cells": total_nulls
            },
            timestamp=datetime.utcnow()
        )
    
    async def _check_uniqueness(
        self,
        table_name: str,
        sample_size: Optional[int] = None
    ) -> QualityCheckResult:
        """Check data uniqueness across columns."""
        
        sample_clause = f"TABLESAMPLE ({sample_size} ROWS)" if sample_size else ""
        
        # Check overall row uniqueness
        query = f"""
        WITH row_counts AS (
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT *) as unique_rows
            FROM {table_name} {sample_clause}
        )
        SELECT 
            total_rows,
            unique_rows,
            (unique_rows::FLOAT / total_rows) as uniqueness_ratio
        FROM row_counts
        """
        
        result = await self.connection_manager.execute_query(query)
        
        if not result:
            raise ValueError("Failed to execute uniqueness check query")
        
        row = result[0]
        total_rows = row["TOTAL_ROWS"]
        unique_rows = row["UNIQUE_ROWS"]
        uniqueness_ratio = row["UNIQUENESS_RATIO"]
        
        duplicate_rows = total_rows - unique_rows
        threshold = self.quality_thresholds[QualityCheckType.UNIQUENESS]
        passed = uniqueness_ratio >= threshold
        
        return QualityCheckResult(
            check_type=QualityCheckType.UNIQUENESS,
            table_name=table_name,
            column_name=None,
            passed=passed,
            score=uniqueness_ratio,
            total_records=total_rows,
            failed_records=duplicate_rows,
            details={
                "unique_rows": unique_rows,
                "duplicate_rows": duplicate_rows,
                "uniqueness_ratio": uniqueness_ratio,
                "threshold": threshold
            },
            timestamp=datetime.utcnow()
        )
    
    async def _check_duplicates(
        self,
        table_name: str,
        sample_size: Optional[int] = None
    ) -> QualityCheckResult:
        """Check for duplicate records."""
        
        sample_clause = f"TABLESAMPLE ({sample_size} ROWS)" if sample_size else ""
        
        # Find duplicate records based on all columns
        query = f"""
        WITH duplicate_check AS (
            SELECT 
                *,
                COUNT(*) OVER (PARTITION BY *) as dup_count
            FROM {table_name} {sample_clause}
        ),
        summary AS (
            SELECT 
                COUNT(*) as total_rows,
                SUM(CASE WHEN dup_count > 1 THEN 1 ELSE 0 END) as duplicate_rows
            FROM duplicate_check
        )
        SELECT 
            total_rows,
            duplicate_rows,
            (1.0 - (duplicate_rows::FLOAT / total_rows)) as quality_score
        FROM summary
        """
        
        result = await self.connection_manager.execute_query(query)
        
        if not result:
            raise ValueError("Failed to execute duplicate check query")
        
        row = result[0]
        total_rows = row["TOTAL_ROWS"]
        duplicate_rows = row["DUPLICATE_ROWS"]
        quality_score = row["QUALITY_SCORE"]
        
        duplicate_threshold = self.quality_thresholds[QualityCheckType.DUPLICATES]
        duplicate_rate = duplicate_rows / total_rows if total_rows > 0 else 0
        passed = duplicate_rate <= duplicate_threshold
        
        return QualityCheckResult(
            check_type=QualityCheckType.DUPLICATES,
            table_name=table_name,
            column_name=None,
            passed=passed,
            score=quality_score,
            total_records=total_rows,
            failed_records=duplicate_rows,
            details={
                "duplicate_rows": duplicate_rows,
                "duplicate_rate": duplicate_rate,
                "threshold": duplicate_threshold
            },
            timestamp=datetime.utcnow()
        )
    
    async def _check_validity(
        self,
        table_name: str,
        sample_size: Optional[int] = None
    ) -> QualityCheckResult:
        """Check data validity (format, range, etc.)."""
        
        # Get table columns with their data types
        columns = await self._get_table_columns(table_name)
        
        sample_clause = f"TABLESAMPLE ({sample_size} ROWS)" if sample_size else ""
        
        # Build validity checks based on data types
        validity_checks = []
        for column in columns:
            col_name = column["COLUMN_NAME"]
            data_type = column["DATA_TYPE"].upper()
            
            if "DATE" in data_type or "TIMESTAMP" in data_type:
                # Check for valid dates
                validity_checks.append(f"""
                SUM(CASE 
                    WHEN {col_name} IS NULL THEN 0
                    WHEN TRY_TO_DATE({col_name}) IS NULL AND {col_name} IS NOT NULL THEN 1
                    ELSE 0 
                END) as {col_name}_invalid
                """)
            elif "NUMBER" in data_type or "DECIMAL" in data_type or "FLOAT" in data_type:
                # Check for valid numbers
                validity_checks.append(f"""
                SUM(CASE 
                    WHEN {col_name} IS NULL THEN 0
                    WHEN TRY_TO_NUMBER({col_name}) IS NULL AND {col_name} IS NOT NULL THEN 1
                    ELSE 0 
                END) as {col_name}_invalid
                """)
            elif "EMAIL" in col_name.upper():
                # Basic email validation
                validity_checks.append(f"""
                SUM(CASE 
                    WHEN {col_name} IS NULL THEN 0
                    WHEN {col_name} NOT RLIKE '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{{2,}}$' THEN 1
                    ELSE 0 
                END) as {col_name}_invalid
                """)
        
        if not validity_checks:
            # If no specific validity checks, assume all data is valid
            return QualityCheckResult(
                check_type=QualityCheckType.VALIDITY,
                table_name=table_name,
                column_name=None,
                passed=True,
                score=1.0,
                total_records=0,
                failed_records=0,
                details={"message": "No specific validity checks applicable"},
                timestamp=datetime.utcnow()
            )
        
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            {', '.join(validity_checks)}
        FROM {table_name} {sample_clause}
        """
        
        result = await self.connection_manager.execute_query(query)
        
        if not result:
            raise ValueError("Failed to execute validity check query")
        
        row = result[0]
        total_rows = row["TOTAL_ROWS"]
        
        # Calculate validity scores
        total_invalid = 0
        column_validity = {}
        
        for column in columns:
            col_name = column["COLUMN_NAME"]
            invalid_key = f"{col_name}_INVALID"
            if invalid_key in row:
                invalid_count = row[invalid_key]
                total_invalid += invalid_count
                column_validity[col_name] = {
                    "invalid_count": invalid_count,
                    "validity_rate": (total_rows - invalid_count) / total_rows if total_rows > 0 else 1.0
                }
        
        total_checks = total_rows * len(column_validity)
        overall_score = (total_checks - total_invalid) / total_checks if total_checks > 0 else 1.0
        
        threshold = self.quality_thresholds[QualityCheckType.VALIDITY]
        passed = overall_score >= threshold
        
        return QualityCheckResult(
            check_type=QualityCheckType.VALIDITY,
            table_name=table_name,
            column_name=None,
            passed=passed,
            score=overall_score,
            total_records=total_rows,
            failed_records=total_invalid,
            details={
                "column_validity": column_validity,
                "threshold": threshold,
                "checks_performed": len(column_validity)
            },
            timestamp=datetime.utcnow()
        )
    
    async def _check_consistency(
        self,
        table_name: str,
        sample_size: Optional[int] = None
    ) -> QualityCheckResult:
        """Check data consistency patterns."""
        
        # For now, implement a basic consistency check
        # This could be expanded to check referential integrity, 
        # format consistency, etc.
        
        sample_clause = f"TABLESAMPLE ({sample_size} ROWS)" if sample_size else ""
        
        # Basic consistency: check for mixed case in string columns
        columns = await self._get_table_columns(table_name)
        string_columns = [
            col for col in columns 
            if any(t in col["DATA_TYPE"].upper() for t in ["VARCHAR", "STRING", "TEXT", "CHAR"])
        ]
        
        if not string_columns:
            return QualityCheckResult(
                check_type=QualityCheckType.CONSISTENCY,
                table_name=table_name,
                column_name=None,
                passed=True,
                score=1.0,
                total_records=0,
                failed_records=0,
                details={"message": "No string columns for consistency checking"},
                timestamp=datetime.utcnow()
            )
        
        consistency_checks = []
        for column in string_columns:
            col_name = column["COLUMN_NAME"]
            consistency_checks.append(f"""
            SUM(CASE 
                WHEN {col_name} IS NULL THEN 0
                WHEN {col_name} != UPPER({col_name}) AND {col_name} != LOWER({col_name}) THEN 1
                ELSE 0 
            END) as {col_name}_mixed_case
            """)
        
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            {', '.join(consistency_checks)}
        FROM {table_name} {sample_clause}
        """
        
        result = await self.connection_manager.execute_query(query)
        
        if not result:
            raise ValueError("Failed to execute consistency check query")
        
        row = result[0]
        total_rows = row["TOTAL_ROWS"]
        
        # Calculate consistency score
        total_inconsistent = 0
        column_consistency = {}
        
        for column in string_columns:
            col_name = column["COLUMN_NAME"]
            mixed_case_key = f"{col_name}_MIXED_CASE"
            if mixed_case_key in row:
                mixed_case_count = row[mixed_case_key]
                total_inconsistent += mixed_case_count
                column_consistency[col_name] = {
                    "mixed_case_count": mixed_case_count,
                    "consistency_rate": (total_rows - mixed_case_count) / total_rows if total_rows > 0 else 1.0
                }
        
        total_checks = total_rows * len(string_columns)
        overall_score = (total_checks - total_inconsistent) / total_checks if total_checks > 0 else 1.0
        
        threshold = self.quality_thresholds[QualityCheckType.CONSISTENCY]
        passed = overall_score >= threshold
        
        return QualityCheckResult(
            check_type=QualityCheckType.CONSISTENCY,
            table_name=table_name,
            column_name=None,
            passed=passed,
            score=overall_score,
            total_records=total_rows,
            failed_records=total_inconsistent,
            details={
                "column_consistency": column_consistency,
                "threshold": threshold,
                "check_type": "case_consistency"
            },
            timestamp=datetime.utcnow()
        )
    
    async def _get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table column information."""
        # Parse table name
        table_parts = table_name.split('.')
        if len(table_parts) == 2:
            schema_name, table_name_only = table_parts
            where_clause = "table_schema = %(schema)s AND table_name = %(table)s"
            params = {"schema": schema_name.upper(), "table": table_name_only.upper()}
        else:
            where_clause = "table_name = %(table)s"
            params = {"table": table_name.upper()}
        
        query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            ordinal_position
        FROM information_schema.columns 
        WHERE {where_clause}
        ORDER BY ordinal_position
        """
        
        result = await self.connection_manager.execute_query(query, params)
        return result or []
    
    def _calculate_overall_score(self, check_results: List[QualityCheckResult]) -> float:
        """Calculate overall quality score from individual check results."""
        if not check_results:
            return 0.0
        
        # Weight different check types
        weights = {
            QualityCheckType.COMPLETENESS: 0.3,
            QualityCheckType.UNIQUENESS: 0.2,
            QualityCheckType.VALIDITY: 0.2,
            QualityCheckType.CONSISTENCY: 0.15,
            QualityCheckType.DUPLICATES: 0.15
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in check_results:
            weight = weights.get(result.check_type, 0.1)
            weighted_score += result.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 0.95:
            return "A"
        elif score >= 0.85:
            return "B"
        elif score >= 0.75:
            return "C"
        elif score >= 0.65:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, check_results: List[QualityCheckResult]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        for result in check_results:
            if not result.passed:
                if result.check_type == QualityCheckType.COMPLETENESS:
                    recommendations.append(
                        f"Address NULL values in table {result.table_name}. "
                        f"Consider setting default values or making fields required."
                    )
                elif result.check_type == QualityCheckType.UNIQUENESS:
                    recommendations.append(
                        f"Remove duplicate records from table {result.table_name}. "
                        f"Consider adding unique constraints."
                    )
                elif result.check_type == QualityCheckType.VALIDITY:
                    recommendations.append(
                        f"Fix data format issues in table {result.table_name}. "
                        f"Implement data validation rules."
                    )
                elif result.check_type == QualityCheckType.CONSISTENCY:
                    recommendations.append(
                        f"Standardize data formats in table {result.table_name}. "
                        f"Implement consistent naming conventions."
                    )
                elif result.check_type == QualityCheckType.DUPLICATES:
                    recommendations.append(
                        f"Implement deduplication process for table {result.table_name}."
                    )
        
        if not recommendations:
            recommendations.append("Data quality is excellent! No immediate improvements needed.")
        
        return recommendations 