"""
Data Quality Analyzer for Data Intelligence Agent

This module analyzes data quality and completeness, providing comprehensive
quality assessments and recommendations for data improvement.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import Counter

from pydantic import BaseModel, Field
from shared.config.settings import Settings
from shared.utils.caching import get_cache_manager
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.validation import validate_input, ValidationError

logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Data quality dimensions for assessment."""
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    CONFORMITY = "conformity"

class QualityIssueType(Enum):
    """Types of data quality issues."""
    MISSING_VALUES = "missing_values"
    DUPLICATE_RECORDS = "duplicate_records"
    INVALID_FORMAT = "invalid_format"
    OUTLIERS = "outliers"
    INCONSISTENT_VALUES = "inconsistent_values"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"

class QualitySeverity(Enum):
    """Severity levels for quality issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class QualityIssue:
    """Individual data quality issue."""
    issue_type: QualityIssueType
    dimension: QualityDimension
    severity: QualitySeverity
    column: Optional[str]
    description: str
    affected_rows: int
    percentage: float
    examples: List[Any] = field(default_factory=list)
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityScore:
    """Quality score for a dimension or overall."""
    dimension: QualityDimension
    score: float  # 0-100
    grade: str  # A, B, C, D, F
    issues_count: int
    critical_issues: int
    recommendations: List[str] = field(default_factory=list)

class QualityAnalysisRequest(BaseModel):
    """Request for quality analysis."""
    data: Dict[str, Any]
    table_name: Optional[str] = None
    schema_info: Dict[str, Any] = Field(default_factory=dict)
    business_rules: Dict[str, Any] = Field(default_factory=dict)
    dimensions: List[QualityDimension] = Field(default_factory=lambda: list(QualityDimension))
    include_samples: bool = True
    sample_size: int = 10
    
    class Config:
        validate_assignment = True

class QualityAnalysisResponse(BaseModel):
    """Response from quality analysis."""
    overall_score: float
    overall_grade: str
    dimension_scores: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True

class QualityAnalyzer:
    """
    Data Quality Analyzer for comprehensive data quality assessment.
    
    Analyzes data across multiple quality dimensions and provides
    actionable recommendations for improvement.
    """
    
    def __init__(self, settings: Settings):
        """Initialize quality analyzer."""
        self.settings = settings
        self.cache = get_cache_manager()
        self.metrics = get_metrics_collector()
        
        # Configuration
        self.config = {
            'cache_ttl': settings.QUALITY_ANALYZER.get('cache_ttl', 3600),
            'max_sample_size': settings.QUALITY_ANALYZER.get('max_sample_size', 100),
            'outlier_threshold': settings.QUALITY_ANALYZER.get('outlier_threshold', 3),  # Z-score
            'duplicate_threshold': settings.QUALITY_ANALYZER.get('duplicate_threshold', 0.05),
            'missing_threshold': settings.QUALITY_ANALYZER.get('missing_threshold', 0.1)
        }
        
        # Quality rules and thresholds
        self._setup_quality_rules()
        self._setup_grading_scheme()
        
        logger.info("Quality Analyzer initialized")

    def _setup_quality_rules(self):
        """Setup quality assessment rules and patterns."""
        self.quality_rules = {
            'completeness': {
                'excellent': 0.99,
                'good': 0.95,
                'fair': 0.90,
                'poor': 0.80
            },
            'uniqueness': {
                'excellent': 0.99,
                'good': 0.95,
                'fair': 0.90,
                'poor': 0.80
            },
            'validity': {
                'patterns': {
                    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    'phone': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
                    'date': r'^\d{4}-\d{2}-\d{2}$',
                    'currency': r'^\$?[\d,]+\.?\d{0,2}$'
                },
                'thresholds': {
                    'excellent': 0.99,
                    'good': 0.95,
                    'fair': 0.90,
                    'poor': 0.80
                }
            },
            'consistency': {
                'case_consistency': 0.95,
                'format_consistency': 0.95,
                'value_consistency': 0.90
            }
        }

    def _setup_grading_scheme(self):
        """Setup grading scheme for quality scores."""
        self.grading_scheme = {
            'A': (90, 100),
            'B': (80, 90),
            'C': (70, 80),
            'D': (60, 70),
            'F': (0, 60)
        }

    @track_performance(tags={"operation": "analyze_quality"})
    async def analyze_quality(self, request: QualityAnalysisRequest) -> QualityAnalysisResponse:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            request: Quality analysis request
            
        Returns:
            Quality analysis response with scores and recommendations
        """
        try:
            # Prepare DataFrame
            df = self._prepare_dataframe(request.data)
            if df.empty:
                return self._create_empty_response()
            
            # Analyze each dimension
            dimension_results = {}
            all_issues = []
            
            for dimension in request.dimensions:
                score, issues = await self._analyze_dimension(df, dimension, request)
                dimension_results[dimension.value] = {
                    'score': score.score,
                    'grade': score.grade,
                    'issues_count': score.issues_count,
                    'critical_issues': score.critical_issues,
                    'recommendations': score.recommendations
                }
                all_issues.extend(issues)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(dimension_results)
            overall_grade = self._assign_grade(overall_score)
            
            # Generate summary and recommendations
            summary = self._generate_summary(df, dimension_results, overall_score)
            recommendations = self._generate_recommendations(all_issues, dimension_results)
            
            # Record metrics
            await self._record_quality_metrics(request, df, overall_score, len(all_issues))
            
            return QualityAnalysisResponse(
                overall_score=overall_score,
                overall_grade=overall_grade,
                dimension_scores=dimension_results,
                issues=[issue.__dict__ for issue in all_issues],
                summary=summary,
                recommendations=recommendations,
                metadata={
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'total_issues': len(all_issues),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing quality: {str(e)}")
            return self._create_error_response(str(e))

    def _prepare_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert data to DataFrame for analysis."""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict) and 'data' in data:
                return pd.DataFrame(data['data'])
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {str(e)}")
            return pd.DataFrame()

    async def _analyze_dimension(self, df: pd.DataFrame, dimension: QualityDimension, request: QualityAnalysisRequest) -> Tuple[QualityScore, List[QualityIssue]]:
        """Analyze a specific quality dimension."""
        if dimension == QualityDimension.COMPLETENESS:
            return await self._analyze_completeness(df, request)
        elif dimension == QualityDimension.UNIQUENESS:
            return await self._analyze_uniqueness(df, request)
        elif dimension == QualityDimension.VALIDITY:
            return await self._analyze_validity(df, request)
        elif dimension == QualityDimension.CONSISTENCY:
            return await self._analyze_consistency(df, request)
        elif dimension == QualityDimension.ACCURACY:
            return await self._analyze_accuracy(df, request)
        else:
            # Default analysis for other dimensions
            return QualityScore(dimension, 85.0, "B", 0, 0), []

    async def _analyze_completeness(self, df: pd.DataFrame, request: QualityAnalysisRequest) -> Tuple[QualityScore, List[QualityIssue]]:
        """Analyze data completeness."""
        issues = []
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells) if total_cells > 0 else 1
        
        # Check each column
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = missing_count / len(df) * 100
            
            if missing_count > 0:
                severity = self._determine_missing_severity(missing_percentage)
                issue = QualityIssue(
                    issue_type=QualityIssueType.MISSING_VALUES,
                    dimension=QualityDimension.COMPLETENESS,
                    severity=severity,
                    column=col,
                    description=f"Column {col} has {missing_count} missing values ({missing_percentage:.2f}%)",
                    affected_rows=missing_count,
                    percentage=missing_percentage,
                    recommendation=self._get_missing_value_recommendation(missing_percentage)
                )
                issues.append(issue)
        
        # Calculate score
        score = completeness_ratio * 100
        grade = self._assign_grade(score)
        critical_issues = len([i for i in issues if i.severity == QualitySeverity.CRITICAL])
        
        quality_score = QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            grade=grade,
            issues_count=len(issues),
            critical_issues=critical_issues,
            recommendations=self._get_completeness_recommendations(completeness_ratio)
        )
        
        return quality_score, issues

    async def _analyze_uniqueness(self, df: pd.DataFrame, request: QualityAnalysisRequest) -> Tuple[QualityScore, List[QualityIssue]]:
        """Analyze data uniqueness."""
        issues = []
        total_score = 0
        columns_analyzed = 0
        
        for col in df.columns:
            if df[col].dtype == 'object' or 'id' in col.lower():
                # Analyze uniqueness for text columns and ID columns
                total_values = len(df[col].dropna())
                unique_values = df[col].nunique()
                uniqueness_ratio = unique_values / total_values if total_values > 0 else 1
                
                # Check for duplicates
                duplicates = df[col].value_counts()
                duplicate_values = duplicates[duplicates > 1]
                
                if len(duplicate_values) > 0:
                    duplicate_percentage = (total_values - unique_values) / total_values * 100
                    severity = self._determine_duplicate_severity(duplicate_percentage)
                    
                    issue = QualityIssue(
                        issue_type=QualityIssueType.DUPLICATE_RECORDS,
                        dimension=QualityDimension.UNIQUENESS,
                        severity=severity,
                        column=col,
                        description=f"Column {col} has {len(duplicate_values)} duplicate values ({duplicate_percentage:.2f}%)",
                        affected_rows=total_values - unique_values,
                        percentage=duplicate_percentage,
                        examples=duplicate_values.head(5).index.tolist(),
                        recommendation=self._get_duplicate_recommendation(duplicate_percentage, col)
                    )
                    issues.append(issue)
                
                total_score += uniqueness_ratio * 100
                columns_analyzed += 1
        
        # Calculate overall uniqueness score
        score = total_score / columns_analyzed if columns_analyzed > 0 else 100
        grade = self._assign_grade(score)
        critical_issues = len([i for i in issues if i.severity == QualitySeverity.CRITICAL])
        
        quality_score = QualityScore(
            dimension=QualityDimension.UNIQUENESS,
            score=score,
            grade=grade,
            issues_count=len(issues),
            critical_issues=critical_issues,
            recommendations=self._get_uniqueness_recommendations(score)
        )
        
        return quality_score, issues

    async def _analyze_validity(self, df: pd.DataFrame, request: QualityAnalysisRequest) -> Tuple[QualityScore, List[QualityIssue]]:
        """Analyze data validity (format compliance)."""
        issues = []
        total_score = 0
        columns_analyzed = 0
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
                
            # Determine expected format based on column name and data
            expected_format = self._infer_column_format(col, col_data)
            
            if expected_format:
                pattern = self.quality_rules['validity']['patterns'].get(expected_format)
                if pattern:
                    import re
                    valid_count = col_data.astype(str).str.match(pattern).sum()
                    validity_ratio = valid_count / len(col_data)
                    
                    if validity_ratio < 1.0:
                        invalid_count = len(col_data) - valid_count
                        invalid_percentage = invalid_count / len(col_data) * 100
                        severity = self._determine_validity_severity(validity_ratio)
                        
                        # Get examples of invalid values
                        invalid_mask = ~col_data.astype(str).str.match(pattern)
                        invalid_examples = col_data[invalid_mask].head(3).tolist()
                        
                        issue = QualityIssue(
                            issue_type=QualityIssueType.INVALID_FORMAT,
                            dimension=QualityDimension.VALIDITY,
                            severity=severity,
                            column=col,
                            description=f"Column {col} has {invalid_count} values with invalid {expected_format} format ({invalid_percentage:.2f}%)",
                            affected_rows=invalid_count,
                            percentage=invalid_percentage,
                            examples=invalid_examples,
                            recommendation=f"Validate and correct {expected_format} format in column {col}"
                        )
                        issues.append(issue)
                    
                    total_score += validity_ratio * 100
                    columns_analyzed += 1
            
            # Check for outliers in numeric columns
            if col_data.dtype in ['int64', 'float64']:
                outliers = self._detect_outliers(col_data)
                if len(outliers) > 0:
                    outlier_percentage = len(outliers) / len(col_data) * 100
                    if outlier_percentage > 5:  # More than 5% outliers
                        severity = QualitySeverity.MEDIUM if outlier_percentage < 10 else QualitySeverity.HIGH
                        
                        issue = QualityIssue(
                            issue_type=QualityIssueType.OUTLIERS,
                            dimension=QualityDimension.VALIDITY,
                            severity=severity,
                            column=col,
                            description=f"Column {col} has {len(outliers)} statistical outliers ({outlier_percentage:.2f}%)",
                            affected_rows=len(outliers),
                            percentage=outlier_percentage,
                            examples=outliers[:3].tolist(),
                            recommendation=f"Review outlier values in {col} for data entry errors"
                        )
                        issues.append(issue)
        
        # Calculate overall validity score
        score = total_score / columns_analyzed if columns_analyzed > 0 else 100
        grade = self._assign_grade(score)
        critical_issues = len([i for i in issues if i.severity == QualitySeverity.CRITICAL])
        
        quality_score = QualityScore(
            dimension=QualityDimension.VALIDITY,
            score=score,
            grade=grade,
            issues_count=len(issues),
            critical_issues=critical_issues,
            recommendations=self._get_validity_recommendations(score)
        )
        
        return quality_score, issues

    async def _analyze_consistency(self, df: pd.DataFrame, request: QualityAnalysisRequest) -> Tuple[QualityScore, List[QualityIssue]]:
        """Analyze data consistency."""
        issues = []
        consistency_scores = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                # Check case consistency
                case_consistency = self._check_case_consistency(col_data)
                consistency_scores.append(case_consistency)
                
                if case_consistency < 0.9:
                    inconsistent_percentage = (1 - case_consistency) * 100
                    issue = QualityIssue(
                        issue_type=QualityIssueType.INCONSISTENT_VALUES,
                        dimension=QualityDimension.CONSISTENCY,
                        severity=QualitySeverity.MEDIUM,
                        column=col,
                        description=f"Column {col} has inconsistent case formatting ({inconsistent_percentage:.1f}% inconsistent)",
                        affected_rows=int(len(col_data) * (1 - case_consistency)),
                        percentage=inconsistent_percentage,
                        recommendation=f"Standardize case formatting in column {col}"
                    )
                    issues.append(issue)
                
                # Check format consistency (for common patterns)
                format_consistency = self._check_format_consistency(col_data)
                consistency_scores.append(format_consistency)
                
                if format_consistency < 0.9:
                    inconsistent_percentage = (1 - format_consistency) * 100
                    issue = QualityIssue(
                        issue_type=QualityIssueType.INCONSISTENT_VALUES,
                        dimension=QualityDimension.CONSISTENCY,
                        severity=QualitySeverity.MEDIUM,
                        column=col,
                        description=f"Column {col} has inconsistent value formatting ({inconsistent_percentage:.1f}% inconsistent)",
                        affected_rows=int(len(col_data) * (1 - format_consistency)),
                        percentage=inconsistent_percentage,
                        recommendation=f"Standardize value formatting in column {col}"
                    )
                    issues.append(issue)
        
        # Calculate overall consistency score
        score = np.mean(consistency_scores) * 100 if consistency_scores else 100
        grade = self._assign_grade(score)
        critical_issues = len([i for i in issues if i.severity == QualitySeverity.CRITICAL])
        
        quality_score = QualityScore(
            dimension=QualityDimension.CONSISTENCY,
            score=score,
            grade=grade,
            issues_count=len(issues),
            critical_issues=critical_issues,
            recommendations=self._get_consistency_recommendations(score)
        )
        
        return quality_score, issues

    async def _analyze_accuracy(self, df: pd.DataFrame, request: QualityAnalysisRequest) -> Tuple[QualityScore, List[QualityIssue]]:
        """Analyze data accuracy (business rules validation)."""
        issues = []
        business_rules = request.business_rules
        
        # Default accuracy assessment without business rules
        score = 85.0  # Assume good accuracy without specific rules
        
        # Apply business rules if provided
        if business_rules:
            rule_scores = []
            for rule_name, rule_config in business_rules.items():
                rule_score = self._validate_business_rule(df, rule_name, rule_config)
                rule_scores.append(rule_score)
                
                if rule_score < 0.9:
                    violation_percentage = (1 - rule_score) * 100
                    issue = QualityIssue(
                        issue_type=QualityIssueType.BUSINESS_RULE_VIOLATION,
                        dimension=QualityDimension.ACCURACY,
                        severity=QualitySeverity.HIGH,
                        column=rule_config.get('column'),
                        description=f"Business rule '{rule_name}' violated in {violation_percentage:.1f}% of records",
                        affected_rows=int(len(df) * (1 - rule_score)),
                        percentage=violation_percentage,
                        recommendation=f"Review and correct violations of business rule: {rule_name}"
                    )
                    issues.append(issue)
            
            if rule_scores:
                score = np.mean(rule_scores) * 100
        
        grade = self._assign_grade(score)
        critical_issues = len([i for i in issues if i.severity == QualitySeverity.CRITICAL])
        
        quality_score = QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=score,
            grade=grade,
            issues_count=len(issues),
            critical_issues=critical_issues,
            recommendations=self._get_accuracy_recommendations(score, bool(business_rules))
        )
        
        return quality_score, issues

    def _infer_column_format(self, col_name: str, col_data: pd.Series) -> Optional[str]:
        """Infer expected format for a column."""
        col_lower = col_name.lower()
        
        if 'email' in col_lower:
            return 'email'
        elif 'phone' in col_lower or 'tel' in col_lower:
            return 'phone'
        elif 'date' in col_lower:
            return 'date'
        elif 'price' in col_lower or 'amount' in col_lower or 'cost' in col_lower:
            return 'currency'
        
        return None

    def _detect_outliers(self, data: pd.Series) -> pd.Series:
        """Detect statistical outliers using Z-score."""
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[z_scores > self.config['outlier_threshold']]

    def _check_case_consistency(self, data: pd.Series) -> float:
        """Check case consistency in text data."""
        if len(data) == 0:
            return 1.0
        
        # Count different case patterns
        lower_count = data.str.islower().sum()
        upper_count = data.str.isupper().sum()
        title_count = data.str.istitle().sum()
        
        total = len(data)
        max_pattern = max(lower_count, upper_count, title_count)
        
        return max_pattern / total

    def _check_format_consistency(self, data: pd.Series) -> float:
        """Check format consistency in text data."""
        if len(data) == 0:
            return 1.0
        
        # Group by length and character patterns
        lengths = data.str.len()
        most_common_length = lengths.mode().iloc[0] if not lengths.mode().empty else 0
        
        # Calculate consistency as percentage with most common length
        consistency = (lengths == most_common_length).sum() / len(data)
        
        return consistency

    def _validate_business_rule(self, df: pd.DataFrame, rule_name: str, rule_config: Dict[str, Any]) -> float:
        """Validate a business rule against the data."""
        try:
            column = rule_config.get('column')
            rule_type = rule_config.get('type')
            
            if not column or column not in df.columns:
                return 1.0
            
            if rule_type == 'range':
                min_val = rule_config.get('min')
                max_val = rule_config.get('max')
                valid_mask = (df[column] >= min_val) & (df[column] <= max_val)
                return valid_mask.sum() / len(df)
            
            elif rule_type == 'not_null':
                return df[column].notna().sum() / len(df)
            
            elif rule_type == 'values':
                allowed_values = rule_config.get('allowed_values', [])
                valid_mask = df[column].isin(allowed_values)
                return valid_mask.sum() / len(df)
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Error validating business rule {rule_name}: {str(e)}")
            return 1.0

    def _determine_missing_severity(self, percentage: float) -> QualitySeverity:
        """Determine severity of missing values."""
        if percentage > 50:
            return QualitySeverity.CRITICAL
        elif percentage > 20:
            return QualitySeverity.HIGH
        elif percentage > 10:
            return QualitySeverity.MEDIUM
        else:
            return QualitySeverity.LOW

    def _determine_duplicate_severity(self, percentage: float) -> QualitySeverity:
        """Determine severity of duplicate values."""
        if percentage > 30:
            return QualitySeverity.HIGH
        elif percentage > 10:
            return QualitySeverity.MEDIUM
        else:
            return QualitySeverity.LOW

    def _determine_validity_severity(self, validity_ratio: float) -> QualitySeverity:
        """Determine severity of validity issues."""
        if validity_ratio < 0.8:
            return QualitySeverity.HIGH
        elif validity_ratio < 0.9:
            return QualitySeverity.MEDIUM
        else:
            return QualitySeverity.LOW

    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score."""
        for grade, (min_score, max_score) in self.grading_scheme.items():
            if min_score <= score <= max_score:
                return grade
        return 'F'

    def _calculate_overall_score(self, dimension_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall quality score."""
        if not dimension_results:
            return 0.0
        
        scores = [result['score'] for result in dimension_results.values()]
        return np.mean(scores)

    def _get_missing_value_recommendation(self, percentage: float) -> str:
        """Get recommendation for missing values."""
        if percentage > 50:
            return "Critical: Investigate data collection process - over 50% missing"
        elif percentage > 20:
            return "High priority: Implement data validation at source"
        elif percentage > 10:
            return "Medium priority: Consider imputation strategies"
        else:
            return "Low priority: Monitor for increasing trend"

    def _get_duplicate_recommendation(self, percentage: float, column: str) -> str:
        """Get recommendation for duplicate values."""
        if 'id' in column.lower():
            return f"Critical: Remove duplicate IDs in {column}"
        elif percentage > 20:
            return f"High priority: Investigate duplicate records in {column}"
        else:
            return f"Review duplicate handling policy for {column}"

    def _get_completeness_recommendations(self, ratio: float) -> List[str]:
        """Get recommendations for completeness."""
        recommendations = []
        if ratio < 0.9:
            recommendations.append("Implement mandatory field validation at data entry")
            recommendations.append("Review data collection processes")
        if ratio < 0.8:
            recommendations.append("Consider data imputation strategies")
        return recommendations

    def _get_uniqueness_recommendations(self, score: float) -> List[str]:
        """Get recommendations for uniqueness."""
        recommendations = []
        if score < 90:
            recommendations.append("Implement duplicate detection and prevention")
            recommendations.append("Review data deduplication processes")
        return recommendations

    def _get_validity_recommendations(self, score: float) -> List[str]:
        """Get recommendations for validity."""
        recommendations = []
        if score < 90:
            recommendations.append("Implement format validation at data entry")
            recommendations.append("Create data quality rules for common formats")
        return recommendations

    def _get_consistency_recommendations(self, score: float) -> List[str]:
        """Get recommendations for consistency."""
        recommendations = []
        if score < 90:
            recommendations.append("Standardize data entry formats")
            recommendations.append("Implement data transformation rules")
        return recommendations

    def _get_accuracy_recommendations(self, score: float, has_business_rules: bool) -> List[str]:
        """Get recommendations for accuracy."""
        recommendations = []
        if not has_business_rules:
            recommendations.append("Define business rules for data validation")
        if score < 90:
            recommendations.append("Implement business rule validation")
            recommendations.append("Regular data accuracy audits")
        return recommendations

    def _generate_summary(self, df: pd.DataFrame, dimension_results: Dict[str, Dict[str, Any]], overall_score: float) -> str:
        """Generate quality analysis summary."""
        total_issues = sum(result['issues_count'] for result in dimension_results.values())
        critical_issues = sum(result['critical_issues'] for result in dimension_results.values())
        
        summary = f"Data Quality Assessment: Overall score {overall_score:.1f}% "
        summary += f"({self._assign_grade(overall_score)} grade) for {len(df):,} records across {len(df.columns)} columns. "
        
        if total_issues == 0:
            summary += "No significant quality issues detected."
        else:
            summary += f"Found {total_issues} quality issues"
            if critical_issues > 0:
                summary += f" including {critical_issues} critical issues"
            summary += " requiring attention."
        
        return summary

    def _generate_recommendations(self, issues: List[QualityIssue], dimension_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # Critical issues first
        critical_issues = [i for i in issues if i.severity == QualitySeverity.CRITICAL]
        if critical_issues:
            recommendations.append("CRITICAL: Address critical data quality issues immediately")
            for issue in critical_issues[:3]:  # Top 3 critical issues
                recommendations.append(f"- {issue.recommendation}")
        
        # High priority issues
        high_issues = [i for i in issues if i.severity == QualitySeverity.HIGH]
        if high_issues:
            recommendations.append("HIGH PRIORITY: Address high-impact quality issues")
            for issue in high_issues[:3]:  # Top 3 high issues
                recommendations.append(f"- {issue.recommendation}")
        
        # General recommendations
        if len(issues) > 5:
            recommendations.append("Implement comprehensive data quality monitoring")
        
        return recommendations

    def _create_empty_response(self) -> QualityAnalysisResponse:
        """Create response for empty data."""
        return QualityAnalysisResponse(
            overall_score=0.0,
            overall_grade='F',
            dimension_scores={},
            issues=[],
            summary="No data available for quality analysis",
            recommendations=["Provide data for quality assessment"],
            metadata={'empty_data': True}
        )

    def _create_error_response(self, error_message: str) -> QualityAnalysisResponse:
        """Create response for analysis errors."""
        return QualityAnalysisResponse(
            overall_score=0.0,
            overall_grade='F',
            dimension_scores={},
            issues=[],
            summary=f"Quality analysis error: {error_message}",
            recommendations=["Review data format and analysis requirements"],
            metadata={'error': True, 'error_message': error_message}
        )

    async def _record_quality_metrics(self, request: QualityAnalysisRequest, df: pd.DataFrame, overall_score: float, issues_count: int):
        """Record quality analysis metrics."""
        try:
            metrics_data = {
                'overall_score': overall_score,
                'row_count': len(df),
                'column_count': len(df.columns),
                'issues_count': issues_count,
                'dimensions_analyzed': len(request.dimensions)
            }
            
            await self.metrics.record_event('quality_analysis', metrics_data)
        except Exception as e:
            logger.warning(f"Error recording metrics: {str(e)}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of quality analyzer."""
        return {
            'service': 'quality_analyzer',
            'status': 'healthy',
            'cache_enabled': self.cache is not None,
            'metrics_enabled': self.metrics is not None,
            'config': {
                'cache_ttl': self.config['cache_ttl'],
                'outlier_threshold': self.config['outlier_threshold'],
                'duplicate_threshold': self.config['duplicate_threshold']
            },
            'timestamp': datetime.now().isoformat()
        } 