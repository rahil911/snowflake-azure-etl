"""
Insight Extractor for Data Intelligence Agent

This module analyzes query results and data to extract meaningful business insights,
trends, anomalies, and actionable recommendations for decision making.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
import numpy as np
from pathlib import Path

from pydantic import BaseModel, Field, validator
import pandas as pd

from shared.schemas.data_models import QueryResult, BusinessEntity, DataQualityMetric
from shared.schemas.data_models import AnalysisResult
from shared.utils.caching import get_cache_manager
from shared.utils.validation import ValidationHelper
from shared.utils.retry import RetryStrategy, retry_on_exception
from shared.config.settings import Settings


logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of business insights."""
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    SEASONAL = "seasonal"
    THRESHOLD = "threshold"
    PERFORMANCE = "performance"
    OPPORTUNITY = "opportunity"
    RISK = "risk"


class InsightSeverity(Enum):
    """Severity levels for insights."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class InsightCategory(Enum):
    """Categories of business insights."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    CUSTOMER = "customer"
    PRODUCT = "product"
    MARKET = "market"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"


@dataclass
class BusinessInsight:
    """Individual business insight with metadata."""
    insight_type: InsightType
    category: InsightCategory
    severity: InsightSeverity
    title: str
    description: str
    value: Any
    confidence: float
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "insight_type": self.insight_type.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "value": self.value,
            "confidence": self.confidence,
            "supporting_data": self.supporting_data,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class TrendAnalysis(BaseModel):
    """Trend analysis results."""
    trend_direction: str  # up, down, flat, volatile
    trend_strength: float  # 0-1
    trend_duration: Optional[str] = None
    growth_rate: Optional[float] = None
    seasonal_component: Optional[Dict[str, Any]] = None
    forecast: Optional[Dict[str, Any]] = None
    
    class Config:
        validate_assignment = True


class AnomalyDetection(BaseModel):
    """Anomaly detection results."""
    anomalies_found: List[Dict[str, Any]] = Field(default_factory=list)
    anomaly_score: float = 0.0
    detection_method: str = "statistical"
    threshold_settings: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True


class CorrelationAnalysis(BaseModel):
    """Correlation analysis results."""
    correlations: Dict[str, float] = Field(default_factory=dict)
    significant_correlations: List[Dict[str, Any]] = Field(default_factory=list)
    correlation_strength: str = "weak"  # weak, moderate, strong
    
    class Config:
        validate_assignment = True


class InsightExtractionRequest(BaseModel):
    """Request for insight extraction."""
    data: Dict[str, Any]
    query_context: Optional[Dict[str, Any]] = None
    business_context: Optional[Dict[str, Any]] = None
    analysis_types: List[InsightType] = Field(default_factory=lambda: list(InsightType))
    include_recommendations: bool = True
    min_confidence: float = 0.6
    
    class Config:
        validate_assignment = True


class InsightExtractionResponse(BaseModel):
    """Response from insight extraction."""
    insights: List[Dict[str, Any]]
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_score: float = 0.0
    
    class Config:
        validate_assignment = True


class InsightExtractor:
    """
    Advanced insight extraction and analysis for business intelligence data.
    
    This class handles:
    - Trend analysis and forecasting
    - Anomaly detection
    - Correlation analysis
    - Performance insights
    - Business opportunity identification
    - Risk assessment
    """
    
    def __init__(self, settings: Settings):
        """Initialize the insight extractor."""
        self.settings = settings
        self.cache = get_cache_manager()
        self.validator = ValidationHelper()
        self.retry_strategy = RetryStrategy.EXPONENTIAL
        
        # Analysis configuration
        self.config = {
            'anomaly_threshold': 2.0,  # Standard deviations
            'trend_min_points': 3,
            'correlation_threshold': 0.5,
            'confidence_threshold': 0.6,
            'forecast_periods': 12
        }
        
        # Business rules and thresholds
        self._setup_business_rules()
        
        logger.info("Insight extractor initialized")
    
    def _setup_business_rules(self):
        """Setup business rules and thresholds."""
        self.business_rules = {
            'financial': {
                'revenue_growth_threshold': 0.05,  # 5% growth
                'profit_margin_threshold': 0.10,   # 10% margin
                'cost_increase_threshold': 0.15,   # 15% cost increase alarm
            },
            'operational': {
                'efficiency_threshold': 0.85,      # 85% efficiency
                'error_rate_threshold': 0.02,      # 2% error rate
                'uptime_threshold': 0.99,          # 99% uptime
            },
            'customer': {
                'satisfaction_threshold': 4.0,     # 4.0/5.0 satisfaction
                'churn_rate_threshold': 0.05,      # 5% churn rate
                'retention_threshold': 0.90,       # 90% retention
            },
            'quality': {
                'defect_rate_threshold': 0.01,     # 1% defect rate
                'completion_rate_threshold': 0.95, # 95% completion
                'accuracy_threshold': 0.98,        # 98% accuracy
            }
        }
        
        self.kpi_mappings = {
            'revenue': {'category': InsightCategory.FINANCIAL, 'rules': 'financial'},
            'profit': {'category': InsightCategory.FINANCIAL, 'rules': 'financial'},
            'sales': {'category': InsightCategory.FINANCIAL, 'rules': 'financial'},
            'cost': {'category': InsightCategory.FINANCIAL, 'rules': 'financial'},
            'efficiency': {'category': InsightCategory.OPERATIONAL, 'rules': 'operational'},
            'error_rate': {'category': InsightCategory.QUALITY, 'rules': 'quality'},
            'customer_satisfaction': {'category': InsightCategory.CUSTOMER, 'rules': 'customer'},
            'churn_rate': {'category': InsightCategory.CUSTOMER, 'rules': 'customer'}
        }
    
    async def extract_insights(self, request: InsightExtractionRequest) -> InsightExtractionResponse:
        """
        Extract comprehensive business insights from data.
        
        Args:
            request: Insight extraction request with data and context
            
        Returns:
            Extracted insights with recommendations and metadata
        """
        try:
            start_time = datetime.now()
            logger.info("Starting insight extraction")
            
            # Convert data to DataFrame for analysis
            df = self._prepare_dataframe(request.data)
            
            if df.empty:
                return InsightExtractionResponse(
                    insights=[],
                    summary="No data available for analysis",
                    analysis_metadata={'error': 'empty_dataset'}
                )
            
            # Extract different types of insights
            insights = []
            
            for insight_type in request.analysis_types:
                if insight_type == InsightType.TREND:
                    trend_insights = await self._extract_trend_insights(df, request)
                    insights.extend(trend_insights)
                
                elif insight_type == InsightType.ANOMALY:
                    anomaly_insights = await self._extract_anomaly_insights(df, request)
                    insights.extend(anomaly_insights)
                
                elif insight_type == InsightType.CORRELATION:
                    correlation_insights = await self._extract_correlation_insights(df, request)
                    insights.extend(correlation_insights)
                
                elif insight_type == InsightType.PERFORMANCE:
                    performance_insights = await self._extract_performance_insights(df, request)
                    insights.extend(performance_insights)
                
                elif insight_type == InsightType.THRESHOLD:
                    threshold_insights = await self._extract_threshold_insights(df, request)
                    insights.extend(threshold_insights)
                
                elif insight_type == InsightType.OPPORTUNITY:
                    opportunity_insights = await self._extract_opportunity_insights(df, request)
                    insights.extend(opportunity_insights)
                
                elif insight_type == InsightType.RISK:
                    risk_insights = await self._extract_risk_insights(df, request)
                    insights.extend(risk_insights)
            
            # Filter by confidence threshold
            filtered_insights = [
                insight for insight in insights
                if insight.confidence >= request.min_confidence
            ]
            
            # Sort by severity and confidence
            filtered_insights.sort(
                key=lambda x: (self._severity_weight(x.severity), -x.confidence),
                reverse=True
            )
            
            # Generate summary and recommendations
            summary = self._generate_summary(filtered_insights)
            key_findings = self._extract_key_findings(filtered_insights)
            recommendations = []
            
            if request.include_recommendations:
                recommendations = self._generate_recommendations(filtered_insights, request)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(filtered_insights, df)
            
            response = InsightExtractionResponse(
                insights=[insight.to_dict() for insight in filtered_insights],
                summary=summary,
                key_findings=key_findings,
                recommendations=recommendations,
                analysis_metadata={
                    'total_insights': len(filtered_insights),
                    'data_points': len(df),
                    'analysis_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'insight_types_analyzed': [it.value for it in request.analysis_types]
                },
                quality_score=quality_score
            )
            
            logger.info(f"Insight extraction completed with {len(filtered_insights)} insights")
            return response
            
        except Exception as e:
            logger.error(f"Error in insight extraction: {str(e)}")
            raise
    
    def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert input data to pandas DataFrame."""
        try:
            if isinstance(data, dict):
                if 'rows' in data and 'columns' in data:
                    # Structured format with rows and columns
                    return pd.DataFrame(data['rows'], columns=data['columns'])
                elif 'data' in data:
                    # Nested data format
                    return pd.DataFrame(data['data'])
                else:
                    # Direct data format
                    return pd.DataFrame(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {str(e)}")
            return pd.DataFrame()
    
    async def _extract_trend_insights(self, df: pd.DataFrame, request: InsightExtractionRequest) -> List[BusinessInsight]:
        """Extract trend-based insights."""
        insights = []
        
        try:
            # Identify numeric columns for trend analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Look for date/time columns
            date_cols = []
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_cols.append(col)
                    except:
                        continue
            
            if not date_cols or not numeric_cols.any():
                return insights
            
            # Analyze trends for each numeric column
            for metric_col in numeric_cols:
                for date_col in date_cols:
                    trend_insight = await self._analyze_trend(df, metric_col, date_col, request)
                    if trend_insight:
                        insights.append(trend_insight)
            
        except Exception as e:
            logger.error(f"Error extracting trend insights: {str(e)}")
        
        return insights
    
    async def _analyze_trend(self, df: pd.DataFrame, metric_col: str, date_col: str, request: InsightExtractionRequest) -> Optional[BusinessInsight]:
        """Analyze trend for a specific metric."""
        try:
            # Sort by date and calculate trend
            sorted_df = df.sort_values(date_col)
            values = sorted_df[metric_col].dropna()
            
            if len(values) < self.config['trend_min_points']:
                return None
            
            # Calculate trend direction and strength
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            slope = z[0]
            
            # Determine trend direction
            if abs(slope) < values.std() * 0.1:
                direction = "flat"
                strength = 0.3
            elif slope > 0:
                direction = "increasing"
                strength = min(abs(slope) / values.std(), 1.0)
            else:
                direction = "decreasing"
                strength = min(abs(slope) / values.std(), 1.0)
            
            # Calculate growth rate
            if len(values) >= 2:
                growth_rate = (values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100
            else:
                growth_rate = 0
            
            # Determine insight category
            category = self._get_insight_category(metric_col)
            
            # Determine severity based on trend and business rules
            severity = self._determine_trend_severity(metric_col, direction, growth_rate, strength)
            
            # Create insight
            insight = BusinessInsight(
                insight_type=InsightType.TREND,
                category=category,
                severity=severity,
                title=f"{metric_col.title()} Trend Analysis",
                description=f"{metric_col.title()} shows a {direction} trend with {strength:.1%} strength",
                value={
                    'direction': direction,
                    'strength': strength,
                    'growth_rate': growth_rate,
                    'current_value': float(values.iloc[-1]),
                    'previous_value': float(values.iloc[0]) if len(values) > 1 else None
                },
                confidence=min(strength + 0.2, 1.0),
                supporting_data={
                    'data_points': len(values),
                    'time_period': f"{sorted_df[date_col].min()} to {sorted_df[date_col].max()}",
                    'slope': float(slope),
                    'trend_line': z.tolist()
                },
                recommendations=self._generate_trend_recommendations(metric_col, direction, growth_rate)
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {metric_col}: {str(e)}")
            return None
    
    async def _extract_anomaly_insights(self, df: pd.DataFrame, request: InsightExtractionRequest) -> List[BusinessInsight]:
        """Extract anomaly-based insights."""
        insights = []
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                anomaly_insight = await self._detect_anomalies(df, col, request)
                if anomaly_insight:
                    insights.append(anomaly_insight)
        
        except Exception as e:
            logger.error(f"Error extracting anomaly insights: {str(e)}")
        
        return insights
    
    async def _detect_anomalies(self, df: pd.DataFrame, col: str, request: InsightExtractionRequest) -> Optional[BusinessInsight]:
        """Detect anomalies in a specific column."""
        try:
            values = df[col].dropna()
            
            if len(values) < 3:
                return None
            
            # Statistical anomaly detection using z-score
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val == 0:
                return None
            
            z_scores = np.abs((values - mean_val) / std_val)
            anomalies = values[z_scores > self.config['anomaly_threshold']]
            
            if len(anomalies) == 0:
                return None
            
            # Calculate anomaly severity
            max_z_score = z_scores.max()
            anomaly_severity = self._determine_anomaly_severity(max_z_score)
            
            # Get insight category
            category = self._get_insight_category(col)
            
            # Create insight
            insight = BusinessInsight(
                insight_type=InsightType.ANOMALY,
                category=category,
                severity=anomaly_severity,
                title=f"Anomalies Detected in {col.title()}",
                description=f"Found {len(anomalies)} anomalous values in {col} (max z-score: {max_z_score:.2f})",
                value={
                    'anomaly_count': len(anomalies),
                    'anomalous_values': anomalies.tolist(),
                    'max_z_score': float(max_z_score),
                    'threshold': self.config['anomaly_threshold']
                },
                confidence=min(max_z_score / 5.0, 1.0),
                supporting_data={
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'total_values': len(values),
                    'anomaly_percentage': len(anomalies) / len(values) * 100
                },
                recommendations=self._generate_anomaly_recommendations(col, len(anomalies), max_z_score)
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {col}: {str(e)}")
            return None
    
    async def _extract_correlation_insights(self, df: pd.DataFrame, request: InsightExtractionRequest) -> List[BusinessInsight]:
        """Extract correlation-based insights."""
        insights = []
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return insights
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Find significant correlations
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i >= j:  # Skip diagonal and duplicates
                        continue
                    
                    correlation = corr_matrix.loc[col1, col2]
                    
                    if abs(correlation) >= self.config['correlation_threshold']:
                        correlation_insight = self._create_correlation_insight(col1, col2, correlation)
                        if correlation_insight:
                            insights.append(correlation_insight)
        
        except Exception as e:
            logger.error(f"Error extracting correlation insights: {str(e)}")
        
        return insights
    
    def _create_correlation_insight(self, col1: str, col2: str, correlation: float) -> Optional[BusinessInsight]:
        """Create a correlation insight."""
        try:
            # Determine correlation strength
            abs_corr = abs(correlation)
            if abs_corr >= 0.8:
                strength = "strong"
                severity = InsightSeverity.HIGH
            elif abs_corr >= 0.6:
                strength = "moderate"
                severity = InsightSeverity.MEDIUM
            else:
                strength = "weak"
                severity = InsightSeverity.LOW
            
            direction = "positive" if correlation > 0 else "negative"
            
            insight = BusinessInsight(
                insight_type=InsightType.CORRELATION,
                category=InsightCategory.OPERATIONAL,  # Default category
                severity=severity,
                title=f"{strength.title()} {direction.title()} Correlation",
                description=f"{col1.title()} and {col2.title()} show a {strength} {direction} correlation ({correlation:.3f})",
                value={
                    'correlation_coefficient': float(correlation),
                    'strength': strength,
                    'direction': direction,
                    'variable1': col1,
                    'variable2': col2
                },
                confidence=abs_corr,
                supporting_data={
                    'correlation_value': float(correlation),
                    'absolute_correlation': float(abs_corr)
                },
                recommendations=self._generate_correlation_recommendations(col1, col2, correlation, strength)
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error creating correlation insight: {str(e)}")
            return None
    
    async def _extract_performance_insights(self, df: pd.DataFrame, request: InsightExtractionRequest) -> List[BusinessInsight]:
        """Extract performance-based insights."""
        insights = []
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                performance_insight = await self._analyze_performance(df, col, request)
                if performance_insight:
                    insights.append(performance_insight)
        
        except Exception as e:
            logger.error(f"Error extracting performance insights: {str(e)}")
        
        return insights
    
    async def _analyze_performance(self, df: pd.DataFrame, col: str, request: InsightExtractionRequest) -> Optional[BusinessInsight]:
        """Analyze performance for a specific metric."""
        try:
            values = df[col].dropna()
            
            if len(values) == 0:
                return None
            
            # Calculate basic statistics
            mean_val = values.mean()
            median_val = values.median()
            std_val = values.std()
            min_val = values.min()
            max_val = values.max()
            
            # Determine performance based on business rules
            category = self._get_insight_category(col)
            rules_key = self.kpi_mappings.get(col.lower(), {}).get('rules', 'operational')
            
            if rules_key in self.business_rules:
                rules = self.business_rules[rules_key]
                performance_insight = self._evaluate_performance_against_rules(
                    col, mean_val, rules, category
                )
                if performance_insight:
                    performance_insight.supporting_data.update({
                        'mean': float(mean_val),
                        'median': float(median_val),
                        'std': float(std_val),
                        'min': float(min_val),
                        'max': float(max_val),
                        'data_points': len(values)
                    })
                    return performance_insight
        
        except Exception as e:
            logger.error(f"Error analyzing performance for {col}: {str(e)}")
        
        return None
    
    def _evaluate_performance_against_rules(self, metric: str, value: float, rules: Dict[str, float], category: InsightCategory) -> Optional[BusinessInsight]:
        """Evaluate performance against business rules."""
        try:
            # Find relevant threshold
            threshold_key = None
            for key in rules.keys():
                if key.replace('_threshold', '').replace('_', ' ') in metric.lower():
                    threshold_key = key
                    break
            
            if not threshold_key:
                return None
            
            threshold = rules[threshold_key]
            
            # Determine if metric meets threshold
            meets_threshold = value >= threshold
            
            # Determine severity based on performance
            if meets_threshold:
                severity = InsightSeverity.INFO
                performance_status = "meets expectations"
            else:
                performance_gap = (threshold - value) / threshold
                if performance_gap > 0.2:
                    severity = InsightSeverity.HIGH
                    performance_status = "significantly below expectations"
                elif performance_gap > 0.1:
                    severity = InsightSeverity.MEDIUM
                    performance_status = "below expectations"
                else:
                    severity = InsightSeverity.LOW
                    performance_status = "slightly below expectations"
            
            insight = BusinessInsight(
                insight_type=InsightType.PERFORMANCE,
                category=category,
                severity=severity,
                title=f"{metric.title()} Performance Analysis",
                description=f"{metric.title()} {performance_status} (current: {value:.3f}, threshold: {threshold:.3f})",
                value={
                    'current_value': float(value),
                    'threshold': float(threshold),
                    'meets_threshold': meets_threshold,
                    'performance_gap': float((threshold - value) / threshold) if not meets_threshold else 0.0
                },
                confidence=0.85,
                recommendations=self._generate_performance_recommendations(metric, value, threshold, meets_threshold)
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {str(e)}")
            return None
    
    async def _extract_threshold_insights(self, df: pd.DataFrame, request: InsightExtractionRequest) -> List[BusinessInsight]:
        """Extract threshold-based insights."""
        insights = []
        # Implementation would analyze values against predefined thresholds
        return insights
    
    async def _extract_opportunity_insights(self, df: pd.DataFrame, request: InsightExtractionRequest) -> List[BusinessInsight]:
        """Extract opportunity-based insights."""
        insights = []
        # Implementation would identify business opportunities in the data
        return insights
    
    async def _extract_risk_insights(self, df: pd.DataFrame, request: InsightExtractionRequest) -> List[BusinessInsight]:
        """Extract risk-based insights."""
        insights = []
        # Implementation would identify potential risks in the data
        return insights
    
    def _get_insight_category(self, metric_name: str) -> InsightCategory:
        """Determine insight category based on metric name."""
        metric_lower = metric_name.lower()
        
        if any(term in metric_lower for term in ['revenue', 'profit', 'cost', 'sales', 'price']):
            return InsightCategory.FINANCIAL
        elif any(term in metric_lower for term in ['customer', 'satisfaction', 'churn', 'retention']):
            return InsightCategory.CUSTOMER
        elif any(term in metric_lower for term in ['product', 'quality', 'defect', 'error']):
            return InsightCategory.QUALITY
        elif any(term in metric_lower for term in ['efficiency', 'performance', 'utilization']):
            return InsightCategory.EFFICIENCY
        else:
            return InsightCategory.OPERATIONAL
    
    def _determine_trend_severity(self, metric: str, direction: str, growth_rate: float, strength: float) -> InsightSeverity:
        """Determine severity of trend insight."""
        if strength < 0.3:
            return InsightSeverity.LOW
        
        abs_growth = abs(growth_rate)
        
        if abs_growth > 50 or strength > 0.8:
            return InsightSeverity.HIGH
        elif abs_growth > 20 or strength > 0.6:
            return InsightSeverity.MEDIUM
        else:
            return InsightSeverity.LOW
    
    def _determine_anomaly_severity(self, max_z_score: float) -> InsightSeverity:
        """Determine severity of anomaly insight."""
        if max_z_score > 4:
            return InsightSeverity.CRITICAL
        elif max_z_score > 3:
            return InsightSeverity.HIGH
        elif max_z_score > 2.5:
            return InsightSeverity.MEDIUM
        else:
            return InsightSeverity.LOW
    
    def _severity_weight(self, severity: InsightSeverity) -> int:
        """Convert severity to numeric weight for sorting."""
        weights = {
            InsightSeverity.CRITICAL: 5,
            InsightSeverity.HIGH: 4,
            InsightSeverity.MEDIUM: 3,
            InsightSeverity.LOW: 2,
            InsightSeverity.INFO: 1
        }
        return weights.get(severity, 1)
    
    def _generate_trend_recommendations(self, metric: str, direction: str, growth_rate: float) -> List[str]:
        """Generate recommendations for trend insights."""
        recommendations = []
        
        if direction == "increasing" and growth_rate > 20:
            recommendations.append(f"Monitor {metric} closely as rapid growth may indicate opportunities or risks")
            recommendations.append(f"Consider scaling resources to support continued {metric} growth")
        elif direction == "decreasing" and growth_rate < -10:
            recommendations.append(f"Investigate root causes of declining {metric}")
            recommendations.append(f"Implement corrective measures to reverse negative {metric} trend")
        elif direction == "flat":
            recommendations.append(f"Analyze factors that could stimulate {metric} growth")
        
        return recommendations
    
    def _generate_anomaly_recommendations(self, metric: str, anomaly_count: int, max_z_score: float) -> List[str]:
        """Generate recommendations for anomaly insights."""
        recommendations = []
        
        if max_z_score > 3:
            recommendations.append(f"Immediate investigation required for extreme {metric} values")
            recommendations.append("Check data quality and collection processes")
        
        if anomaly_count > 1:
            recommendations.append(f"Review multiple anomalous {metric} values for patterns")
        
        recommendations.append(f"Consider setting up alerts for {metric} anomalies")
        
        return recommendations
    
    def _generate_correlation_recommendations(self, col1: str, col2: str, correlation: float, strength: str) -> List[str]:
        """Generate recommendations for correlation insights."""
        recommendations = []
        
        if strength in ["strong", "moderate"]:
            if correlation > 0:
                recommendations.append(f"Leverage positive relationship between {col1} and {col2}")
                recommendations.append(f"Consider using {col1} as a leading indicator for {col2}")
            else:
                recommendations.append(f"Monitor inverse relationship between {col1} and {col2}")
                recommendations.append(f"Balance improvements in {col1} with potential impacts on {col2}")
        
        return recommendations
    
    def _generate_performance_recommendations(self, metric: str, value: float, threshold: float, meets_threshold: bool) -> List[str]:
        """Generate recommendations for performance insights."""
        recommendations = []
        
        if not meets_threshold:
            gap = (threshold - value) / threshold * 100
            recommendations.append(f"Improve {metric} by {gap:.1f}% to meet threshold")
            recommendations.append(f"Analyze factors contributing to {metric} underperformance")
            recommendations.append(f"Set action plan to achieve {metric} targets")
        else:
            recommendations.append(f"Maintain current {metric} performance levels")
            recommendations.append(f"Consider setting higher targets for {metric}")
        
        return recommendations
    
    def _generate_summary(self, insights: List[BusinessInsight]) -> str:
        """Generate a summary of all insights."""
        if not insights:
            return "No significant insights found in the data."
        
        # Count insights by severity and type
        severity_counts = {}
        type_counts = {}
        
        for insight in insights:
            severity_counts[insight.severity.value] = severity_counts.get(insight.severity.value, 0) + 1
            type_counts[insight.insight_type.value] = type_counts.get(insight.insight_type.value, 0) + 1
        
        summary_parts = [f"Found {len(insights)} insights"]
        
        if severity_counts:
            severity_summary = ", ".join([f"{count} {sev}" for sev, count in severity_counts.items()])
            summary_parts.append(f"Severity distribution: {severity_summary}")
        
        if type_counts:
            type_summary = ", ".join([f"{count} {typ}" for typ, count in sorted(type_counts.items())])
            summary_parts.append(f"Analysis types: {type_summary}")
        
        return "; ".join(summary_parts)
    
    def _extract_key_findings(self, insights: List[BusinessInsight]) -> List[str]:
        """Extract key findings from insights."""
        key_findings = []
        
        # Get top insights by severity and confidence
        top_insights = sorted(
            insights,
            key=lambda x: (self._severity_weight(x.severity), x.confidence),
            reverse=True
        )[:5]
        
        for insight in top_insights:
            key_findings.append(insight.description)
        
        return key_findings
    
    def _generate_recommendations(self, insights: List[BusinessInsight], request: InsightExtractionRequest) -> List[str]:
        """Generate overall recommendations based on all insights."""
        all_recommendations = []
        
        # Collect all individual recommendations
        for insight in insights:
            all_recommendations.extend(insight.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        # Return top recommendations
        return unique_recommendations[:10]
    
    def _calculate_quality_score(self, insights: List[BusinessInsight], df: pd.DataFrame) -> float:
        """Calculate overall quality score for the analysis."""
        if not insights:
            return 0.0
        
        # Factors affecting quality score
        avg_confidence = sum(insight.confidence for insight in insights) / len(insights)
        data_completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        insight_diversity = len(set(insight.insight_type for insight in insights)) / len(InsightType)
        
        # Weighted quality score
        quality_score = (
            avg_confidence * 0.4 +
            data_completeness * 0.3 +
            insight_diversity * 0.3
        )
        
        return min(quality_score, 1.0)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the insight extractor."""
        return {
            'status': 'healthy',
            'business_rules_loaded': bool(self.business_rules),
            'kpi_mappings_loaded': bool(self.kpi_mappings),
            'config_loaded': bool(self.config)
        } 