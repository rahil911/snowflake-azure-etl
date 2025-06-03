"""
Result Processor for Data Intelligence Agent

This module processes and formats SQL query results with business context,
transforming raw data into meaningful insights for business users.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from pydantic import BaseModel, Field
from shared.config.settings import Settings
from shared.utils.caching import get_cache_manager
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.validation import validate_input, ValidationError

logger = logging.getLogger(__name__)

class ResultFormat(Enum):
    """Output format types for processed results."""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    BUSINESS_NARRATIVE = "business_narrative"
    TABULAR = "tabular"
    CHART_DATA = "chart_data"
    KPI_DASHBOARD = "kpi_dashboard"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"

class BusinessContext(Enum):
    """Business context types for result interpretation."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    CUSTOMER = "customer"
    PRODUCT = "product"
    SALES = "sales"
    MARKETING = "marketing"
    QUALITY = "quality"
    PERFORMANCE = "performance"

@dataclass
class ProcessedResult:
    """Processed query result with business interpretation."""
    raw_data: pd.DataFrame
    formatted_data: Dict[str, Any]
    business_summary: str
    key_insights: List[str]
    recommendations: List[str] = field(default_factory=list)
    charts_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class ResultProcessingRequest(BaseModel):
    """Request for result processing."""
    query_result: Dict[str, Any]
    query_context: Dict[str, Any]
    business_context: BusinessContext
    output_format: ResultFormat = ResultFormat.BUSINESS_NARRATIVE
    include_recommendations: bool = True
    include_charts: bool = True
    executive_level: bool = False
    max_insights: int = 5
    
    class Config:
        validate_assignment = True

class ResultProcessingResponse(BaseModel):
    """Response from result processing."""
    processed_result: Dict[str, Any]
    summary: str
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    charts: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True

class ResultProcessor:
    """
    Result Processor for transforming SQL results into business insights.
    
    Processes raw query results and formats them with business context
    for various stakeholder audiences.
    """
    
    def __init__(self, settings: Settings):
        """Initialize result processor."""
        self.settings = settings
        self.cache = get_cache_manager()
        self.metrics = get_metrics_collector()
        
        # Configuration
        self.config = {
            'max_rows_summary': settings.RESULT_PROCESSOR.get('max_rows_summary', 1000),
            'max_insights': settings.RESULT_PROCESSOR.get('max_insights', 10),
            'enable_charts': settings.RESULT_PROCESSOR.get('enable_charts', True),
            'cache_ttl': settings.RESULT_PROCESSOR.get('cache_ttl', 1800),
            'executive_threshold': settings.RESULT_PROCESSOR.get('executive_threshold', 100)
        }
        
        # Business formatting templates
        self._setup_formatting_templates()
        self._setup_business_mappings()
        
        logger.info("Result Processor initialized")

    def _setup_formatting_templates(self):
        """Setup formatting templates for different output types."""
        self.templates = {
            'executive_summary': {
                'structure': ['headline', 'key_metrics', 'critical_insights', 'recommendations'],
                'max_length': 500,
                'tone': 'executive'
            },
            'detailed_analysis': {
                'structure': ['overview', 'detailed_findings', 'data_breakdown', 'methodology', 'appendix'],
                'max_length': 2000,
                'tone': 'analytical'
            },
            'business_narrative': {
                'structure': ['context', 'findings', 'implications', 'next_steps'],
                'max_length': 1000,
                'tone': 'business'
            },
            'kpi_dashboard': {
                'structure': ['current_performance', 'trends', 'comparisons', 'alerts'],
                'max_length': 300,
                'tone': 'metric_focused'
            }
        }

    def _setup_business_mappings(self):
        """Setup business context mappings and interpretations."""
        self.business_mappings = {
            'financial': {
                'key_metrics': ['revenue', 'profit', 'cost', 'margin', 'growth'],
                'terminology': {
                    'sales_amount': 'Revenue',
                    'quantity_sold': 'Units Sold',
                    'avg_price': 'Average Selling Price'
                },
                'thresholds': {
                    'growth_good': 0.05,
                    'growth_excellent': 0.15,
                    'margin_good': 0.2,
                    'margin_excellent': 0.3
                }
            },
            'operational': {
                'key_metrics': ['efficiency', 'quality', 'throughput', 'utilization'],
                'terminology': {
                    'processing_time': 'Cycle Time',
                    'error_rate': 'Quality Score',
                    'capacity_used': 'Utilization Rate'
                },
                'thresholds': {
                    'efficiency_good': 0.8,
                    'quality_good': 0.95,
                    'utilization_optimal': 0.85
                }
            },
            'customer': {
                'key_metrics': ['satisfaction', 'retention', 'acquisition', 'lifetime_value'],
                'terminology': {
                    'customer_count': 'Customer Base',
                    'repeat_purchases': 'Customer Retention',
                    'avg_order_value': 'Average Order Value'
                },
                'thresholds': {
                    'retention_good': 0.8,
                    'satisfaction_good': 4.0,
                    'nps_good': 50
                }
            },
            'sales': {
                'key_metrics': ['volume', 'value', 'conversion', 'pipeline'],
                'terminology': {
                    'total_sales': 'Sales Volume',
                    'sales_amount': 'Sales Value',
                    'unique_customers': 'Customer Reach'
                },
                'thresholds': {
                    'conversion_good': 0.15,
                    'growth_good': 0.1,
                    'pipeline_healthy': 3.0
                }
            }
        }

    @track_performance(tags={"operation": "process_result"})
    async def process_result(self, request: ResultProcessingRequest) -> ResultProcessingResponse:
        """
        Process SQL query result into business-friendly format.
        
        Args:
            request: Processing request with query result and context
            
        Returns:
            Processed result with business interpretation
        """
        start_time = datetime.now()
        
        try:
            # Validate input
            if not request.query_result or 'data' not in request.query_result:
                raise ValidationError("Query result must contain 'data' field")
            
            # Convert to DataFrame
            df = self._prepare_dataframe(request.query_result['data'])
            if df.empty:
                return self._create_empty_response("No data found for the specified criteria")
            
            # Apply business context processing
            processed_data = await self._apply_business_context(df, request)
            
            # Generate insights
            insights = await self._generate_insights(df, request)
            
            # Create summary
            summary = await self._generate_summary(df, processed_data, request)
            
            # Generate recommendations
            recommendations = []
            if request.include_recommendations:
                recommendations = await self._generate_recommendations(df, insights, request)
            
            # Create charts data
            charts = {}
            if request.include_charts:
                charts = await self._generate_charts_data(df, request)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            await self._record_processing_metrics(request, df, processing_time)
            
            return ResultProcessingResponse(
                processed_result={
                    'data': processed_data,
                    'row_count': len(df),
                    'columns': list(df.columns),
                    'data_types': df.dtypes.to_dict()
                },
                summary=summary,
                insights=insights[:request.max_insights],
                recommendations=recommendations,
                charts=charts,
                metadata={
                    'processing_time': processing_time,
                    'business_context': request.business_context.value,
                    'output_format': request.output_format.value,
                    'row_count': len(df),
                    'column_count': len(df.columns)
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing result: {str(e)}")
            return self._create_error_response(str(e))

    def _prepare_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert query result data to DataFrame."""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {str(e)}")
            return pd.DataFrame()

    async def _apply_business_context(self, df: pd.DataFrame, request: ResultProcessingRequest) -> Dict[str, Any]:
        """Apply business context to raw data."""
        context = request.business_context.value
        mapping = self.business_mappings.get(context, {})
        
        # Rename columns using business terminology
        terminology = mapping.get('terminology', {})
        display_names = {}
        for col in df.columns:
            display_names[col] = terminology.get(col.lower(), col.replace('_', ' ').title())
        
        # Calculate business metrics
        business_metrics = self._calculate_business_metrics(df, mapping)
        
        # Format data for display
        formatted_data = self._format_data_for_display(df, request.output_format)
        
        return {
            'display_data': formatted_data,
            'column_mappings': display_names,
            'business_metrics': business_metrics,
            'context_applied': context
        }

    def _calculate_business_metrics(self, df: pd.DataFrame, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate context-specific business metrics."""
        metrics = {}
        
        try:
            # Numeric columns for calculations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if not numeric_cols.empty:
                # Basic statistics
                metrics['totals'] = df[numeric_cols].sum().to_dict()
                metrics['averages'] = df[numeric_cols].mean().to_dict()
                metrics['ranges'] = {
                    col: {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'std': df[col].std()
                    } for col in numeric_cols
                }
                
                # Growth calculations if date column exists
                date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
                for date_col in date_cols:
                    if 'date' in date_col.lower() or 'time' in date_col.lower():
                        try:
                            df_sorted = df.sort_values(date_col)
                            for num_col in numeric_cols:
                                first_val = df_sorted[num_col].iloc[0]
                                last_val = df_sorted[num_col].iloc[-1]
                                if first_val != 0:
                                    growth = (last_val - first_val) / first_val
                                    metrics[f'{num_col}_growth'] = growth
                        except:
                            continue
                        break
                
        except Exception as e:
            logger.warning(f"Error calculating business metrics: {str(e)}")
        
        return metrics

    def _format_data_for_display(self, df: pd.DataFrame, format_type: ResultFormat) -> List[Dict[str, Any]]:
        """Format data based on requested format type."""
        try:
            if format_type == ResultFormat.EXECUTIVE_SUMMARY:
                # Show only top-level aggregates
                return self._create_executive_format(df)
            elif format_type == ResultFormat.KPI_DASHBOARD:
                # Show key metrics in dashboard format
                return self._create_kpi_format(df)
            elif format_type == ResultFormat.TABULAR:
                # Show full tabular data
                return df.head(100).to_dict('records')
            else:
                # Default business narrative format
                return df.head(50).to_dict('records')
        except Exception as e:
            logger.error(f"Error formatting data: {str(e)}")
            return df.head(10).to_dict('records')

    def _create_executive_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create executive summary format."""
        summary = []
        
        # Aggregate numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            totals = df[numeric_cols].sum()
            summary.append({
                'metric_type': 'Totals',
                **totals.to_dict()
            })
            
            averages = df[numeric_cols].mean()
            summary.append({
                'metric_type': 'Averages',
                **averages.to_dict()
            })
        
        return summary

    def _create_kpi_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create KPI dashboard format."""
        kpis = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            kpi = {
                'metric_name': col.replace('_', ' ').title(),
                'value': df[col].sum(),
                'average': df[col].mean(),
                'trend': 'stable'  # Would calculate actual trend with time series
            }
            kpis.append(kpi)
        
        return kpis

    async def _generate_insights(self, df: pd.DataFrame, request: ResultProcessingRequest) -> List[str]:
        """Generate business insights from the data."""
        insights = []
        
        try:
            # Row count insight
            row_count = len(df)
            if row_count > 1000:
                insights.append(f"Large dataset with {row_count:,} records suggests comprehensive coverage")
            elif row_count < 10:
                insights.append(f"Limited data with only {row_count} records - consider expanding criteria")
            
            # Numeric insights
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Variance insight
                    cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                    if cv > 1:
                        insights.append(f"{col.replace('_', ' ').title()} shows high variability (CV: {cv:.2f})")
                    elif cv < 0.1:
                        insights.append(f"{col.replace('_', ' ').title()} shows consistent values (CV: {cv:.2f})")
                    
                    # Outlier insight
                    q75, q25 = np.percentile(col_data, [75, 25])
                    iqr = q75 - q25
                    outliers = col_data[(col_data < q25 - 1.5 * iqr) | (col_data > q75 + 1.5 * iqr)]
                    if len(outliers) > 0:
                        insights.append(f"{col.replace('_', ' ').title()} has {len(outliers)} outlier values requiring attention")
            
            # Categorical insights
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                if unique_count / total_count < 0.1:
                    insights.append(f"{col.replace('_', ' ').title()} shows high concentration in few categories")
                elif unique_count / total_count > 0.8:
                    insights.append(f"{col.replace('_', ' ').title()} shows high diversity across categories")
            
            # Business context specific insights
            context_insights = self._generate_context_insights(df, request.business_context)
            insights.extend(context_insights)
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append("Data analysis completed successfully")
        
        return insights

    def _generate_context_insights(self, df: pd.DataFrame, context: BusinessContext) -> List[str]:
        """Generate context-specific business insights."""
        insights = []
        
        try:
            if context == BusinessContext.SALES:
                # Sales-specific insights
                if 'sales_amount' in df.columns:
                    total_sales = df['sales_amount'].sum()
                    avg_sale = df['sales_amount'].mean()
                    insights.append(f"Total sales volume: ${total_sales:,.2f} with average transaction of ${avg_sale:.2f}")
                
                if 'quantity_sold' in df.columns:
                    total_qty = df['quantity_sold'].sum()
                    insights.append(f"Total units sold: {total_qty:,}")
                
            elif context == BusinessContext.CUSTOMER:
                # Customer-specific insights
                if 'customer_id' in df.columns:
                    unique_customers = df['customer_id'].nunique()
                    insights.append(f"Analysis covers {unique_customers:,} unique customers")
                
            elif context == BusinessContext.FINANCIAL:
                # Financial-specific insights
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                revenue_cols = [col for col in numeric_cols if 'revenue' in col.lower() or 'sales' in col.lower()]
                for col in revenue_cols:
                    total = df[col].sum()
                    insights.append(f"Total {col.replace('_', ' ').title()}: ${total:,.2f}")
                    
        except Exception as e:
            logger.warning(f"Error generating context insights: {str(e)}")
        
        return insights

    async def _generate_summary(self, df: pd.DataFrame, processed_data: Dict[str, Any], request: ResultProcessingRequest) -> str:
        """Generate executive summary of results."""
        try:
            context = request.business_context.value
            row_count = len(df)
            col_count = len(df.columns)
            
            if request.executive_level:
                summary = f"Executive Summary: Analysis of {row_count:,} records across {col_count} metrics in {context} domain. "
            else:
                summary = f"Data Analysis: Processed {row_count:,} records with {col_count} data points for {context} analysis. "
            
            # Add key metric if available
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                primary_metric = numeric_cols[0]
                total_value = df[primary_metric].sum()
                summary += f"Primary metric ({primary_metric.replace('_', ' ').title()}): {total_value:,.2f}. "
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Analysis completed for {len(df)} records in {request.business_context.value} context."

    async def _generate_recommendations(self, df: pd.DataFrame, insights: List[str], request: ResultProcessingRequest) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        try:
            context = request.business_context.value
            
            # Data quality recommendations
            missing_data = df.isnull().sum().sum()
            if missing_data > 0:
                recommendations.append("Review data collection processes to address missing values")
            
            # Context-specific recommendations
            if context == 'sales':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                sales_cols = [col for col in numeric_cols if 'sales' in col.lower() or 'revenue' in col.lower()]
                for col in sales_cols:
                    growth_potential = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                    if growth_potential > 0.5:
                        recommendations.append(f"High variability in {col.replace('_', ' ')} suggests opportunity for performance optimization")
            
            elif context == 'customer':
                if 'customer_id' in df.columns:
                    unique_customers = df['customer_id'].nunique()
                    total_records = len(df)
                    avg_interactions = total_records / unique_customers if unique_customers > 0 else 0
                    if avg_interactions > 5:
                        recommendations.append("High customer engagement levels - consider loyalty program expansion")
                    elif avg_interactions < 2:
                        recommendations.append("Low customer engagement - implement retention strategies")
            
            # General recommendations based on data characteristics
            if len(df) > 10000:
                recommendations.append("Consider implementing data sampling for faster analysis iterations")
            
            if len(recommendations) == 0:
                recommendations.append("Continue monitoring these metrics for ongoing business optimization")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Review analysis results for actionable insights")
        
        return recommendations

    async def _generate_charts_data(self, df: pd.DataFrame, request: ResultProcessingRequest) -> Dict[str, Any]:
        """Generate data structures for chart creation."""
        charts = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Bar chart data for categorical vs numeric
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                chart_data = df.groupby(cat_col)[num_col].sum().reset_index()
                charts['bar_chart'] = {
                    'type': 'bar',
                    'data': chart_data.to_dict('records'),
                    'x_axis': cat_col,
                    'y_axis': num_col,
                    'title': f"{num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}"
                }
            
            # Line chart data for time series
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols and len(numeric_cols) > 0:
                date_col = date_cols[0]
                num_col = numeric_cols[0]
                # Try to convert to datetime
                try:
                    df_chart = df.copy()
                    df_chart[date_col] = pd.to_datetime(df_chart[date_col])
                    chart_data = df_chart.groupby(df_chart[date_col].dt.date)[num_col].sum().reset_index()
                    charts['line_chart'] = {
                        'type': 'line',
                        'data': chart_data.to_dict('records'),
                        'x_axis': date_col,
                        'y_axis': num_col,
                        'title': f"{num_col.replace('_', ' ').title()} Trend Over Time"
                    }
                except:
                    pass
            
            # Pie chart for categorical distribution
            if len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                value_counts = df[cat_col].value_counts().head(10)
                charts['pie_chart'] = {
                    'type': 'pie',
                    'data': [{'label': idx, 'value': val} for idx, val in value_counts.items()],
                    'title': f"Distribution of {cat_col.replace('_', ' ').title()}"
                }
                
        except Exception as e:
            logger.error(f"Error generating chart data: {str(e)}")
        
        return charts

    def _create_empty_response(self, message: str) -> ResultProcessingResponse:
        """Create response for empty results."""
        return ResultProcessingResponse(
            processed_result={'data': [], 'row_count': 0, 'columns': []},
            summary=message,
            insights=[message],
            recommendations=["Adjust query criteria to retrieve relevant data"],
            charts={},
            metadata={'empty_result': True}
        )

    def _create_error_response(self, error_message: str) -> ResultProcessingResponse:
        """Create response for processing errors."""
        return ResultProcessingResponse(
            processed_result={'data': [], 'row_count': 0, 'columns': []},
            summary=f"Processing error: {error_message}",
            insights=[],
            recommendations=["Review data format and processing requirements"],
            charts={},
            metadata={'error': True, 'error_message': error_message}
        )

    async def _record_processing_metrics(self, request: ResultProcessingRequest, df: pd.DataFrame, processing_time: float):
        """Record processing metrics."""
        try:
            metrics_data = {
                'business_context': request.business_context.value,
                'output_format': request.output_format.value,
                'row_count': len(df),
                'column_count': len(df.columns),
                'processing_time': processing_time,
                'executive_level': request.executive_level,
                'include_charts': request.include_charts,
                'include_recommendations': request.include_recommendations
            }
            
            await self.metrics.record_event('result_processing', metrics_data)
        except Exception as e:
            logger.warning(f"Error recording metrics: {str(e)}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of result processor."""
        return {
            'service': 'result_processor',
            'status': 'healthy',
            'cache_enabled': self.cache is not None,
            'metrics_enabled': self.metrics is not None,
            'config': {
                'max_rows_summary': self.config['max_rows_summary'],
                'max_insights': self.config['max_insights'],
                'enable_charts': self.config['enable_charts']
            },
            'timestamp': datetime.now().isoformat()
        } 