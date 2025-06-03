"""
Pattern Detector for Data Intelligence Agent

This module identifies and analyzes patterns in business data including:
- Seasonal patterns and cyclical trends
- Recurring behaviors and periodic patterns
- Data distribution patterns
- Time-based patterns and anomalies
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
from collections import defaultdict

from pydantic import BaseModel, Field, validator
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

from shared.schemas.data_models import QueryResult, BusinessEntity, DataQualityMetric
from shared.schemas.data_models import AnalysisResult
from shared.utils.caching import get_cache_manager
from shared.utils.validation import ValidationHelper
from shared.utils.retry import RetryStrategy, retry_on_exception
from shared.config.settings import Settings


logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns that can be detected."""
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    TRENDING = "trending"
    PERIODIC = "periodic"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    OUTLIER = "outlier"
    CLUSTERING = "clustering"


class PatternSeverity(Enum):
    """Severity levels for pattern significance."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SeasonalType(Enum):
    """Types of seasonal patterns."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class DetectedPattern:
    """Individual detected pattern with metadata."""
    pattern_type: PatternType
    severity: PatternSeverity
    confidence: float
    title: str
    description: str
    pattern_data: Dict[str, Any]
    statistical_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_type": self.pattern_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "pattern_data": self.pattern_data,
            "statistical_metrics": self.statistical_metrics,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class SeasonalPattern(BaseModel):
    """Seasonal pattern analysis results."""
    seasonal_type: SeasonalType
    strength: float  # 0-1
    period_length: int
    peak_periods: List[str] = Field(default_factory=list)
    low_periods: List[str] = Field(default_factory=list)
    seasonal_decomposition: Optional[Dict[str, Any]] = None
    
    class Config:
        validate_assignment = True


class CyclicalPattern(BaseModel):
    """Cyclical pattern analysis results."""
    cycle_length: float
    amplitude: float
    frequency: float
    phase: float
    regularity_score: float  # 0-1
    
    class Config:
        validate_assignment = True


class DistributionPattern(BaseModel):
    """Data distribution pattern analysis."""
    distribution_type: str  # normal, skewed, bimodal, etc.
    skewness: float
    kurtosis: float
    outlier_percentage: float
    distribution_params: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True


class PatternDetectionRequest(BaseModel):
    """Request for pattern detection."""
    data: Dict[str, Any]
    pattern_types: List[PatternType] = Field(default_factory=lambda: list(PatternType))
    time_column: Optional[str] = None
    value_columns: List[str] = Field(default_factory=list)
    min_confidence: float = 0.6
    include_statistical_details: bool = True
    
    class Config:
        validate_assignment = True


class PatternDetectionResponse(BaseModel):
    """Response from pattern detection."""
    patterns: List[Dict[str, Any]]
    summary: str
    pattern_insights: List[str] = Field(default_factory=list)
    statistical_summary: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True


class PatternDetector:
    """
    Advanced pattern detection and analysis for business intelligence data.
    
    This class handles:
    - Seasonal and cyclical pattern detection
    - Time series pattern analysis
    - Data distribution pattern identification
    - Correlation pattern discovery
    - Outlier and anomaly pattern detection
    - Business pattern recognition
    """
    
    def __init__(self, settings: Settings):
        """Initialize the pattern detector."""
        self.settings = settings
        self.cache = get_cache_manager()
        self.validator = ValidationHelper()
        self.retry_strategy = RetryStrategy.EXPONENTIAL
        
        # Detection configuration
        self.config = {
            'min_data_points': 10,
            'seasonal_min_periods': 2,
            'outlier_threshold': 2.5,  # Standard deviations
            'correlation_threshold': 0.5,
            'peak_prominence': 0.1,
            'cycle_min_length': 3
        }
        
        # Pattern thresholds and rules
        self._setup_pattern_rules()
        
        logger.info("Pattern detector initialized")
    
    def _setup_pattern_rules(self):
        """Setup pattern detection rules and thresholds."""
        self.pattern_rules = {
            'seasonal': {
                'strong_threshold': 0.7,
                'moderate_threshold': 0.5,
                'weak_threshold': 0.3
            },
            'cyclical': {
                'regular_threshold': 0.8,
                'moderate_threshold': 0.6,
                'irregular_threshold': 0.4
            },
            'distribution': {
                'normal_skew_threshold': 0.5,
                'high_skew_threshold': 1.0,
                'outlier_concern_threshold': 0.05  # 5%
            },
            'correlation': {
                'strong_threshold': 0.7,
                'moderate_threshold': 0.5,
                'weak_threshold': 0.3
            }
        }
    
    async def detect_patterns(self, request: PatternDetectionRequest) -> PatternDetectionResponse:
        """
        Detect comprehensive patterns in business data.
        
        Args:
            request: Pattern detection request with data and parameters
            
        Returns:
            Detected patterns with analysis and recommendations
        """
        try:
            start_time = datetime.now()
            logger.info("Starting pattern detection")
            
            # Prepare data for analysis
            df = self._prepare_dataframe(request.data)
            
            if df.empty:
                return PatternDetectionResponse(
                    patterns=[],
                    summary="No data available for pattern analysis",
                    analysis_metadata={'error': 'empty_dataset'}
                )
            
            # Detect different types of patterns
            patterns = []
            
            for pattern_type in request.pattern_types:
                if pattern_type == PatternType.SEASONAL:
                    seasonal_patterns = await self._detect_seasonal_patterns(df, request)
                    patterns.extend(seasonal_patterns)
                
                elif pattern_type == PatternType.CYCLICAL:
                    cyclical_patterns = await self._detect_cyclical_patterns(df, request)
                    patterns.extend(cyclical_patterns)
                
                elif pattern_type == PatternType.TRENDING:
                    trending_patterns = await self._detect_trending_patterns(df, request)
                    patterns.extend(trending_patterns)
                
                elif pattern_type == PatternType.PERIODIC:
                    periodic_patterns = await self._detect_periodic_patterns(df, request)
                    patterns.extend(periodic_patterns)
                
                elif pattern_type == PatternType.DISTRIBUTION:
                    distribution_patterns = await self._detect_distribution_patterns(df, request)
                    patterns.extend(distribution_patterns)
                
                elif pattern_type == PatternType.CORRELATION:
                    correlation_patterns = await self._detect_correlation_patterns(df, request)
                    patterns.extend(correlation_patterns)
                
                elif pattern_type == PatternType.OUTLIER:
                    outlier_patterns = await self._detect_outlier_patterns(df, request)
                    patterns.extend(outlier_patterns)
                
                elif pattern_type == PatternType.CLUSTERING:
                    clustering_patterns = await self._detect_clustering_patterns(df, request)
                    patterns.extend(clustering_patterns)
            
            # Filter by confidence threshold
            filtered_patterns = [
                pattern for pattern in patterns
                if pattern.confidence >= request.min_confidence
            ]
            
            # Sort patterns by significance
            filtered_patterns.sort(
                key=lambda x: (self._severity_weight(x.severity), x.confidence),
                reverse=True
            )
            
            # Generate analysis results
            summary = self._generate_summary(filtered_patterns)
            pattern_insights = self._extract_pattern_insights(filtered_patterns)
            statistical_summary = self._generate_statistical_summary(df, filtered_patterns)
            recommendations = self._generate_recommendations(filtered_patterns)
            
            response = PatternDetectionResponse(
                patterns=[pattern.to_dict() for pattern in filtered_patterns],
                summary=summary,
                pattern_insights=pattern_insights,
                statistical_summary=statistical_summary,
                recommendations=recommendations,
                analysis_metadata={
                    'total_patterns': len(filtered_patterns),
                    'data_points': len(df),
                    'columns_analyzed': len(df.columns),
                    'analysis_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'pattern_types_analyzed': [pt.value for pt in request.pattern_types]
                }
            )
            
            logger.info(f"Pattern detection completed with {len(filtered_patterns)} patterns")
            return response
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {str(e)}")
            raise
    
    def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert input data to pandas DataFrame."""
        try:
            if isinstance(data, dict):
                if 'rows' in data and 'columns' in data:
                    return pd.DataFrame(data['rows'], columns=data['columns'])
                elif 'data' in data:
                    return pd.DataFrame(data['data'])
                else:
                    return pd.DataFrame(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {str(e)}")
            return pd.DataFrame()
    
    async def _detect_seasonal_patterns(self, df: pd.DataFrame, request: PatternDetectionRequest) -> List[DetectedPattern]:
        """Detect seasonal patterns in time series data."""
        patterns = []
        
        try:
            # Find time and value columns
            time_col = request.time_column or self._find_time_column(df)
            value_cols = request.value_columns or self._find_numeric_columns(df)
            
            if not time_col or not value_cols:
                return patterns
            
            # Convert time column to datetime
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col)
            
            for value_col in value_cols:
                seasonal_pattern = await self._analyze_seasonality(df, time_col, value_col)
                if seasonal_pattern:
                    patterns.append(seasonal_pattern)
        
        except Exception as e:
            logger.error(f"Error detecting seasonal patterns: {str(e)}")
        
        return patterns
    
    async def _analyze_seasonality(self, df: pd.DataFrame, time_col: str, value_col: str) -> Optional[DetectedPattern]:
        """Analyze seasonality for a specific time series."""
        try:
            values = df[value_col].dropna()
            
            if len(values) < self.config['min_data_points']:
                return None
            
            # Detect different seasonal patterns
            seasonal_results = {}
            
            # Daily seasonality (if hourly data)
            if self._has_intraday_data(df[time_col]):
                daily_strength = self._detect_daily_seasonality(df, time_col, value_col)
                if daily_strength > self.pattern_rules['seasonal']['weak_threshold']:
                    seasonal_results['daily'] = daily_strength
            
            # Weekly seasonality
            weekly_strength = self._detect_weekly_seasonality(df, time_col, value_col)
            if weekly_strength > self.pattern_rules['seasonal']['weak_threshold']:
                seasonal_results['weekly'] = weekly_strength
            
            # Monthly seasonality
            monthly_strength = self._detect_monthly_seasonality(df, time_col, value_col)
            if monthly_strength > self.pattern_rules['seasonal']['weak_threshold']:
                seasonal_results['monthly'] = monthly_strength
            
            # Quarterly seasonality
            quarterly_strength = self._detect_quarterly_seasonality(df, time_col, value_col)
            if quarterly_strength > self.pattern_rules['seasonal']['weak_threshold']:
                seasonal_results['quarterly'] = quarterly_strength
            
            if not seasonal_results:
                return None
            
            # Find strongest seasonal pattern
            strongest_pattern = max(seasonal_results.items(), key=lambda x: x[1])
            pattern_type, strength = strongest_pattern
            
            # Determine severity and confidence
            severity = self._determine_seasonal_severity(strength)
            confidence = min(strength + 0.1, 1.0)
            
            # Generate pattern details
            pattern_data = {
                'seasonal_type': pattern_type,
                'strength': strength,
                'all_seasonal_strengths': seasonal_results,
                'peak_periods': self._find_seasonal_peaks(df, time_col, value_col, pattern_type),
                'low_periods': self._find_seasonal_lows(df, time_col, value_col, pattern_type)
            }
            
            pattern = DetectedPattern(
                pattern_type=PatternType.SEASONAL,
                severity=severity,
                confidence=confidence,
                title=f"{pattern_type.title()} Seasonal Pattern in {value_col}",
                description=f"Strong {pattern_type} seasonal pattern detected with {strength:.1%} strength",
                pattern_data=pattern_data,
                statistical_metrics={
                    'seasonal_strength': strength,
                    'data_points': len(values),
                    'time_span_days': (df[time_col].max() - df[time_col].min()).days
                },
                recommendations=self._generate_seasonal_recommendations(pattern_type, strength, value_col)
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {str(e)}")
            return None
    
    def _has_intraday_data(self, time_series: pd.Series) -> bool:
        """Check if data has intraday granularity."""
        try:
            time_diff = time_series.diff().dropna()
            min_diff = time_diff.min()
            return min_diff < timedelta(days=1)
        except:
            return False
    
    def _detect_daily_seasonality(self, df: pd.DataFrame, time_col: str, value_col: str) -> float:
        """Detect daily seasonal patterns."""
        try:
            df['hour'] = df[time_col].dt.hour
            hourly_means = df.groupby('hour')[value_col].mean()
            
            if len(hourly_means) < 12:  # Need at least half day coverage
                return 0.0
            
            # Calculate coefficient of variation as seasonality strength
            cv = hourly_means.std() / hourly_means.mean() if hourly_means.mean() != 0 else 0
            return min(cv, 1.0)
            
        except:
            return 0.0
    
    def _detect_weekly_seasonality(self, df: pd.DataFrame, time_col: str, value_col: str) -> float:
        """Detect weekly seasonal patterns."""
        try:
            df['day_of_week'] = df[time_col].dt.dayofweek
            weekly_means = df.groupby('day_of_week')[value_col].mean()
            
            if len(weekly_means) < 5:  # Need most days of week
                return 0.0
            
            cv = weekly_means.std() / weekly_means.mean() if weekly_means.mean() != 0 else 0
            return min(cv, 1.0)
            
        except:
            return 0.0
    
    def _detect_monthly_seasonality(self, df: pd.DataFrame, time_col: str, value_col: str) -> float:
        """Detect monthly seasonal patterns."""
        try:
            df['month'] = df[time_col].dt.month
            monthly_means = df.groupby('month')[value_col].mean()
            
            if len(monthly_means) < 6:  # Need at least half year
                return 0.0
            
            cv = monthly_means.std() / monthly_means.mean() if monthly_means.mean() != 0 else 0
            return min(cv, 1.0)
            
        except:
            return 0.0
    
    def _detect_quarterly_seasonality(self, df: pd.DataFrame, time_col: str, value_col: str) -> float:
        """Detect quarterly seasonal patterns."""
        try:
            df['quarter'] = df[time_col].dt.quarter
            quarterly_means = df.groupby('quarter')[value_col].mean()
            
            if len(quarterly_means) < 3:  # Need at least 3 quarters
                return 0.0
            
            cv = quarterly_means.std() / quarterly_means.mean() if quarterly_means.mean() != 0 else 0
            return min(cv, 1.0)
            
        except:
            return 0.0
    
    def _find_seasonal_peaks(self, df: pd.DataFrame, time_col: str, value_col: str, pattern_type: str) -> List[str]:
        """Find peak periods for seasonal patterns."""
        try:
            if pattern_type == 'daily':
                grouped = df.groupby(df[time_col].dt.hour)[value_col].mean()
                time_label = 'hour'
            elif pattern_type == 'weekly':
                grouped = df.groupby(df[time_col].dt.dayofweek)[value_col].mean()
                time_label = 'day_of_week'
            elif pattern_type == 'monthly':
                grouped = df.groupby(df[time_col].dt.month)[value_col].mean()
                time_label = 'month'
            elif pattern_type == 'quarterly':
                grouped = df.groupby(df[time_col].dt.quarter)[value_col].mean()
                time_label = 'quarter'
            else:
                return []
            
            # Find peaks
            values = grouped.values
            peaks, _ = find_peaks(values, prominence=self.config['peak_prominence'] * values.std())
            
            peak_periods = []
            for peak in peaks:
                period_key = grouped.index[peak]
                peak_periods.append(f"{time_label}_{period_key}")
            
            return peak_periods
            
        except:
            return []
    
    def _find_seasonal_lows(self, df: pd.DataFrame, time_col: str, value_col: str, pattern_type: str) -> List[str]:
        """Find low periods for seasonal patterns."""
        try:
            if pattern_type == 'daily':
                grouped = df.groupby(df[time_col].dt.hour)[value_col].mean()
                time_label = 'hour'
            elif pattern_type == 'weekly':
                grouped = df.groupby(df[time_col].dt.dayofweek)[value_col].mean()
                time_label = 'day_of_week'
            elif pattern_type == 'monthly':
                grouped = df.groupby(df[time_col].dt.month)[value_col].mean()
                time_label = 'month'
            elif pattern_type == 'quarterly':
                grouped = df.groupby(df[time_col].dt.quarter)[value_col].mean()
                time_label = 'quarter'
            else:
                return []
            
            # Find troughs (inverted peaks)
            values = -grouped.values
            peaks, _ = find_peaks(values, prominence=self.config['peak_prominence'] * values.std())
            
            low_periods = []
            for peak in peaks:
                period_key = grouped.index[peak]
                low_periods.append(f"{time_label}_{period_key}")
            
            return low_periods
            
        except:
            return []
    
    async def _detect_cyclical_patterns(self, df: pd.DataFrame, request: PatternDetectionRequest) -> List[DetectedPattern]:
        """Detect cyclical patterns using frequency analysis."""
        patterns = []
        
        try:
            value_cols = request.value_columns or self._find_numeric_columns(df)
            
            for value_col in value_cols:
                cyclical_pattern = await self._analyze_cycles(df, value_col)
                if cyclical_pattern:
                    patterns.append(cyclical_pattern)
        
        except Exception as e:
            logger.error(f"Error detecting cyclical patterns: {str(e)}")
        
        return patterns
    
    async def _analyze_cycles(self, df: pd.DataFrame, value_col: str) -> Optional[DetectedPattern]:
        """Analyze cyclical patterns using FFT."""
        try:
            values = df[value_col].dropna().values
            
            if len(values) < self.config['min_data_points']:
                return None
            
            # Perform FFT to find dominant frequencies
            fft_values = fft(values)
            freqs = fftfreq(len(values))
            
            # Find dominant frequency (excluding DC component)
            magnitude = np.abs(fft_values[1:len(fft_values)//2])
            frequencies = freqs[1:len(freqs)//2]
            
            if len(magnitude) == 0:
                return None
            
            dominant_freq_idx = np.argmax(magnitude)
            dominant_freq = frequencies[dominant_freq_idx]
            
            if dominant_freq == 0:
                return None
            
            # Calculate cycle characteristics
            cycle_length = 1 / abs(dominant_freq)
            amplitude = magnitude[dominant_freq_idx]
            
            # Calculate regularity score
            regularity_score = amplitude / np.sum(magnitude)
            
            if regularity_score < self.pattern_rules['cyclical']['irregular_threshold']:
                return None
            
            # Determine severity
            severity = self._determine_cyclical_severity(regularity_score)
            confidence = min(regularity_score + 0.1, 1.0)
            
            pattern_data = {
                'cycle_length': float(cycle_length),
                'amplitude': float(amplitude),
                'frequency': float(dominant_freq),
                'regularity_score': float(regularity_score),
                'dominant_frequencies': frequencies[:5].tolist(),
                'magnitudes': magnitude[:5].tolist()
            }
            
            pattern = DetectedPattern(
                pattern_type=PatternType.CYCLICAL,
                severity=severity,
                confidence=confidence,
                title=f"Cyclical Pattern in {value_col}",
                description=f"Cyclical pattern with {cycle_length:.1f} period length and {regularity_score:.1%} regularity",
                pattern_data=pattern_data,
                statistical_metrics={
                    'cycle_length': float(cycle_length),
                    'regularity_score': float(regularity_score),
                    'data_points': len(values)
                },
                recommendations=self._generate_cyclical_recommendations(cycle_length, regularity_score, value_col)
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error analyzing cycles: {str(e)}")
            return None
    
    async def _detect_trending_patterns(self, df: pd.DataFrame, request: PatternDetectionRequest) -> List[DetectedPattern]:
        """Detect trending patterns in data."""
        patterns = []
        # Implementation for trend detection
        return patterns
    
    async def _detect_periodic_patterns(self, df: pd.DataFrame, request: PatternDetectionRequest) -> List[DetectedPattern]:
        """Detect periodic patterns in data."""
        patterns = []
        # Implementation for periodic pattern detection
        return patterns
    
    async def _detect_distribution_patterns(self, df: pd.DataFrame, request: PatternDetectionRequest) -> List[DetectedPattern]:
        """Detect data distribution patterns."""
        patterns = []
        
        try:
            numeric_cols = self._find_numeric_columns(df)
            
            for col in numeric_cols:
                distribution_pattern = await self._analyze_distribution(df, col)
                if distribution_pattern:
                    patterns.append(distribution_pattern)
        
        except Exception as e:
            logger.error(f"Error detecting distribution patterns: {str(e)}")
        
        return patterns
    
    async def _analyze_distribution(self, df: pd.DataFrame, col: str) -> Optional[DetectedPattern]:
        """Analyze distribution pattern for a column."""
        try:
            values = df[col].dropna()
            
            if len(values) < self.config['min_data_points']:
                return None
            
            # Calculate distribution statistics
            skewness = stats.skew(values)
            kurtosis = stats.kurtosis(values)
            
            # Detect outliers
            z_scores = np.abs(stats.zscore(values))
            outliers = values[z_scores > self.config['outlier_threshold']]
            outlier_percentage = len(outliers) / len(values)
            
            # Classify distribution type
            distribution_type = self._classify_distribution(skewness, kurtosis)
            
            # Determine if pattern is significant
            if abs(skewness) < self.pattern_rules['distribution']['normal_skew_threshold'] and \
               outlier_percentage < self.pattern_rules['distribution']['outlier_concern_threshold']:
                return None  # Normal distribution, not interesting
            
            # Determine severity
            severity = self._determine_distribution_severity(skewness, outlier_percentage)
            confidence = min(abs(skewness) / 2 + outlier_percentage * 2, 1.0)
            
            pattern_data = {
                'distribution_type': distribution_type,
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'outlier_count': len(outliers),
                'outlier_percentage': float(outlier_percentage),
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std())
            }
            
            pattern = DetectedPattern(
                pattern_type=PatternType.DISTRIBUTION,
                severity=severity,
                confidence=confidence,
                title=f"{distribution_type.title()} Distribution in {col}",
                description=f"{col} shows {distribution_type} distribution with {outlier_percentage:.1%} outliers",
                pattern_data=pattern_data,
                statistical_metrics={
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'outlier_percentage': float(outlier_percentage)
                },
                recommendations=self._generate_distribution_recommendations(distribution_type, skewness, outlier_percentage, col)
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error analyzing distribution: {str(e)}")
            return None
    
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution type based on statistical properties."""
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return "normal"
        elif skewness > 1:
            return "highly_right_skewed"
        elif skewness > 0.5:
            return "right_skewed"
        elif skewness < -1:
            return "highly_left_skewed"
        elif skewness < -0.5:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        else:
            return "irregular"
    
    async def _detect_correlation_patterns(self, df: pd.DataFrame, request: PatternDetectionRequest) -> List[DetectedPattern]:
        """Detect correlation patterns between variables."""
        patterns = []
        # Implementation for correlation pattern detection
        return patterns
    
    async def _detect_outlier_patterns(self, df: pd.DataFrame, request: PatternDetectionRequest) -> List[DetectedPattern]:
        """Detect outlier patterns in data."""
        patterns = []
        # Implementation for outlier pattern detection
        return patterns
    
    async def _detect_clustering_patterns(self, df: pd.DataFrame, request: PatternDetectionRequest) -> List[DetectedPattern]:
        """Detect clustering patterns in data."""
        patterns = []
        # Implementation for clustering pattern detection
        return patterns
    
    def _find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the time column in the DataFrame."""
        for col in df.columns:
            if any(time_word in col.lower() for time_word in ['date', 'time', 'timestamp']):
                return col
        return None
    
    def _find_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Find numeric columns in the DataFrame."""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _determine_seasonal_severity(self, strength: float) -> PatternSeverity:
        """Determine severity of seasonal pattern."""
        if strength >= self.pattern_rules['seasonal']['strong_threshold']:
            return PatternSeverity.HIGH
        elif strength >= self.pattern_rules['seasonal']['moderate_threshold']:
            return PatternSeverity.MEDIUM
        else:
            return PatternSeverity.LOW
    
    def _determine_cyclical_severity(self, regularity_score: float) -> PatternSeverity:
        """Determine severity of cyclical pattern."""
        if regularity_score >= self.pattern_rules['cyclical']['regular_threshold']:
            return PatternSeverity.HIGH
        elif regularity_score >= self.pattern_rules['cyclical']['moderate_threshold']:
            return PatternSeverity.MEDIUM
        else:
            return PatternSeverity.LOW
    
    def _determine_distribution_severity(self, skewness: float, outlier_percentage: float) -> PatternSeverity:
        """Determine severity of distribution pattern."""
        if abs(skewness) > self.pattern_rules['distribution']['high_skew_threshold'] or \
           outlier_percentage > self.pattern_rules['distribution']['outlier_concern_threshold'] * 2:
            return PatternSeverity.HIGH
        elif abs(skewness) > self.pattern_rules['distribution']['normal_skew_threshold'] or \
             outlier_percentage > self.pattern_rules['distribution']['outlier_concern_threshold']:
            return PatternSeverity.MEDIUM
        else:
            return PatternSeverity.LOW
    
    def _severity_weight(self, severity: PatternSeverity) -> int:
        """Convert severity to numeric weight for sorting."""
        weights = {
            PatternSeverity.CRITICAL: 5,
            PatternSeverity.HIGH: 4,
            PatternSeverity.MEDIUM: 3,
            PatternSeverity.LOW: 2,
            PatternSeverity.INFO: 1
        }
        return weights.get(severity, 1)
    
    def _generate_seasonal_recommendations(self, pattern_type: str, strength: float, metric: str) -> List[str]:
        """Generate recommendations for seasonal patterns."""
        recommendations = []
        
        if strength > self.pattern_rules['seasonal']['strong_threshold']:
            recommendations.append(f"Leverage strong {pattern_type} seasonality for {metric} planning")
            recommendations.append(f"Implement {pattern_type}-based forecasting for {metric}")
        
        recommendations.append(f"Monitor {pattern_type} patterns in {metric} for business planning")
        recommendations.append(f"Consider seasonal adjustments in {metric} analysis")
        
        return recommendations
    
    def _generate_cyclical_recommendations(self, cycle_length: float, regularity_score: float, metric: str) -> List[str]:
        """Generate recommendations for cyclical patterns."""
        recommendations = []
        
        recommendations.append(f"Monitor {cycle_length:.1f}-period cycles in {metric}")
        
        if regularity_score > self.pattern_rules['cyclical']['regular_threshold']:
            recommendations.append(f"Use cyclical pattern for {metric} forecasting")
            recommendations.append(f"Plan resources based on {cycle_length:.1f}-period cycle")
        
        return recommendations
    
    def _generate_distribution_recommendations(self, distribution_type: str, skewness: float, outlier_percentage: float, metric: str) -> List[str]:
        """Generate recommendations for distribution patterns."""
        recommendations = []
        
        if "skewed" in distribution_type:
            recommendations.append(f"Consider transformations for skewed {metric} distribution")
            recommendations.append(f"Use median instead of mean for {metric} analysis")
        
        if outlier_percentage > self.pattern_rules['distribution']['outlier_concern_threshold']:
            recommendations.append(f"Investigate {outlier_percentage:.1%} outliers in {metric}")
            recommendations.append(f"Consider outlier treatment for {metric} analysis")
        
        return recommendations
    
    def _generate_summary(self, patterns: List[DetectedPattern]) -> str:
        """Generate summary of detected patterns."""
        if not patterns:
            return "No significant patterns detected in the data."
        
        pattern_counts = defaultdict(int)
        for pattern in patterns:
            pattern_counts[pattern.pattern_type.value] += 1
        
        summary_parts = [f"Detected {len(patterns)} patterns"]
        
        if pattern_counts:
            pattern_summary = ", ".join([f"{count} {ptype}" for ptype, count in pattern_counts.items()])
            summary_parts.append(f"Pattern types: {pattern_summary}")
        
        return "; ".join(summary_parts)
    
    def _extract_pattern_insights(self, patterns: List[DetectedPattern]) -> List[str]:
        """Extract key insights from detected patterns."""
        insights = []
        
        # Get top patterns by confidence
        top_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)[:5]
        
        for pattern in top_patterns:
            insights.append(pattern.description)
        
        return insights
    
    def _generate_statistical_summary(self, df: pd.DataFrame, patterns: List[DetectedPattern]) -> Dict[str, Any]:
        """Generate statistical summary of the analysis."""
        return {
            'data_shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'pattern_distribution': {
                pattern_type.value: len([p for p in patterns if p.pattern_type == pattern_type])
                for pattern_type in PatternType
            },
            'average_confidence': sum(p.confidence for p in patterns) / len(patterns) if patterns else 0.0,
            'high_confidence_patterns': len([p for p in patterns if p.confidence > 0.8])
        }
    
    def _generate_recommendations(self, patterns: List[DetectedPattern]) -> List[str]:
        """Generate overall recommendations based on detected patterns."""
        all_recommendations = []
        
        for pattern in patterns:
            all_recommendations.extend(pattern.recommendations)
        
        # Remove duplicates
        unique_recommendations = list(dict.fromkeys(all_recommendations))
        
        return unique_recommendations[:10]  # Return top 10
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the pattern detector."""
        return {
            'status': 'healthy',
            'pattern_rules_loaded': bool(self.pattern_rules),
            'config_loaded': bool(self.config)
        } 