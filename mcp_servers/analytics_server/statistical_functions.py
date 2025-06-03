#!/usr/bin/env python3
"""
Statistical Analysis Functions
==============================

Statistical analysis and computation tools for the Analytics MCP server.
Provides descriptive statistics, hypothesis testing, correlation analysis,
and distribution fitting capabilities.

Integrates with Session A foundation for caching, validation, and metrics.
"""

import asyncio
import logging
import statistics
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau

# Session A Foundation imports
from shared.utils.caching import cache_analytics
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.validation import sanitize_input
from shared.utils.data_processing import DataProcessor


class StatisticalTest(str, Enum):
    """Available statistical tests."""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    KOLMOGOROV_SMIRNOV = "ks_test"
    SHAPIRO_WILK = "shapiro_wilk"
    MANN_WHITNEY = "mann_whitney"


class CorrelationMethod(str, Enum):
    """Correlation analysis methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


@dataclass
class StatisticalResult:
    """Statistical analysis result."""
    statistic_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    test_statistic: Optional[float] = None
    interpretation: Optional[str] = None


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis engine for the Analytics MCP server.
    
    Provides descriptive statistics, correlation analysis, hypothesis testing,
    and distribution analysis capabilities with caching and performance monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("StatisticalAnalyzer")
        self.metrics = get_metrics_collector()
        self.data_processor = DataProcessor()
        
        # Statistical configuration
        self.default_confidence_level = 0.95
        self.max_data_size = 1000000  # Maximum data points for analysis
        
        # State
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize statistical analyzer."""
        if self._initialized:
            return
        
        self.logger.info("Initializing statistical analyzer...")
        self._initialized = True
        self.logger.info("Statistical analyzer initialized")
    
    async def cleanup(self) -> None:
        """Cleanup statistical analyzer."""
        self.logger.info("Cleaning up statistical analyzer...")
        self._initialized = False
        self.logger.info("Statistical analyzer cleanup complete")
    
    @track_performance(tags={"component": "statistical_analyzer", "operation": "calculate_statistics"})
    async def calculate_statistics(
        self,
        data: List[Union[int, float]],
        statistics: List[str] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics for numerical data.
        
        Args:
            data: List of numerical values
            statistics: List of statistics to calculate
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary containing statistical measures
        """
        try:
            self.logger.info(f"Calculating statistics for {len(data)} data points")
            
            # Validate and clean data
            cleaned_data = self._validate_numerical_data(data)
            
            if not cleaned_data:
                raise ValueError("No valid numerical data provided")
            
            # Default statistics to calculate
            if statistics is None:
                statistics = ["mean", "median", "std", "min", "max", "count", "variance"]
            
            results = {
                "data_summary": {
                    "total_count": len(data),
                    "valid_count": len(cleaned_data),
                    "missing_count": len(data) - len(cleaned_data),
                    "missing_percentage": (len(data) - len(cleaned_data)) / len(data) * 100
                },
                "descriptive_statistics": {},
                "distribution_analysis": {},
                "confidence_level": confidence_level,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Calculate requested statistics
            np_data = np.array(cleaned_data)
            
            if "mean" in statistics:
                results["descriptive_statistics"]["mean"] = float(np.mean(np_data))
            
            if "median" in statistics:
                results["descriptive_statistics"]["median"] = float(np.median(np_data))
            
            if "std" in statistics:
                results["descriptive_statistics"]["standard_deviation"] = float(np.std(np_data, ddof=1))
            
            if "variance" in statistics:
                results["descriptive_statistics"]["variance"] = float(np.var(np_data, ddof=1))
            
            if "min" in statistics:
                results["descriptive_statistics"]["minimum"] = float(np.min(np_data))
            
            if "max" in statistics:
                results["descriptive_statistics"]["maximum"] = float(np.max(np_data))
            
            if "count" in statistics:
                results["descriptive_statistics"]["count"] = len(cleaned_data)
            
            if "skewness" in statistics:
                results["descriptive_statistics"]["skewness"] = float(stats.skew(np_data))
            
            if "kurtosis" in statistics:
                results["descriptive_statistics"]["kurtosis"] = float(stats.kurtosis(np_data))
            
            # Calculate quartiles and percentiles
            if any(stat in statistics for stat in ["quartiles", "percentiles", "q1", "q3", "iqr"]):
                q1 = float(np.percentile(np_data, 25))
                q3 = float(np.percentile(np_data, 75))
                iqr = q3 - q1
                
                results["descriptive_statistics"]["quartiles"] = {
                    "q1": q1,
                    "q2": results["descriptive_statistics"].get("median", float(np.median(np_data))),
                    "q3": q3,
                    "iqr": iqr
                }
                
                results["descriptive_statistics"]["percentiles"] = {
                    "p5": float(np.percentile(np_data, 5)),
                    "p10": float(np.percentile(np_data, 10)),
                    "p25": q1,
                    "p50": float(np.percentile(np_data, 50)),
                    "p75": q3,
                    "p90": float(np.percentile(np_data, 90)),
                    "p95": float(np.percentile(np_data, 95))
                }
            
            # Calculate confidence intervals
            if "confidence_intervals" in statistics:
                results["confidence_intervals"] = await self._calculate_confidence_intervals(
                    np_data, confidence_level
                )
            
            # Distribution analysis
            if "distribution" in statistics:
                results["distribution_analysis"] = await self._analyze_distribution(np_data)
            
            # Outlier detection
            if "outliers" in statistics:
                results["outlier_analysis"] = await self._detect_outliers(np_data)
            
            # Update metrics
            self.metrics.counter("analytics.statistics.calculations").increment()
            self.metrics.histogram("analytics.statistics.data_size").update(len(cleaned_data))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Statistical calculation failed: {str(e)}")
            self.metrics.counter("analytics.statistics.errors").increment()
            raise
    
    @track_performance(tags={"component": "statistical_analyzer", "operation": "correlation_analysis"})
    async def correlation_analysis(
        self,
        data: Dict[str, List[Union[int, float]]],
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        Perform correlation analysis between variables.
        
        Args:
            data: Dictionary of variable names to value arrays
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            Correlation analysis results
        """
        try:
            self.logger.info(f"Performing {method} correlation analysis on {len(data)} variables")
            
            # Validate correlation method
            try:
                corr_method = CorrelationMethod(method.lower())
            except ValueError:
                raise ValueError(f"Unsupported correlation method: {method}")
            
            # Validate and prepare data
            variable_names = list(data.keys())
            if len(variable_names) < 2:
                raise ValueError("At least two variables required for correlation analysis")
            
            # Clean and align data
            aligned_data = self._align_numerical_data(data)
            
            if len(aligned_data) == 0:
                raise ValueError("No valid data pairs found for correlation analysis")
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(aligned_data)
            
            results = {
                "method": method,
                "variables": variable_names,
                "data_summary": {
                    "total_observations": len(df),
                    "variables_count": len(variable_names)
                },
                "correlation_matrix": {},
                "pairwise_correlations": [],
                "strongest_correlations": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Calculate correlation matrix
            if corr_method == CorrelationMethod.PEARSON:
                corr_matrix = df.corr(method='pearson')
            elif corr_method == CorrelationMethod.SPEARMAN:
                corr_matrix = df.corr(method='spearman')
            elif corr_method == CorrelationMethod.KENDALL:
                corr_matrix = df.corr(method='kendall')
            
            # Convert correlation matrix to dictionary
            results["correlation_matrix"] = corr_matrix.to_dict()
            
            # Calculate pairwise correlations with significance tests
            pairwise_results = []
            for i, var1 in enumerate(variable_names):
                for j, var2 in enumerate(variable_names):
                    if i < j:  # Avoid duplicates and self-correlation
                        x = df[var1].dropna()
                        y = df[var2].dropna()
                        
                        if len(x) > 2 and len(y) > 2:
                            # Align the data (remove pairs where either is NaN)
                            combined = pd.DataFrame({'x': x, 'y': y}).dropna()
                            if len(combined) > 2:
                                x_clean = combined['x'].values
                                y_clean = combined['y'].values
                                
                                if corr_method == CorrelationMethod.PEARSON:
                                    correlation, p_value = pearsonr(x_clean, y_clean)
                                elif corr_method == CorrelationMethod.SPEARMAN:
                                    correlation, p_value = spearmanr(x_clean, y_clean)
                                elif corr_method == CorrelationMethod.KENDALL:
                                    correlation, p_value = kendalltau(x_clean, y_clean)
                                
                                pairwise_results.append({
                                    "variable_1": var1,
                                    "variable_2": var2,
                                    "correlation": float(correlation),
                                    "p_value": float(p_value),
                                    "sample_size": len(x_clean),
                                    "significance": "significant" if p_value < 0.05 else "not_significant",
                                    "strength": self._interpret_correlation_strength(abs(correlation))
                                })
            
            results["pairwise_correlations"] = pairwise_results
            
            # Identify strongest correlations
            if pairwise_results:
                sorted_correlations = sorted(
                    pairwise_results,
                    key=lambda x: abs(x["correlation"]),
                    reverse=True
                )
                results["strongest_correlations"] = sorted_correlations[:5]
            
            # Summary statistics
            if pairwise_results:
                correlations = [abs(r["correlation"]) for r in pairwise_results]
                results["summary"] = {
                    "average_correlation": float(np.mean(correlations)),
                    "max_correlation": float(np.max(correlations)),
                    "min_correlation": float(np.min(correlations)),
                    "significant_correlations": len([r for r in pairwise_results if r["significance"] == "significant"])
                }
            
            # Update metrics
            self.metrics.counter("analytics.correlation.analyses").increment()
            self.metrics.histogram("analytics.correlation.variables_count").update(len(variable_names))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {str(e)}")
            self.metrics.counter("analytics.correlation.errors").increment()
            raise
    
    async def hypothesis_test(
        self,
        test_type: str,
        data: Union[List[float], Dict[str, List[float]]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform statistical hypothesis tests.
        
        Args:
            test_type: Type of statistical test
            data: Data for testing
            **kwargs: Additional test parameters
            
        Returns:
            Test results with statistics and interpretation
        """
        try:
            test_enum = StatisticalTest(test_type.lower())
            
            if test_enum == StatisticalTest.T_TEST:
                return await self._perform_t_test(data, **kwargs)
            elif test_enum == StatisticalTest.CHI_SQUARE:
                return await self._perform_chi_square_test(data, **kwargs)
            elif test_enum == StatisticalTest.ANOVA:
                return await self._perform_anova(data, **kwargs)
            elif test_enum == StatisticalTest.KOLMOGOROV_SMIRNOV:
                return await self._perform_ks_test(data, **kwargs)
            elif test_enum == StatisticalTest.SHAPIRO_WILK:
                return await self._perform_shapiro_wilk_test(data, **kwargs)
            elif test_enum == StatisticalTest.MANN_WHITNEY:
                return await self._perform_mann_whitney_test(data, **kwargs)
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
                
        except Exception as e:
            self.logger.error(f"Hypothesis test failed: {str(e)}")
            raise
    
    def _validate_numerical_data(self, data: List[Any]) -> List[float]:
        """Validate and clean numerical data."""
        cleaned = []
        for value in data:
            try:
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    float_val = float(value)
                    if not math.isnan(float_val) and not math.isinf(float_val):
                        cleaned.append(float_val)
            except (ValueError, TypeError):
                continue
        return cleaned
    
    def _align_numerical_data(self, data: Dict[str, List[Any]]) -> Dict[str, List[float]]:
        """Align numerical data across variables, removing rows with any missing values."""
        # First, validate each variable
        validated_data = {}
        for var_name, values in data.items():
            validated_data[var_name] = []
            for value in values:
                try:
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        float_val = float(value)
                        if not math.isnan(float_val) and not math.isinf(float_val):
                            validated_data[var_name].append(float_val)
                        else:
                            validated_data[var_name].append(None)
                    else:
                        validated_data[var_name].append(None)
                except (ValueError, TypeError):
                    validated_data[var_name].append(None)
        
        # Find the minimum length
        min_length = min(len(values) for values in validated_data.values())
        
        # Align data by removing rows with any None values
        aligned_data = {var_name: [] for var_name in validated_data.keys()}
        
        for i in range(min_length):
            row_values = [validated_data[var_name][i] for var_name in validated_data.keys()]
            if all(value is not None for value in row_values):
                for j, var_name in enumerate(validated_data.keys()):
                    aligned_data[var_name].append(row_values[j])
        
        return aligned_data
    
    async def _calculate_confidence_intervals(
        self,
        data: np.ndarray,
        confidence_level: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for various statistics."""
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        
        # t-distribution for mean confidence interval
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        mean_ci = (mean - t_critical * std_err, mean + t_critical * std_err)
        
        return {
            "mean": mean_ci
        }
    
    async def _analyze_distribution(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze data distribution characteristics."""
        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Shapiro-Wilk limited to 5000 samples
        
        # Calculate distribution parameters
        return {
            "normality_test": {
                "shapiro_wilk_statistic": float(shapiro_stat),
                "shapiro_wilk_p_value": float(shapiro_p),
                "is_normal": shapiro_p > 0.05
            },
            "distribution_fit": {
                "normal": {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data))
                }
            }
        }
    
    async def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        # IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = data[z_scores > 3]
        
        return {
            "iqr_method": {
                "outliers_count": len(iqr_outliers),
                "outlier_percentage": len(iqr_outliers) / len(data) * 100,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            },
            "zscore_method": {
                "outliers_count": len(z_outliers),
                "outlier_percentage": len(z_outliers) / len(data) * 100,
                "threshold": 3.0
            }
        }
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength based on absolute value."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    async def _perform_t_test(self, data: Union[List[float], Dict[str, List[float]]], **kwargs) -> Dict[str, Any]:
        """Perform t-test (one-sample or two-sample)."""
        if isinstance(data, list):
            # One-sample t-test
            population_mean = kwargs.get('population_mean', 0)
            cleaned_data = self._validate_numerical_data(data)
            
            t_stat, p_value = stats.ttest_1samp(cleaned_data, population_mean)
            
            return {
                "test_type": "one_sample_t_test",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": len(cleaned_data) - 1,
                "sample_mean": float(np.mean(cleaned_data)),
                "population_mean": population_mean,
                "significant": p_value < 0.05
            }
        else:
            # Two-sample t-test
            groups = list(data.keys())
            if len(groups) != 2:
                raise ValueError("Two-sample t-test requires exactly two groups")
            
            group1_data = self._validate_numerical_data(data[groups[0]])
            group2_data = self._validate_numerical_data(data[groups[1]])
            
            equal_var = kwargs.get('equal_var', True)
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
            
            return {
                "test_type": "two_sample_t_test",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": len(group1_data) + len(group2_data) - 2,
                "group1_mean": float(np.mean(group1_data)),
                "group2_mean": float(np.mean(group2_data)),
                "equal_variances": equal_var,
                "significant": p_value < 0.05
            }
    
    async def _perform_chi_square_test(self, data: List[List[int]], **kwargs) -> Dict[str, Any]:
        """Perform chi-square test of independence."""
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(data)
        
        return {
            "test_type": "chi_square_independence",
            "chi2_statistic": float(chi2_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "expected_frequencies": expected.tolist(),
            "significant": p_value < 0.05
        }
    
    async def _perform_anova(self, data: Dict[str, List[float]], **kwargs) -> Dict[str, Any]:
        """Perform one-way ANOVA."""
        groups = [self._validate_numerical_data(group_data) for group_data in data.values()]
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            "test_type": "one_way_anova",
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "groups_count": len(groups),
            "significant": p_value < 0.05
        }
    
    async def _perform_ks_test(self, data: List[float], **kwargs) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for normality."""
        cleaned_data = self._validate_numerical_data(data)
        
        # Test against normal distribution
        ks_stat, p_value = stats.kstest(cleaned_data, 'norm')
        
        return {
            "test_type": "kolmogorov_smirnov_normality",
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05
        }
    
    async def _perform_shapiro_wilk_test(self, data: List[float], **kwargs) -> Dict[str, Any]:
        """Perform Shapiro-Wilk test for normality."""
        cleaned_data = self._validate_numerical_data(data)
        
        # Limit to 5000 samples for Shapiro-Wilk
        if len(cleaned_data) > 5000:
            cleaned_data = cleaned_data[:5000]
        
        shapiro_stat, p_value = stats.shapiro(cleaned_data)
        
        return {
            "test_type": "shapiro_wilk_normality",
            "shapiro_statistic": float(shapiro_stat),
            "p_value": float(p_value),
            "sample_size": len(cleaned_data),
            "is_normal": p_value > 0.05
        }
    
    async def _perform_mann_whitney_test(self, data: Dict[str, List[float]], **kwargs) -> Dict[str, Any]:
        """Perform Mann-Whitney U test."""
        groups = list(data.keys())
        if len(groups) != 2:
            raise ValueError("Mann-Whitney test requires exactly two groups")
        
        group1_data = self._validate_numerical_data(data[groups[0]])
        group2_data = self._validate_numerical_data(data[groups[1]])
        
        u_stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        
        return {
            "test_type": "mann_whitney_u",
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "group1_median": float(np.median(group1_data)),
            "group2_median": float(np.median(group2_data)),
            "significant": p_value < 0.05
        } 