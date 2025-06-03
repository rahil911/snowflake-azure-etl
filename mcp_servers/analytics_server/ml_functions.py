#!/usr/bin/env python3
"""
Machine Learning Functions
===========================

Basic machine learning and analysis tools for the Analytics MCP server.
Provides clustering, trend analysis, outlier detection, and time series
analysis capabilities.

Integrates with Session A foundation for caching, validation, and metrics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import zscore

# Session A Foundation imports
from shared.utils.caching import cache_analytics
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.validation import sanitize_input
from shared.utils.data_processing import DataProcessor


class ClusteringAlgorithm(str, Enum):
    """Available clustering algorithms."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    ISOLATION_FOREST = "isolation_forest"


class OutlierMethod(str, Enum):
    """Outlier detection methods."""
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation"
    MODIFIED_ZSCORE = "modified_zscore"


class TrendDirection(str, Enum):
    """Trend direction types."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class MLAnalyzer:
    """
    Machine learning and advanced analytics engine for the Analytics MCP server.
    
    Provides clustering, outlier detection, trend analysis, and time series
    analysis capabilities with caching and performance monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MLAnalyzer")
        self.metrics = get_metrics_collector()
        self.data_processor = DataProcessor()
        
        # ML configuration
        self.default_random_state = 42
        self.max_data_size = 100000  # Maximum data points for ML operations
        
        # Model state
        self._fitted_models = {}
        self._analysis_history = []
        self.max_history_size = 50
        
        # State
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize ML analyzer."""
        if self._initialized:
            return
        
        self.logger.info("Initializing ML analyzer...")
        self._initialized = True
        self.logger.info("ML analyzer initialized")
    
    async def cleanup(self) -> None:
        """Cleanup ML analyzer."""
        self.logger.info("Cleaning up ML analyzer...")
        self._fitted_models.clear()
        self._analysis_history.clear()
        self._initialized = False
        self.logger.info("ML analyzer cleanup complete")
    
    @track_performance(tags={"component": "ml_analyzer", "operation": "cluster_analysis"})
    async def cluster_analysis(
        self,
        data: Union[List[Dict[str, Any]], List[List[float]]],
        n_clusters: int = 3,
        algorithm: str = "kmeans"
    ) -> Dict[str, Any]:
        """
        Perform clustering analysis on data.
        
        Args:
            data: Input data for clustering
            n_clusters: Number of clusters (for algorithms that require it)
            algorithm: Clustering algorithm to use
            
        Returns:
            Clustering results with cluster assignments and metrics
        """
        try:
            self.logger.info(f"Performing {algorithm} clustering with {n_clusters} clusters")
            
            # Validate algorithm
            try:
                cluster_algo = ClusteringAlgorithm(algorithm.lower())
            except ValueError:
                raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
            
            # Prepare data
            df = self._prepare_data_for_ml(data)
            
            if df.empty:
                raise ValueError("No valid data provided for clustering")
            
            # Select numerical columns only
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numerical_columns:
                raise ValueError("No numerical columns found for clustering")
            
            X = df[numerical_columns].values
            
            # Check data size
            if len(X) > self.max_data_size:
                self.logger.warning(f"Data size {len(X)} exceeds maximum, sampling to {self.max_data_size}")
                sample_indices = np.random.choice(len(X), self.max_data_size, replace=False)
                X = X[sample_indices]
                df_sampled = df.iloc[sample_indices]
            else:
                df_sampled = df
            
            # Standardize data for clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            cluster_result = await self._perform_clustering(X_scaled, cluster_algo, n_clusters)
            
            # Calculate clustering metrics
            metrics = await self._calculate_clustering_metrics(X_scaled, cluster_result["labels"])
            
            # Analyze clusters
            cluster_analysis = await self._analyze_clusters(df_sampled, cluster_result["labels"], numerical_columns)
            
            # Store model
            model_id = f"cluster_{algorithm}_{datetime.utcnow().timestamp()}"
            self._fitted_models[model_id] = {
                "model": cluster_result["model"],
                "scaler": scaler,
                "columns": numerical_columns,
                "algorithm": algorithm,
                "type": "clustering"
            }
            
            result = {
                "algorithm": algorithm,
                "n_clusters": cluster_result.get("n_clusters", n_clusters),
                "cluster_labels": cluster_result["labels"].tolist(),
                "cluster_centers": cluster_result.get("centers", []),
                "clustering_metrics": metrics,
                "cluster_analysis": cluster_analysis,
                "model_id": model_id,
                "data_summary": {
                    "total_points": len(X_scaled),
                    "features": len(numerical_columns),
                    "feature_names": numerical_columns
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add to history
            self._add_to_history({
                "operation": "clustering",
                "algorithm": algorithm,
                "n_clusters": n_clusters,
                "data_points": len(X_scaled),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update metrics
            self.metrics.counter("analytics.clustering.analyses").increment()
            self.metrics.counter(f"analytics.clustering.{algorithm}").increment()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Clustering analysis failed: {str(e)}")
            self.metrics.counter("analytics.clustering.errors").increment()
            raise
    
    @track_performance(tags={"component": "ml_analyzer", "operation": "outlier_detection"})
    async def outlier_detection(
        self,
        data: Union[List[float], List[Dict[str, Any]]],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """
        Detect outliers in numerical data.
        
        Args:
            data: Input data for outlier detection
            method: Detection method (iqr, zscore, isolation, modified_zscore)
            threshold: Threshold for outlier detection
            
        Returns:
            Outlier detection results with outlier indices and statistics
        """
        try:
            self.logger.info(f"Detecting outliers using {method} method with threshold {threshold}")
            
            # Validate method
            try:
                outlier_method = OutlierMethod(method.lower())
            except ValueError:
                raise ValueError(f"Unsupported outlier detection method: {method}")
            
            # Prepare data
            if isinstance(data, list) and data and not isinstance(data[0], dict):
                # Simple numerical list
                values = [float(x) for x in data if x is not None and not (isinstance(x, float) and np.isnan(x))]
                df = pd.DataFrame({"value": values})
                target_column = "value"
            else:
                # Complex data structure
                df = self._prepare_data_for_ml(data)
                numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numerical_columns:
                    raise ValueError("No numerical columns found for outlier detection")
                # Use first numerical column as default
                target_column = numerical_columns[0]
            
            if df.empty:
                raise ValueError("No valid data provided for outlier detection")
            
            values = df[target_column].dropna().values
            
            if len(values) < 5:
                raise ValueError("Insufficient data points for outlier detection")
            
            # Perform outlier detection
            outlier_result = await self._detect_outliers_by_method(values, outlier_method, threshold)
            
            # Calculate statistics
            outlier_stats = {
                "total_points": len(values),
                "outliers_count": len(outlier_result["outlier_indices"]),
                "outliers_percentage": len(outlier_result["outlier_indices"]) / len(values) * 100,
                "clean_data_points": len(values) - len(outlier_result["outlier_indices"])
            }
            
            # Analyze outliers
            outlier_values = values[outlier_result["outlier_indices"]]
            clean_values = np.delete(values, outlier_result["outlier_indices"])
            
            analysis = {
                "outlier_statistics": {
                    "outlier_min": float(np.min(outlier_values)) if len(outlier_values) > 0 else None,
                    "outlier_max": float(np.max(outlier_values)) if len(outlier_values) > 0 else None,
                    "outlier_mean": float(np.mean(outlier_values)) if len(outlier_values) > 0 else None
                },
                "clean_data_statistics": {
                    "clean_min": float(np.min(clean_values)) if len(clean_values) > 0 else None,
                    "clean_max": float(np.max(clean_values)) if len(clean_values) > 0 else None,
                    "clean_mean": float(np.mean(clean_values)) if len(clean_values) > 0 else None,
                    "clean_std": float(np.std(clean_values)) if len(clean_values) > 0 else None
                }
            }
            
            result = {
                "method": method,
                "threshold": threshold,
                "outlier_indices": outlier_result["outlier_indices"].tolist(),
                "outlier_values": outlier_values.tolist() if len(outlier_values) > 0 else [],
                "detection_metadata": outlier_result.get("metadata", {}),
                "statistics": outlier_stats,
                "analysis": analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update metrics
            self.metrics.counter("analytics.outlier_detection.analyses").increment()
            self.metrics.counter(f"analytics.outlier_detection.{method}").increment()
            self.metrics.histogram("analytics.outlier_detection.outliers_found").update(len(outlier_result["outlier_indices"]))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Outlier detection failed: {str(e)}")
            self.metrics.counter("analytics.outlier_detection.errors").increment()
            raise
    
    @track_performance(tags={"component": "ml_analyzer", "operation": "trend_analysis"})
    async def trend_analysis(
        self,
        data: List[float],
        timestamps: List[str],
        period: str = "monthly"
    ) -> Dict[str, Any]:
        """
        Analyze trends in time series data.
        
        Args:
            data: Time series values
            timestamps: Corresponding timestamps
            period: Analysis period (daily, weekly, monthly, yearly)
            
        Returns:
            Trend analysis results with direction, strength, and predictions
        """
        try:
            self.logger.info(f"Performing trend analysis with {period} period")
            
            # Validate and prepare data
            if len(data) != len(timestamps):
                raise ValueError("Data and timestamps must have the same length")
            
            if len(data) < 3:
                raise ValueError("At least 3 data points required for trend analysis")
            
            # Create DataFrame
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamps),
                "value": data
            })
            
            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Remove duplicates and NaN values
            df = df.dropna().drop_duplicates(subset=["timestamp"])
            
            if len(df) < 3:
                raise ValueError("Insufficient valid data points after cleaning")
            
            # Perform trend analysis
            trend_result = await self._analyze_trend(df, period)
            
            # Calculate trend metrics
            trend_metrics = await self._calculate_trend_metrics(df["value"].values)
            
            # Generate trend predictions (simple linear extrapolation)
            predictions = await self._generate_trend_predictions(df, period)
            
            result = {
                "period": period,
                "data_summary": {
                    "total_points": len(df),
                    "time_span": {
                        "start": df["timestamp"].min().isoformat(),
                        "end": df["timestamp"].max().isoformat(),
                        "duration_days": (df["timestamp"].max() - df["timestamp"].min()).days
                    },
                    "value_range": {
                        "min": float(df["value"].min()),
                        "max": float(df["value"].max()),
                        "mean": float(df["value"].mean())
                    }
                },
                "trend_analysis": trend_result,
                "trend_metrics": trend_metrics,
                "predictions": predictions,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add to history
            self._add_to_history({
                "operation": "trend_analysis",
                "period": period,
                "data_points": len(df),
                "trend_direction": trend_result.get("direction"),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update metrics
            self.metrics.counter("analytics.trend_analysis.analyses").increment()
            self.metrics.histogram("analytics.trend_analysis.data_points").update(len(df))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {str(e)}")
            self.metrics.counter("analytics.trend_analysis.errors").increment()
            raise
    
    def _prepare_data_for_ml(self, data: Union[List[Dict[str, Any]], List[List[float]]]) -> pd.DataFrame:
        """Prepare data for ML operations."""
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return pd.DataFrame(data)
            elif data and isinstance(data[0], (list, tuple)):
                # List of lists/tuples - convert to DataFrame with column names
                return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(len(data[0]))])
            else:
                # Simple list of values
                return pd.DataFrame({"value": data})
        else:
            raise ValueError("Data must be a list")
    
    async def _perform_clustering(
        self,
        X: np.ndarray,
        algorithm: ClusteringAlgorithm,
        n_clusters: int
    ) -> Dict[str, Any]:
        """Perform clustering using specified algorithm."""
        
        if algorithm == ClusteringAlgorithm.KMEANS:
            model = KMeans(n_clusters=n_clusters, random_state=self.default_random_state, n_init=10)
            labels = model.fit_predict(X)
            return {
                "model": model,
                "labels": labels,
                "centers": model.cluster_centers_.tolist(),
                "n_clusters": n_clusters
            }
        
        elif algorithm == ClusteringAlgorithm.DBSCAN:
            # Auto-determine eps using heuristic
            eps = self._estimate_dbscan_eps(X)
            model = DBSCAN(eps=eps, min_samples=5)
            labels = model.fit_predict(X)
            unique_labels = np.unique(labels)
            n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            return {
                "model": model,
                "labels": labels,
                "n_clusters": n_clusters_found,
                "eps": eps,
                "noise_points": np.sum(labels == -1)
            }
        
        elif algorithm == ClusteringAlgorithm.HIERARCHICAL:
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)
            return {
                "model": model,
                "labels": labels,
                "n_clusters": n_clusters
            }
        
        else:
            raise ValueError(f"Clustering algorithm {algorithm} not implemented")
    
    def _estimate_dbscan_eps(self, X: np.ndarray) -> float:
        """Estimate eps parameter for DBSCAN using k-distance graph."""
        from sklearn.neighbors import NearestNeighbors
        
        # Use k=4 as a heuristic
        k = min(4, len(X) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Sort distances and use the knee point
        k_distances = np.sort(distances[:, -1])
        
        # Simple heuristic: use 90th percentile
        eps = np.percentile(k_distances, 90)
        return eps
    
    async def _calculate_clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering quality metrics."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        metrics = {
            "n_clusters_found": int(n_clusters),
            "n_noise_points": int(np.sum(labels == -1)) if -1 in unique_labels else 0
        }
        
        # Silhouette score (only if we have more than 1 cluster and not all points are noise)
        if n_clusters > 1 and len(X) > n_clusters:
            try:
                # Remove noise points for silhouette calculation
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette_avg = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                    metrics["silhouette_score"] = float(silhouette_avg)
            except Exception as e:
                self.logger.warning(f"Could not calculate silhouette score: {str(e)}")
        
        # Inertia for K-means (if applicable)
        if hasattr(self, '_last_kmeans_model'):
            metrics["inertia"] = float(self._last_kmeans_model.inertia_)
        
        return metrics
    
    async def _analyze_clusters(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Analyze characteristics of each cluster."""
        df_with_clusters = df.copy()
        df_with_clusters["cluster"] = labels
        
        cluster_analysis = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_data = df_with_clusters[df_with_clusters["cluster"] == cluster_id]
            
            analysis = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(df_with_clusters) * 100,
                "feature_means": {},
                "feature_stds": {}
            }
            
            for col in feature_columns:
                if col in cluster_data.columns:
                    analysis["feature_means"][col] = float(cluster_data[col].mean())
                    analysis["feature_stds"][col] = float(cluster_data[col].std())
            
            cluster_analysis[f"cluster_{cluster_id}"] = analysis
        
        return cluster_analysis
    
    async def _detect_outliers_by_method(
        self,
        values: np.ndarray,
        method: OutlierMethod,
        threshold: float
    ) -> Dict[str, Any]:
        """Detect outliers using specified method."""
        
        if method == OutlierMethod.IQR:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
            
            return {
                "outlier_indices": outlier_indices,
                "metadata": {
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
            }
        
        elif method == OutlierMethod.ZSCORE:
            z_scores = np.abs(zscore(values))
            outlier_indices = np.where(z_scores > threshold)[0]
            
            return {
                "outlier_indices": outlier_indices,
                "metadata": {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "threshold": threshold
                }
            }
        
        elif method == OutlierMethod.MODIFIED_ZSCORE:
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
            
            return {
                "outlier_indices": outlier_indices,
                "metadata": {
                    "median": float(median),
                    "mad": float(mad),
                    "threshold": threshold
                }
            }
        
        elif method == OutlierMethod.ISOLATION_FOREST:
            contamination = min(0.1, threshold / 100)  # Convert threshold to contamination rate
            iso_forest = IsolationForest(contamination=contamination, random_state=self.default_random_state)
            
            # Reshape for sklearn
            X = values.reshape(-1, 1)
            outlier_predictions = iso_forest.fit_predict(X)
            outlier_indices = np.where(outlier_predictions == -1)[0]
            
            return {
                "outlier_indices": outlier_indices,
                "metadata": {
                    "contamination": contamination,
                    "n_estimators": iso_forest.n_estimators
                }
            }
        
        else:
            raise ValueError(f"Outlier detection method {method} not implemented")
    
    async def _analyze_trend(self, df: pd.DataFrame, period: str) -> Dict[str, Any]:
        """Analyze trend direction and characteristics."""
        # Simple linear regression for trend
        x = np.arange(len(df))
        y = df["value"].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if abs(slope) < std_err:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Calculate volatility
        volatility = float(np.std(y))
        mean_value = float(np.mean(y))
        coefficient_of_variation = volatility / mean_value if mean_value != 0 else 0
        
        # If volatility is high relative to trend, mark as volatile
        if coefficient_of_variation > 0.5:
            direction = TrendDirection.VOLATILE
        
        return {
            "direction": direction.value,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "standard_error": float(std_err),
            "volatility": volatility,
            "coefficient_of_variation": coefficient_of_variation,
            "trend_strength": "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
        }
    
    async def _calculate_trend_metrics(self, values: np.ndarray) -> Dict[str, Any]:
        """Calculate additional trend metrics."""
        # Calculate moving averages
        if len(values) >= 5:
            ma_5 = np.convolve(values, np.ones(5)/5, mode='valid')
            current_ma = float(ma_5[-1]) if len(ma_5) > 0 else None
        else:
            current_ma = None
        
        # Calculate rate of change
        if len(values) >= 2:
            rate_of_change = float((values[-1] - values[0]) / len(values))
        else:
            rate_of_change = 0.0
        
        return {
            "current_moving_average": current_ma,
            "rate_of_change": rate_of_change,
            "total_change": float(values[-1] - values[0]) if len(values) > 1 else 0.0,
            "total_change_percentage": float((values[-1] - values[0]) / values[0] * 100) if len(values) > 1 and values[0] != 0 else 0.0
        }
    
    async def _generate_trend_predictions(self, df: pd.DataFrame, period: str) -> Dict[str, Any]:
        """Generate simple trend predictions."""
        # Simple linear extrapolation
        x = np.arange(len(df))
        y = df["value"].values
        
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        # Predict next few points
        future_points = 5
        future_x = np.arange(len(df), len(df) + future_points)
        future_y = slope * future_x + intercept
        
        # Generate future timestamps
        last_timestamp = df["timestamp"].iloc[-1]
        if period == "daily":
            future_timestamps = [last_timestamp + timedelta(days=i+1) for i in range(future_points)]
        elif period == "weekly":
            future_timestamps = [last_timestamp + timedelta(weeks=i+1) for i in range(future_points)]
        elif period == "monthly":
            future_timestamps = [last_timestamp + timedelta(days=(i+1)*30) for i in range(future_points)]
        else:
            future_timestamps = [last_timestamp + timedelta(days=i+1) for i in range(future_points)]
        
        return {
            "method": "linear_extrapolation",
            "predictions": [
                {
                    "timestamp": ts.isoformat(),
                    "predicted_value": float(val)
                }
                for ts, val in zip(future_timestamps, future_y)
            ],
            "confidence": "low"  # Simple linear prediction has low confidence
        }
    
    def _add_to_history(self, analysis_record: Dict[str, Any]) -> None:
        """Add analysis to history."""
        self._analysis_history.append(analysis_record)
        
        # Trim history if too large
        if len(self._analysis_history) > self.max_history_size:
            self._analysis_history = self._analysis_history[-self.max_history_size:]
    
    async def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self._analysis_history.copy()
    
    async def get_fitted_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about fitted models."""
        return {
            model_id: {
                "type": info["type"],
                "algorithm": info.get("algorithm"),
                "columns": info.get("columns"),
                "fitted_at": model_id.split("_")[-1]
            }
            for model_id, info in self._fitted_models.items()
        } 