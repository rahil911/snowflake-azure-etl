#!/usr/bin/env python3
"""
Data Transformation Functions
=============================

Data transformation and aggregation tools for the Analytics MCP server.
Provides data normalization, scaling, aggregation, and reshaping capabilities.

Extends and complements the shared data processing utilities from Session A
with analytics-specific transformations.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA

# Session A Foundation imports
from shared.utils.caching import cache_analytics
from shared.utils.metrics import get_metrics_collector, track_performance
from shared.utils.validation import sanitize_input
from shared.utils.data_processing import DataProcessor, DataTransformer as SharedTransformer


class TransformationType(str, Enum):
    """Available data transformation types."""
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    ROBUST_SCALE = "robust_scale"
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    BOX_COX = "box_cox"
    QUANTILE_TRANSFORM = "quantile_transform"
    POLYNOMIAL_FEATURES = "polynomial_features"
    PCA_TRANSFORM = "pca_transform"
    LABEL_ENCODE = "label_encode"
    ONE_HOT_ENCODE = "one_hot_encode"


class AggregationFunction(str, Enum):
    """Available aggregation functions."""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"
    MODE = "mode"


class DataTransformer:
    """
    Advanced data transformation engine for the Analytics MCP server.
    
    Provides data normalization, scaling, aggregation, and advanced transformations
    with caching and performance monitoring. Extends the shared data processing
    utilities with analytics-specific capabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("DataTransformer")
        self.metrics = get_metrics_collector()
        self.data_processor = DataProcessor()
        self.shared_transformer = SharedTransformer()
        
        # Transformation state
        self._fitted_transformers = {}
        self._transformation_history = []
        self.max_history_size = 100
        
        # State
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize data transformer."""
        if self._initialized:
            return
        
        self.logger.info("Initializing data transformer...")
        self._initialized = True
        self.logger.info("Data transformer initialized")
    
    async def cleanup(self) -> None:
        """Cleanup data transformer."""
        self.logger.info("Cleaning up data transformer...")
        self._fitted_transformers.clear()
        self._transformation_history.clear()
        self._initialized = False
        self.logger.info("Data transformer cleanup complete")
    
    @track_performance(tags={"component": "data_transformer", "operation": "transform_data"})
    async def transform_data(
        self,
        data: Union[List[Any], Dict[str, Any]],
        transformation: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Apply data transformation to input data.
        
        Args:
            data: Input data (list of values or dictionary)
            transformation: Type of transformation to apply
            parameters: Transformation-specific parameters
            
        Returns:
            Transformed data with metadata
        """
        try:
            self.logger.info(f"Applying {transformation} transformation")
            
            if parameters is None:
                parameters = {}
            
            # Validate transformation type
            try:
                transform_type = TransformationType(transformation.lower())
            except ValueError:
                raise ValueError(f"Unsupported transformation type: {transformation}")
            
            # Convert data to appropriate format
            df = self._prepare_data_for_transformation(data)
            
            if df.empty:
                raise ValueError("No valid data provided for transformation")
            
            # Apply transformation
            result = await self._apply_transformation(df, transform_type, parameters)
            
            # Record transformation in history
            transformation_record = {
                "transformation": transformation,
                "parameters": parameters,
                "timestamp": datetime.utcnow().isoformat(),
                "input_shape": df.shape,
                "output_shape": result["transformed_data"].shape if hasattr(result["transformed_data"], 'shape') else None
            }
            
            self._add_to_history(transformation_record)
            
            # Update metrics
            self.metrics.counter("analytics.transformations.applied").increment()
            self.metrics.counter(f"analytics.transformations.{transformation}").increment()
            
            return {
                "transformation": transformation,
                "parameters": parameters,
                "original_data_shape": list(df.shape),
                "transformed_data": result["transformed_data"],
                "transformation_metadata": result.get("metadata", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Data transformation failed: {str(e)}")
            self.metrics.counter("analytics.transformations.errors").increment()
            raise
    
    @track_performance(tags={"component": "data_transformer", "operation": "aggregate_data"})
    async def aggregate_data(
        self,
        data: List[Dict[str, Any]],
        group_by: str,
        aggregations: Dict[str, Union[str, List[str]]]
    ) -> Dict[str, Any]:
        """
        Aggregate data by groups with specified functions.
        
        Args:
            data: List of data records (dictionaries)
            group_by: Column name to group by
            aggregations: Dictionary mapping column names to aggregation functions
            
        Returns:
            Aggregated data with summary statistics
        """
        try:
            self.logger.info(f"Aggregating data by {group_by} column")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if df.empty:
                raise ValueError("No data provided for aggregation")
            
            if group_by not in df.columns:
                raise ValueError(f"Group by column '{group_by}' not found in data")
            
            # Validate aggregation functions
            validated_aggregations = {}
            for column, functions in aggregations.items():
                if column not in df.columns:
                    self.logger.warning(f"Column '{column}' not found, skipping")
                    continue
                
                if isinstance(functions, str):
                    functions = [functions]
                
                valid_functions = []
                for func in functions:
                    try:
                        AggregationFunction(func.lower())
                        valid_functions.append(func.lower())
                    except ValueError:
                        self.logger.warning(f"Unknown aggregation function '{func}', skipping")
                
                if valid_functions:
                    validated_aggregations[column] = valid_functions
            
            if not validated_aggregations:
                raise ValueError("No valid aggregation functions specified")
            
            # Perform aggregation
            grouped = df.groupby(group_by)
            
            results = {}
            summary_stats = {
                "total_groups": len(grouped),
                "total_records": len(df),
                "group_sizes": {}
            }
            
            # Calculate group sizes
            group_sizes = grouped.size().to_dict()
            summary_stats["group_sizes"] = {str(k): int(v) for k, v in group_sizes.items()}
            
            # Apply aggregations
            for column, functions in validated_aggregations.items():
                if column not in results:
                    results[column] = {}
                
                for func in functions:
                    try:
                        if func == "sum":
                            agg_result = grouped[column].sum()
                        elif func == "mean":
                            agg_result = grouped[column].mean()
                        elif func == "median":
                            agg_result = grouped[column].median()
                        elif func == "min":
                            agg_result = grouped[column].min()
                        elif func == "max":
                            agg_result = grouped[column].max()
                        elif func == "count":
                            agg_result = grouped[column].count()
                        elif func == "std":
                            agg_result = grouped[column].std()
                        elif func == "var":
                            agg_result = grouped[column].var()
                        elif func == "first":
                            agg_result = grouped[column].first()
                        elif func == "last":
                            agg_result = grouped[column].last()
                        elif func == "mode":
                            # Mode requires special handling
                            agg_result = grouped[column].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                        
                        # Convert to dictionary with string keys
                        results[column][func] = {str(k): v for k, v in agg_result.to_dict().items()}
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate {func} for column {column}: {str(e)}")
            
            # Update metrics
            self.metrics.counter("analytics.aggregations.performed").increment()
            self.metrics.histogram("analytics.aggregations.group_count").update(len(grouped))
            
            return {
                "group_by": group_by,
                "aggregations": aggregations,
                "results": results,
                "summary": summary_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Data aggregation failed: {str(e)}")
            self.metrics.counter("analytics.aggregations.errors").increment()
            raise
    
    async def pivot_data(
        self,
        data: List[Dict[str, Any]],
        index: str,
        columns: str,
        values: str,
        aggfunc: str = "mean"
    ) -> Dict[str, Any]:
        """
        Create pivot table from data.
        
        Args:
            data: Input data records
            index: Column to use as index
            columns: Column to use as columns
            values: Column to use as values
            aggfunc: Aggregation function for duplicate entries
            
        Returns:
            Pivoted data with metadata
        """
        try:
            df = pd.DataFrame(data)
            
            # Create pivot table
            pivot_table = df.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc,
                fill_value=0
            )
            
            return {
                "pivot_table": pivot_table.to_dict(),
                "shape": list(pivot_table.shape),
                "index": index,
                "columns": columns,
                "values": values,
                "aggfunc": aggfunc,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Pivot operation failed: {str(e)}")
            raise
    
    def _prepare_data_for_transformation(self, data: Union[List[Any], Dict[str, Any]]) -> pd.DataFrame:
        """Prepare data for transformation operations."""
        if isinstance(data, list):
            # Handle list of values or list of dictionaries
            if data and isinstance(data[0], dict):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame({"value": data})
        elif isinstance(data, dict):
            # Handle dictionary of arrays
            return pd.DataFrame(data)
        else:
            raise ValueError("Data must be a list or dictionary")
    
    async def _apply_transformation(
        self,
        df: pd.DataFrame,
        transform_type: TransformationType,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply specific transformation to DataFrame."""
        
        if transform_type == TransformationType.NORMALIZE:
            return await self._normalize_data(df, parameters)
        elif transform_type == TransformationType.STANDARDIZE:
            return await self._standardize_data(df, parameters)
        elif transform_type == TransformationType.ROBUST_SCALE:
            return await self._robust_scale_data(df, parameters)
        elif transform_type == TransformationType.LOG_TRANSFORM:
            return await self._log_transform_data(df, parameters)
        elif transform_type == TransformationType.SQRT_TRANSFORM:
            return await self._sqrt_transform_data(df, parameters)
        elif transform_type == TransformationType.PCA_TRANSFORM:
            return await self._pca_transform_data(df, parameters)
        elif transform_type == TransformationType.LABEL_ENCODE:
            return await self._label_encode_data(df, parameters)
        elif transform_type == TransformationType.ONE_HOT_ENCODE:
            return await self._one_hot_encode_data(df, parameters)
        else:
            raise ValueError(f"Transformation {transform_type} not implemented")
    
    async def _normalize_data(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data to [0, 1] range."""
        feature_range = parameters.get("feature_range", (0, 1))
        columns = parameters.get("columns", None)
        
        if columns:
            target_columns = [col for col in columns if col in df.columns]
        else:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not target_columns:
            raise ValueError("No numeric columns found for normalization")
        
        scaler = MinMaxScaler(feature_range=feature_range)
        df_transformed = df.copy()
        
        df_transformed[target_columns] = scaler.fit_transform(df[target_columns])
        
        # Store fitted transformer
        transformer_id = f"normalize_{datetime.utcnow().timestamp()}"
        self._fitted_transformers[transformer_id] = {
            "transformer": scaler,
            "columns": target_columns,
            "type": "normalize"
        }
        
        return {
            "transformed_data": df_transformed,
            "metadata": {
                "transformer_id": transformer_id,
                "feature_range": feature_range,
                "transformed_columns": target_columns,
                "original_ranges": {col: {"min": float(df[col].min()), "max": float(df[col].max())} for col in target_columns}
            }
        }
    
    async def _standardize_data(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize data to zero mean and unit variance."""
        columns = parameters.get("columns", None)
        
        if columns:
            target_columns = [col for col in columns if col in df.columns]
        else:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not target_columns:
            raise ValueError("No numeric columns found for standardization")
        
        scaler = StandardScaler()
        df_transformed = df.copy()
        
        df_transformed[target_columns] = scaler.fit_transform(df[target_columns])
        
        # Store fitted transformer
        transformer_id = f"standardize_{datetime.utcnow().timestamp()}"
        self._fitted_transformers[transformer_id] = {
            "transformer": scaler,
            "columns": target_columns,
            "type": "standardize"
        }
        
        return {
            "transformed_data": df_transformed,
            "metadata": {
                "transformer_id": transformer_id,
                "transformed_columns": target_columns,
                "means": {col: float(scaler.mean_[i]) for i, col in enumerate(target_columns)},
                "scales": {col: float(scaler.scale_[i]) for i, col in enumerate(target_columns)}
            }
        }
    
    async def _robust_scale_data(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply robust scaling using median and IQR."""
        columns = parameters.get("columns", None)
        
        if columns:
            target_columns = [col for col in columns if col in df.columns]
        else:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not target_columns:
            raise ValueError("No numeric columns found for robust scaling")
        
        scaler = RobustScaler()
        df_transformed = df.copy()
        
        df_transformed[target_columns] = scaler.fit_transform(df[target_columns])
        
        # Store fitted transformer
        transformer_id = f"robust_scale_{datetime.utcnow().timestamp()}"
        self._fitted_transformers[transformer_id] = {
            "transformer": scaler,
            "columns": target_columns,
            "type": "robust_scale"
        }
        
        return {
            "transformed_data": df_transformed,
            "metadata": {
                "transformer_id": transformer_id,
                "transformed_columns": target_columns,
                "centers": {col: float(scaler.center_[i]) for i, col in enumerate(target_columns)},
                "scales": {col: float(scaler.scale_[i]) for i, col in enumerate(target_columns)}
            }
        }
    
    async def _log_transform_data(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply logarithmic transformation."""
        columns = parameters.get("columns", None)
        base = parameters.get("base", "natural")  # "natural", "10", or "2"
        
        if columns:
            target_columns = [col for col in columns if col in df.columns]
        else:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not target_columns:
            raise ValueError("No numeric columns found for log transformation")
        
        df_transformed = df.copy()
        
        for col in target_columns:
            # Check for non-positive values
            if (df[col] <= 0).any():
                self.logger.warning(f"Column {col} contains non-positive values, adding constant")
                min_val = df[col].min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                data_col = df[col] + offset
            else:
                data_col = df[col]
            
            if base == "natural":
                df_transformed[col] = np.log(data_col)
            elif base == "10":
                df_transformed[col] = np.log10(data_col)
            elif base == "2":
                df_transformed[col] = np.log2(data_col)
            else:
                try:
                    base_num = float(base)
                    df_transformed[col] = np.log(data_col) / np.log(base_num)
                except ValueError:
                    raise ValueError(f"Invalid base for log transformation: {base}")
        
        return {
            "transformed_data": df_transformed,
            "metadata": {
                "transformed_columns": target_columns,
                "base": base
            }
        }
    
    async def _sqrt_transform_data(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply square root transformation."""
        columns = parameters.get("columns", None)
        
        if columns:
            target_columns = [col for col in columns if col in df.columns]
        else:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not target_columns:
            raise ValueError("No numeric columns found for sqrt transformation")
        
        df_transformed = df.copy()
        
        for col in target_columns:
            # Check for negative values
            if (df[col] < 0).any():
                self.logger.warning(f"Column {col} contains negative values, taking absolute value")
                df_transformed[col] = np.sqrt(np.abs(df[col]))
            else:
                df_transformed[col] = np.sqrt(df[col])
        
        return {
            "transformed_data": df_transformed,
            "metadata": {
                "transformed_columns": target_columns
            }
        }
    
    async def _pca_transform_data(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Principal Component Analysis transformation."""
        n_components = parameters.get("n_components", 2)
        columns = parameters.get("columns", None)
        
        if columns:
            target_columns = [col for col in columns if col in df.columns]
        else:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not target_columns:
            raise ValueError("No numeric columns found for PCA transformation")
        
        if len(target_columns) < n_components:
            n_components = len(target_columns)
        
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(df[target_columns])
        
        # Create DataFrame with PCA components
        component_columns = [f"PC{i+1}" for i in range(n_components)]
        df_transformed = df.copy()
        
        # Remove original columns and add PCA components
        df_transformed = df_transformed.drop(columns=target_columns)
        for i, col in enumerate(component_columns):
            df_transformed[col] = transformed_data[:, i]
        
        # Store fitted transformer
        transformer_id = f"pca_{datetime.utcnow().timestamp()}"
        self._fitted_transformers[transformer_id] = {
            "transformer": pca,
            "columns": target_columns,
            "type": "pca"
        }
        
        return {
            "transformed_data": df_transformed,
            "metadata": {
                "transformer_id": transformer_id,
                "n_components": n_components,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "total_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "original_columns": target_columns,
                "component_columns": component_columns
            }
        }
    
    async def _label_encode_data(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply label encoding to categorical columns."""
        columns = parameters.get("columns", None)
        
        if columns:
            target_columns = [col for col in columns if col in df.columns]
        else:
            target_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not target_columns:
            raise ValueError("No categorical columns found for label encoding")
        
        df_transformed = df.copy()
        encoders = {}
        mappings = {}
        
        for col in target_columns:
            encoder = LabelEncoder()
            df_transformed[col] = encoder.fit_transform(df[col].astype(str))
            
            encoders[col] = encoder
            mappings[col] = {str(label): int(code) for code, label in enumerate(encoder.classes_)}
        
        # Store fitted transformers
        transformer_id = f"label_encode_{datetime.utcnow().timestamp()}"
        self._fitted_transformers[transformer_id] = {
            "transformer": encoders,
            "columns": target_columns,
            "type": "label_encode"
        }
        
        return {
            "transformed_data": df_transformed,
            "metadata": {
                "transformer_id": transformer_id,
                "transformed_columns": target_columns,
                "mappings": mappings
            }
        }
    
    async def _one_hot_encode_data(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply one-hot encoding to categorical columns."""
        columns = parameters.get("columns", None)
        
        if columns:
            target_columns = [col for col in columns if col in df.columns]
        else:
            target_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not target_columns:
            raise ValueError("No categorical columns found for one-hot encoding")
        
        df_transformed = df.copy()
        
        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df_transformed, columns=target_columns, prefix=target_columns)
        
        new_columns = [col for col in df_encoded.columns if col not in df.columns]
        
        return {
            "transformed_data": df_encoded,
            "metadata": {
                "transformed_columns": target_columns,
                "new_columns": new_columns,
                "total_new_columns": len(new_columns)
            }
        }
    
    def _add_to_history(self, transformation_record: Dict[str, Any]) -> None:
        """Add transformation to history."""
        self._transformation_history.append(transformation_record)
        
        # Trim history if too large
        if len(self._transformation_history) > self.max_history_size:
            self._transformation_history = self._transformation_history[-self.max_history_size:]
    
    async def get_transformation_history(self) -> List[Dict[str, Any]]:
        """Get transformation history."""
        return self._transformation_history.copy()
    
    async def get_fitted_transformers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about fitted transformers."""
        return {
            transformer_id: {
                "type": info["type"],
                "columns": info["columns"],
                "fitted_at": transformer_id.split("_")[-1]
            }
            for transformer_id, info in self._fitted_transformers.items()
        } 