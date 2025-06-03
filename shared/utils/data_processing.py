"""
Data processing utilities for the multi-agent data intelligence platform.

This module provides common data processing, transformation, and validation
functions used across ETL operations and analytics.
"""

import re
import logging
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


@dataclass
class ProcessingStats:
    """Statistics for data processing operations."""
    total_records: int = 0
    processed_records: int = 0
    error_records: int = 0
    transformation_time_ms: float = 0.0
    validation_time_ms: float = 0.0


class DataProcessor:
    """General-purpose data processor with common transformations."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats = ProcessingStats()
    
    def process_batch(
        self,
        data: List[Dict[str, Any]],
        transformations: List[Callable],
        validate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of records with transformations.
        
        Args:
            data: List of records to process
            transformations: List of transformation functions
            validate: Whether to validate records
            
        Returns:
            Processed records
        """
        self.stats.total_records = len(data)
        processed = []
        
        for record in data:
            try:
                # Apply transformations
                for transform in transformations:
                    record = transform(record)
                
                # Validate if requested
                if validate:
                    self._validate_record(record)
                
                processed.append(record)
                self.stats.processed_records += 1
                
            except Exception as e:
                self.logger.error(f"Error processing record: {e}")
                self.stats.error_records += 1
        
        return processed
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Basic record validation."""
        if not isinstance(record, dict):
            raise ValueError("Record must be a dictionary")
        
        if not record:
            raise ValueError("Record cannot be empty")
        
        return True


class DataTransformer:
    """Common data transformation functions."""
    
    @staticmethod
    def clean_string(value: Any) -> Optional[str]:
        """Clean and normalize string values."""
        if value is None:
            return None
        
        if not isinstance(value, str):
            value = str(value)
        
        # Remove extra whitespace and normalize
        cleaned = value.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned if cleaned else None
    
    @staticmethod
    def parse_numeric(value: Any, decimal_places: int = 2) -> Optional[Decimal]:
        """Parse numeric values safely."""
        if value is None or value == '':
            return None
        
        try:
            # Handle string representations
            if isinstance(value, str):
                # Remove common formatting
                cleaned = value.replace(',', '').replace('$', '').replace('%', '')
                cleaned = cleaned.strip()
                
                if not cleaned:
                    return None
            else:
                cleaned = value
            
            result = Decimal(str(cleaned))
            return result.quantize(Decimal('0.' + '0' * decimal_places))
            
        except (InvalidOperation, ValueError):
            return None
    
    @staticmethod
    def parse_boolean(value: Any) -> Optional[bool]:
        """Parse boolean values from various formats."""
        if value is None:
            return None
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, (int, float)):
            return bool(value)
        
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ('true', 't', 'yes', 'y', '1', 'on'):
                return True
            elif value in ('false', 'f', 'no', 'n', '0', 'off'):
                return False
        
        return None
    
    @staticmethod
    def standardize_phone(phone: str) -> Optional[str]:
        """Standardize phone number format."""
        if not phone:
            return None
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"1-({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        
        return phone  # Return original if can't standardize
    
    @staticmethod
    def categorize_amount(amount: Union[int, float, Decimal], 
                         thresholds: List[tuple]) -> str:
        """Categorize amounts based on thresholds."""
        if amount is None:
            return "Unknown"
        
        amount = float(amount)
        
        for threshold, category in sorted(thresholds):
            if amount <= threshold:
                return category
        
        return "High"


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        if not email:
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_date_range(date_value: date, 
                           min_date: Optional[date] = None,
                           max_date: Optional[date] = None) -> bool:
        """Validate date is within range."""
        if min_date and date_value < min_date:
            return False
        
        if max_date and date_value > max_date:
            return False
        
        return True
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float, Decimal],
                              min_value: Optional[float] = None,
                              max_value: Optional[float] = None) -> bool:
        """Validate numeric value is within range."""
        value = float(value)
        
        if min_value is not None and value < min_value:
            return False
        
        if max_value is not None and value > max_value:
            return False
        
        return True
    
    @staticmethod
    def check_required_fields(record: Dict[str, Any], 
                             required_fields: List[str]) -> List[str]:
        """Check for missing required fields."""
        missing = []
        for field in required_fields:
            if field not in record or record[field] is None or record[field] == '':
                missing.append(field)
        return missing


# Utility functions

def format_currency(amount: Union[int, float, Decimal], currency: str = 'USD') -> str:
    """Format amount as currency."""
    if amount is None:
        return "N/A"
    
    amount = float(amount)
    
    if currency.upper() == 'USD':
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: Union[int, float, Decimal], 
                     decimal_places: int = 1) -> str:
    """Format value as percentage."""
    if value is None:
        return "N/A"
    
    value = float(value)
    return f"{value:.{decimal_places}f}%"


def safe_divide(numerator: Union[int, float, Decimal],
                denominator: Union[int, float, Decimal],
                default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return default


def parse_date_flexible(date_string: str) -> Optional[datetime]:
    """Parse date from various string formats."""
    if not date_string:
        return None
    
    # Common date formats to try
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%B %d, %Y',
        '%b %d, %Y',
        '%d %B %Y',
        '%d %b %Y'
    ]
    
    date_string = date_string.strip()
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    return None


def normalize_string(text: str, 
                    lowercase: bool = True,
                    remove_special: bool = False,
                    max_length: Optional[int] = None) -> str:
    """Normalize string with various options."""
    if not text:
        return ""
    
    # Basic cleanup
    normalized = text.strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    
    if lowercase:
        normalized = normalized.lower()
    
    if remove_special:
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', normalized)
    
    if max_length and len(normalized) > max_length:
        normalized = normalized[:max_length].strip()
    
    return normalized


def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect appropriate data types for DataFrame columns."""
    type_suggestions = {}
    
    for column in df.columns:
        series = df[column].dropna()
        
        if series.empty:
            type_suggestions[column] = 'object'
            continue
        
        # Check for numeric types
        try:
            pd.to_numeric(series)
            if series.dtype in ['int64', 'int32']:
                type_suggestions[column] = 'integer'
            else:
                type_suggestions[column] = 'float'
            continue
        except (ValueError, TypeError):
            pass
        
        # Check for datetime
        try:
            pd.to_datetime(series)
            type_suggestions[column] = 'datetime'
            continue
        except (ValueError, TypeError):
            pass
        
        # Check for boolean
        unique_values = series.str.lower().unique() if hasattr(series, 'str') else series.unique()
        if set(unique_values).issubset({'true', 'false', 't', 'f', '1', '0', 'yes', 'no'}):
            type_suggestions[column] = 'boolean'
            continue
        
        # Default to string
        type_suggestions[column] = 'string'
    
    return type_suggestions


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate data profile with statistics and quality metrics."""
    profile = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'columns': {}
    }
    
    for column in df.columns:
        col_profile = {
            'dtype': str(df[column].dtype),
            'non_null_count': df[column].count(),
            'null_count': df[column].isnull().sum(),
            'null_percentage': (df[column].isnull().sum() / len(df)) * 100,
            'unique_count': df[column].nunique(),
            'duplicate_count': df[column].duplicated().sum()
        }
        
        # Add numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(df[column]):
            col_profile.update({
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'q25': df[column].quantile(0.25),
                'q75': df[column].quantile(0.75)
            })
        
        # Add string statistics if applicable
        elif pd.api.types.is_string_dtype(df[column]):
            str_lengths = df[column].str.len()
            col_profile.update({
                'avg_length': str_lengths.mean(),
                'min_length': str_lengths.min(),
                'max_length': str_lengths.max(),
                'empty_strings': (df[column] == '').sum()
            })
        
        profile['columns'][column] = col_profile
    
    return profile


class DataAggregator:
    """Data aggregation utilities."""
    
    def __init__(self, aggregation_functions: Dict[str, str] = None):
        self.aggregation_functions = aggregation_functions or {
            'sum': 'sum',
            'avg': 'mean', 
            'count': 'count',
            'min': 'min',
            'max': 'max'
        }
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def aggregate_data(self, data: List[Dict[str, Any]], group_by: List[str], agg_columns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Aggregate data by specified columns."""
        try:
            if not data:
                return []
            
            # Group data
            groups = {}
            for row in data:
                key = tuple(row.get(col) for col in group_by)
                if key not in groups:
                    groups[key] = []
                groups[key].append(row)
            
            # Aggregate each group
            results = []
            for key, group_data in groups.items():
                result = {}
                
                # Add grouping columns
                for i, col in enumerate(group_by):
                    result[col] = key[i]
                
                # Add aggregated columns
                for col, agg_func in agg_columns.items():
                    values = [row.get(col) for row in group_data if row.get(col) is not None]
                    if values:
                        if agg_func == 'sum':
                            result[f"{col}_{agg_func}"] = sum(values)
                        elif agg_func == 'mean':
                            result[f"{col}_{agg_func}"] = sum(values) / len(values)
                        elif agg_func == 'count':
                            result[f"{col}_{agg_func}"] = len(values)
                        elif agg_func == 'min':
                            result[f"{col}_{agg_func}"] = min(values)
                        elif agg_func == 'max':
                            result[f"{col}_{agg_func}"] = max(values)
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Data aggregation failed: {e}")
            raise AggregationError(f"Failed to aggregate data: {e}")


def process_data_batch(data_batch: List[Dict[str, Any]], processing_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Process a batch of data records."""
    try:
        if not data_batch:
            return []
        
        config = processing_config or {}
        processor = DataProcessor()
        
        results = []
        for record in data_batch:
            try:
                # Apply transformations
                if config.get('normalize_strings', True):
                    for key, value in record.items():
                        if isinstance(value, str):
                            record[key] = normalize_string(value)
                
                # Apply data type conversions
                if 'type_conversions' in config:
                    for field, target_type in config['type_conversions'].items():
                        if field in record:
                            record[field] = processor.convert_type(record[field], target_type)
                
                results.append(record)
                
            except Exception as e:
                logging.warning(f"Failed to process record: {e}")
                if not config.get('skip_errors', True):
                    raise
        
        return results
        
    except Exception as e:
        raise DataProcessingError(f"Batch processing failed: {e}")


def validate_data_quality(data: List[Dict[str, Any]], quality_rules: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate data quality against specified rules."""
    try:
        rules = quality_rules or {}
        validator = DataValidator()
        
        quality_report = {
            'total_records': len(data),
            'valid_records': 0,
            'invalid_records': 0,
            'quality_score': 0.0,
            'issues': [],
            'field_quality': {}
        }
        
        for i, record in enumerate(data):
            record_valid = True
            
            # Check required fields
            for field in rules.get('required_fields', []):
                if field not in record or record[field] is None:
                    quality_report['issues'].append({
                        'record_index': i,
                        'field': field,
                        'issue': 'missing_required_field'
                    })
                    record_valid = False
            
            # Check field formats
            for field, format_rule in rules.get('field_formats', {}).items():
                if field in record and record[field] is not None:
                    if not validator.validate_format(record[field], format_rule):
                        quality_report['issues'].append({
                            'record_index': i,
                            'field': field,
                            'issue': 'invalid_format',
                            'value': record[field]
                        })
                        record_valid = False
            
            if record_valid:
                quality_report['valid_records'] += 1
            else:
                quality_report['invalid_records'] += 1
        
        # Calculate quality score
        if quality_report['total_records'] > 0:
            quality_report['quality_score'] = quality_report['valid_records'] / quality_report['total_records']
        
        return quality_report
        
    except Exception as e:
        raise DataQualityError(f"Data quality validation failed: {e}")


def transform_data_pipeline(data: List[Dict[str, Any]], pipeline_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply a pipeline of data transformations."""
    try:
        transformer = DataTransformer()
        result = data
        
        for step in pipeline_config:
            step_type = step.get('type')
            step_config = step.get('config', {})
            
            if step_type == 'filter':
                # Filter records based on criteria
                result = [record for record in result if transformer.meets_criteria(record, step_config)]
            
            elif step_type == 'map':
                # Transform fields
                for record in result:
                    for field, transformation in step_config.items():
                        if field in record:
                            record[field] = transformer.apply_transformation(record[field], transformation)
            
            elif step_type == 'aggregate':
                # Aggregate data
                aggregator = DataAggregator()
                result = aggregator.aggregate_data(
                    result,
                    step_config.get('group_by', []),
                    step_config.get('aggregations', {})
                )
            
            elif step_type == 'sort':
                # Sort records
                sort_key = step_config.get('key')
                reverse = step_config.get('reverse', False)
                result = sorted(result, key=lambda x: x.get(sort_key), reverse=reverse)
        
        return result
        
    except Exception as e:
        raise TransformationError(f"Data pipeline transformation failed: {e}")


def aggregate_metrics(data: List[Dict[str, Any]], metric_definitions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregated metrics from data."""
    try:
        metrics = {}
        
        for metric_name, metric_config in metric_definitions.items():
            field = metric_config.get('field')
            operation = metric_config.get('operation', 'sum')
            
            values = [record.get(field) for record in data if record.get(field) is not None]
            
            if not values:
                metrics[metric_name] = None
                continue
            
            if operation == 'sum':
                metrics[metric_name] = sum(values)
            elif operation == 'avg':
                metrics[metric_name] = sum(values) / len(values)
            elif operation == 'count':
                metrics[metric_name] = len(values)
            elif operation == 'min':
                metrics[metric_name] = min(values)
            elif operation == 'max':
                metrics[metric_name] = max(values)
            elif operation == 'distinct_count':
                metrics[metric_name] = len(set(values))
            else:
                metrics[metric_name] = None
        
        return metrics
        
    except Exception as e:
        raise AggregationError(f"Metrics aggregation failed: {e}")


def create_data_processor(config: Dict[str, Any] = None) -> DataProcessor:
    """Factory function to create a configured DataProcessor."""
    try:
        config = config or {}
        
        processor = DataProcessor()
        
        # Configure processor based on config
        if 'date_formats' in config:
            processor.date_formats = config['date_formats']
        
        if 'number_formats' in config:
            processor.number_formats = config['number_formats']
        
        return processor
        
    except Exception as e:
        raise DataProcessingError(f"Failed to create data processor: {e}")


# Add missing exception classes
class DataProcessingError(Exception):
    """Base exception for data processing errors."""
    pass


class TransformationError(DataProcessingError):
    """Exception for data transformation errors."""
    pass


class AggregationError(DataProcessingError):
    """Exception for data aggregation errors."""
    pass 