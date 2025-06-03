"""
Validation utilities for the multi-agent data intelligence platform.

This module provides comprehensive validation functionality including input
sanitization, data quality checks, SQL validation, and validation decorators.
"""

import re
import logging
import ipaddress
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass
from functools import wraps

from pydantic import BaseModel, Field, ConfigDict, validator
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Function
from sqlparse.tokens import Keyword, DML


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, error_code: str = "VALIDATION_ERROR"):
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value
        self.error_code = error_code
        self.timestamp = datetime.utcnow()


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    cleaned_value: Any = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


class DataQualityProfile(BaseModel):
    """Data quality assessment profile."""
    model_config = ConfigDict(extra='forbid')
    
    field_name: str
    total_records: int
    non_null_records: int
    unique_records: int
    
    # Quality metrics
    completeness: float = Field(ge=0.0, le=1.0)  # non-null / total
    uniqueness: float = Field(ge=0.0, le=1.0)    # unique / total
    validity: float = Field(ge=0.0, le=1.0)      # valid / total
    consistency: float = Field(ge=0.0, le=1.0)   # consistent / total
    
    # Specific issues
    format_errors: int = 0
    range_errors: int = 0
    business_rule_errors: int = 0
    
    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score."""
        return (self.completeness + self.uniqueness + self.validity + self.consistency) / 4


class InputValidator:
    """Comprehensive input validation utilities."""
    
    # Common regex patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PHONE_PATTERN = re.compile(r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$')
    URL_PATTERN = re.compile(r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$')
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9]+$')
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_email(self, email: str, required: bool = True) -> ValidationResult:
        """Validate email address format."""
        if not email:
            if required:
                return ValidationResult(is_valid=False, errors=['Email is required'])
            return ValidationResult(is_valid=True)
        
        if not isinstance(email, str):
            return ValidationResult(is_valid=False, errors=['Email must be a string'])
        
        email = email.strip().lower()
        
        if len(email) > 254:  # RFC 5321 limit
            return ValidationResult(is_valid=False, errors=['Email address too long'])
        
        if not self.EMAIL_PATTERN.match(email):
            return ValidationResult(is_valid=False, errors=['Invalid email format'])
        
        # Additional checks
        local, domain = email.rsplit('@', 1)
        if len(local) > 64:  # RFC 5321 limit
            return ValidationResult(is_valid=False, errors=['Email local part too long'])
        
        return ValidationResult(is_valid=True, cleaned_value=email)
    
    def validate_phone(self, phone: str, required: bool = True) -> ValidationResult:
        """Validate phone number format."""
        if not phone:
            if required:
                return ValidationResult(is_valid=False, errors=['Phone number is required'])
            return ValidationResult(is_valid=True)
        
        if not isinstance(phone, str):
            return ValidationResult(is_valid=False, errors=['Phone must be a string'])
        
        # Clean phone number
        cleaned = re.sub(r'\D', '', phone)
        
        if not cleaned:
            return ValidationResult(is_valid=False, errors=['Phone number contains no digits'])
        
        # Validate US phone numbers
        if len(cleaned) == 10:
            formatted = f"({cleaned[:3]}) {cleaned[3:6]}-{cleaned[6:]}"
        elif len(cleaned) == 11 and cleaned[0] == '1':
            formatted = f"1-({cleaned[1:4]}) {cleaned[4:7]}-{cleaned[7:]}"
        else:
            return ValidationResult(is_valid=False, errors=['Invalid phone number length'])
        
        return ValidationResult(is_valid=True, cleaned_value=formatted)
    
    def validate_url(self, url: str, required: bool = True, allowed_schemes: Set[str] = None) -> ValidationResult:
        """Validate URL format."""
        if not url:
            if required:
                return ValidationResult(is_valid=False, errors=['URL is required'])
            return ValidationResult(is_valid=True)
        
        if not isinstance(url, str):
            return ValidationResult(is_valid=False, errors=['URL must be a string'])
        
        url = url.strip()
        allowed_schemes = allowed_schemes or {'http', 'https'}
        
        if not self.URL_PATTERN.match(url):
            return ValidationResult(is_valid=False, errors=['Invalid URL format'])
        
        # Check scheme
        scheme = url.split('://', 1)[0].lower()
        if scheme not in allowed_schemes:
            return ValidationResult(is_valid=False, errors=[f'URL scheme must be one of: {", ".join(allowed_schemes)}'])
        
        return ValidationResult(is_valid=True, cleaned_value=url)
    
    def validate_ip_address(self, ip: str, required: bool = True, version: Optional[int] = None) -> ValidationResult:
        """Validate IP address format."""
        if not ip:
            if required:
                return ValidationResult(is_valid=False, errors=['IP address is required'])
            return ValidationResult(is_valid=True)
        
        if not isinstance(ip, str):
            return ValidationResult(is_valid=False, errors=['IP address must be a string'])
        
        ip = ip.strip()
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check version if specified
            if version and ip_obj.version != version:
                return ValidationResult(is_valid=False, errors=[f'IP address must be IPv{version}'])
            
            return ValidationResult(is_valid=True, cleaned_value=str(ip_obj))
            
        except ValueError as e:
            return ValidationResult(is_valid=False, errors=[f'Invalid IP address: {str(e)}'])
    
    def validate_numeric_range(
        self, 
        value: Union[int, float, str, Decimal], 
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        required: bool = True
    ) -> ValidationResult:
        """Validate numeric value within range."""
        if value is None or value == '':
            if required:
                return ValidationResult(is_valid=False, errors=['Numeric value is required'])
            return ValidationResult(is_valid=True)
        
        try:
            if isinstance(value, str):
                # Try to parse as number
                cleaned = value.strip().replace(',', '')
                numeric_value = float(cleaned)
            else:
                numeric_value = float(value)
            
            errors = []
            if min_value is not None and numeric_value < min_value:
                errors.append(f'Value must be at least {min_value}')
            
            if max_value is not None and numeric_value > max_value:
                errors.append(f'Value must be at most {max_value}')
            
            if errors:
                return ValidationResult(is_valid=False, errors=errors)
            
            return ValidationResult(is_valid=True, cleaned_value=numeric_value)
            
        except (ValueError, TypeError, InvalidOperation) as e:
            return ValidationResult(is_valid=False, errors=[f'Invalid numeric value: {str(e)}'])
    
    def validate_date_range(
        self,
        date_value: Union[str, date, datetime],
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
        date_format: str = '%Y-%m-%d',
        required: bool = True
    ) -> ValidationResult:
        """Validate date within range."""
        if not date_value:
            if required:
                return ValidationResult(is_valid=False, errors=['Date is required'])
            return ValidationResult(is_valid=True)
        
        try:
            if isinstance(date_value, str):
                parsed_date = datetime.strptime(date_value.strip(), date_format).date()
            elif isinstance(date_value, datetime):
                parsed_date = date_value.date()
            elif isinstance(date_value, date):
                parsed_date = date_value
            else:
                return ValidationResult(is_valid=False, errors=['Invalid date type'])
            
            errors = []
            if min_date and parsed_date < min_date:
                errors.append(f'Date must be on or after {min_date}')
            
            if max_date and parsed_date > max_date:
                errors.append(f'Date must be on or before {max_date}')
            
            if errors:
                return ValidationResult(is_valid=False, errors=errors)
            
            return ValidationResult(is_valid=True, cleaned_value=parsed_date)
            
        except ValueError as e:
            return ValidationResult(is_valid=False, errors=[f'Invalid date format: {str(e)}'])
    
    def validate_string_length(
        self,
        text: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        required: bool = True
    ) -> ValidationResult:
        """Validate string length."""
        if not text:
            if required:
                return ValidationResult(is_valid=False, errors=['Text is required'])
            return ValidationResult(is_valid=True)
        
        if not isinstance(text, str):
            return ValidationResult(is_valid=False, errors=['Value must be a string'])
        
        length = len(text)
        errors = []
        
        if min_length is not None and length < min_length:
            errors.append(f'Text must be at least {min_length} characters')
        
        if max_length is not None and length > max_length:
            errors.append(f'Text must be at most {max_length} characters')
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors)
        
        return ValidationResult(is_valid=True, cleaned_value=text)
    
    def validate_choice(
        self,
        value: Any,
        choices: Union[List, Set, Tuple],
        required: bool = True,
        case_sensitive: bool = True
    ) -> ValidationResult:
        """Validate value is in allowed choices."""
        if value is None or value == '':
            if required:
                return ValidationResult(is_valid=False, errors=['Value is required'])
            return ValidationResult(is_valid=True)
        
        check_value = value
        check_choices = choices
        
        if isinstance(value, str) and not case_sensitive:
            check_value = value.lower()
            check_choices = [c.lower() if isinstance(c, str) else c for c in choices]
        
        if check_value not in check_choices:
            return ValidationResult(is_valid=False, errors=[f'Value must be one of: {", ".join(map(str, choices))}'])
        
        return ValidationResult(is_valid=True, cleaned_value=value)


class SQLValidator:
    """
    SQL query validation and security checks.

    IMPORTANT: This validator acts as a policy checker for SQL query structure
    (e.g., disallowing certain keywords in read-only contexts, checking for complexity)
    and aims to detect *some* common anti-patterns or potential vulnerabilities.
    It is **NOT a substitute for proper SQL injection prevention mechanisms
    like parameterized queries (prepared statements)**. It should be used as a
    defense-in-depth measure, not as the primary security control against
    SQL injection.
    """
    
    DANGEROUS_KEYWORDS = {
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
        'EXEC', 'EXECUTE', 'xp_', 'sp_', 'GRANT', 'REVOKE'
    }
    
    ALLOWED_KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
        'GROUP', 'ORDER', 'BY', 'HAVING', 'UNION', 'INTERSECT', 'EXCEPT',
        'WITH', 'AS', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS',
        'NULL', 'DISTINCT', 'TOP', 'LIMIT', 'OFFSET', 'CASE', 'WHEN', 'THEN',
        'ELSE', 'END', 'CAST', 'CONVERT'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_sql_query(self, query: str, read_only: bool = True) -> ValidationResult:
        """Validate SQL query for safety and syntax."""
        if not query or not query.strip():
            return ValidationResult(is_valid=False, errors=['SQL query is required'])
        
        query = query.strip()
        errors = []
        warnings = []
        
        try:
            # Parse the SQL
            parsed = sqlparse.parse(query)
            if not parsed:
                return ValidationResult(is_valid=False, errors=['Could not parse SQL query'])
            
            # Check for dangerous operations if read-only
            if read_only:
                dangerous_found = self._check_dangerous_keywords(query.upper())
                if dangerous_found:
                    errors.append(f'Dangerous SQL operations not allowed: {", ".join(dangerous_found)}')
            
            # Check for SQL injection patterns
            injection_risks = self._check_sql_injection_patterns(query)
            if injection_risks:
                warnings.extend(injection_risks)
            
            # Validate table/column references
            table_issues = self._validate_identifiers(parsed[0])
            if table_issues:
                warnings.extend(table_issues)
            
            # Check query complexity
            complexity_issues = self._check_query_complexity(parsed[0])
            if complexity_issues:
                warnings.extend(complexity_issues)
            
        except Exception as e:
            errors.append(f'SQL parsing error: {str(e)}')
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cleaned_value=query if is_valid else None
        )
    
    def _check_dangerous_keywords(self, query: str) -> List[str]:
        """Check for dangerous SQL keywords."""
        found_dangerous = []
        query_upper = query.upper()
        
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword in query_upper:
                found_dangerous.append(keyword)
        
        return found_dangerous
    
    def _check_sql_injection_patterns(self, query: str) -> List[str]:
        """Check for potential SQL injection patterns."""
        risks = []
        query_lower = query.lower()
        
        # Common injection patterns
        injection_patterns = [
            r"'.*'.*=.*'.*'",  # String comparison that might be injected
            r";\s*--",         # Command termination with comment
            r"union.*select",  # Union-based injection
            r"or.*1\s*=\s*1",  # Always true condition
            r"and.*1\s*=\s*1", # Always true condition
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query_lower):
                risks.append(f'Potential injection pattern detected: {pattern}')
        
        return risks
    
    def _validate_identifiers(self, parsed_query) -> List[str]:
        """Validate table and column identifiers."""
        issues = []
        
        # This is a simplified check - in production you'd validate against actual schema
        try:
            for token in parsed_query.flatten():
                if token.ttype is sqlparse.tokens.Name:
                    identifier = str(token).strip()
                    # Check for suspicious identifier patterns
                    if re.search(r'[;<>]', identifier):
                        issues.append(f'Suspicious identifier: {identifier}')
        except Exception as e:
            issues.append(f'Error validating identifiers: {str(e)}')
        
        return issues
    
    def _check_query_complexity(self, parsed_query) -> List[str]:
        """Check query complexity and performance implications."""
        warnings = []
        
        query_str = str(parsed_query).upper()
        
        # Count joins
        join_count = len(re.findall(r'\bJOIN\b', query_str))
        if join_count > 5:
            warnings.append(f'High number of joins ({join_count}) may impact performance')
        
        # Check for missing WHERE clauses on large tables
        if 'FROM' in query_str and 'WHERE' not in query_str:
            warnings.append('Query without WHERE clause may return large result set')
        
        # Check for SELECT *
        if 'SELECT *' in query_str:
            warnings.append('SELECT * may impact performance - consider specifying columns')
        
        return warnings


class DataQualityChecker:
    """Comprehensive data quality assessment."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def assess_data_quality(self, data: List[Dict[str, Any]], field_definitions: Optional[Dict[str, Dict]] = None) -> Dict[str, DataQualityProfile]:
        """Assess data quality for all fields."""
        if not data:
            return {}
        
        profiles = {}
        total_records = len(data)
        
        # Get all unique fields
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())
        
        for field in all_fields:
            profiles[field] = self._assess_field_quality(data, field, total_records, field_definitions)
        
        return profiles
    
    def _assess_field_quality(self, data: List[Dict[str, Any]], field: str, total_records: int, field_definitions: Optional[Dict[str, Dict]] = None) -> DataQualityProfile:
        """Assess quality for a specific field."""
        values = [record.get(field) for record in data]
        non_null_values = [v for v in values if v is not None and v != '']
        unique_values = set(non_null_values)
        
        # Basic metrics
        completeness = len(non_null_values) / total_records
        uniqueness = len(unique_values) / total_records if total_records > 0 else 0
        
        # Validate values if field definition provided
        validity = 1.0
        consistency = 1.0
        format_errors = 0
        range_errors = 0
        business_rule_errors = 0
        
        if field_definitions and field in field_definitions:
            field_def = field_definitions[field]
            validator = InputValidator()
            
            valid_count = 0
            for value in non_null_values:
                is_valid = True
                
                # Type validation
                expected_type = field_def.get('type')
                if expected_type == 'email':
                    result = validator.validate_email(str(value))
                    if not result.is_valid:
                        format_errors += 1
                        is_valid = False
                elif expected_type == 'phone':
                    result = validator.validate_phone(str(value))
                    if not result.is_valid:
                        format_errors += 1
                        is_valid = False
                elif expected_type == 'numeric':
                    result = validator.validate_numeric_range(
                        value,
                        field_def.get('min_value'),
                        field_def.get('max_value')
                    )
                    if not result.is_valid:
                        range_errors += 1
                        is_valid = False
                
                # Business rules
                business_rules = field_def.get('business_rules', [])
                for rule in business_rules:
                    if not self._check_business_rule(value, rule):
                        business_rule_errors += 1
                        is_valid = False
                
                if is_valid:
                    valid_count += 1
            
            validity = valid_count / len(non_null_values) if non_null_values else 1.0
        
        return DataQualityProfile(
            field_name=field,
            total_records=total_records,
            non_null_records=len(non_null_values),
            unique_records=len(unique_values),
            completeness=completeness,
            uniqueness=uniqueness,
            validity=validity,
            consistency=consistency,
            format_errors=format_errors,
            range_errors=range_errors,
            business_rule_errors=business_rule_errors
        )
    
    def _check_business_rule(self, value: Any, rule: Dict[str, Any]) -> bool:
        """Check a business rule against a value."""
        rule_type = rule.get('type')
        
        if rule_type == 'range':
            try:
                num_value = float(value)
                min_val = rule.get('min')
                max_val = rule.get('max')
                if min_val is not None and num_value < min_val:
                    return False
                if max_val is not None and num_value > max_val:
                    return False
            except (ValueError, TypeError):
                return False
        
        elif rule_type == 'pattern':
            pattern = rule.get('pattern')
            if pattern and not re.match(pattern, str(value)):
                return False
        
        elif rule_type == 'choices':
            choices = rule.get('choices', [])
            if value not in choices:
                return False
        
        return True


def sanitize_input(text: str, max_length: Optional[int] = None, allow_html: bool = False) -> str:
    """
    Sanitize user input for general cleaning and basic security.

    IMPORTANT: This function is intended for general input cleaning (e.g., for
    display purposes, simple string inputs, or preventing basic Cross-Site Scripting (XSS)
    if `allow_html=False`). It is **NOT A DEFENSE AGAINST SQL INJECTION**.
    For constructing SQL queries with variable data, parameterized queries
    (prepared statements) **MUST** be used. Relying on this function for
    SQL security will leave your application vulnerable.
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Strip whitespace
    text = text.strip()
    
    # Remove/escape HTML if not allowed
    if not allow_html:
        text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
    
    # Truncate if max_length specified
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_sql_query(query: str, read_only: bool = True) -> ValidationResult:
    """Convenience function for SQL validation."""
    validator = SQLValidator()
    return validator.validate_sql_query(query, read_only)


def validate_email(email: str, required: bool = True) -> ValidationResult:
    """Convenience function for email validation."""
    validator = InputValidator()
    return validator.validate_email(email, required)


def validate_phone(phone: str, required: bool = True) -> ValidationResult:
    """Convenience function for phone validation."""
    validator = InputValidator()
    return validator.validate_phone(phone, required)


def check_data_quality(data: List[Dict[str, Any]], field_definitions: Optional[Dict[str, Dict]] = None) -> Dict[str, DataQualityProfile]:
    """Convenience function for data quality assessment."""
    checker = DataQualityChecker()
    return checker.assess_data_quality(data, field_definitions)


# Validation decorators

def validate_input(**validators):
    """
    Decorator to validate function inputs.
    
    Usage:
        @validate_input(email=validate_email, age=lambda x: validate_numeric_range(x, 0, 150))
        def create_user(email: str, age: int):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map args to parameter names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator_func in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    if callable(validator_func):
                        result = validator_func(value)
                        if hasattr(result, 'is_valid') and not result.is_valid:
                            raise ValidationError(
                                f"Validation failed for {param_name}: {'; '.join(result.errors)}",
                                field=param_name,
                                value=value
                            )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_fields(*required_fields):
    """
    Decorator to ensure required fields are present in dict arguments.
    
    Usage:
        @require_fields('name', 'email')
        def create_user(user_data: dict):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find dict arguments
            for arg in args:
                if isinstance(arg, dict):
                    missing_fields = [field for field in required_fields if field not in arg or not arg[field]]
                    if missing_fields:
                        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
            
            for value in kwargs.values():
                if isinstance(value, dict):
                    missing_fields = [field for field in required_fields if field not in value or not value[field]]
                    if missing_fields:
                        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Add missing classes that are being imported

class DataQualityValidator(InputValidator):
    """Extended validator for comprehensive data quality checking."""
    
    def __init__(self):
        super().__init__()
        self.quality_rules = {
            'completeness': 0.95,  # 95% of required fields should be present
            'validity': 0.90,      # 90% of values should be valid
            'consistency': 0.95,   # 95% consistency across related fields
            'uniqueness': 1.0      # 100% uniqueness for unique fields
        }
    
    def check_data_quality(self, data: List[Dict[str, Any]], rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        try:
            rules = rules or self.quality_rules
            
            quality_report = {
                'overall_score': 0.0,
                'completeness_score': 0.0,
                'validity_score': 0.0,
                'consistency_score': 0.0,
                'uniqueness_score': 0.0,
                'total_records': len(data),
                'issues': []
            }
            
            if not data:
                return quality_report
            
            # Check completeness
            quality_report['completeness_score'] = self._check_completeness(data, rules)
            
            # Check validity
            quality_report['validity_score'] = self._check_validity(data, rules)
            
            # Check consistency
            quality_report['consistency_score'] = self._check_consistency(data, rules)
            
            # Check uniqueness
            quality_report['uniqueness_score'] = self._check_uniqueness(data, rules)
            
            # Calculate overall score
            quality_report['overall_score'] = (
                quality_report['completeness_score'] * 0.3 +
                quality_report['validity_score'] * 0.3 +
                quality_report['consistency_score'] * 0.2 +
                quality_report['uniqueness_score'] * 0.2
            )
            
            return quality_report
            
        except Exception as e:
            raise DataQualityError(f"Data quality check failed: {e}")
    
    def _check_completeness(self, data: List[Dict[str, Any]], rules: Dict[str, Any]) -> float:
        """Check data completeness."""
        if not data:
            return 1.0
        
        required_fields = rules.get('required_fields', [])
        if not required_fields:
            return 1.0
        
        total_expected = len(data) * len(required_fields)
        total_present = 0
        
        for record in data:
            for field in required_fields:
                if field in record and record[field] is not None:
                    total_present += 1
        
        return total_present / total_expected if total_expected > 0 else 1.0
    
    def _check_validity(self, data: List[Dict[str, Any]], rules: Dict[str, Any]) -> float:
        """Check data validity."""
        if not data:
            return 1.0
        
        field_rules = rules.get('field_rules', {})
        if not field_rules:
            return 1.0
        
        total_checks = 0
        valid_checks = 0
        
        for record in data:
            for field, field_rule in field_rules.items():
                if field in record and record[field] is not None:
                    total_checks += 1
                    if self.validate_format(record[field], field_rule):
                        valid_checks += 1
        
        return valid_checks / total_checks if total_checks > 0 else 1.0
    
    def _check_consistency(self, data: List[Dict[str, Any]], rules: Dict[str, Any]) -> float:
        """Check data consistency."""
        # Simplified consistency check
        return 0.95  # Placeholder
    
    def _check_uniqueness(self, data: List[Dict[str, Any]], rules: Dict[str, Any]) -> float:
        """Check data uniqueness."""
        unique_fields = rules.get('unique_fields', [])
        if not unique_fields or not data:
            return 1.0
        
        total_uniqueness = 0.0
        
        for field in unique_fields:
            values = [record.get(field) for record in data if record.get(field) is not None]
            if values:
                unique_ratio = len(set(values)) / len(values)
                total_uniqueness += unique_ratio
        
        return total_uniqueness / len(unique_fields) if unique_fields else 1.0


class InputSanitizer:
    """Input sanitization utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def sanitize_input(self, value: Any, input_type: str = 'string') -> Any:
        """Sanitize input based on type."""
        try:
            if value is None:
                return None
            
            if input_type == 'string':
                return self.sanitize_string(str(value))
            elif input_type == 'html':
                return self.sanitize_html(str(value))
            elif input_type == 'sql':
                return self.sanitize_sql_identifier(str(value))
            elif input_type == 'email':
                return self.sanitize_email(str(value))
            elif input_type == 'phone':
                return self.sanitize_phone(str(value))
            else:
                return str(value)
                
        except Exception as e:
            self.logger.error(f"Input sanitization failed: {e}")
            raise SanitizationError(f"Failed to sanitize input: {e}")
    
    def sanitize_string(self, value: str) -> str:
        """Basic string sanitization."""
        if not isinstance(value, str):
            value = str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Strip whitespace
        value = value.strip()
        
        # Remove control characters except newline and tab
        import re
        value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        return value
    
    def sanitize_html(self, value: str) -> str:
        """HTML sanitization to prevent XSS."""
        if not isinstance(value, str):
            value = str(value)
        
        # Basic HTML escape
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
            ">": "&gt;",
            "<": "&lt;",
        }
        
        for char, escape in html_escape_table.items():
            value = value.replace(char, escape)
        
        return value
    
    def sanitize_sql_identifier(self, value: str) -> str:
        """Sanitize SQL identifiers."""
        if not isinstance(value, str):
            value = str(value)
        
        # Allow only alphanumeric characters and underscores
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', value)
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        
        return sanitized
    
    def sanitize_email(self, value: str) -> str:
        """Basic email sanitization."""
        if not isinstance(value, str):
            value = str(value)
        
        return value.strip().lower()
    
    def sanitize_phone(self, value: str) -> str:
        """Basic phone number sanitization."""
        if not isinstance(value, str):
            value = str(value)
        
        # Remove all non-digit characters
        import re
        return re.sub(r'[^\d+]', '', value)


# Add missing exception classes
class DataQualityError(ValidationError):
    """Exception for data quality validation errors."""
    pass


class SanitizationError(ValidationError):
    """Exception for input sanitization errors."""
    pass


# Add missing standalone functions that are being imported

def validate_ip_address(ip: str, required: bool = True, version: Optional[int] = None) -> ValidationResult:
    """Standalone IP address validation function."""
    validator = InputValidator()
    return validator.validate_ip_address(ip, required, version)


def validate_json(json_str: str, required: bool = True) -> ValidationResult:
    """Validate JSON string format."""
    import json as json_module
    
    if not required and not json_str:
        return ValidationResult(is_valid=True, value=None)
    
    if not json_str:
        return ValidationResult(
            is_valid=False,
            error="JSON string is required",
            error_code="MISSING_JSON"
        )
    
    try:
        parsed = json_module.loads(json_str)
        return ValidationResult(is_valid=True, value=parsed)
    except json_module.JSONDecodeError as e:
        return ValidationResult(
            is_valid=False,
            error=f"Invalid JSON format: {str(e)}",
            error_code="INVALID_JSON",
            value=json_str
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error=f"JSON validation error: {str(e)}",
            error_code="JSON_ERROR",
            value=json_str
        )


def sanitize_sql_identifier(identifier: str) -> str:
    """Standalone SQL identifier sanitization function."""
    sanitizer = InputSanitizer()
    return sanitizer.sanitize_sql_identifier(identifier)


# Add SecurityValidationError exception
class SecurityValidationError(ValidationError):
    """Exception for security validation errors."""
    pass


# Add SQLValidationError exception
class SQLValidationError(ValidationError):
    """Exception for SQL validation errors."""
    pass


class ValidationHelper:
    """Helper class for common validation operations."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.sql_validator = SQLValidator()
        self.sanitizer = InputSanitizer()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_and_sanitize_input(self, value: Any, input_type: str = 'string', **kwargs) -> ValidationResult:
        """Validate and sanitize input in one step."""
        try:
            # First sanitize
            sanitized_value = self.sanitizer.sanitize_input(value, input_type)
            
            # Then validate based on type
            if input_type == 'email':
                return self.input_validator.validate_email(sanitized_value, **kwargs)
            elif input_type == 'phone':
                return self.input_validator.validate_phone(sanitized_value, **kwargs)
            elif input_type == 'url':
                return self.input_validator.validate_url(sanitized_value, **kwargs)
            elif input_type == 'sql':
                return self.sql_validator.validate_sql_query(sanitized_value, **kwargs)
            else:
                # For string and other types, just return sanitized value
                return ValidationResult(is_valid=True, value=sanitized_value)
                
        except Exception as e:
            self.logger.error(f"Validation helper error: {e}")
            return ValidationResult(
                is_valid=False,
                error=f"Validation failed: {str(e)}",
                error_code="VALIDATION_HELPER_ERROR"
            )
    
    def validate_business_entity(self, entity_data: Dict[str, Any]) -> ValidationResult:
        """Validate business entity data."""
        errors = []
        
        # Check required fields
        required_fields = ['entity_id', 'entity_type', 'entity_name']
        for field in required_fields:
            if field not in entity_data or not entity_data[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate entity_id format
        if 'entity_id' in entity_data:
            entity_id = str(entity_data['entity_id'])
            if not entity_id.replace('_', '').replace('-', '').isalnum():
                errors.append("Entity ID must be alphanumeric with underscores or hyphens")
        
        # Validate confidence if present
        if 'confidence' in entity_data:
            try:
                confidence = float(entity_data['confidence'])
                if not 0.0 <= confidence <= 1.0:
                    errors.append("Confidence must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("Confidence must be a valid number")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            value=entity_data
        )
    
    def validate_query_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate query parameters."""
        errors = []
        sanitized_params = {}
        
        for key, value in params.items():
            # Sanitize parameter name
            clean_key = self.sanitizer.sanitize_sql_identifier(key)
            if clean_key != key:
                errors.append(f"Parameter name '{key}' contains invalid characters")
                continue
            
            # Sanitize parameter value
            if isinstance(value, str):
                sanitized_value = self.sanitizer.sanitize_string(value)
                sanitized_params[clean_key] = sanitized_value
            else:
                sanitized_params[clean_key] = value
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            value=sanitized_params
        ) 