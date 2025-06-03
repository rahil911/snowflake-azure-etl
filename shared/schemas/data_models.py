"""
Data Models Schema
==================

Pydantic models representing Snowflake database structures for type safety
and validation. Based on the existing DDL schemas in private_ddl/ directory.

These models provide:
- Type validation for database operations
- Serialization/deserialization for agent communication
- Schema introspection capabilities
- Data quality validation
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# Base Models
# =============================================================================

class BaseTableModel(BaseModel):
    """Base class for all database table models."""
    
    class Config:
        # Allow population by field name or alias
        populate_by_name = True
        # Use enum values instead of names
        use_enum_values = True
        # Enable ORM mode for SQLAlchemy integration
        from_attributes = True
        # Validate on assignment
        validate_assignment = True


class AuditFieldsMixin(BaseModel):
    """Mixin for common audit fields."""
    created_date: Optional[str] = Field(None, description="Record creation date")
    created_by: Optional[str] = Field(None, description="User who created the record")
    modified_date: Optional[str] = Field(None, description="Last modification date")
    modified_by: Optional[str] = Field(None, description="User who last modified the record")


# =============================================================================
# Staging Table Models
# =============================================================================

class StagingCustomer(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_CUSTOMER table."""
    
    customer_id: str = Field(..., alias="CUSTOMERID", description="Unique customer identifier")
    subsegment_id: Optional[int] = Field(None, alias="SUBSEGMENTID", description="Customer subsegment")
    first_name: Optional[str] = Field(None, alias="FIRSTNAME", description="Customer first name")
    last_name: Optional[str] = Field(None, alias="LASTNAME", description="Customer last name")
    gender: Optional[str] = Field(None, alias="GENDER", description="Customer gender")
    email_address: Optional[str] = Field(None, alias="EMAILADDRESS", description="Email address")
    address: Optional[str] = Field(None, alias="ADDRESS", description="Street address")
    city: Optional[str] = Field(None, alias="CITY", description="City")
    state_province: Optional[str] = Field(None, alias="STATEPROVINCE", description="State or province")
    country: Optional[str] = Field(None, alias="COUNTRY", description="Country")
    postal_code: Optional[int] = Field(None, alias="POSTALCODE", description="Postal code")
    phone_number: Optional[str] = Field(None, alias="PHONENUMBER", description="Phone number")
    
    @validator('email_address')
    def validate_email(cls, v):
        """Basic email validation."""
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v


class StagingProduct(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_PRODUCT table."""
    
    product_id: str = Field(..., alias="PRODUCTID", description="Unique product identifier")
    product_type_id: Optional[int] = Field(None, alias="PRODUCTTYPEID", description="Product type")
    product_category_id: Optional[int] = Field(None, alias="PRODUCTCATEGORYID", description="Product category")
    product_name: Optional[str] = Field(None, alias="PRODUCTNAME", description="Product name")
    product_description: Optional[str] = Field(None, alias="PRODUCTDESCRIPTION", description="Product description")
    product_brand: Optional[str] = Field(None, alias="PRODUCTBRAND", description="Product brand")
    product_sku: Optional[str] = Field(None, alias="PRODUCTSKU", description="Stock keeping unit")
    product_retail_price: Optional[float] = Field(None, alias="PRODUCTRETAILPRICE", description="Retail price")
    product_wholesale_price: Optional[float] = Field(None, alias="PRODUCTWHOLESALEPRICE", description="Wholesale price")
    
    @validator('product_retail_price', 'product_wholesale_price')
    def validate_positive_price(cls, v):
        """Ensure prices are positive."""
        if v is not None and v < 0:
            raise ValueError('Price must be positive')
        return v


class StagingStore(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_STORE table."""
    
    store_id: str = Field(..., alias="STOREID", description="Unique store identifier")
    store_number: Optional[str] = Field(None, alias="STORENUMBER", description="Store number")
    store_name: Optional[str] = Field(None, alias="STORENAME", description="Store name")
    address: Optional[str] = Field(None, alias="ADDRESS", description="Store address")
    city: Optional[str] = Field(None, alias="CITY", description="City")
    state_province: Optional[str] = Field(None, alias="STATEPROVINCE", description="State or province")
    country: Optional[str] = Field(None, alias="COUNTRY", description="Country")
    postal_code: Optional[str] = Field(None, alias="POSTALCODE", description="Postal code")
    store_manager: Optional[str] = Field(None, alias="STOREMANAGER", description="Store manager name")
    store_phone: Optional[str] = Field(None, alias="STOREPHONE", description="Store phone number")
    store_fax: Optional[str] = Field(None, alias="STOREFAX", description="Store fax number")


class StagingSalesHeader(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_SALESHEADER table."""
    
    sales_header_id: str = Field(..., alias="SALESHEADERID", description="Unique sales header identifier")
    date: Optional[str] = Field(None, alias="DATE", description="Sales date")
    customer_id: Optional[str] = Field(None, alias="CUSTOMERID", description="Customer identifier")
    store_id: Optional[str] = Field(None, alias="STOREID", description="Store identifier")
    reseller_id: Optional[str] = Field(None, alias="RESELLERID", description="Reseller identifier")
    channel_id: Optional[str] = Field(None, alias="CHANNELID", description="Channel identifier")


class StagingSalesDetail(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_SALESDETAIL table."""
    
    sales_detail_id: str = Field(..., alias="SALESDETAILID", description="Unique sales detail identifier")
    sales_header_id: Optional[str] = Field(None, alias="SALESHEADERID", description="Sales header identifier")
    product_id: Optional[str] = Field(None, alias="PRODUCTID", description="Product identifier")
    sale_quantity: Optional[int] = Field(None, alias="SALEQUANTITY", description="Quantity sold")
    sale_unit_price: Optional[float] = Field(None, alias="SALEUNITPRICE", description="Unit price")
    sale_amount: Optional[float] = Field(None, alias="SALEAMOUNT", description="Total sale amount")
    
    @validator('sale_quantity')
    def validate_quantity(cls, v):
        """Ensure quantity is positive."""
        if v is not None and v < 0:
            raise ValueError('Quantity must be positive')
        return v


class StagingChannel(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_CHANNEL table."""
    
    channel_id: str = Field(..., alias="CHANNELID", description="Unique channel identifier")
    channel_category_id: Optional[int] = Field(None, alias="CHANNELCATEGORYID", description="Channel category")
    channel_name: Optional[str] = Field(None, alias="CHANNELNAME", description="Channel name")


class StagingChannelCategory(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_CHANNELCATEGORY table."""
    channel_category_id: Optional[int] = Field(None, alias="CHANNELCATEGORYID", description="Channel category identifier")
    channel_category: Optional[str] = Field(None, alias="CHANNELCATEGORY", description="Channel category name")


class StagingProductCategory(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_PRODUCTCATEGORY table."""
    product_category_id: Optional[int] = Field(None, alias="PRODUCTCATEGORYID", description="Product category identifier")
    product_category: Optional[str] = Field(None, alias="PRODUCTCATEGORY", description="Product category name")


class StagingProductType(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_PRODUCTTYPE table."""
    product_type_id: Optional[int] = Field(None, alias="PRODUCTTYPEID", description="Product type identifier")
    product_category_id: Optional[int] = Field(None, alias="PRODUCTCATEGORYID", description="Associated product category identifier")
    product_type: Optional[str] = Field(None, alias="PRODUCTTYPE", description="Product type name")


class StagingReseller(BaseTableModel, AuditFieldsMixin):
    """Model for STAGING_RESELLER table."""
    reseller_id: Optional[str] = Field(None, alias="RESELLERID", description="Unique reseller identifier")
    contact: Optional[str] = Field(None, alias="CONTACT", description="Contact person name")
    email_address: Optional[str] = Field(None, alias="EMAILADDRESS", description="Email address")
    address: Optional[str] = Field(None, alias="ADDRESS", description="Street address")
    city: Optional[str] = Field(None, alias="CITY", description="City")
    state_province: Optional[str] = Field(None, alias="STATEPROVINCE", description="State or province")
    country: Optional[str] = Field(None, alias="COUNTRY", description="Country")
    postal_code: Optional[int] = Field(None, alias="POSTALCODE", description="Postal code") # DDL: INTEGER
    phone_number: Optional[str] = Field(None, alias="PHONENUMBER", description="Phone number")
    reseller_name: Optional[str] = Field(None, alias="RESELLERNAME", description="Reseller name")


class StagingTargetDataChannel(BaseTableModel): # No AuditFieldsMixin
    """Model for STAGING_TARGETDATACHANNEL table."""
    year: Optional[int] = Field(None, alias="YEAR", description="Target year")
    channel_name: Optional[str] = Field(None, alias="CHANNELNAME", description="Channel name")
    target_name: Optional[str] = Field(None, alias="TARGETNAME", description="Name of the target")
    target_sales_amount: Optional[int] = Field(None, alias="TARGETSALESAMOUNT", description="Target sales amount") # DDL: INTEGER


class StagingTargetDataProduct(BaseTableModel): # No AuditFieldsMixin
    """Model for STAGING_TARGETDATAPRODUCT table."""
    product_id: Optional[int] = Field(None, alias="PRODUCTID", description="Product identifier") # DDL: INTEGER
    product: Optional[str] = Field(None, alias="PRODUCT", description="Product name or description")
    year: Optional[int] = Field(None, alias="YEAR", description="Target year")
    sales_quantity_target: Optional[int] = Field(None, alias="SALESQUANTITYTARGET", description="Target sales quantity") # DDL: INTEGER


# =============================================================================
# Dimension Table Models
# =============================================================================

class DimCustomer(BaseTableModel):
    """Model for DIM_CUSTOMER dimension table."""
    
    dim_customer_id: int = Field(..., alias="DIMCUSTOMERID", description="Dimension customer ID")
    customer_id: str = Field(..., alias="CUSTOMERID", description="Business customer ID")
    dim_location_id: Optional[int] = Field(None, alias="DIMLOCATIONID", description="Dimension location ID")
    customer_full_name: Optional[str] = Field(None, alias="CUSTOMERFULLNAME", description="Customer full name")
    first_name: Optional[str] = Field(None, alias="FIRSTNAME", description="First name")
    last_name: Optional[str] = Field(None, alias="LASTNAME", description="Last name")
    gender: Optional[str] = Field(None, alias="GENDER", description="Gender")


class DimProduct(BaseTableModel):
    """Model for DIM_PRODUCT dimension table."""
    
    dim_product_id: int = Field(..., alias="DIMPRODUCTID", description="Dimension product ID")
    product_id: Optional[int] = Field(None, alias="PRODUCTID", description="Business product ID") # Changed type from str
    product_type_id: Optional[int] = Field(None, alias="PRODUCTTYPEID", description="Product type ID from DDL")
    product_category_id: Optional[int] = Field(None, alias="PRODUCTCATEGORYID", description="Product category ID from DDL")
    product_name: Optional[str] = Field(None, alias="PRODUCTNAME", description="Product name")
    # product_brand removed
    product_retail_price: Optional[float] = Field(None, alias="PRODUCTRETAILPRICE", description="Retail price")
    product_wholesale_price: Optional[float] = Field(None, alias="PRODUCTWHOLESALEPRICE", description="Wholesale price")
    product_type: Optional[str] = Field(None, alias="PRODUCTTYPE", description="Product type")
    product_category: Optional[str] = Field(None, alias="PRODUCTCATEGORY", description="Product category")
    product_cost: Optional[float] = Field(None, alias="PRODUCTCOST", description="Product cost")
    product_retail_profit: Optional[float] = Field(None, alias="PRODUCTRETAILPROFIT", description="Product retail profit")
    product_wholesale_unit_profit: Optional[float] = Field(None, alias="PRODUCTWHOLESALEUNITPROFIT", description="Product wholesale unit profit")
    product_profit_margin_unit_percent: Optional[float] = Field(None, alias="PRODUCTPROFITMARGINUNITPERCENT", description="Product profit margin unit percent")


class DimStore(BaseTableModel):
    """Model for DIM_STORE dimension table."""
    
    dim_store_id: int = Field(..., alias="DIMSTOREID", description="Dimension store ID")
    store_id: Optional[int] = Field(None, alias="STOREID", description="Business store ID") # Changed type from str
    dim_location_id: Optional[int] = Field(None, alias="DIMLOCATIONID", description="Dimension location ID")
    source_store_id: Optional[int] = Field(None, alias="SOURCESTOREID", description="Source store ID")
    store_number: Optional[str] = Field(None, alias="STORENUMBER", description="Store number")
    store_name: Optional[str] = Field(None, alias="STORENAME", description="Store name")
    store_manager: Optional[str] = Field(None, alias="STOREMANAGER", description="Store manager")


class DimDate(BaseTableModel):
    """Model for DIM_DATE dimension table."""
    
    date_pkey: int = Field(..., alias="DATE_PKEY", description="Date primary key")
    calendar_date: date = Field(..., alias="DATE", description="Calendar date") # DDL: DATE date not null
    full_date_desc: str = Field(..., alias="FULL_DATE_DESC", description="Full date description") # DDL: varchar(64) not null
    day_num_in_week: int = Field(..., alias="DAY_NUM_IN_WEEK", description="Day number in week") # DDL: number(1) not null
    day_name: str = Field(..., alias="DAY_NAME", description="Day name") # DDL: varchar(10) not null
    day_num_in_month: int = Field(..., alias="DAY_NUM_IN_MONTH", description="Day number in month") # DDL: number(2) not null
    day_num_in_year: int = Field(..., alias="DAY_NUM_IN_YEAR", description="Day number in year") # DDL: number(3) not null
    week_num_in_year: int = Field(..., alias="WEEK_NUM_IN_YEAR", description="Week number in year") # DDL: number(9) not null
    month_name: str = Field(..., alias="MONTH_NAME", description="Month name") # DDL: varchar(10) not null
    month_num_in_year: int = Field(..., alias="MONTH_NUM_IN_YEAR", description="Month number in year") # DDL: number(2) not null
    quarter: int = Field(..., alias="QUARTER", description="Quarter") # DDL: number(1) not null
    year: int = Field(..., alias="YEAR", description="Year") # DDL: number(5) not null
    weekday_ind: str = Field(..., alias="WEEKDAY_IND", description="Indicator if it is a weekday or not") # DDL: varchar(64) not null


class DimChannel(BaseTableModel):
    """Model for DIM_CHANNEL dimension table."""
    
    dim_channel_id: int = Field(..., alias="DIMCHANNELID", description="Dimension channel ID")
    channel_id: Optional[int] = Field(None, alias="CHANNELID", description="Business channel ID") # Changed type from str, DDL: INT
    channel_category_id: Optional[int] = Field(None, alias="CHANNELCATEGORYID", description="Channel category ID") # DDL: INT
    channel_name: Optional[str] = Field(None, alias="CHANNELNAME", description="Channel name") # DDL: VARCHAR
    channel_category: Optional[str] = Field(None, alias="CHANNELCATEGORY", description="Channel category name") # DDL: VARCHAR


class DimLocation(BaseTableModel):
    """Model for DIM_LOCATION dimension table."""
    dim_location_id: int = Field(..., alias="DIMLOCATIONID", description="Dimension location ID (Primary Key)")
    address: Optional[str] = Field(None, alias="ADDRESS", description="Street address")
    city: Optional[str] = Field(None, alias="CITY", description="City name")
    postal_code: Optional[str] = Field(None, alias="POSTALCODE", description="Postal code")
    state_province: Optional[str] = Field(None, alias="STATE_PROVINCE", description="State or province name")
    country: Optional[str] = Field(None, alias="COUNTRY", description="Country name")


class DimReseller(BaseTableModel):
    """Model for DIM_RESELLER dimension table."""
    dim_reseller_id: int = Field(..., alias="DIMRESELLERID", description="Dimension reseller ID (Primary Key)")
    reseller_id: Optional[str] = Field(None, alias="RESELLERID", description="Business reseller ID")
    dim_location_id: Optional[int] = Field(None, alias="DIMLOCATIONID", description="Dimension location ID (Foreign Key)")
    reseller_name: Optional[str] = Field(None, alias="RESELLERNAME", description="Reseller name")
    contact_name: Optional[str] = Field(None, alias="CONTACTNAME", description="Contact person's name")
    phone_number: Optional[str] = Field(None, alias="PHONENUMBER", description="Phone number")
    email: Optional[str] = Field(None, alias="EMAIL", description="Email address")


# =============================================================================
# Fact Table Models
# =============================================================================

class FactSalesActual(BaseTableModel):
    """Model for FACT_SALESACTUAL fact table."""
    
    dim_product_id: int = Field(..., alias="DIMPRODUCTID", description="Product dimension ID")
    dim_store_id: int = Field(..., alias="DIMSTOREID", description="Store dimension ID")
    dim_reseller_id: Optional[int] = Field(None, alias="DIMRESELLERID", description="Reseller dimension ID")
    dim_customer_id: int = Field(..., alias="DIMCUSTOMERID", description="Customer dimension ID")
    dim_channel_id: int = Field(..., alias="DIMCHANNELID", description="Channel dimension ID")
    dim_sale_date_id: int = Field(..., alias="DIMSALEDATEID", description="Sale date dimension ID")
    dim_location_id: Optional[int] = Field(None, alias="DIMLOCATIONID", description="Location dimension ID")
    sales_header_id: int = Field(..., alias="SALESHEADERID", description="Sales header ID")
    sales_detail_id: int = Field(..., alias="SALESDETAILID", description="Sales detail ID")
    sale_amount: Optional[float] = Field(None, alias="SALEAMOUNT", description="Sale amount")
    sale_quantity: Optional[int] = Field(None, alias="SALEQUANTITY", description="Sale quantity")
    sale_unit_price: Optional[float] = Field(None, alias="SALEUNITPRICE", description="Sale unit price")
    sale_extended_cost: Optional[float] = Field(None, alias="SALEEXTENDEDCOST", description="Extended cost")
    sale_total_profit: Optional[float] = Field(None, alias="SALETOTALPROFIT", description="Total profit")
    
    @validator('sale_amount', 'sale_unit_price', 'sale_extended_cost')
    def validate_monetary_values(cls, v):
        """Ensure monetary values are reasonable."""
        if v is not None and v < 0:
            raise ValueError('Monetary values should be positive')
        return v


class FactProductSalesTarget(BaseTableModel):
    """Model for FACT_PRODUCTSALESTARGET fact table."""
    dim_product_id: int = Field(..., alias="DIMPRODUCTID", description="Dimension Product ID")
    dim_target_date_id: int = Field(..., alias="DIMTARGETDATEID", description="Dimension Target Date ID")
    product_target_sales_quantity: Optional[int] = Field(None, alias="PRODUCTTARGETSALESQUANTITY", description="Target sales quantity for the product")


class FactSrcSalesTarget(BaseTableModel):
    """Model for FACT_SRCSALESTARGET fact table."""
    dim_store_id: int = Field(..., alias="DIMSTOREID", description="Dimension Store ID")
    dim_reseller_id: int = Field(..., alias="DIMRESELLERID", description="Dimension Reseller ID")
    dim_channel_id: int = Field(..., alias="DIMCHANNELID", description="Dimension Channel ID")
    dim_target_date_id: int = Field(..., alias="DIMTARGETDATEID", description="Dimension Target Date ID")
    sales_target_amount: Optional[float] = Field(None, alias="SALESTARGETAMOUNT", description="Target sales amount")


# =============================================================================
# Query Result Models
# =============================================================================

class QueryMetadata(BaseModel):
    """Metadata about a database query execution."""
    
    query_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    execution_time: Optional[float] = Field(None, description="Query execution time in seconds")
    row_count: Optional[int] = Field(None, description="Number of rows returned")
    columns: List[str] = Field(default_factory=list, description="Column names")
    query_text: Optional[str] = Field(None, description="SQL query text")
    cache_hit: bool = Field(default=False, description="Whether result was cached")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryResult(BaseModel):
    """Container for database query results."""
    
    data: List[Dict[str, Any]] = Field(..., description="Query result data")
    metadata: QueryMetadata = Field(..., description="Query execution metadata")
    success: bool = Field(default=True, description="Whether query was successful")
    error_message: Optional[str] = Field(None, description="Error message if query failed")
    
    def to_pandas(self):
        """Convert result to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(self.data)
        except ImportError:
            raise ImportError("pandas is required to convert results to DataFrame")


# =============================================================================
# Data Quality Models
# =============================================================================

class DataQualityRule(BaseModel):
    """Definition of a data quality rule."""
    
    rule_id: str = Field(..., description="Unique rule identifier")
    rule_name: str = Field(..., description="Human-readable rule name")
    table_name: str = Field(..., description="Target table name")
    column_name: Optional[str] = Field(None, description="Target column (None for table-level rules)")
    rule_type: str = Field(..., description="Type of rule (completeness, validity, etc)")
    condition: str = Field(..., description="SQL condition for the rule")
    threshold: Optional[float] = Field(None, description="Acceptable threshold (0-1)")
    severity: str = Field(default="warning", description="Rule severity level")
    is_active: bool = Field(default=True, description="Whether rule is active")


class DataQualityResult(BaseModel):
    """Result of a data quality check."""
    
    rule_id: str = Field(..., description="Rule that was checked")
    table_name: str = Field(..., description="Table that was checked")
    passed: bool = Field(..., description="Whether the rule passed")
    score: Optional[float] = Field(None, description="Quality score (0-1)")
    failed_records: Optional[int] = Field(None, description="Number of failed records")
    total_records: Optional[int] = Field(None, description="Total number of records")
    details: Optional[str] = Field(None, description="Additional details")
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class DataQualityReport(BaseModel):
    """Comprehensive data quality report."""
    
    report_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    table_name: str = Field(..., description="Table analyzed")
    overall_score: float = Field(..., description="Overall quality score (0-1)")
    results: List[DataQualityResult] = Field(..., description="Individual rule results")
    summary: str = Field(..., description="Quality summary")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class BusinessMetric(BaseModel):
    """Business metric definition and value."""
    
    metric_id: str = Field(..., description="Unique metric identifier")
    metric_name: str = Field(..., description="Human-readable metric name")
    metric_type: str = Field(..., description="Type of metric (revenue, count, ratio, etc)")
    value: Union[float, int, str] = Field(..., description="Metric value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    period: Optional[str] = Field(None, description="Time period for the metric")
    target: Optional[float] = Field(None, description="Target value")
    variance: Optional[float] = Field(None, description="Variance from target")
    trend: Optional[str] = Field(None, description="Trend direction (up, down, stable)")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class AnalysisResult(BaseModel):
    """Result of data analysis operation."""
    
    analysis_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    analysis_type: str = Field(..., description="Type of analysis performed")
    query_result: Optional[QueryResult] = Field(None, description="Raw query results")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    metrics: List[BusinessMetric] = Field(default_factory=list, description="Calculated metrics")
    confidence: float = Field(default=0.8, description="Confidence in results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class InsightRecommendation(BaseModel):
    """Business insight and recommendation."""
    
    recommendation_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    category: str = Field(..., description="Category (operational, strategic, etc)")
    priority: str = Field(default="medium", description="Priority level")
    impact: str = Field(..., description="Expected impact")
    effort: str = Field(..., description="Implementation effort")
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Supporting analysis")
    confidence: float = Field(default=0.8, description="Confidence in recommendation")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ETLPipelineStatus(BaseModel):
    """Status of ETL pipeline execution."""
    
    pipeline_id: str = Field(..., description="Pipeline identifier")
    pipeline_name: str = Field(..., description="Pipeline name")
    status: str = Field(..., description="Current status (running, completed, failed)")
    stage: str = Field(..., description="Current stage")
    progress: float = Field(default=0.0, description="Progress percentage (0-1)")
    records_processed: int = Field(default=0, description="Number of records processed")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    started_at: Optional[datetime] = Field(None, description="Pipeline start time")
    completed_at: Optional[datetime] = Field(None, description="Pipeline completion time")
    duration: Optional[float] = Field(None, description="Execution duration in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BusinessEntity(BaseModel):
    """Business entity for context extraction."""
    
    entity_id: str = Field(..., description="Unique entity identifier")
    entity_type: str = Field(..., description="Type of entity (customer, product, store, etc)")
    entity_name: str = Field(..., description="Human-readable entity name")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    confidence: float = Field(default=1.0, description="Confidence in entity extraction")
    source_text: Optional[str] = Field(None, description="Source text where entity was found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DataQualityMetric(BaseModel):
    """Data quality metric definition."""
    
    metric_id: str = Field(..., description="Unique metric identifier")
    metric_name: str = Field(..., description="Human-readable metric name")
    table_name: str = Field(..., description="Target table")
    column_name: Optional[str] = Field(None, description="Target column (None for table-level)")
    metric_type: str = Field(..., description="Type of metric (completeness, accuracy, etc)")
    value: float = Field(..., description="Metric value (0-1)")
    threshold: Optional[float] = Field(None, description="Acceptable threshold")
    status: str = Field(..., description="Status (pass, fail, warning)")
    description: str = Field(..., description="Metric description")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Schema Introspection Models
# =============================================================================

class ColumnInfo(BaseModel):
    """Information about a database column."""
    
    column_name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Column data type")
    is_nullable: bool = Field(..., description="Whether column allows nulls")
    default_value: Optional[str] = Field(None, description="Default value")
    max_length: Optional[int] = Field(None, description="Maximum length for string types")
    precision: Optional[int] = Field(None, description="Precision for numeric types")
    scale: Optional[int] = Field(None, description="Scale for numeric types")
    is_primary_key: bool = Field(default=False, description="Whether column is primary key")
    is_foreign_key: bool = Field(default=False, description="Whether column is foreign key")
    foreign_key_table: Optional[str] = Field(None, description="Referenced table if foreign key")


class TableSchema(BaseModel):
    """Schema information for a database table."""
    
    table_name: str = Field(..., description="Table name")
    schema_name: str = Field(..., description="Schema name")
    table_type: str = Field(..., description="Table type (TABLE, VIEW, etc)")
    columns: List[ColumnInfo] = Field(..., description="Column information")
    row_count: Optional[int] = Field(None, description="Approximate row count")
    created_date: Optional[datetime] = Field(None, description="Table creation date")
    last_modified: Optional[datetime] = Field(None, description="Last modification date")


class DatabaseSchema(BaseModel):
    """Complete database schema information."""
    
    database_name: str = Field(..., description="Database name")
    schemas: Dict[str, List[TableSchema]] = Field(..., description="Schema definitions")
    version: Optional[str] = Field(None, description="Database version")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def get_table(self, schema_name: str, table_name: str) -> Optional[TableSchema]:
        """Get a specific table schema."""
        if schema_name in self.schemas:
            for table in self.schemas[schema_name]:
                if table.table_name == table_name:
                    return table
        return None
    
    def get_all_tables(self) -> List[TableSchema]:
        """Get all tables across all schemas."""
        all_tables = []
        for schema_tables in self.schemas.values():
            all_tables.extend(schema_tables)
        return all_tables


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_for_table(table_name: str) -> Optional[type]:
    """Get the Pydantic model class for a given table name."""
    table_models = {
        'STAGING_CUSTOMER': StagingCustomer,
        'STAGING_PRODUCT': StagingProduct,
        'STAGING_STORE': StagingStore,
        'STAGING_SALESHEADER': StagingSalesHeader,
        'STAGING_SALESDETAIL': StagingSalesDetail,
        'STAGING_CHANNEL': StagingChannel,
        'STAGING_CHANNELCATEGORY': StagingChannelCategory,
        'STAGING_PRODUCTCATEGORY': StagingProductCategory,
        'STAGING_PRODUCTTYPE': StagingProductType,
        'STAGING_RESELLER': StagingReseller,
        'STAGING_TARGETDATACHANNEL': StagingTargetDataChannel,
        'STAGING_TARGETDATAPRODUCT': StagingTargetDataProduct,
        'DIM_CUSTOMER': DimCustomer,
        'DIM_PRODUCT': DimProduct,
        'DIM_STORE': DimStore,
        'DIM_DATE': DimDate,
        'DIM_CHANNEL': DimChannel,
        'DIM_LOCATION': DimLocation,
        'DIM_RESELLER': DimReseller,
        'FACT_SALESACTUAL': FactSalesActual,
        'FACT_PRODUCTSALESTARGET': FactProductSalesTarget,
        'FACT_SRCSALESTARGET': FactSrcSalesTarget,
    }
    return table_models.get(table_name.upper())


def validate_record(table_name: str, record_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a record against the appropriate table model."""
    model_class = get_model_for_table(table_name)
    if not model_class:
        raise ValueError(f"No model found for table: {table_name}")
    
    try:
        validated_record = model_class(**record_data)
        return validated_record.dict(by_alias=True)
    except Exception as e:
        raise ValueError(f"Validation failed for {table_name}: {str(e)}")


def get_table_columns(table_name: str) -> List[str]:
    """Get the column names for a table."""
    model_class = get_model_for_table(table_name)
    if not model_class:
        return []
    
    # Get field names, preferring aliases
    columns = []
    for field_name, field_info in model_class.__fields__.items():
        alias = field_info.alias or field_name
        columns.append(alias)
    
    return columns 