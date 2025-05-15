# Dimensional Model ETL for Snowflake

This package provides a complete ETL (Extract, Transform, Load) solution for creating a dimensional model in Snowflake. The ETL process loads data from previously created staging tables into a star schema dimensional model with dimension and fact tables.

## Overview

This ETL pipeline automates the following process:

1. **Database Creation**: Creates a Snowflake dimensional database if it doesn't exist
2. **Dimension Tables Creation**: Creates dimension tables with proper schemas
3. **Dim_Date Loading**: Loads the date dimension table using the provided SQL script
4. **Dimension Tables Loading**: Populates dimension tables from staging data
5. **Fact Tables Creation**: Creates fact tables with proper schemas and relationships
6. **Fact Tables Loading**: Populates fact tables from staging data and dimension tables

## Dimensional Model Structure

The dimensional model follows a star schema design with these components:

### Dimension Tables
- **Dim_Product**: Product dimension with hierarchies and attributes
- **Dim_Store**: Store dimension with store details
- **Dim_Reseller**: Reseller dimension with reseller details
- **Dim_Location**: Location dimension (shared by Store, Reseller, and Customer)
- **Dim_Customer**: Customer dimension with customer details
- **Dim_Channel**: Sales channel dimension with channel categories
- **Dim_Date**: Date dimension with calendar and fiscal attributes

### Fact Tables
- **Fact_SalesActual**: Sales transactions with measures like amount, quantity, profit
- **Fact_ProductSalesTarget**: Product sales targets
- **Fact_SRCSalesTarget**: Store/Reseller/Channel sales targets

## Data Quality Enhancements

This implementation includes several data quality and robustness enhancements:

### 1. Unknown Member Handling
- Each dimension table has a designated "Unknown" record (e.g., "Unknown Customer", "Unknown Product") 
- These records are automatically inserted during dimension table creation
- Fact tables reference these unknown members when source data contains NULL values or missing references
- This ensures no orphaned facts or broken references in the dimensional model

### 2. Data Type Handling
- Special handling for UUID-style identifiers (CustomerID, ResellerID) using VARCHAR(255) instead of INT
- Proper type casting between numeric and string formats for fields like PostalCode
- Consistent type conversion in JOIN conditions to prevent type mismatch errors

### 3. NULL Value Protection
- Comprehensive use of COALESCE functions to replace NULL values with appropriate defaults
- LEFT JOIN operations instead of INNER JOIN to preserve records with missing data
- WHERE clauses that filter out completely invalid records while keeping partially valid ones
- Default values defined for all fields to prevent NULL handling issues

### 4. Error Handling and Prevention
- Division by zero protection in calculated fields (unit price, profit margins, etc.)
- Data validation filters at each stage of the ETL process
- Detailed error reporting with specific exception messages
- Transaction handling to prevent partial loads

## Setup Instructions

1. **Install required packages**:
   ```bash
   pip install python-dotenv snowflake-connector-python tabulate
   ```

2. **Ensure your `.env` file is set up with credentials**:
   ```
   # Snowflake credentials
   SNOWFLAKE_ACCOUNT=your_account_here
   SNOWFLAKE_USER=your_username_here
   SNOWFLAKE_PASSWORD=your_password_here
   SNOWFLAKE_WAREHOUSE=COMPUTE_WH
   SNOWFLAKE_ROLE=ACCOUNTADMIN
   SNOWFLAKE_SCHEMA=PUBLIC
   
   # User configuration - change this to your name
   USER_NAME=YOUR_NAME_HERE
   ```

3. **Make sure the staging database is already created and populated**:
   The ETL process expects staging tables in the database `IMT577_DW_{USER_NAME}_STAGING`

## Running the Dimensional ETL Process

### Complete ETL Process

Run the entire dimensional ETL process with a single command:
```bash
python -m rahil.run_dimensional_etl 2>&1 | tee rahil/logs/dim_etl_run_$(date +%Y%m%d_%H%M%S).log
```

This will:
1. Create a dimensional database named `IMT577_DW_{USER_NAME}_DIMENSION`
2. Create dimension tables with appropriate schemas
3. Load the Dim_Date table using the DIM_DATE.sql script
4. Load data from staging tables to dimension tables
5. Create fact tables with appropriate schemas and relationships
6. Load data from staging tables to fact tables
7. Log the entire process to a timestamped file in rahil/logs/

### Individual Steps

You can also run each step separately if needed:

1. **Create dimensional database only**:
   ```bash
   python -m rahil.create_dimension_database
   ```

2. **Create dimension tables only**:
   ```bash
   python -m rahil.create_dimension_tables
   ```

3. **Load Dim_Date table only**:
   ```bash
   python -m rahil.load_dim_date
   ```

4. **Load dimension tables only**:
   ```bash
   python -m rahil.load_dimension_tables
   ```

5. **Create fact tables only**:
   ```bash
   python -m rahil.create_fact_tables
   ```

6. **Load fact tables only**:
   ```bash
   python -m rahil.load_fact_tables
   ```

## Data Flow

The ETL process follows this data flow:

1. **Location Data**: Combined from Customer, Store, and Reseller staging tables into Dim_Location
2. **Customer Data**: Mapped from staging to Dim_Customer with references to Dim_Location
3. **Store Data**: Mapped from staging to Dim_Store with references to Dim_Location
4. **Reseller Data**: Mapped from staging to Dim_Reseller with references to Dim_Location
5. **Product Data**: Combined from Product, ProductType, and ProductCategory staging tables into Dim_Product
6. **Channel Data**: Combined from Channel and ChannelCategory staging tables into Dim_Channel
7. **Sales Data**: Combined from SalesHeader and SalesDetail staging tables into Fact_SalesActual
8. **Target Data**: Mapped from TargetDataChannel and TargetDataProduct staging tables into fact tables

## Schema Design Decisions

1. **Shared Location Dimension**: A single Dim_Location table is used for all location data, reducing redundancy
2. **Surrogate Keys**: All dimension tables use identity columns as surrogate keys
3. **Star Schema**: The design follows a pure star schema for simplicity and query performance
4. **Profit Calculations**: Pre-calculated profit metrics are stored in dimension and fact tables
5. **Date Keys**: Date keys are stored in YYYYMMDD format for easy reference
6. **Type Flexibility**: Data type selection optimized for heterogeneous source data (e.g., VARCHAR for IDs)
7. **No Hard Constraints**: Foreign key constraints implemented logically rather than enforced by the database, allowing for more flexible data loading

## Implementation Details

### Dimension Tables Implementation
- Each dimension includes both business keys and surrogate keys
- Consistent naming conventions for all tables and columns
- Default "Unknown" members in all dimensions to handle NULLs and missing data
- Properly derived attributes like profit calculations and full names

### Fact Tables Implementation
- Surrogate key references to all dimensions
- Measures isolated from dimensions for proper star schema design
- Aggregatable measures with consistent data types
- Derived measures pre-calculated during load for query performance

## Logging System

All output from the ETL process is captured in log files stored in the `rahil/logs/` directory. These logs include:
- Detailed information about each step
- Success/failure status for each operation
- Row counts for loaded data
- Sample data from each table
- Any errors encountered

Log files are named with timestamps (e.g., `dim_etl_run_20250508_233839.log`) for easy reference.

## Troubleshooting

### Connection Issues
- Verify your Snowflake credentials in the .env file
- Check that your account name is correct
- Ensure your warehouse and role have the necessary permissions

### Missing Staging Data
- Ensure the staging database exists and is populated
- Check that the staging tables have the expected structure
- Verify the staging database name matches the configuration

### Data Type Mismatch Issues
- If you encounter data type errors, check the source data for unexpected formats
- The ETL handles most common type conversions automatically
- For CustomerID or ResellerID issues, verify they follow UUID format as expected

### NULL or Missing Data Errors
- The implementation handles NULLs with default "Unknown" dimension members
- If specific error messages mention NULL values, check the source data quality

## Security Notes

This package uses `.env` for storing credentials, which provides better security by:
- Keeping sensitive information out of source code
- Allowing different users to have their own credentials
- Preventing credentials from being accidentally committed to version control 