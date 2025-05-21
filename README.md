# Snowflake ETL Framework with Schema Versioning

This package provides a complete ETL (Extract, Transform, Load) solution for Snowflake that manages both staging data and a dimensional data warehouse. The pipeline processes data from Azure Blob Storage into staging tables, then transforms it into a star schema dimensional model with full versioning and customization support.

## Overview

This ETL framework provides:

1. **Schema Management**:
   - SQLAlchemy models for all tables in `rahil/schemas/`
   - Alembic migrations to version and deploy schema changes
   - Reusable, customizable table definitions 

2. **Staging Layer ETL**:
   - Database creation 
   - External Azure Blob stages setup
   - Staging table creation
   - Data loading from stages to tables
   - Data verification

3. **Dimensional Model ETL**:
   - Dimension tables creation and loading
   - Date dimension generation
   - Fact tables creation and loading
   - Star schema implementation
   - Data quality enhancements

## Schema Management with SQLAlchemy and Alembic

A key feature of this framework is its modular, version-controlled approach to table definitions:

### Schema Definition Structure

```
rahil/schemas/
├── __init__.py                  # Base declarative setup
├── channel.py                   # Staging table definitions (one per file)
├── channel_category.py
├── customer.py
├── ...
├── dimension/                   # Dimension table definitions
│   ├── __init__.py
│   ├── channel.py
│   ├── customer.py
│   ├── date.py
│   ├── location.py
│   ├── product.py
│   ├── reseller.py
│   └── store.py
└── fact/                        # Fact table definitions
    ├── __init__.py
    ├── product_sales_target.py
    ├── sales_actual.py
    └── src_sales_target.py
```

### Customizing Table Schemas as a Student

You can easily override the default table definitions without modifying the original code:

1. **Create your custom schema definitions**:

   Create a `.schema_customizations/` directory (this is in `.gitignore`):
   ```bash
   mkdir -p .schema_customizations/
   ```

2. **Override an existing table definition**:

   For example, to add a new column to the Customer table:
   ```python
   # .schema_customizations/customer.py
   from sqlalchemy import Column, Integer, String, Float
   from sqlalchemy.ext.declarative import declarative_base
   
   Base = declarative_base()
   
   class Customer(Base):
       __tablename__ = 'STAGING_CUSTOMER'
       
       # Original columns
       CUSTOMERID = Column(String, primary_key=True)
       SUBSEGMENTID = Column(Integer)
       FIRSTNAME = Column(String)
       LASTNAME = Column(String)
       GENDER = Column(String)
       
       # Add your custom column
       CUSTOMER_SEGMENT = Column(String)
   ```

3. **Run migrations to apply changes**:
   ```bash
   alembic revision --autogenerate -m "Add customer segment column"
   alembic upgrade head
   ```

## Project Structure

```
rahil/
├── __init__.py               # Package initialization
├── config.py                 # Staging configuration
├── dim_config.py             # Dimensional model configuration
├── connection.py             # Snowflake connection handling
├── create_database.py        # Staging database creation
├── create_dimension_database.py # Dimension database creation
├── create_stages.py          # External stages creation
├── create_tables.py          # Staging tables creation
├── create_dimension_tables.py # Dimension tables creation
├── create_fact_tables.py     # Fact tables creation
├── load_data.py              # Staging data loading
├── load_dim_date.py          # Date dimension loading
├── load_dimension_tables.py  # Dimension tables loading
├── load_fact_tables.py       # Fact tables loading
├── view_sample_data.py       # Sample data display
├── run_etl.py                # Staging ETL runner
├── run_dimensional_etl.py    # Dimensional ETL runner
├── schemas/                  # SQLAlchemy table definitions
├── example.env               # Environment variables template
├── .env                      # Your actual credentials (not in git)
└── logs/                     # ETL process logs
    └── etl_run_*.log         # Timestamped log files
```

## Data Architecture

### Staging Layer
- Raw data loaded from Azure Blob Storage
- 12 staging tables reflecting source format
- Minimal transformations
- Source tables for the dimensional model

### Dimensional Model
- Star schema design with dimension and fact tables
- Date dimension for time analysis
- Location dimension shared across entities
- Pre-calculated measures
- "Unknown" members for data quality

## Setup Instructions

1. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file based on the template**:
   ```bash
   cp rahil/example.env rahil/.env
   ```

3. **Edit the `.env` file with your credentials**:
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
   
   # Azure Storage account
   AZURE_STORAGE_ACCOUNT=your_storage_account.blob.core.windows.net
   ```

## Running the ETL Processes

### Staging ETL

Run the entire staging ETL process:
```bash
python -m rahil.run_etl 2>&1 | tee rahil/logs/etl_run_$(date +%Y%m%d_%H%M%S).log
```

This creates a database named `IMT577_DW_{USER_NAME}_STAGING` with:
- External stages for 12 entities
- 12 staging tables with data from Azure Blob Storage

### Dimensional Model ETL

Run the dimensional model ETL process:
```bash
python -m rahil.run_dimensional_etl 2>&1 | tee rahil/logs/dim_etl_run_$(date +%Y%m%d_%H%M%S).log
```

This creates a database named `IMT577_DW_{USER_NAME}_DIMENSION` with:
- Dimension tables (Product, Store, Reseller, Customer, etc.)
- Fact tables (SalesActual, ProductSalesTarget, SRCSalesTarget)
- Data loaded from the staging tables

## Schema Migrations with Alembic

This project uses Alembic for database schema migrations, allowing you to version control your schema changes:

### View Current Migration Status

```bash
alembic current
```

### Create a New Migration

After modifying or adding schema files:

```bash
alembic revision --autogenerate -m "Description of your changes"
```

### Apply Migrations

```bash
alembic upgrade head
```

### Revert Migrations

```bash
alembic downgrade -1  # Go back one revision
```

## Individual ETL Steps

### Staging Layer

1. **Create database only**:
   ```bash
   python -m rahil.create_database
   ```

2. **Create external stages only**:
   ```bash
   python -m rahil.create_stages
   ```

3. **Create staging tables only**:
   ```bash
   python -m rahil.create_tables
   ```

4. **Load data from stages to tables only**:
   ```bash
   python -m rahil.load_data
   ```

5. **View sample data only**:
   ```bash
   python -m rahil.view_sample_data
   ```

### Dimensional Model Layer

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

## Dimensional Model Design

### Dimension Tables
- **Dim_Product**: Product hierarchy and attributes
- **Dim_Store**: Store details and location reference
- **Dim_Reseller**: Reseller details and location reference
- **Dim_Location**: Shared location dimension
- **Dim_Customer**: Customer details and location reference
- **Dim_Channel**: Sales channel hierarchy
- **Dim_Date**: Date dimension with calendar attributes

### Fact Tables
- **Fact_SalesActual**: Sales transactions with measures
- **Fact_ProductSalesTarget**: Product sales targets
- **Fact_SRCSalesTarget**: Store/Reseller/Channel sales targets

## Data Quality Enhancements

- **Unknown Member Handling**: Each dimension has an "Unknown" record
- **Data Type Handling**: Proper type conversion
- **NULL Value Protection**: COALESCE functions and LEFT JOINs
- **Error Prevention**: Division by zero protection and validation

## Customizing for Your Project

### Using a Different User Name

Update the USER_NAME in your .env file:
```
USER_NAME=YOUR_NAME
```

### Adding New Entities or Tables

1. Create a new schema definition file in `rahil/schemas/`
2. Add it to the imports in `rahil/schemas/__init__.py`
3. Generate and apply migrations with Alembic
4. Update the ETL process to handle the new entity

### Advanced Schema Customizations

For advanced customizations:

1. Create a `.schema_customizations/` directory
2. Add Python files with SQLAlchemy models that mirror the ones in `rahil/schemas/`
3. Your custom models will override the defaults
4. Generate migrations to apply your changes

For more customization examples see [SCHEMA_CUSTOMIZATION.md](SCHEMA_CUSTOMIZATION.md).

## Troubleshooting

### Connection Issues
- Verify your Snowflake credentials
- Check that your warehouse and role have the necessary permissions

### Migration Errors
- Check that your SQLAlchemy models are properly defined
- Verify that there are no syntax errors in your schema files
- Use `alembic history` to see migration history

### Data Loading Issues
- Verify that Azure Blob Storage paths are correct
- Check that your stage definitions match the expected format
- Ensure you have appropriate permissions

## Security Notes

This package uses `.env` for storing credentials, which provides better security by:
- Keeping sensitive information out of source code
- Allowing different users to have their own credentials
- Preventing credentials from being accidentally committed to version control

For production use, consider:
- Using environment variables instead of .env files
- Implementing key-pair authentication
- Using role-based access control 