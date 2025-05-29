# Snowflake ETL Pipeline for Azure Blob Storage

This package provides a complete ETL (Extract, Transform, Load) solution for loading data from Azure Blob Storage into Snowflake staging tables. The pipeline creates a database, sets up external stages, creates staging tables, and loads data with full logging and sample data verification.

## Overview

This ETL pipeline automates the following process:

1. **Database Creation**: Creates a Snowflake database if it doesn't exist
2. **External Stages**: Sets up Azure Blob Storage external stages for 12 entities
3. **Table Creation**: Creates 12 staging tables with proper schemas
4. **Data Loading**: Loads data from external stages into staging tables
5. **Data Verification**: Displays sample data from each table for verification

## Project Structure

```
rahil/
├── __init__.py             # Package initialization
├── config.py               # Configuration settings (loads from .env)
├── connection.py           # Snowflake connection handling
├── create_database.py      # Database creation script
├── create_stages.py        # External stages creation
├── create_tables.py        # Staging tables creation
├── load_data.py            # Data loading from stages to tables
├── view_sample_data.py     # Sample data display
├── run_etl.py              # Main ETL runner
├── example.env             # Environment variables template
├── .env                    # Your actual credentials (not in git)
├── .gitignore              # Prevents .env from being committed
├── README.md               # Documentation
└── logs/                   # Directory for ETL process logs
    └── etl_run_*.log       # Timestamped log files
```

### Script Overview

| Script | Purpose |
|--------|---------|
|`create_database.py`|Create the staging database and schema if they do not already exist.|
|`create_stages.py`|Define external stages pointing to Azure Blob Storage for each entity.|
|`create_tables.py`|Create staging tables from SQL definitions in `private_ddl/`; copies example files if none are present.|
|`load_data.py`|Load data from the external stages into the staging tables.|
|`view_sample_data.py`|Display sample rows from each staging table.|
|`run_etl.py`|Run all staging steps in order.|
|`create_dimension_database.py`|Create the dimensional model database.|
|`create_dimension_tables.py`|Create all dimension tables and insert default "Unknown" members.|
|`load_dim_date.py`|Create and populate the `Dim_Date` table.|
|`load_dimension_tables.py`|Transform data from staging tables into dimension tables.|
|`create_fact_tables.py`|Create fact tables from SQL files.|
|`load_fact_tables.py`|Populate fact tables from staging and dimension data.|
|`run_dimensional_etl.py`|Run the dimensional ETL sequence.|
|`verify_sql.py`|Validate that SQL definition files match the code.|

## Entities Processed

This ETL pipeline processes entities that you configure in your `.env` file. You can process any subset of the available entities based on what's available in your Azure Blob Storage.

### Available Entities
The system supports these entities:
- channel
- channelcategory  
- customer
- product
- productcategory
- producttype
- reseller
- salesdetail
- salesheader
- store
- targetdatachannel
- targetdataproduct

### Configuring Entities

In your `.env` file, specify which entities you want to process:

```bash
# Process all entities
ENTITIES=channel,channelcategory,customer,product,productcategory,producttype,reseller,salesdetail,salesheader,store,targetdatachannel,targetdataproduct

# Process only a few entities
ENTITIES=channel,customer,product,salesdetail,salesheader

# Process just core sales data
ENTITIES=customer,product,salesdetail,salesheader
```

**Important**: The entity names must match your Azure Blob Storage container/folder names exactly.

## Azure Blob Storage Structure

Your Azure Blob Storage should be organized with containers/folders that match the entity names you configure in your `.env` file.

### Expected Structure

```
your_storage_account.blob.core.windows.net/
├── channel/
│   └── channel.csv (or other CSV files)
├── customer/
│   └── customer.csv
├── product/
│   └── product.csv
├── salesdetail/
│   └── salesdetail.csv
├── salesheader/
│   └── salesheader.csv
└── ... (other entity folders as configured)
```

### URL Pattern

The system creates Snowflake external stages using this pattern:
```
azure://{AZURE_STORAGE_ACCOUNT}/{ENTITY_NAME}
```

**Example**: If your storage account is `mystorageaccount.blob.core.windows.net` and you have `ENTITIES=channel,customer`, the system creates:
- `azure://mystorageaccount.blob.core.windows.net/channel`
- `azure://mystorageaccount.blob.core.windows.net/customer`

### File Format Requirements

- **Format**: CSV files
- **Headers**: First row should contain column names (automatically skipped)
- **Delimiter**: Comma (`,`)
- **Null values**: Use `NULL` or `null` for missing values

## Setup Instructions

1. **Install required packages**:
   ```bash
   pip install python-dotenv snowflake-connector-python tabulate
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
   
   # Entities to process (comma-separated, matching your Azure blob containers)
   ENTITIES=channel,customer,product,salesdetail,salesheader
   ```

## Running the ETL Process

### Complete ETL Process

Run the entire ETL process with a single command:
```bash
python -m rahil.run_etl 2>&1 | tee rahil/logs/etl_run_$(date +%Y%m%d_%H%M%S).log
```

This will:
1. Create a database named `IMT577_DW_{USER_NAME}_STAGING`
2. Create 12 external stages pointing to Azure Blob Storage
3. Create 12 staging tables with appropriate schemas
4. Load data from stages into tables
5. Display sample data from each table
6. Log the entire process to a timestamped file in rahil/logs/

### Individual Steps

You can also run each step separately if needed:

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

## How the ETL Process Works

### 1. Database Creation (Step 0)
The process begins by checking if your database exists. If not, it creates a new Snowflake database named `IMT577_DW_{USER_NAME}_STAGING` where `{USER_NAME}` is the value from your .env file. This allows different users to have their own separate databases.

### 2. External Stages (Step 1)
For each of the 12 entities, the system creates an external stage in Snowflake that points to the corresponding blob in Azure Blob Storage. The URL pattern used is:
```
azure://{your_storage_account}.blob.core.windows.net/{entity}
```

For example, the CHANNEL_STAGE points to `azure://{your_storage_account}.blob.core.windows.net/channel`

### 3. Table Creation (Step 2)
The system creates 12 staging tables with schemas specifically designed for each entity. Each table's schema includes all necessary columns with appropriate data types.

### 4. Data Loading (Step 3)
Data is loaded from the external stages into the corresponding staging tables using Snowflake's COPY command. The system tracks how many rows are loaded into each table.

### 5. Sample Data Display (Step 4)
As a verification step, the system displays the top 5 rows from each table in a nicely formatted table, along with the total row count. This helps verify that the data was loaded correctly and has the expected structure.

## Logging System

All output from the ETL process is captured in log files stored in the `rahil/logs/` directory. These logs include:
- Detailed information about each step
- Success/failure status for each operation
- Row counts for loaded data
- Sample data from each table
- Any errors encountered

Log files are named with timestamps (e.g., `etl_run_20250508_233839.log`) for easy reference.

## Customization

### Using for Different Users
To run this ETL process for a different user (e.g., "VERO_SMITH"), simply update the USER_NAME in your .env file:
```
USER_NAME=VERO_SMITH
```

The system will automatically create and use the database `IMT577_DW_VERO_SMITH_STAGING`.

### Modifying the Entities List
If you need to process different entities, modify the ENTITIES list in the `config.py` file.

### Changing Table Schemas
Table schemas are defined in the `create_tables.py` file. You can modify the table creation SQL statements if needed.

## Troubleshooting

### Connection Issues
- Verify your Snowflake credentials in the .env file
- Check that your account name is correct
- Ensure your warehouse and role have the necessary permissions

### Missing Data
- Confirm that the Azure Blob Storage paths are correct
- Check that the blob data exists and is formatted correctly
- Verify file format settings in the create_stages.py file

### Permission Errors
- Make sure your Snowflake role has CREATE DATABASE privileges
- Check that you have permissions to create stages, tables, and load data

## Security Notes

This package uses `.env` for storing credentials, which provides better security by:
- Keeping sensitive information out of source code
- Allowing different users to have their own credentials
- Preventing credentials from being accidentally committed to version control

For additional security in production environments:
- Add `.env` to your `.gitignore` file 
- Use environment variables instead of .env files
- Consider using key-pair authentication for Snowflake
- Use the Snowflake OAuth integration when possible 

## SQL Table Definitions

The ETL pipeline now loads table definitions from SQL files in the `private_ddl/` directory. This improves the system by:

1. **Separating schema from code**: Table definitions are now in separate SQL files rather than embedded in code
2. **Reusability**: You can customize table structures without modifying Python code
3. **Maintainability**: Each table definition can be maintained independently
4. **Academic integrity**: Table definitions are not committed to git, allowing students to create their own schemas

### How SQL files are loaded

The system looks for the following SQL files in the `private_ddl/` directory:

- `staging_*.sql`: Definitions for staging tables
- `dim_*.sql`: Definitions for dimension tables
- `fact_*.sql`: Definitions for fact tables

If these files don't exist, the system will automatically copy them from the backup directory `private_ddl/rahil/` if available.

### Creating your own schema

To modify the table structures:

1. Create a `.sql` file in the `private_ddl/` directory with the appropriate naming pattern
2. Define your table structure using standard Snowflake SQL syntax
3. Run the ETL process as normal - it will use your custom definitions

Example of a table definition file (`private_ddl/staging_customer.sql`):

```sql
CREATE OR REPLACE TABLE STAGING_CUSTOMER (
  CUSTOMERID           VARCHAR,
  SUBSEGMENTID         INTEGER,
  FIRSTNAME            VARCHAR,
  LASTNAME             VARCHAR,
  GENDER               VARCHAR,
  -- ... other columns ...
  MODIFIEDBY           VARCHAR
);
```

## Dimensional Model ETL

After loading data into staging tables, you can run the dimensional ETL process to create and populate a dimensional model. This includes dimension tables, the date dimension, and fact tables.

### Running the Dimensional ETL Process

To run the complete dimensional ETL process:

```bash
python -m rahil.run_dimensional_etl
```

This will:
1. Create a dimensional database named `IMT577_DW_{USER_NAME}_DIMENSION`
2. Create dimension tables (Product, Location, Customer, Channel, Reseller, Store, Date)
3. Load the Date dimension with 2 years of data
4. Load data from staging tables into dimension tables
5. Create fact tables (SalesActual, ProductSalesTarget, SRCSalesTarget)
6. Load data from staging tables into fact tables
7. Log the entire process

### Dimensional Model Architecture

The dimensional model follows a star schema with:

- **Dimension Tables**: Store descriptive attributes for the different business entities
  - Dim_Product: Product information with pricing and profitability metrics
  - Dim_Location: Geographic locations for customers, stores, and resellers
  - Dim_Customer: Customer details with demographics
  - Dim_Channel: Sales channels and categories
  - Dim_Reseller: Reseller information with contact details
  - Dim_Store: Store details with location references
  - Dim_Date: Calendar date dimension with fiscal periods

- **Fact Tables**: Store transactional and measurement data
  - Fact_SalesActual: Detailed sales transactions
  - Fact_ProductSalesTarget: Product-level sales targets
  - Fact_SRCSalesTarget: Store/Reseller/Channel sales targets

### SQL Table Definitions for Dimensional Model

The dimensional ETL also uses SQL files for table definitions, located in:

- `private_ddl/dim_*.sql`: Definitions for dimension tables
- `private_ddl/DIM_DATE.sql`: Special file for the date dimension that includes data generation
- `private_ddl/fact_*.sql`: Definitions for fact tables

These files follow the same pattern as the staging SQL definitions and allow you to customize the dimensional model structure.

### Date Dimension

The Date dimension is populated using a special SQL script that:
- Creates the Dim_Date table with calendar and fiscal attributes
- Generates 730 days (2 years) of date records
- Includes business logic for holidays, weekends, fiscal periods, etc.

## Complete ETL Workflow

The complete ETL workflow consists of two main phases:

1. **Staging ETL** (from Azure Blob to staging tables)
   ```bash
   python -m rahil.run_etl
   ```

2. **Dimensional ETL** (from staging to dimensional model)
   ```bash
   python -m rahil.run_dimensional_etl
   ```

### Running the Complete ETL Pipeline

To process data end-to-end run both ETL phases back to back:

```bash
python -m rahil.run_etl
python -m rahil.run_dimensional_etl
```

This sequence loads the staging tables, then builds the dimensional model and facts. Each step logs its output under `rahil/logs/` with a timestamp.

Always run the staging phase first so the dimensional ETL has up to date data available.

## Secure Views Data Access Layer

After creating the dimensional model, you can create a comprehensive secure views system that provides both pass-through views and analytical views for data access and visualization.

### Creating Secure Views

To create all secure views:

```bash
python -m rahil.run_views_etl
```

This will:
1. Create 10 pass-through secure views for all dimension and fact tables
2. Create 3 analytical secure views for business intelligence
3. Verify all views with sample data display
4. Provide comprehensive logging and documentation

### View Categories

#### Pass-Through Views (10 views)
These are exact copies of dimension and fact tables using explicit column lists (no SELECT *):

**Dimension Views**: `VW_Dim_Product`, `VW_Dim_Customer`, `VW_Dim_Location`, `VW_Dim_Channel`, `VW_Dim_Store`, `VW_Dim_Reseller`, `VW_Dim_Date`

**Fact Views**: `VW_Fact_SalesActual`, `VW_Fact_ProductSalesTarget`, `VW_Fact_SRCSalesTarget`

#### Analytical Views (3 views)
These provide pre-aggregated data optimized for visualization:

- **VW_SalesPerformanceSummary**: Product sales performance by time periods with profit margins and pricing analysis
- **VW_CustomerSalesAnalysis**: Customer demographics analysis with sales patterns by geography, gender, and channel
- **VW_TargetVsActualPerformance**: Target vs actual performance comparison with achievement percentages

### Benefits of Secure Views

1. **Security**: All views are created as `SECURE VIEWS` for sharing with professors/graders
2. **Data Protection**: Views isolate the warehouse from direct table queries
3. **Change Insulation**: Downstream objects are protected from table structure changes
4. **Explicit Columns**: No `SELECT *` usage for better security and performance
5. **Business Intelligence**: Pre-aggregated analytical views optimize visualization tools

### Individual View Operations

Create views only:
```bash
python -m rahil.create_views
```

Verify views only:
```bash
python -m rahil.view_sample_views
```

### Using Views in Tableau

1. Connect to Snowflake using your credentials
2. Select the dimensional database (`IMT577_DW_{USER_NAME}_DIMENSION`)
3. Use the secure views instead of direct table access
4. Drag and drop pre-calculated metrics from analytical views
5. Create relationships between pass-through views if needed

For detailed documentation on the secure views system, see [VIEWS_README.md](VIEWS_README.md).

## Complete Data Pipeline Workflow

The complete data pipeline consists of three main phases:

1. **Staging ETL** (from Azure Blob to staging tables)
   ```bash
   python -m rahil.run_etl
   ```

2. **Dimensional ETL** (from staging to dimensional model)
   ```bash
   python -m rahil.run_dimensional_etl
   ```

3. **Views Creation** (secure data access layer)
   ```bash
   python -m rahil.run_views_etl
   ```

### Running the Complete Pipeline

To process data end-to-end with full data access layer:

```bash
python -m rahil.run_etl
python -m rahil.run_dimensional_etl
python -m rahil.run_views_etl
```

This sequence creates a complete data warehouse with staging tables, dimensional model, and secure views ready for business intelligence and sharing with professors/graders.
