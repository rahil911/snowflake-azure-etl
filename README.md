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

## Entities Processed

This ETL pipeline processes 12 entities:
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
   AZURE_STORAGE_ACCOUNT=sp72storage.blob.core.windows.net
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
azure://sp72storage.blob.core.windows.net/{entity}
```

For example, the CHANNEL_STAGE points to `azure://sp72storage.blob.core.windows.net/channel`

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