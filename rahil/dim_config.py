#!/usr/bin/env python3
"""
Configuration settings for Dimensional Model ETL process
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if not env_path.exists():
    print("ERROR: .env file not found. Please create one based on example.env")
    print("Copy example.env to .env and update with your credentials")
    sys.exit(1)

load_dotenv(dotenv_path=env_path)

# User configuration from .env
USER_NAME = os.getenv("USER_NAME")
if not USER_NAME:
    print("ERROR: USER_NAME not found in .env file")
    sys.exit(1)

# Database names
STAGING_DB_NAME = f"IMT577_DW_{USER_NAME}_STAGING"
DIMENSION_DB_NAME = f"IMT577_DW_{USER_NAME}_DIMENSION"

# Snowflake connection parameters from .env
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")

# Check for required credentials
if not all([SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD]):
    print("ERROR: Missing required Snowflake credentials in .env file")
    sys.exit(1)

# Dimension tables to create
DIMENSION_TABLES = [
    'Dim_Product',
    'Dim_Store',
    'Dim_Reseller',
    'Dim_Location',
    'Dim_Customer',
    'Dim_Channel',
    'Dim_Date'
]

# Fact tables to create
FACT_TABLES = [
    'Fact_SalesActual',
    'Fact_ProductSalesTarget',
    'Fact_SRCSalesTarget'
] 