#!/usr/bin/env python3
"""
Configuration settings for Snowflake ETL process
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

DATABASE_NAME = f"IMT577_DW_{USER_NAME}_STAGING"

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

# Azure Blob Storage
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
if not AZURE_STORAGE_ACCOUNT:
    print("ERROR: AZURE_STORAGE_ACCOUNT not found in .env file")
    sys.exit(1)

# Entities list - all entities to process in the ETL pipeline
ENTITIES = [
    'channel', 
    'channelcategory', 
    'customer', 
    'product', 
    'productcategory', 
    'producttype', 
    'reseller', 
    'salesdetail', 
    'salesheader', 
    'store', 
    'targetdatachannel', 
    'targetdataproduct'
] 