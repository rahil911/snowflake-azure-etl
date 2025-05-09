#!/usr/bin/env python3
"""
Snowflake connection module
"""
import sys
import snowflake.connector
from snowflake.connector.errors import OperationalError, DatabaseError
from . import config

def get_snowflake_connection():
    """
    Establishes and returns a connection to Snowflake using credentials from .env
    """
    print("Connecting to Snowflake...")
    
    try:
        # Connect to Snowflake using credentials from .env (via config)
        conn = snowflake.connector.connect(
            user=config.SNOWFLAKE_USER,
            password=config.SNOWFLAKE_PASSWORD,
            account=config.SNOWFLAKE_ACCOUNT,
            warehouse=config.SNOWFLAKE_WAREHOUSE,
            role=config.SNOWFLAKE_ROLE,
            database=config.DATABASE_NAME,
            schema=config.SNOWFLAKE_SCHEMA
        )
        
        # Test the connection
        cursor = conn.cursor()
        cursor.execute("SELECT current_version()")
        version = cursor.fetchone()[0]
        print(f"Connected to Snowflake version: {version}")
        cursor.close()
        
        return conn
        
    except OperationalError as e:
        print(f"ERROR: Failed to connect to Snowflake: {str(e)}")
        print("Please check your credentials in the .env file")
        sys.exit(1)
    except DatabaseError as e:
        print(f"ERROR: Database error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {str(e)}")
        sys.exit(1) 