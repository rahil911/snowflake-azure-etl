#!/usr/bin/env python3
"""
Script to create the database if it doesn't exist
"""
import sys
import snowflake.connector
from snowflake.connector.errors import OperationalError, DatabaseError
from . import config

def create_database():
    """
    Create the database if it doesn't exist
    """
    print(f"Creating database: {config.DATABASE_NAME}")
    
    try:
        # Connect to Snowflake (without specifying database)
        conn = snowflake.connector.connect(
            user=config.SNOWFLAKE_USER,
            password=config.SNOWFLAKE_PASSWORD,
            account=config.SNOWFLAKE_ACCOUNT,
            warehouse=config.SNOWFLAKE_WAREHOUSE,
            role=config.SNOWFLAKE_ROLE
        )
        
        # Test the connection
        cursor = conn.cursor()
        cursor.execute("SELECT current_version()")
        version = cursor.fetchone()[0]
        print(f"Connected to Snowflake version: {version}")
        
        # Check if database exists
        cursor.execute(f"SHOW DATABASES LIKE '{config.DATABASE_NAME}'")
        db_exists = cursor.fetchone() is not None
        
        if db_exists:
            print(f"Database {config.DATABASE_NAME} already exists")
        else:
            # Create the database
            print(f"Creating database {config.DATABASE_NAME}")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.DATABASE_NAME}")
            print(f"Database {config.DATABASE_NAME} created successfully")
        
        # Create the schema
        cursor.execute(f"USE DATABASE {config.DATABASE_NAME}")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {config.SNOWFLAKE_SCHEMA}")
        print(f"Schema {config.SNOWFLAKE_SCHEMA} created successfully")
        
        # Close connections
        cursor.close()
        conn.close()
        print("Database setup completed!")
        
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

if __name__ == "__main__":
    create_database() 