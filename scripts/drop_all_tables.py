#!/usr/bin/env python3
"""
Script to drop all existing tables in Snowflake
"""
import sys
sys.path.append('.')
from scripts.connection import get_snowflake_connection

def drop_all_tables():
    """
    Drop all existing tables in the database
    """
    try:
        # Connect to Snowflake
        print("Connecting to Snowflake...")
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Use our database
        cursor.execute("USE DATABASE IMT577_DW_RAHIL_HARIHAR_STAGING")
        cursor.execute("USE SCHEMA PUBLIC")
        
        # Get all existing tables
        print("Fetching all existing tables...")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found to drop.")
            return
        
        # Drop each table
        for table in tables:
            table_name = table[1]
            print(f"Dropping table: {table_name}")
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        print(f"\nSuccessfully dropped {len(tables)} tables.")
        
        # Close connections
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    drop_all_tables() 