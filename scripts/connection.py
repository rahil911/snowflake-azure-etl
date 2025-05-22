#!/usr/bin/env python3
"""
Step 1: Snowflake Connection Setup
This script establishes a connection to Snowflake with the provided credentials.
"""
import snowflake.connector

def get_snowflake_connection():
    """
    Connect to Snowflake using the provided credentials.
    
    Returns:
        Connection: Snowflake connection object
    """
    # Snowflake connection parameters
    conn_params = {
        'account': 'DMWLLLA-SW69144',
        'user': 'RAHILMHARIHAR',
        'password': 'Rahil0911112358132134#',
        'database': 'IMT577_DW_RAHIL_HARIHAR_STAGING',
        'schema': 'PUBLIC',
        'warehouse': 'COMPUTE_WH',
        'role': 'ACCOUNTADMIN'
    }
    
    # Connect to Snowflake
    print("Connecting to Snowflake...")
    conn = snowflake.connector.connect(**conn_params)
    
    return conn

if __name__ == "__main__":
    # Test the connection
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Execute a simple query to test the connection
        cursor.execute("SELECT current_version()")
        version = cursor.fetchone()[0]
        print(f"Connected to Snowflake version: {version}")
        
        # Show current account info
        cursor.execute("SELECT current_account(), current_database(), current_schema(), current_warehouse(), current_role()")
        account, database, schema, warehouse, role = cursor.fetchone()
        print(f"Account: {account}")
        print(f"Database: {database}")
        print(f"Schema: {schema}")
        print(f"Warehouse: {warehouse}")
        print(f"Role: {role}")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        print("Connection test successful!")
        
    except Exception as e:
        print(f"Error connecting to Snowflake: {str(e)}") 