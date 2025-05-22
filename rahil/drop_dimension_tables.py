#!/usr/bin/env python3
"""
Script to drop all existing tables in the dimension database
"""
import sys
sys.path.append('.') # To allow importing from parent directory
from rahil.dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA
)
import snowflake.connector

def drop_dimension_tables():
    """
    Drop all existing tables in the dimension database
    """
    try:
        # Connect to Snowflake
        print(f"Connecting to Snowflake to drop tables in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}...")
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE,
            database=DIMENSION_DB_NAME,
            schema=SNOWFLAKE_SCHEMA
        )
        cursor = conn.cursor()
        
        # Get all existing tables in the current schema
        print(f"Fetching all existing tables in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}...")
        cursor.execute("SHOW TABLES")
        tables_info = cursor.fetchall()
        headers = [desc[0] for desc in cursor.description] # Get headers

        if not tables_info:
            print("No tables found to drop.")
            return True
        
        table_count = 0
        # Drop each table
        for table_row_values in tables_info:
            table_data = dict(zip(headers, table_row_values)) # Create a dict from headers and values
            table_name = table_data["name"] 
            db_name_from_show = table_data["database_name"]
            schema_name_from_show = table_data["schema_name"]
            
            if db_name_from_show.upper() == DIMENSION_DB_NAME.upper() and \
               schema_name_from_show.upper() == SNOWFLAKE_SCHEMA.upper():
                print(f"Dropping table: {table_name}")
                # Use fully qualified name for dropping
                cursor.execute(f'DROP TABLE IF EXISTS "{db_name_from_show}"."{schema_name_from_show}"."{table_name}"')
                table_count +=1
            else:
                print(f"Skipping table {table_name} as it is in a different db/schema: {db_name_from_show}.{schema_name_from_show}")

        if table_count > 0:
            print(f"\nSuccessfully dropped {table_count} tables from {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.")
        else:
            print("No tables were dropped (either none found or none matched the target database/schema).")

        # Close connections
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if not drop_dimension_tables():
        sys.exit(1) 