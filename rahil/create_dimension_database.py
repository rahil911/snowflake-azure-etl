#!/usr/bin/env python3
"""
Create the dimensional model database
"""
import snowflake.connector
from tabulate import tabulate
from .dim_config import SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME

def create_dimension_database():
    """Create the dimensional model database if it doesn't exist"""
    print(f"Step 0: Creating dimensional model database {DIMENSION_DB_NAME} if it doesn't exist")
    
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE
        )
        
        # Create a cursor object
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DIMENSION_DB_NAME}")
        
        # Use the database
        cursor.execute(f"USE DATABASE {DIMENSION_DB_NAME}")
        
        # Show databases to verify creation
        cursor.execute("SHOW DATABASES")
        
        # Fetch and display results
        results = cursor.fetchall()
        print("\nVerifying database creation:")
        headers = [column[0] for column in cursor.description]
        print(tabulate(results, headers=headers, tablefmt="grid"))
        
        print(f"\n✅ Database {DIMENSION_DB_NAME} is ready.")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error creating database: {e}")
        return False

if __name__ == "__main__":
    create_dimension_database() 