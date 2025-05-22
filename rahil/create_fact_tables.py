#!/usr/bin/env python3
"""
Create fact tables for the dimensional model
"""
import os
import shutil
from pathlib import Path
import snowflake.connector
from tabulate import tabulate
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA
)

def copy_fact_sql_from_backup_if_needed():
    """
    Copy SQL files from rahil backup directory if the main private_ddl directory is empty
    """
    private_ddl_dir = Path(__file__).parents[1] / "private_ddl"
    rahil_backup_dir = private_ddl_dir / "rahil"
    
    # Make sure the directories exist
    private_ddl_dir.mkdir(exist_ok=True)
    
    # Check if there are fact_*.sql files in the main private_ddl directory
    fact_files = list(private_ddl_dir.glob("fact_*.sql"))
    
    # If no fact files exist and rahil backup directory exists, copy them over
    if not fact_files and rahil_backup_dir.exists():
        print("No fact SQL definition files found. Copying from backup...")
        for sql_file in rahil_backup_dir.glob("fact_*.sql"):
            target_file = private_ddl_dir / sql_file.name
            shutil.copy2(sql_file, target_file)
            print(f"Copied {sql_file.name} to {private_ddl_dir}")

def create_fact_tables():
    """Create fact tables for the dimensional model"""
    print(f"Step 4: Creating fact tables in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
    
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE,
            database=DIMENSION_DB_NAME,
            schema=SNOWFLAKE_SCHEMA
        )
        
        # Create a cursor object
        cursor = conn.cursor()
        
        # Path to SQL definition files
        sql_dir = Path(__file__).parents[1] / "private_ddl"
        
        # Copy SQL files from backup if needed
        copy_fact_sql_from_backup_if_needed()
        
        # Get all fact table SQL files
        sql_files = list(sql_dir.glob("fact_*.sql"))
        
        if not sql_files:
            print(f"Error: No SQL definition files found in {sql_dir}")
            print("Please add your SQL table definition files with 'fact_*.sql' naming pattern")
            return False
        
        # Execute each SQL file to create fact tables
        for sql_file in sql_files:
            table_name = sql_file.stem.replace("_", "").capitalize()
            print(f"\nCreating {table_name} table...")
            
            # Read SQL from file
            with open(sql_file, 'r') as f:
                sql = f.read()
            
            # Execute the SQL
            cursor.execute(sql)
        
        # Show tables to verify creation
        cursor.execute("SHOW TABLES")
        
        # Fetch and display results
        results = cursor.fetchall()
        print("\nVerifying table creation:")
        headers = [column[0] for column in cursor.description]
        print(tabulate(results, headers=headers, tablefmt="grid"))
        
        print(f"\n✅ Fact tables created successfully in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error creating fact tables: {e}")
        return False

if __name__ == "__main__":
    create_fact_tables() 