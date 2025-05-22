#!/usr/bin/env python3
"""
Script to create staging tables in Snowflake
"""
import os
import shutil
from pathlib import Path
from . import config
from .connection import get_snowflake_connection

def copy_sql_from_backup_if_needed():
    """
    Copy SQL files from rahil backup directory if the main private_ddl directory is empty
    """
    private_ddl_dir = Path(__file__).parents[1] / "private_ddl"
    rahil_backup_dir = private_ddl_dir / "rahil"
    
    # Make sure the directories exist
    private_ddl_dir.mkdir(exist_ok=True)
    
    # Check if there are staging_*.sql files in the main private_ddl directory
    staging_files = list(private_ddl_dir.glob("staging_*.sql"))
    
    # If no staging files exist and rahil backup directory exists, copy them over
    if not staging_files and rahil_backup_dir.exists():
        print("No SQL definition files found. Copying from backup...")
        for sql_file in rahil_backup_dir.glob("staging_*.sql"):
            target_file = private_ddl_dir / sql_file.name
            shutil.copy2(sql_file, target_file)
            print(f"Copied {sql_file.name} to {private_ddl_dir}")

def create_staging_tables():
    """
    Create staging tables in Snowflake
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Use our database
        cursor.execute(f"USE DATABASE {config.DATABASE_NAME}")
        cursor.execute("USE SCHEMA PUBLIC")
        
        # Path to SQL definition files
        sql_dir = Path(__file__).parents[1] / "private_ddl"
        
        # Copy SQL files from backup if needed
        copy_sql_from_backup_if_needed()
        
        # Get all staging table SQL files
        sql_files = list(sql_dir.glob("staging_*.sql"))
        
        if not sql_files:
            print(f"Error: No SQL definition files found in {sql_dir}")
            print("Please add your SQL table definition files with 'staging_*.sql' naming pattern")
            return
        
        created_tables = []
        
        # Execute each SQL file
        for i, sql_file in enumerate(sql_files, 1):
            # Extract table name from file name
            table_name = "STAGING_" + sql_file.stem.replace("staging_", "").upper()
            print(f"\nCreating table {i}: {table_name}")
            
            # Read SQL from file
            with open(sql_file, 'r') as f:
                sql = f.read()
            
            # Execute the SQL
            cursor.execute(sql)
            created_tables.append(table_name)
            print(f"Table {table_name} created successfully.")
        
        # Verify tables exist
        print("\nVerifying tables...")
        cursor.execute("SHOW TABLES")
        existing_tables = [row[1] for row in cursor.fetchall()]
        
        print("\nSummary:")
        print(f"Tables created: {len(created_tables)}")
        for table in created_tables:
            status = "✅ Exists" if table in existing_tables else "❌ Not found"
            print(f"- {table}: {status}")
        
        # Close connections
        cursor.close()
        conn.close()
        print("\nTable creation process completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    create_staging_tables() 