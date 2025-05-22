#!/usr/bin/env python3
"""
Script to create staging tables in Snowflake.
This version uses external SQL files instead of inline DDL.
"""
import os
from pathlib import Path
from . import config
from .connection import get_snowflake_connection

def load_sql_file(entity):
    """
    Load SQL file for a given entity
    """
    base_dir = Path(__file__).parent.parent
    sql_path = base_dir / "local_schemas" / f"{entity}.sql"
    
    if not sql_path.exists():
        # Try the example file if actual file doesn't exist
        sql_path = base_dir / "local_schemas" / f"{entity}.sql.example"
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file for {entity} not found")
    
    sql_content = sql_path.read_text()
    # Replace placeholders with actual values
    sql_content = sql_content.replace("${DB_NAME}", config.DATABASE_NAME)
    sql_content = sql_content.replace("${SCHEMA}", config.SNOWFLAKE_SCHEMA)
    
    return sql_content

def create_staging_tables():
    """
    Create staging tables in Snowflake by loading and executing external SQL files
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Use our database
        cursor.execute(f"USE DATABASE {config.DATABASE_NAME}")
        cursor.execute("USE SCHEMA PUBLIC")
        
        created_tables = []
        
        # Create tables for each entity
        for i, entity in enumerate(config.ENTITIES, 1):
            try:
                sql = load_sql_file(entity)
                table_name = f"STAGING_{entity.upper()}"
                print(f"\nCreating table {i}: {table_name}")
                cursor.execute(sql)
                created_tables.append(table_name)
                print(f"Table {table_name} created successfully.")
            except Exception as e:
                print(f"Error creating table {entity}: {str(e)}")
        
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