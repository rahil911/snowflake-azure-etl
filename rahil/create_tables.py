#!/usr/bin/env python3
"""
Script to create staging tables in Snowflake
"""
from sqlalchemy.schema import CreateTable

from . import config
from .connection import get_snowflake_connection
from .schemas import Base

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
        
        # Create tables using SQLAlchemy models
        created_tables = []
        for i, table in enumerate(Base.metadata.sorted_tables, 1):
            sql = str(CreateTable(table))
            print(f"\nCreating table {i}: {table.name}")
            cursor.execute(sql)
            created_tables.append(table.name)
            print(f"Table {table.name} created successfully.")
        
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