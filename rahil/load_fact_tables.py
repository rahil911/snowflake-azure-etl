#!/usr/bin/env python3
"""
Load data from staging tables to fact tables
"""
import snowflake.connector
import os
from tabulate import tabulate
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA, STAGING_DB_NAME
)

def read_sql_file(file_path, **kwargs):
    """Read and format SQL file with provided parameters"""
    try:
        with open(file_path, 'r') as file:
            sql_content = file.read()
        # Format SQL with provided parameters
        return sql_content.format(**kwargs)
    except Exception as e:
        raise Exception(f"Error reading SQL file {file_path}: {e}")

def execute_sql_file(cursor, file_path, **kwargs):
    """Execute SQL from file with parameter substitution"""
    try:
        sql_query = read_sql_file(file_path, **kwargs)
        cursor.execute(sql_query)
        return True
    except Exception as e:
        print(f"Error executing SQL file {file_path}: {e}")
        return False

def load_fact_tables():
    """Load data from staging tables to fact tables"""
    print(f"Step 5: Loading data from {STAGING_DB_NAME} to {DIMENSION_DB_NAME} fact tables")
    
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
        
        # Check if the staging database exists
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        
        # Find available staging databases
        actual_staging_db = None
        
        # Look for the exact staging database name first
        for db in databases:
            if db[1].upper() == STAGING_DB_NAME.upper():
                actual_staging_db = db[1]
                break
        
        # If not found, look for databases with "STAGING" in the name
        if not actual_staging_db:
            staging_dbs = []
            for db in databases:
                if "STAGING" in db[1].upper() and "HIDDEN" in db[1].upper():
                    staging_dbs.append(db[1])
            
            if staging_dbs:
                # Use the first staging database found
                actual_staging_db = staging_dbs[0]
                print(f"⚠️ Warning: {STAGING_DB_NAME} not found. Using {actual_staging_db} instead.")
            else:
                # If still not found, look for any database with "STAGING" in the name
                for db in databases:
                    if "STAGING" in db[1].upper():
                        staging_dbs.append(db[1])
                
                if staging_dbs:
                    # Use the first staging database found
                    actual_staging_db = staging_dbs[0]
                    print(f"⚠️ Warning: {STAGING_DB_NAME} not found. Using {actual_staging_db} instead.")
                else:
                    raise Exception(f"No staging database found. Please run the staging ETL process first.")
        
        # Use the actual staging database name for queries
        staging_db = actual_staging_db if actual_staging_db else STAGING_DB_NAME
        
        # Get the directory path for SQL files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sql_dir = os.path.join(current_dir, '..', 'private_ddl', 'rahil')
        
        # Common parameters for SQL file formatting
        sql_params = {
            'DIMENSION_DB_NAME': DIMENSION_DB_NAME,
            'SNOWFLAKE_SCHEMA': SNOWFLAKE_SCHEMA,
            'staging_db': staging_db
        }
        
        # Load Fact_SalesActual table
        print("\nLoading Fact_SalesActual table...")
        fact_salesactual_path = os.path.join(sql_dir, 'load_fact_salesactual.sql')
        if execute_sql_file(cursor, fact_salesactual_path, **sql_params):
            # Get row count for Fact_SalesActual
            cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_SalesActual")
            sales_count = cursor.fetchone()[0]
            print(f"Loaded {sales_count} rows into Fact_SalesActual")
        else:
            raise Exception("Failed to load Fact_SalesActual")
        
        # Load Fact_ProductSalesTarget table
        print("\nLoading Fact_ProductSalesTarget table...")
        fact_productsalestarget_path = os.path.join(sql_dir, 'load_fact_productsalestarget.sql')
        if execute_sql_file(cursor, fact_productsalestarget_path, **sql_params):
            # Get row count for Fact_ProductSalesTarget
            cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_ProductSalesTarget")
            product_target_count = cursor.fetchone()[0]
            print(f"Loaded {product_target_count} rows into Fact_ProductSalesTarget")
        else:
            raise Exception("Failed to load Fact_ProductSalesTarget")
        
        # Load Fact_SRCSalesTarget table
        print("\nLoading Fact_SRCSalesTarget table...")
        fact_srcsalestarget_path = os.path.join(sql_dir, 'load_fact_srcsalestarget.sql')
        if execute_sql_file(cursor, fact_srcsalestarget_path, **sql_params):
            # Get row count for Fact_SRCSalesTarget
            cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_SRCSalesTarget")
            src_target_count = cursor.fetchone()[0]
            print(f"Loaded {src_target_count} rows into Fact_SRCSalesTarget")
        else:
            raise Exception("Failed to load Fact_SRCSalesTarget")

        # Display sample data from each fact table
        tables = ['Fact_SalesActual', 'Fact_ProductSalesTarget', 'Fact_SRCSalesTarget']
        
        for table in tables:
            print(f"\nSample data from {table}:")
            cursor.execute(f"SELECT * FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.{table} LIMIT 5")
            results = cursor.fetchall()
            headers = [column[0] for column in cursor.description]
            print(tabulate(results, headers=headers, tablefmt="grid"))
        
        print(f"\n✅ Fact tables loaded successfully from staging tables")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error loading fact tables: {e}")
        return False

if __name__ == "__main__":
    load_fact_tables() 