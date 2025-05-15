#!/usr/bin/env python3
"""
Load Dim_Date table using the provided SQL script
"""
import os
import snowflake.connector
from tabulate import tabulate
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA
)

def load_dim_date():
    """Load Dim_Date table using the provided SQL script"""
    print(f"Step 2: Loading Dim_Date table in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
    
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
        
        # Read the SQL script file
        sql_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DIM_DATE.sql')
        print(f"Reading SQL script from: {sql_script_path}")
        
        with open(sql_script_path, 'r') as file:
            sql_script = file.read()
        
        # Extract create table statement
        create_table_part = sql_script.split('create or replace table DIM_DATE')[1].split('insert into DIM_DATE')[0]
        create_table_sql = f"create or replace table DIM_DATE{create_table_part}"
        
        # Extract insert statement
        insert_part = sql_script.split('insert into DIM_DATE')[1].split('--Miscellaneous queries')[0]
        insert_sql = f"insert into DIM_DATE{insert_part}"
        
        # Execute create table statement
        print("Executing CREATE TABLE statement...")
        try:
            cursor.execute(create_table_sql)
            print("Table DIM_DATE created successfully.")
        except Exception as e:
            if "already exists" in str(e):
                print("Table DIM_DATE already exists, continuing with insert...")
            else:
                raise
        
        # Execute insert statement
        print("Executing INSERT statement...")
        cursor.execute(insert_sql)
        print("Data inserted into DIM_DATE successfully.")
        
        # Verify that the table exists and has data
        cursor.execute("SELECT COUNT(*) FROM DIM_DATE")
        count = cursor.fetchone()[0]
        print(f"Dim_Date table has {count} rows")
        
        # Show sample data
        cursor.execute("SELECT * FROM DIM_DATE LIMIT 5")
        results = cursor.fetchall()
        
        if results:
            print("\nSample data from Dim_Date:")
            headers = [column[0] for column in cursor.description]
            print(tabulate(results, headers=headers, tablefmt="grid"))
            
            print(f"\n✅ Dim_Date table loaded successfully in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
        else:
            print("\n❌ Dim_Date table was created but no data was loaded")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error loading Dim_Date table: {e}")
        return False

if __name__ == "__main__":
    load_dim_date() 