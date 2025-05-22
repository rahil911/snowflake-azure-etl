#!/usr/bin/env python3
"""
Create dimension tables for the dimensional model
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

def copy_dim_sql_from_backup_if_needed():
    """
    Copy SQL files from rahil backup directory if the main private_ddl directory is empty
    """
    private_ddl_dir = Path(__file__).parents[1] / "private_ddl"
    rahil_backup_dir = private_ddl_dir / "rahil"
    
    # Make sure the directories exist
    private_ddl_dir.mkdir(exist_ok=True)
    
    # Check if there are dim_*.sql files in the main private_ddl directory
    dim_files = list(private_ddl_dir.glob("dim_*.sql"))
    
    # If no dimension files exist and rahil backup directory exists, copy them over
    if not dim_files and rahil_backup_dir.exists():
        print("No dimension SQL definition files found. Copying from backup...")
        for sql_file in rahil_backup_dir.glob("dim_*.sql"):
            target_file = private_ddl_dir / sql_file.name
            shutil.copy2(sql_file, target_file)
            print(f"Copied {sql_file.name} to {private_ddl_dir}")

def create_dimension_tables():
    """Create dimension tables for the dimensional model"""
    print(f"Step 1: Creating dimension tables in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
    
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
        copy_dim_sql_from_backup_if_needed()
        
        # Get all dimension table SQL files
        sql_files = list(sql_dir.glob("dim_*.sql"))
        
        if not sql_files:
            print(f"Error: No SQL definition files found in {sql_dir}")
            print("Please add your SQL table definition files with 'dim_*.sql' naming pattern")
            return False
        
        # Create Dim_Date table
        # This will be created using the DIM_DATE.sql file separately
        # so we don't need to create it here.
        
        # Execute each SQL file to create dimension tables
        print("\nCreating dimension tables...")
        for sql_file in sql_files:
            table_name = sql_file.stem.replace("_", "").capitalize()
            print(f"Creating {table_name} table...")
            
            # Read SQL from file
            with open(sql_file, 'r') as f:
                sql = f.read()
            
            # Execute the SQL
            cursor.execute(sql)
        
        # Add "Unknown" members to dimension tables for handling NULL references
        print("\nAdding unknown members to dimension tables...")
        
        # Unknown Location
        cursor.execute("""
        INSERT INTO Dim_Location (Address, City, PostalCode, State_Province, Country)
        VALUES ('Unknown', 'Unknown', 'Unknown', 'Unknown', 'Unknown')
        """)
        
        # Unknown Customer
        cursor.execute("""
        INSERT INTO Dim_Customer (CustomerID, DimLocationID, CustomerFullName, 
                                CustomerFirstName, CustomerLastName, CustomerGender)
        VALUES ('UNKNOWN', 1, 'Unknown Customer', 'Unknown', 'Unknown', 'Unknown')
        """)
        
        # Unknown Reseller
        cursor.execute("""
        INSERT INTO Dim_Reseller (ResellerID, DimLocationID, ResellerName, ContactName, 
                               PhoneNumber, Email)
        VALUES ('UNKNOWN', 1, 'Unknown Reseller', 'Unknown', 'Unknown', 'Unknown')
        """)
        
        # Unknown Store
        cursor.execute("""
        INSERT INTO Dim_Store (StoreID, DimLocationID, SourceStoreID, StoreName, 
                            StoreNumber, StoreManager)
        VALUES (-1, 1, -1, 'Unknown Store', 'Unknown', 'Unknown')
        """)
        
        # Unknown Product
        cursor.execute("""
        INSERT INTO Dim_Product (ProductID, ProductTypeID, ProductCategoryID, ProductName, 
                              ProductType, ProductCategory, ProductRetailPrice, 
                              ProductWholesalePrice, ProductCost, ProductRetailProfit,
                              ProductWholesaleUnitProfit, ProductProfitMarginUnitPercent)
        VALUES (-1, -1, -1, 'Unknown Product', 'Unknown', 'Unknown', 0, 0, 0, 0, 0, 0)
        """)
        
        # Unknown Channel
        cursor.execute("""
        INSERT INTO Dim_Channel (ChannelID, ChannelCategoryID, ChannelName, ChannelCategory)
        VALUES (-1, -1, 'Unknown Channel', 'Unknown')
        """)
        
        # Show tables to verify creation
        cursor.execute("SHOW TABLES")
        
        # Fetch and display results
        results = cursor.fetchall()
        print("\nVerifying table creation:")
        headers = [column[0] for column in cursor.description]
        print(tabulate(results, headers=headers, tablefmt="grid"))
        
        print(f"\n✅ Dimension tables created successfully in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error creating dimension tables: {e}")
        return False

if __name__ == "__main__":
    create_dimension_tables() 