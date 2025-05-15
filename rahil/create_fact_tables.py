#!/usr/bin/env python3
"""
Create fact tables for the dimensional model
"""
import snowflake.connector
from tabulate import tabulate
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA
)

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
        
        # Create Fact_SalesActual table
        print("\nCreating Fact_SalesActual table...")
        cursor.execute("""
        CREATE OR REPLACE TABLE Fact_SalesActual (
            DimProductID INT,
            DimStoreID INT,
            DimResellerID INT,
            DimCustomerID INT,
            DimChannelID INT,
            DimSaleDateID INT,
            DimLocationID INT,
            SalesHeaderID INT,
            SalesDetailID INT,
            SaleAmount FLOAT,
            SaleQuantity INT,
            SaleUnitPrice FLOAT,
            SaleExtendedCost FLOAT,
            SaleTotalProfit FLOAT
        )
        """)
        
        # Create Fact_ProductSalesTarget table
        print("Creating Fact_ProductSalesTarget table...")
        cursor.execute("""
        CREATE OR REPLACE TABLE Fact_ProductSalesTarget (
            DimProductID INT,
            DimTargetDateID INT,
            ProductTargetSalesQuantity INT
        )
        """)
        
        # Create Fact_SRCSalesTarget table (Store/Reseller/Channel Sales Target)
        print("Creating Fact_SRCSalesTarget table...")
        cursor.execute("""
        CREATE OR REPLACE TABLE Fact_SRCSalesTarget (
            DimStoreID INT,
            DimResellerID INT,
            DimChannelID INT,
            DimTargetDateID INT,
            SalesTargetAmount FLOAT
        )
        """)
        
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