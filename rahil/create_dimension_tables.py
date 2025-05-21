#!/usr/bin/env python3
"""
Create dimension tables for the dimensional model
"""
import snowflake.connector
from sqlalchemy.schema import CreateTable
from tabulate import tabulate
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA
)
from .schemas.dimension import DimensionBase

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

        # Create tables using SQLAlchemy models
        for table in DimensionBase.metadata.sorted_tables:
            print(f"Creating {table.name} table...")
            cursor.execute(str(CreateTable(table)))
        
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