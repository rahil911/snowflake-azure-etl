#!/usr/bin/env python3
"""
Create dimension tables for the dimensional model
"""
import snowflake.connector
from tabulate import tabulate
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA
)

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
        
        # Create Dim_Date table
        # This will be created using the DIM_DATE.sql file separately
        # so we don't need to create it here.
        
        # Create Dim_Product table
        print("\nCreating Dim_Product table...")
        cursor.execute("""
        CREATE OR REPLACE TABLE Dim_Product (
            DimProductID INT IDENTITY(1,1) PRIMARY KEY,
            ProductID INT,
            ProductTypeID INT,
            ProductCategoryID INT,
            ProductName VARCHAR(255),
            ProductType VARCHAR(255),
            ProductCategory VARCHAR(255),
            ProductRetailPrice FLOAT,
            ProductWholesalePrice FLOAT,
            ProductCost FLOAT,
            ProductRetailProfit FLOAT,
            ProductWholesaleUnitProfit FLOAT,
            ProductProfitMarginUnitPercent FLOAT
        )
        """)
        
        # Create Dim_Store table
        print("Creating Dim_Store table...")
        cursor.execute("""
        CREATE OR REPLACE TABLE Dim_Store (
            DimStoreID INT IDENTITY(1,1) PRIMARY KEY,
            StoreID INT,
            DimLocationID INT,
            SourceStoreID INT,
            StoreName VARCHAR(255),
            StoreNumber VARCHAR(255),
            StoreManager VARCHAR(255)
        )
        """)
        
        # Create Dim_Reseller table
        print("Creating Dim_Reseller table...")
        cursor.execute("""
        CREATE OR REPLACE TABLE Dim_Reseller (
            DimResellerID INT IDENTITY(1,1) PRIMARY KEY,
            ResellerID VARCHAR(255),
            DimLocationID INT,
            ResellerName VARCHAR(255),
            ContactName VARCHAR(255),
            PhoneNumber VARCHAR(255),
            Email VARCHAR(255)
        )
        """)
        
        # Create Dim_Location table
        print("Creating Dim_Location table...")
        cursor.execute("""
        CREATE OR REPLACE TABLE Dim_Location (
            DimLocationID INT IDENTITY(1,1) PRIMARY KEY,
            Address VARCHAR(255),
            City VARCHAR(255),
            PostalCode VARCHAR(255),
            State_Province VARCHAR(255),
            Country VARCHAR(255)
        )
        """)
        
        # Create Dim_Customer table
        print("Creating Dim_Customer table...")
        cursor.execute("""
        CREATE OR REPLACE TABLE Dim_Customer (
            DimCustomerID INT IDENTITY(1,1) PRIMARY KEY,
            CustomerID VARCHAR(255),
            DimLocationID INT,
            CustomerFullName VARCHAR(255),
            CustomerFirstName VARCHAR(255),
            CustomerLastName VARCHAR(255),
            CustomerGender VARCHAR(255)
        )
        """)
        
        # Create Dim_Channel table
        print("Creating Dim_Channel table...")
        cursor.execute("""
        CREATE OR REPLACE TABLE Dim_Channel (
            DimChannelID INT IDENTITY(1,1) PRIMARY KEY,
            ChannelID INT,
            ChannelCategoryID INT,
            ChannelName VARCHAR(255),
            ChannelCategory VARCHAR(255)
        )
        """)
        
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