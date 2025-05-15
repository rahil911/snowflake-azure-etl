#!/usr/bin/env python3
"""
Load data from staging tables to dimension tables
"""
import snowflake.connector
from tabulate import tabulate
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA, STAGING_DB_NAME
)

def load_dimension_tables():
    """Load data from staging tables to dimension tables"""
    print(f"Step 3: Loading data from {STAGING_DB_NAME} to {DIMENSION_DB_NAME} dimension tables")
    
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
        
        # First, load Dim_Location from customer, store, and reseller tables
        print("\nLoading Dim_Location table...")
        cursor.execute(f"""
        INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Location (
            Address, City, PostalCode, State_Province, Country
        )
        -- Location data from Customer
        SELECT DISTINCT
            COALESCE(Address, 'Unknown') as Address, 
            COALESCE(City, 'Unknown') as City, 
            COALESCE(CAST(PostalCode AS VARCHAR(255)), 'Unknown') as PostalCode, 
            COALESCE(StateProvince, 'Unknown') as StateProvince, 
            COALESCE(Country, 'Unknown') as Country
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_CUSTOMER
        WHERE Address IS NOT NULL 
          AND City IS NOT NULL 
          AND Country IS NOT NULL
        
        UNION
        
        -- Location data from Store
        SELECT DISTINCT
            COALESCE(Address, 'Unknown') as Address, 
            COALESCE(City, 'Unknown') as City, 
            COALESCE(CAST(PostalCode AS VARCHAR(255)), 'Unknown') as PostalCode, 
            COALESCE(StateProvince, 'Unknown') as StateProvince, 
            COALESCE(Country, 'Unknown') as Country
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_STORE
        WHERE Address IS NOT NULL 
          AND City IS NOT NULL 
          AND Country IS NOT NULL
        
        UNION
        
        -- Location data from Reseller
        SELECT DISTINCT
            COALESCE(Address, 'Unknown') as Address, 
            COALESCE(City, 'Unknown') as City, 
            COALESCE(CAST(PostalCode AS VARCHAR(255)), 'Unknown') as PostalCode, 
            COALESCE(StateProvince, 'Unknown') as StateProvince, 
            COALESCE(Country, 'Unknown') as Country
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_RESELLER
        WHERE Address IS NOT NULL 
          AND City IS NOT NULL 
          AND Country IS NOT NULL
        """)
        
        # Get row count for Dim_Location
        cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Location")
        location_count = cursor.fetchone()[0]
        print(f"Loaded {location_count} locations into Dim_Location")
        
        # Load Dim_Channel
        print("\nLoading Dim_Channel table...")
        cursor.execute(f"""
        INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Channel (
            ChannelID, ChannelCategoryID, ChannelName, ChannelCategory
        )
        SELECT 
            c.ChannelID,
            c.ChannelCategoryID,
            COALESCE(c.Channel, 'Unknown') as Channel,
            COALESCE(cc.ChannelCategory, 'Unknown') as ChannelCategory
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_CHANNEL c
        JOIN {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_CHANNELCATEGORY cc 
            ON c.ChannelCategoryID = cc.ChannelCategoryID
        WHERE c.Channel IS NOT NULL
        """)
        
        # Get row count for Dim_Channel
        cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Channel")
        channel_count = cursor.fetchone()[0]
        print(f"Loaded {channel_count} channels into Dim_Channel")
        
        # Load Dim_Customer
        print("\nLoading Dim_Customer table...")
        cursor.execute(f"""
        INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Customer (
            CustomerID, DimLocationID, CustomerFullName, CustomerFirstName, CustomerLastName, CustomerGender
        )
        SELECT 
            c.CustomerID,
            COALESCE(l.DimLocationID, 1) as DimLocationID,
            COALESCE(c.FirstName, 'Unknown') || ' ' || COALESCE(c.LastName, 'Unknown') as CustomerFullName,
            COALESCE(c.FirstName, 'Unknown') as FirstName,
            COALESCE(c.LastName, 'Unknown') as LastName,
            COALESCE(c.Gender, 'Unknown') as Gender
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_CUSTOMER c
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Location l 
            ON COALESCE(c.Address, 'Unknown') = l.Address
            AND COALESCE(c.City, 'Unknown') = l.City
            AND COALESCE(CAST(c.PostalCode AS VARCHAR(255)), 'Unknown') = l.PostalCode
            AND COALESCE(c.StateProvince, 'Unknown') = l.State_Province
            AND COALESCE(c.Country, 'Unknown') = l.Country
        WHERE c.CustomerID IS NOT NULL
        """)
        
        # Get row count for Dim_Customer
        cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Customer")
        customer_count = cursor.fetchone()[0]
        print(f"Loaded {customer_count} customers into Dim_Customer")
        
        # Load Dim_Reseller
        print("\nLoading Dim_Reseller table...")
        cursor.execute(f"""
        INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Reseller (
            ResellerID, DimLocationID, ResellerName, ContactName, PhoneNumber, Email
        )
        SELECT 
            r.ResellerID,
            COALESCE(l.DimLocationID, 1) as DimLocationID,
            COALESCE(r.ResellerName, 'Unknown') as ResellerName,
            COALESCE(r.Contact, 'Unknown') as Contact,
            COALESCE(r.PhoneNumber, 'Unknown') as PhoneNumber,
            COALESCE(r.EmailAddress, 'Unknown') as EmailAddress
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_RESELLER r
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Location l 
            ON COALESCE(r.Address, 'Unknown') = l.Address
            AND COALESCE(r.City, 'Unknown') = l.City
            AND COALESCE(CAST(r.PostalCode AS VARCHAR(255)), 'Unknown') = l.PostalCode
            AND COALESCE(r.StateProvince, 'Unknown') = l.State_Province
            AND COALESCE(r.Country, 'Unknown') = l.Country
        WHERE r.ResellerID IS NOT NULL
        """)
        
        # Get row count for Dim_Reseller
        cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Reseller")
        reseller_count = cursor.fetchone()[0]
        print(f"Loaded {reseller_count} resellers into Dim_Reseller")
        
        # Load Dim_Store
        print("\nLoading Dim_Store table...")
        cursor.execute(f"""
        INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Store (
            StoreID, DimLocationID, SourceStoreID, StoreName, StoreNumber, StoreManager
        )
        SELECT 
            s.StoreID,
            COALESCE(l.DimLocationID, 1) as DimLocationID,
            s.StoreID as SourceStoreID,
            'Store ' || COALESCE(s.StoreNumber, 'Unknown') as StoreName,
            COALESCE(CAST(s.StoreNumber AS VARCHAR(255)), 'Unknown') as StoreNumber,
            COALESCE(s.StoreManager, 'Unknown') as StoreManager
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_STORE s
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Location l 
            ON COALESCE(s.Address, 'Unknown') = l.Address
            AND COALESCE(s.City, 'Unknown') = l.City
            AND COALESCE(CAST(s.PostalCode AS VARCHAR(255)), 'Unknown') = l.PostalCode
            AND COALESCE(s.StateProvince, 'Unknown') = l.State_Province
            AND COALESCE(s.Country, 'Unknown') = l.Country
        WHERE s.StoreID IS NOT NULL
        """)
        
        # Get row count for Dim_Store
        cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Store")
        store_count = cursor.fetchone()[0]
        print(f"Loaded {store_count} stores into Dim_Store")
        
        # Load Dim_Product
        print("\nLoading Dim_Product table...")
        cursor.execute(f"""
        INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Product (
            ProductID, ProductTypeID, ProductCategoryID, 
            ProductName, ProductType, ProductCategory,
            ProductRetailPrice, ProductWholesalePrice, ProductCost,
            ProductRetailProfit, ProductWholesaleUnitProfit, ProductProfitMarginUnitPercent
        )
        SELECT 
            p.ProductID,
            p.ProductTypeID,
            pt.ProductCategoryID,
            COALESCE(p.Product, 'Unknown') as ProductName,
            COALESCE(pt.ProductType, 'Unknown') as ProductType,
            COALESCE(pc.ProductCategory, 'Unknown') as ProductCategory,
            COALESCE(CAST(p.Price AS FLOAT), 0) as ProductRetailPrice,
            COALESCE(CAST(p.WholesalePrice AS FLOAT), 0) as ProductWholesalePrice,
            COALESCE(CAST(p.Cost AS FLOAT), 0) as ProductCost,
            COALESCE(CAST(p.Price AS FLOAT), 0) - COALESCE(CAST(p.Cost AS FLOAT), 0) as ProductRetailProfit,
            COALESCE(CAST(p.WholesalePrice AS FLOAT), 0) - COALESCE(CAST(p.Cost AS FLOAT), 0) as ProductWholesaleUnitProfit,
            CASE 
                WHEN COALESCE(CAST(p.Price AS FLOAT), 0) = 0 THEN 0
                ELSE ((COALESCE(CAST(p.Price AS FLOAT), 0) - COALESCE(CAST(p.Cost AS FLOAT), 0)) / COALESCE(CAST(p.Price AS FLOAT), 1)) * 100
            END as ProductProfitMarginUnitPercent
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_PRODUCT p
        JOIN {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_PRODUCTTYPE pt 
            ON p.ProductTypeID = pt.ProductTypeID
        JOIN {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_PRODUCTCATEGORY pc 
            ON pt.ProductCategoryID = pc.ProductCategoryID
        WHERE p.ProductID IS NOT NULL
        """)
        
        # Get row count for Dim_Product
        cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Product")
        product_count = cursor.fetchone()[0]
        print(f"Loaded {product_count} products into Dim_Product")
        
        # Display sample data from each dimension table
        tables = ['Dim_Location', 'Dim_Channel', 'Dim_Customer', 'Dim_Reseller', 'Dim_Store', 'Dim_Product']
        
        for table in tables:
            print(f"\nSample data from {table}:")
            cursor.execute(f"SELECT * FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.{table} LIMIT 5")
            results = cursor.fetchall()
            headers = [column[0] for column in cursor.description]
            print(tabulate(results, headers=headers, tablefmt="grid"))
        
        print(f"\n✅ Dimension tables loaded successfully from staging tables")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error loading dimension tables: {e}")
        return False

if __name__ == "__main__":
    load_dimension_tables() 