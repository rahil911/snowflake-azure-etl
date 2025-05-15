#!/usr/bin/env python3
"""
Load data from staging tables to fact tables
"""
import snowflake.connector
from tabulate import tabulate
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA, STAGING_DB_NAME
)

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
        
        # Load Fact_SalesActual table
        print("\nLoading Fact_SalesActual table...")
        cursor.execute(f"""
        INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_SalesActual (
            DimProductID, DimStoreID, DimResellerID, DimCustomerID, DimChannelID, DimSaleDateID, DimLocationID,
            SalesHeaderID, SalesDetailID, SaleAmount, SaleQuantity, SaleUnitPrice, SaleExtendedCost, SaleTotalProfit
        )
        SELECT 
            -- Use COALESCE to get the Unknown product if the join fails
            COALESCE(dp.DimProductID, (SELECT DimProductID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Product WHERE ProductID = -1)),
            
            -- Store dimension - use CASE WHEN for clarity
            CASE 
                WHEN sh.StoreID IS NOT NULL THEN 
                    COALESCE(ds.DimStoreID, (SELECT DimStoreID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Store WHERE StoreID = -1))
                ELSE 
                    (SELECT DimStoreID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Store WHERE StoreID = -1)
            END as DimStoreID,
            
            -- Reseller dimension
            CASE 
                WHEN sh.ResellerID IS NOT NULL THEN 
                    COALESCE(dr.DimResellerID, (SELECT DimResellerID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Reseller WHERE ResellerID = 'UNKNOWN'))
                ELSE 
                    (SELECT DimResellerID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Reseller WHERE ResellerID = 'UNKNOWN')
            END as DimResellerID,
            
            -- Customer dimension
            CASE 
                WHEN sh.CustomerID IS NOT NULL THEN 
                    COALESCE(dc.DimCustomerID, (SELECT DimCustomerID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Customer WHERE CustomerID = 'UNKNOWN'))
                ELSE 
                    (SELECT DimCustomerID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Customer WHERE CustomerID = 'UNKNOWN')
            END as DimCustomerID,
            
            -- Channel dimension
            COALESCE(dch.DimChannelID, (SELECT DimChannelID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Channel WHERE ChannelID = -1)),
            
            -- Date dimension - convert date or use default if NULL
            CASE
                WHEN sh.Date IS NULL THEN 19000101  -- Default to Jan 1, 1900 if date is NULL
                ELSE TO_NUMBER(TO_CHAR(TO_DATE(sh.Date), 'YYYYMMDD'))
            END as DimSaleDateID,
            
            -- Location dimension based on the entity type (Store, Reseller, Customer)
            CASE 
                WHEN sh.StoreID IS NOT NULL AND ds.DimLocationID IS NOT NULL THEN ds.DimLocationID
                WHEN sh.ResellerID IS NOT NULL AND dr.DimLocationID IS NOT NULL THEN dr.DimLocationID
                WHEN sh.CustomerID IS NOT NULL AND dc.DimLocationID IS NOT NULL THEN dc.DimLocationID
                ELSE 1  -- Default to Unknown Location
            END as DimLocationID,
            
            sh.SalesHeaderID,
            sd.SalesDetailID,
            
            -- Handle financial values - make sure they're not NULL
            CAST(COALESCE(sd.SalesAmount, 0) AS FLOAT) as SalesAmount,
            COALESCE(sd.SalesQuantity, 0) as SalesQuantity,
            
            -- Calculate unit price safely (avoid division by zero)
            CASE
                WHEN COALESCE(sd.SalesQuantity, 0) = 0 THEN 0
                ELSE CAST(COALESCE(sd.SalesAmount, 0) AS FLOAT) / COALESCE(sd.SalesQuantity, 1)
            END as SaleUnitPrice,
            
            -- Extended cost safely
            COALESCE(sd.SalesQuantity, 0) * COALESCE(CAST(dp.ProductCost AS FLOAT), 0) as SaleExtendedCost,
            
            -- Total profit safely
            CAST(COALESCE(sd.SalesAmount, 0) AS FLOAT) - (COALESCE(sd.SalesQuantity, 0) * COALESCE(CAST(dp.ProductCost AS FLOAT), 0)) as SaleTotalProfit
            
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_SALESHEADER sh
        JOIN {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_SALESDETAIL sd 
            ON sh.SalesHeaderID = sd.SalesHeaderID
        -- Use LEFT JOIN for dimension tables to handle missing references
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Product dp 
            ON sd.ProductID = dp.ProductID
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Channel dch 
            ON sh.ChannelID = dch.ChannelID
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Store ds 
            ON sh.StoreID = ds.StoreID
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Reseller dr 
            ON sh.ResellerID = dr.ResellerID
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Customer dc 
            ON sh.CustomerID = dc.CustomerID
        -- Filter out completely invalid records 
        WHERE sd.SalesDetailID IS NOT NULL AND sh.SalesHeaderID IS NOT NULL
        """)
        
        # Get row count for Fact_SalesActual
        cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_SalesActual")
        sales_count = cursor.fetchone()[0]
        print(f"Loaded {sales_count} rows into Fact_SalesActual")
        
        # Load Fact_ProductSalesTarget table
        print("\nLoading Fact_ProductSalesTarget table...")
        cursor.execute(f"""
        INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_ProductSalesTarget (
            DimProductID, DimTargetDateID, ProductTargetSalesQuantity
        )
        SELECT 
            -- Use COALESCE to get the Unknown product if the join fails
            COALESCE(dp.DimProductID, (SELECT DimProductID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Product WHERE ProductID = -1)),
            
            -- Convert Year to first day of year (January 1st)
            -- If year is NULL, default to year 1900
            TO_NUMBER(COALESCE(td.Year, 1900) || '0101'),
            
            -- Ensure target quantity is not NULL
            COALESCE(td.SalesQuantityTarget, 0) as SalesQuantityTarget
            
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_TARGETDATAPRODUCT td
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Product dp 
            ON td.ProductID = dp.ProductID
        WHERE td.ProductID IS NOT NULL
        """)
        
        # Get row count for Fact_ProductSalesTarget
        cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_ProductSalesTarget")
        product_target_count = cursor.fetchone()[0]
        print(f"Loaded {product_target_count} rows into Fact_ProductSalesTarget")
        
        # Load Fact_SRCSalesTarget table
        print("\nLoading Fact_SRCSalesTarget table...")
        cursor.execute(f"""
        INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_SRCSalesTarget (
            DimStoreID, DimResellerID, DimChannelID, DimTargetDateID, SalesTargetAmount
        )
        SELECT
            -- Store dimension
            CASE 
                WHEN UPPER(COALESCE(td.TargetName, 'UNKNOWN')) = 'STORE' THEN 
                    COALESCE(ds.DimStoreID, (SELECT DimStoreID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Store WHERE StoreID = -1))
                ELSE 
                    (SELECT DimStoreID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Store WHERE StoreID = -1)
            END as DimStoreID,
            
            -- Reseller dimension
            CASE 
                WHEN UPPER(COALESCE(td.TargetName, 'UNKNOWN')) = 'RESELLER' THEN 
                    COALESCE(dr.DimResellerID, (SELECT DimResellerID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Reseller WHERE ResellerID = 'UNKNOWN'))
                ELSE 
                    (SELECT DimResellerID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Reseller WHERE ResellerID = 'UNKNOWN')
            END as DimResellerID,
            
            -- Channel dimension
            COALESCE(dc.DimChannelID, (SELECT DimChannelID FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Channel WHERE ChannelID = -1)),
            
            -- Convert Year to first day of year (January 1st)
            -- If year is NULL, default to year 1900
            TO_NUMBER(COALESCE(td.Year, 1900) || '0101'),
            
            -- Ensure target amount is not NULL
            COALESCE(td.TargetSalesAmount, 0) as TargetSalesAmount
            
        FROM {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}.STAGING_TARGETDATACHANNEL td
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Channel dc 
            ON td.ChannelName = dc.ChannelName
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Store ds
            ON td.TargetName = 'Store'
        LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Reseller dr
            ON td.TargetName = 'Reseller'
        -- Ensure we have a valid channel
        WHERE td.ChannelName IS NOT NULL
        """)
        
        # Get row count for Fact_SRCSalesTarget
        cursor.execute(f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_SRCSalesTarget")
        src_target_count = cursor.fetchone()[0]
        print(f"Loaded {src_target_count} rows into Fact_SRCSalesTarget")
        
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