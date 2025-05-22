#!/usr/bin/env python3
"""
Step 2: Create Tables
This script creates tables for all 12 entities with proper schemas.
"""
import sys
sys.path.append('.')  # Add current directory to path
from scripts.connection import get_snowflake_connection

# List of entities to create tables for
entities = [
    'channel', 
    'channelcategory', 
    'customer', 
    'product', 
    'productcategory', 
    'producttype', 
    'reseller', 
    'salesdetail', 
    'salesheader', 
    'store', 
    'targetdatachannel', 
    'targetdataproduct'
]

# Schema definitions for each table based on CSV_schemas.md
table_schemas = {
    'channel': """(
        ChannelID INTEGER,
        ChannelCategoryID INTEGER,
        Channel STRING,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING
    )""",
    'channelcategory': """(
        ChannelCategoryID INTEGER,
        ChannelCategory STRING,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING
    )""",
    'customer': """(
        CustomerID INTEGER,
        SubSegmentID INTEGER,
        FirstName STRING,
        LastName STRING,
        Gender STRING,
        EmailAddress STRING,
        Address STRING,
        City STRING,
        StateProvince STRING,
        Country STRING,
        PostalCode STRING,
        PhoneNumber STRING,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING
    )""",
    'product': """(
        ProductID INTEGER,
        ProductTypeID INTEGER,
        Product STRING,
        Color STRING,
        Style STRING,
        UnitofMeasureID INTEGER,
        Weight FLOAT,
        Price FLOAT,
        Cost FLOAT,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING,
        WholesalePrice FLOAT
    )""",
    'productcategory': """(
        ProductCategoryID INTEGER,
        ProductCategory STRING,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING
    )""",
    'producttype': """(
        ProductTypeID INTEGER,
        ProductCategoryID INTEGER,
        ProductType STRING,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING
    )""",
    'reseller': """(
        ResellerID INTEGER,
        Contact STRING,
        EmailAddress STRING,
        Address STRING,
        City STRING,
        StateProvince STRING,
        Country STRING,
        PostalCode STRING,
        PhoneNumber STRING,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING,
        ResellerName STRING
    )""",
    'salesdetail': """(
        SalesDetailID INTEGER,
        SalesHeaderID INTEGER,
        ProductID INTEGER,
        SalesQuantity INTEGER,
        SalesAmount FLOAT,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING
    )""",
    'salesheader': """(
        SalesHeaderID INTEGER,
        Date TIMESTAMP_NTZ,
        ChannelID INTEGER,
        StoreID INTEGER,
        CustomerID INTEGER,
        ResellerID INTEGER,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING
    )""",
    'store': """(
        StoreID INTEGER,
        SubSegmentID INTEGER,
        StoreNumber STRING,
        StoreManager STRING,
        Address STRING,
        City STRING,
        StateProvince STRING,
        Country STRING,
        PostalCode STRING,
        PhoneNumber STRING,
        CreatedDate TIMESTAMP_NTZ,
        CreatedBy STRING,
        ModifiedDate TIMESTAMP_NTZ,
        ModifiedBy STRING
    )""",
    'targetdatachannel': """(
        Year INTEGER,
        ChannelName STRING,
        TargetName STRING,
        TargetSalesAmount FLOAT
    )""",
    'targetdataproduct': """(
        ProductID INTEGER,
        Product STRING,
        Year INTEGER,
        SalesQuantityTarget INTEGER
    )"""
}

def create_tables():
    """
    Create tables for all entities with proper schemas.
    """
    # Connect to Snowflake
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Create tables for each entity
        created_tables = []
        failed_tables = []
        
        for entity in entities:
            table_name = f"TABLE_{entity.upper()}"
            
            if entity in table_schemas:
                try:
                    print(f"Creating table {table_name}...")
                    cursor.execute(f"""
                    CREATE OR REPLACE TABLE {table_name}
                    {table_schemas[entity]}
                    """)
                    created_tables.append(table_name)
                    print(f"Table {table_name} created successfully.")
                except Exception as e:
                    print(f"Error creating table {table_name}: {str(e)}")
                    failed_tables.append((table_name, str(e)))
            else:
                print(f"Warning: No schema defined for {entity}")
                failed_tables.append((table_name, "No schema defined"))
        
        # Verify that tables exist
        print("\nVerifying tables...")
        cursor.execute("SHOW TABLES")
        existing_tables = [row[1] for row in cursor.fetchall()]
        
        print("\nSummary:")
        print(f"Tables created: {len(created_tables)}")
        for table in created_tables:
            status = "✅ Exists" if table in existing_tables else "❌ Not found"
            print(f"- {table}: {status}")
        
        if failed_tables:
            print(f"\nFailed tables: {len(failed_tables)}")
            for table, error in failed_tables:
                print(f"- {table}: {error}")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        print("\nTable creation process completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    create_tables() 