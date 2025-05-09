#!/usr/bin/env python3
"""
Script to create staging tables in Snowflake
"""
from . import config
from .connection import get_snowflake_connection

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
        
        # Create tables
        table_creation_statements = [
            """
            -- 1. CHANNEL
            CREATE OR REPLACE TABLE STAGING_CHANNEL (
              CHANNELID            INTEGER,
              CHANNELCATEGORYID    INTEGER,
              CHANNEL              VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR
            );
            """,
            
            """
            -- 2. CHANNELCATEGORY
            CREATE OR REPLACE TABLE STAGING_CHANNELCATEGORY (
              CHANNELCATEGORYID    INTEGER,
              CHANNELCATEGORY      VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR
            );
            """,
            
            """
            -- 3. CUSTOMER
            CREATE OR REPLACE TABLE STAGING_CUSTOMER (
              CUSTOMERID           VARCHAR,
              SUBSEGMENTID         INTEGER,
              FIRSTNAME            VARCHAR,
              LASTNAME             VARCHAR,
              GENDER               VARCHAR,
              EMAILADDRESS         VARCHAR,
              ADDRESS              VARCHAR,
              CITY                 VARCHAR,
              STATEPROVINCE        VARCHAR,
              COUNTRY              VARCHAR,
              POSTALCODE           INTEGER,
              PHONENUMBER          VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR
            );
            """,
            
            """
            -- 4. PRODUCT
            CREATE OR REPLACE TABLE STAGING_PRODUCT (
              PRODUCTID            INTEGER,
              PRODUCTTYPEID        INTEGER,
              PRODUCT              VARCHAR,
              COLOR                VARCHAR,
              STYLE                VARCHAR,
              UNITOFMEASUREID      INTEGER,
              WEIGHT               VARCHAR,
              PRICE                VARCHAR,
              COST                 VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR,
              WHOLESALEPRICE       VARCHAR
            );
            """,
            
            """
            -- 5. PRODUCTCATEGORY
            CREATE OR REPLACE TABLE STAGING_PRODUCTCATEGORY (
              PRODUCTCATEGORYID    INTEGER,
              PRODUCTCATEGORY      VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR
            );
            """,
            
            """
            -- 6. PRODUCTTYPE
            CREATE OR REPLACE TABLE STAGING_PRODUCTTYPE (
              PRODUCTTYPEID        INTEGER,
              PRODUCTCATEGORYID    INTEGER,
              PRODUCTTYPE          VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR
            );
            """,
            
            """
            -- 7. RESELLER
            CREATE OR REPLACE TABLE STAGING_RESELLER (
              RESELLERID           VARCHAR,
              CONTACT              VARCHAR,
              EMAILADDRESS         VARCHAR,
              ADDRESS              VARCHAR,
              CITY                 VARCHAR,
              STATEPROVINCE        VARCHAR,
              COUNTRY              VARCHAR,
              POSTALCODE           INTEGER,
              PHONENUMBER          VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR,
              RESELLERNAME         VARCHAR
            );
            """,
            
            """
            -- 8. STORE
            CREATE OR REPLACE TABLE STAGING_STORE (
              STOREID              INTEGER,
              SUBSEGMENTID         INTEGER,
              STORENUMBER          INTEGER,
              STOREMANAGER         VARCHAR,
              ADDRESS              VARCHAR,
              CITY                 VARCHAR,
              STATEPROVINCE        VARCHAR,
              COUNTRY              VARCHAR,
              POSTALCODE           INTEGER,
              PHONENUMBER          VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR
            );
            """,
            
            """
            -- 9. SALESDETAIL
            CREATE OR REPLACE TABLE STAGING_SALESDETAIL (
              SALESDETAILID        INTEGER,
              SALESHEADERID        INTEGER,
              PRODUCTID            INTEGER,
              SALESQUANTITY        INTEGER,
              SALESAMOUNT          VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR
            );
            """,
            
            """
            -- 10. SALESHEADER
            CREATE OR REPLACE TABLE STAGING_SALESHEADER (
              SALESHEADERID        INTEGER,
              DATE                 VARCHAR,
              CHANNELID            INTEGER,
              STOREID              INTEGER,
              CUSTOMERID           VARCHAR,
              RESELLERID           VARCHAR,
              CREATEDDATE          VARCHAR,
              CREATEDBY            VARCHAR,
              MODIFIEDDATE         VARCHAR,
              MODIFIEDBY           VARCHAR
            );
            """,
            
            """
            -- 11. TARGETDATACHANNEL
            CREATE OR REPLACE TABLE STAGING_TARGETDATACHANNEL (
              YEAR                 INTEGER,
              CHANNELNAME          VARCHAR,
              TARGETNAME           VARCHAR,
              TARGETSALESAMOUNT    INTEGER
            );
            """,
            
            """
            -- 12. TARGETDATAPRODUCT
            CREATE OR REPLACE TABLE STAGING_TARGETDATAPRODUCT (
              PRODUCTID            INTEGER,
              PRODUCT              VARCHAR,
              YEAR                 INTEGER,
              SALESQUANTITYTARGET  INTEGER
            );
            """
        ]
        
        created_tables = []
        
        # Execute each SQL statement
        for i, sql in enumerate(table_creation_statements, 1):
            table_name = f"STAGING_{sql.split('STAGING_')[1].split(' ')[0]}"
            print(f"\nCreating table {i}: {table_name}")
            cursor.execute(sql)
            created_tables.append(table_name)
            print(f"Table {table_name} created successfully.")
        
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