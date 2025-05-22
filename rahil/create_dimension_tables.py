#!/usr/bin/env python3
"""
Create dimension tables for the dimensional model
"""
import snowflake.connector
from sqlalchemy.schema import CreateTable
from sqlalchemy import Integer, create_engine
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
import sys
sys.path.append('.') # To allow importing from parent directory
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA, STAGING_DB_NAME # STAGING_DB_NAME might not be needed here
)
from rahil.schemas import DimensionBase
from rahil.schemas.dimension_loader import get_snowflake_engine # For dim_engine
# Import specific dimension models for adding unknown records (MUST BE AFTER DimensionBase is defined and potentially used by them)
from rahil.schemas.dimension.location import DimLocation
from rahil.schemas.dimension.customer import DimCustomer
from rahil.schemas.dimension.reseller import DimReseller
from rahil.schemas.dimension.store import DimStore
from rahil.schemas.dimension.product import DimProduct
from rahil.schemas.dimension.channel import DimChannel

def create_all_dimension_tables_ddl(cursor, conn, engine_for_dialect):
    """
    Generates and executes DDL for all dimension tables.
    Also attempts to add AUTOINCREMENT to integer PKs if autoincrement=True.
    """
    print(f"Step 1: Creating dimension tables in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
    
    # DEBUG: Print all tables to be processed
    all_table_names = [t.name for t in DimensionBase.metadata.sorted_tables]
    print(f"DEBUG: DimensionBase.metadata.sorted_tables initially contains: {all_table_names}")

    try:
        # Create tables using SQLAlchemy models
        for table in DimensionBase.metadata.sorted_tables:
            print(f"DEBUG: ----- Iterating for table: {table.name} -----") # Ensure this prints for each table
            print(f"Processing table object: {table.name} ({table})")
            try:
                # Ensure CreateTable generates for the correct dialect for IF NOT EXISTS
                ddl_compiler = CreateTable(table).compile(dialect=engine_for_dialect.dialect)
                ddl_statement = str(ddl_compiler)
                
                # Make DDL idempotent
                # Basic string replacement; might need refinement for complex DDL
                if ddl_statement.strip().upper().startswith("CREATE TABLE"):
                    ddl_statement = ddl_statement.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)
                
                print(f"Successfully generated DDL for {table.name}.")

                # If the PK is autoincrement, add AUTOINCREMENT to its DDL definition
                pk_column = table.primary_key.columns[0]
                if pk_column.autoincrement == True and isinstance(pk_column.type, Integer):
                    # Use the passed engine's dialect for compiling
                    pk_col_ddl_part = f'"{pk_column.name}" {pk_column.type.compile(dialect=engine_for_dialect.dialect)} NOT NULL'
                    if pk_col_ddl_part in ddl_statement:
                        ddl_statement = ddl_statement.replace(
                            pk_col_ddl_part,
                            f'"{pk_column.name}" {pk_column.type.compile(dialect=engine_for_dialect.dialect)} AUTOINCREMENT NOT NULL'
                        )
                    else:
                        print(f"Warning: Could not reliably add AUTOINCREMENT for {table.name}. PK DDL part not found: {pk_col_ddl_part}")
                
                print(f"Final DDL for {table.name}: {ddl_statement}")
                cursor.execute(ddl_statement)
                print(f"Successfully executed DDL for {table.name}.")

            except Exception as e:
                print(f"ERROR processing DDL for table {table.name}: {e}")
                if "already exists" in str(e).lower():
                    print(f"Table {table.name} already exists, skipping.")
                else:
                    print(f"Critical error creating table {table.name}. Halting DDL operations.")
                    raise
        
        conn.commit()

        # Verify table creation (optional, can be verbose)
        print("\nVerifying table creation:")
        cursor.execute(f"SHOW TABLES IN SCHEMA {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
        rows = cursor.fetchall()
        headers = [desc[0] for desc in cursor.description]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        if rows:
            print(f"\n✅ Dimension tables created successfully in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
            return True
        else:
            print(f"\n⚠️ No dimension tables found after creation attempt in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error creating dimension tables: {e}")
        return False

def add_unknown_records(dim_engine):
    """Adds standard unknown records to dimension tables."""
    print("\nAdding unknown members to dimension tables (via ORM)...")
    Session = sessionmaker(bind=dim_engine)
    session = Session()

    unknown_records_data = {
        DimLocation: {"DimLocationID": 1, "Address": 'Unknown', "City": 'Unknown', "PostalCode": 'Unknown', "State_Province": 'Unknown', "Country": 'Unknown'},
        DimCustomer: {"DimCustomerID": 1, "CustomerID": 'UNKNOWN', "DimLocationID": 1, "CustomerFullName": 'Unknown Customer', "CustomerFirstName": 'Unknown', "CustomerLastName": 'Unknown', "CustomerGender": 'Unknown'},
        DimReseller: {"DimResellerID": 1, "ResellerID": 'UNKNOWN', "DimLocationID": 1, "ResellerName": 'Unknown Reseller', "ContactName": 'Unknown', "PhoneNumber": 'Unknown', "Email": 'Unknown'},
        DimStore: {"DimStoreID": 1, "StoreID": -1, "DimLocationID": 1, "SourceStoreID": -1, "StoreName": 'Unknown Store', "StoreNumber": 'Unknown', "StoreManager": 'Unknown'},
        DimProduct: {"DimProductID": 1, "ProductID": -1, "ProductTypeID": -1, "ProductCategoryID": -1, "ProductName": 'Unknown Product', "ProductType": 'Unknown', "ProductCategory": 'Unknown', "ProductRetailPrice": 0.0, "ProductWholesalePrice": 0.0, "ProductCost": 0.0, "ProductRetailProfit": 0.0, "ProductWholesaleUnitProfit": 0.0, "ProductProfitMarginUnitPercent": 0.0},
        DimChannel: {"DimChannelID": 1, "ChannelID": -1, "ChannelCategoryID": -1, "ChannelName": 'Unknown Channel', "ChannelCategory": 'Unknown'},
    }
    all_added_successfully = True
    try:
        for model_class, data in unknown_records_data.items():
            pk_col_name = model_class.__mapper__.primary_key[0].name
            pk_val = data[pk_col_name]
            
            existing = session.get(model_class, pk_val)
            if not existing:
                try:
                    record = model_class(**data)
                    session.add(record)
                    print(f"Staged unknown record for {model_class.__tablename__}")
                except Exception as e_add:
                    print(f"Error staging unknown record for {model_class.__tablename__}: {e_add}")
                    all_added_successfully = False
                    # Continue to try other unknown records
        
        if session.new: # Check if there's anything to commit
             session.commit()
             print("Committed unknown records.")
        else:
            print("No new unknown records to commit.")

    except Exception as e:
        session.rollback()
        print(f"Error during unknown records operation: {e}")
        all_added_successfully = False
    finally:
        session.close()
    
    return all_added_successfully

def create_dimension_tables():
    """Main function to connect and create dimension tables"""
    conn = None
    dim_engine = None # Define dim_engine here
    try:
        print(f"Connecting to Snowflake ({DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA})...")
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE,
            database=DIMENSION_DB_NAME,
            schema=SNOWFLAKE_SCHEMA
        )
        cursor = conn.cursor()
        print("Connected.")

        # Create a dim_engine for dialect purposes and for add_unknown_members
        dim_engine = get_snowflake_engine(DIMENSION_DB_NAME)

        if not create_all_dimension_tables_ddl(cursor, conn, dim_engine): # Pass dim_engine
            raise Exception("Failed to create dimension tables via DDL.")

        # Step 2: Add Unknown members using ORM (after tables are surely created and committed)
        if not add_unknown_records(dim_engine):
            print("Warning: Failed to add some or all unknown records.")
            # Decide if this is fatal, for now, we continue
        
        print("\nDimension table creation and unknown member population process completed.")
        return True

    except Exception as e:
        print(f"❌ Error creating dimension tables: {e}")
        return False
    finally:
        if conn:
            conn.close()
            print("Connection closed.")
        if dim_engine: # Dispose the engine
            dim_engine.dispose()

if __name__ == "__main__":
    if not create_dimension_tables():
        sys.exit(1)