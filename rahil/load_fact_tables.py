#!/usr/bin/env python3
"""
Load data from staging tables to fact tables using SQLAlchemy
"""
import snowflake.connector
from tabulate import tabulate
from sqlalchemy import create_engine, Table, MetaData, select, func, cast, Float, Integer, text, case, and_, or_
from sqlalchemy.sql.sqltypes import String
from sqlalchemy.orm import sessionmaker
from .dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA, STAGING_DB_NAME
)

# Import fact models
from .schemas.fact import FactBase
from .schemas.fact.sales_actual import FactSalesActual
from .schemas.fact.product_sales_target import FactProductSalesTarget
from .schemas.fact.src_sales_target import FactSRCSalesTarget

# Import dimension models (needed for lookups)
from .schemas.dimension.product import DimProduct
from .schemas.dimension.store import DimStore
from .schemas.dimension.reseller import DimReseller
from .schemas.dimension.customer import DimCustomer
from .schemas.dimension.channel import DimChannel

def get_snowflake_engine(database):
    """Create SQLAlchemy engine for Snowflake"""
    return create_engine(
        f"snowflake://{SNOWFLAKE_USER}:{SNOWFLAKE_PASSWORD}@{SNOWFLAKE_ACCOUNT}/"
        f"{database}/{SNOWFLAKE_SCHEMA}?warehouse={SNOWFLAKE_WAREHOUSE}&role={SNOWFLAKE_ROLE}"
    )

class FactLoader:
    """Base class for fact loaders to abstract loading logic"""
    
    def __init__(self, staging_engine, dim_engine):
        """Initialize with database engines"""
        self.staging_engine = staging_engine
        self.dim_engine = dim_engine
        self.staging_metadata = MetaData()
        self.dimension_metadata = MetaData()
        
        # Create sessions
        Session = sessionmaker(bind=dim_engine)
        self.session = Session()
    
    def get_row_count(self, table_name):
        """Get row count for a fact table"""
        with self.dim_engine.connect() as conn:
            result = conn.execute(text(
                f"SELECT COUNT(*) FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.{table_name}"
            ))
            return result.scalar()
    
    def get_sample_data(self, table_name, limit=5):
        """Get sample data from a fact table"""
        with self.dim_engine.connect() as conn:
            result = conn.execute(text(
                f"SELECT * FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.{table_name} LIMIT {limit}"
            ))
            rows = result.fetchall()
            headers = result.keys()
            return rows, headers
    
    def get_unknown_dimension_id(self, dim_table, id_column, condition_column=None, condition_value=None):
        """Get the ID of the Unknown member from a dimension table"""
        query = select(getattr(dim_table.c, id_column))
        
        if condition_column and condition_value:
            query = query.where(getattr(dim_table.c, condition_column) == condition_value)
        
        with self.dim_engine.connect() as conn:
            result = conn.execute(query).first()
            return result[0] if result else None

class SalesActualLoader(FactLoader):
    """Loader for the SalesActual fact table"""
    
    def load(self):
        """Load data from staging tables to the SalesActual fact table"""
        print("\nLoading Fact_SalesActual table...")
        
        # Reflect necessary tables
        staging_sales_header = Table('STAGING_SALESHEADER', self.staging_metadata, autoload_with=self.staging_engine)
        staging_sales_detail = Table('STAGING_SALESDETAIL', self.staging_metadata, autoload_with=self.staging_engine)
        
        dim_product = Table('DIM_PRODUCT', self.dimension_metadata, autoload_with=self.dim_engine)
        dim_store = Table('DIM_STORE', self.dimension_metadata, autoload_with=self.dim_engine)
        dim_reseller = Table('DIM_RESELLER', self.dimension_metadata, autoload_with=self.dim_engine)
        dim_customer = Table('DIM_CUSTOMER', self.dimension_metadata, autoload_with=self.dim_engine)
        dim_channel = Table('DIM_CHANNEL', self.dimension_metadata, autoload_with=self.dim_engine)
        dim_date = Table('DIM_DATE', self.dimension_metadata, autoload_with=self.dim_engine)
        dim_location = Table('DIM_LOCATION', self.dimension_metadata, autoload_with=self.dim_engine)
        
        # Get unknown dimension IDs
        unknown_product_id = self.get_unknown_dimension_id(dim_product, 'DimProductID', 'ProductID', -1)
        unknown_store_id = self.get_unknown_dimension_id(dim_store, 'DimStoreID', 'StoreID', -1)
        unknown_reseller_id = self.get_unknown_dimension_id(dim_reseller, 'DimResellerID', 'ResellerID', 'UNKNOWN')
        unknown_customer_id = self.get_unknown_dimension_id(dim_customer, 'DimCustomerID', 'CustomerID', 'UNKNOWN')
        unknown_channel_id = self.get_unknown_dimension_id(dim_channel, 'DimChannelID', 'ChannelID', -1)
        unknown_location_id = self.get_unknown_dimension_id(dim_location, 'DimLocationID')
        
        # Check for necessary data to continue
        if None in [unknown_product_id, unknown_store_id, unknown_reseller_id, 
                   unknown_customer_id, unknown_channel_id, unknown_location_id]:
            print("⚠️ One or more Unknown dimension members not found. Please create dimension tables first.")
            return 0
        
        # Build the query to extract sales data
        with self.staging_engine.connect() as staging_conn, self.dim_engine.connect() as dim_conn:
            # Get staging data - join sales header and detail
            sales_data = staging_conn.execute(
                select(
                    staging_sales_detail.c.SalesDetailID,
                    staging_sales_detail.c.SalesHeaderID,
                    staging_sales_detail.c.ProductID,
                    staging_sales_header.c.StoreID,
                    staging_sales_header.c.ResellerID,
                    staging_sales_header.c.CustomerID,
                    staging_sales_header.c.ChannelID,
                    func.to_date(staging_sales_header.c.Date).label('SaleDate'),
                    func.coalesce(cast(staging_sales_detail.c.SalesAmount, Float), 0).label('SaleAmount'),
                    func.coalesce(cast(staging_sales_detail.c.SalesQuantity, Integer), 0).label('SaleQuantity'),
                    func.coalesce(cast(staging_sales_detail.c.UnitPrice, Float), 0).label('SaleUnitPrice'),
                    func.coalesce(cast(staging_sales_detail.c.ExtendedCost, Float), 0).label('SaleExtendedCost'),
                    func.coalesce(cast(staging_sales_detail.c.UnitCost, Float), 0).label('UnitCost')
                ).join(
                    staging_sales_header,
                    staging_sales_detail.c.SalesHeaderID == staging_sales_header.c.SalesHeaderID
                ).where(
                    and_(
                        staging_sales_detail.c.SalesDetailID.isnot(None),
                        staging_sales_header.c.SalesHeaderID.isnot(None)
                    )
                )
            ).fetchall()
            
            # Process each sales record and insert into fact table
            for sale in sales_data:
                # Look up dimension keys
                # Product dimension lookup
                product_query = select(dim_product.c.DimProductID).where(
                    dim_product.c.ProductID == sale.ProductID
                )
                product_result = dim_conn.execute(product_query).first()
                product_id = product_result.DimProductID if product_result else unknown_product_id
                
                # Store dimension lookup (only if StoreID is not null)
                store_id = unknown_store_id
                if sale.StoreID is not None:
                    store_query = select(dim_store.c.DimStoreID).where(
                        dim_store.c.StoreID == sale.StoreID
                    )
                    store_result = dim_conn.execute(store_query).first()
                    store_id = store_result.DimStoreID if store_result else unknown_store_id
                
                # Reseller dimension lookup (only if ResellerID is not null)
                reseller_id = unknown_reseller_id
                if sale.ResellerID is not None:
                    reseller_query = select(dim_reseller.c.DimResellerID).where(
                        dim_reseller.c.ResellerID == sale.ResellerID
                    )
                    reseller_result = dim_conn.execute(reseller_query).first()
                    reseller_id = reseller_result.DimResellerID if reseller_result else unknown_reseller_id
                
                # Customer dimension lookup (only if CustomerID is not null)
                customer_id = unknown_customer_id
                if sale.CustomerID is not None:
                    customer_query = select(dim_customer.c.DimCustomerID).where(
                        dim_customer.c.CustomerID == sale.CustomerID
                    )
                    customer_result = dim_conn.execute(customer_query).first()
                    customer_id = customer_result.DimCustomerID if customer_result else unknown_customer_id
                
                # Channel dimension lookup
                channel_id = unknown_channel_id
                if sale.ChannelID is not None:
                    channel_query = select(dim_channel.c.DimChannelID).where(
                        dim_channel.c.ChannelID == sale.ChannelID
                    )
                    channel_result = dim_conn.execute(channel_query).first()
                    channel_id = channel_result.DimChannelID if channel_result else unknown_channel_id
                
                # Date dimension lookup
                sale_date_id = None
                if sale.SaleDate is not None:
                    date_str = sale.SaleDate.strftime('%Y%m%d')
                    date_query = select(dim_date.c.DateID).where(
                        dim_date.c.DateID == date_str
                    )
                    date_result = dim_conn.execute(date_query).first()
                    sale_date_id = date_result.DateID if date_result else None
                
                # If date not found, use a default
                if sale_date_id is None:
                    sale_date_id = 19000101  # Default to Jan 1, 1900
                
                # Calculate profit
                sale_total_profit = sale.SaleAmount - sale.SaleExtendedCost
                
                # Check if record already exists
                existing = self.session.query(FactSalesActual).filter(
                    FactSalesActual.SalesDetailID == sale.SalesDetailID
                ).first()
                
                if not existing:
                    # Insert into fact table using ORM
                    self.session.add(FactSalesActual(
                        DimProductID=product_id,
                        DimStoreID=store_id,
                        DimResellerID=reseller_id,
                        DimCustomerID=customer_id,
                        DimChannelID=channel_id,
                        DimSaleDateID=sale_date_id,
                        DimLocationID=unknown_location_id,  # Default to Unknown location
                        SalesHeaderID=sale.SalesHeaderID,
                        SalesDetailID=sale.SalesDetailID,
                        SaleAmount=sale.SaleAmount,
                        SaleQuantity=sale.SaleQuantity,
                        SaleUnitPrice=sale.SaleUnitPrice,
                        SaleExtendedCost=sale.SaleExtendedCost,
                        SaleTotalProfit=sale_total_profit
                    ))
            
            # Commit all the inserts at once for better performance
            self.session.commit()
        
        # Get the count of rows loaded
        row_count = self.get_row_count('FACT_SALESACTUAL')
        print(f"Loaded {row_count} rows into Fact_SalesActual")
        return row_count

class ProductSalesTargetLoader(FactLoader):
    """Loader for the ProductSalesTarget fact table"""
    
    def load(self):
        """Load data from staging tables to the ProductSalesTarget fact table"""
        print("\nLoading Fact_ProductSalesTarget table...")
        
        # Reflect necessary tables
        staging_target_product = Table('STAGING_TARGETDATAPRODUCT', self.staging_metadata, autoload_with=self.staging_engine)
        dim_product = Table('DIM_PRODUCT', self.dimension_metadata, autoload_with=self.dim_engine)
        
        # Get unknown product ID
        unknown_product_id = self.get_unknown_dimension_id(dim_product, 'DimProductID', 'ProductID', -1)
        
        if unknown_product_id is None:
            print("⚠️ Unknown product member not found. Please create dimension tables first.")
            return 0
        
        # Build the query to extract target data
        with self.staging_engine.connect() as staging_conn, self.dim_engine.connect() as dim_conn:
            # Get target data
            target_data = staging_conn.execute(
                select(
                    staging_target_product.c.ProductID,
                    staging_target_product.c.Year,
                    func.coalesce(cast(staging_target_product.c.SalesQuantityTarget, Integer), 0).label('SalesQuantityTarget')
                ).where(
                    staging_target_product.c.ProductID.isnot(None)
                )
            ).fetchall()
            
            # Process each target record and insert into fact table
            for target in target_data:
                # Look up product dimension
                product_query = select(dim_product.c.DimProductID).where(
                    dim_product.c.ProductID == target.ProductID
                )
                product_result = dim_conn.execute(product_query).first()
                product_id = product_result.DimProductID if product_result else unknown_product_id
                
                # Create target date ID (first day of year)
                year = target.Year if target.Year is not None else 1900
                target_date_id = int(f"{year}0101")
                
                # Check if record already exists - Product and date are the composite key
                existing = self.session.query(FactProductSalesTarget).filter(
                    FactProductSalesTarget.DimProductID == product_id,
                    FactProductSalesTarget.DimTargetDateID == target_date_id
                ).first()
                
                if not existing:
                    # Insert into fact table using ORM
                    self.session.add(FactProductSalesTarget(
                        DimProductID=product_id,
                        DimTargetDateID=target_date_id,
                        ProductTargetSalesQuantity=target.SalesQuantityTarget
                    ))
            
            # Commit all the inserts at once for better performance
            self.session.commit()
        
        # Get the count of rows loaded
        row_count = self.get_row_count('FACT_PRODUCTSALESTARGET')
        print(f"Loaded {row_count} rows into Fact_ProductSalesTarget")
        return row_count

class SRCSalesTargetLoader(FactLoader):
    """Loader for the SRCSalesTarget fact table"""
    
    def load(self):
        """Load data from staging tables to the SRCSalesTarget fact table"""
        print("\nLoading Fact_SRCSalesTarget table...")
        
        # Reflect necessary tables
        staging_target_channel = Table('STAGING_TARGETDATACHANNEL', self.staging_metadata, autoload_with=self.staging_engine)
        dim_store = Table('DIM_STORE', self.dimension_metadata, autoload_with=self.dim_engine)
        dim_reseller = Table('DIM_RESELLER', self.dimension_metadata, autoload_with=self.dim_engine)
        dim_channel = Table('DIM_CHANNEL', self.dimension_metadata, autoload_with=self.dim_engine)
        
        # Get unknown dimension IDs
        unknown_store_id = self.get_unknown_dimension_id(dim_store, 'DimStoreID', 'StoreID', -1)
        unknown_reseller_id = self.get_unknown_dimension_id(dim_reseller, 'DimResellerID', 'ResellerID', 'UNKNOWN')
        unknown_channel_id = self.get_unknown_dimension_id(dim_channel, 'DimChannelID', 'ChannelID', -1)
        
        if None in [unknown_store_id, unknown_reseller_id, unknown_channel_id]:
            print("⚠️ One or more Unknown dimension members not found. Please create dimension tables first.")
            return 0
        
        # Build the query to extract target data
        with self.staging_engine.connect() as staging_conn, self.dim_engine.connect() as dim_conn:
            # Get target data
            target_data = staging_conn.execute(
                select(
                    staging_target_channel.c.TargetName,
                    staging_target_channel.c.ChannelName,
                    staging_target_channel.c.Year,
                    func.coalesce(cast(staging_target_channel.c.TargetSalesAmount, Float), 0).label('TargetSalesAmount')
                ).where(
                    staging_target_channel.c.ChannelName.isnot(None)
                )
            ).fetchall()
            
            # Process each target record and insert into fact table
            for target in target_data:
                # Determine appropriate dimension IDs based on target type
                store_id = unknown_store_id
                reseller_id = unknown_reseller_id
                
                target_name_upper = target.TargetName.upper() if target.TargetName else 'UNKNOWN'
                
                # Handle Store targets
                if target_name_upper == 'STORE':
                    # Get all stores for this target
                    store_query = select(dim_store.c.DimStoreID)
                    store_results = dim_conn.execute(store_query).fetchall()
                    
                    if not store_results:
                        # No stores found, use unknown
                        store_ids = [unknown_store_id]
                    else:
                        # Use all actual stores
                        store_ids = [r.DimStoreID for r in store_results if r.DimStoreID != unknown_store_id]
                        if not store_ids:
                            store_ids = [unknown_store_id]
                else:
                    # Not a store target
                    store_ids = [unknown_store_id]
                
                # Handle Reseller targets
                if target_name_upper == 'RESELLER':
                    # Get all resellers for this target
                    reseller_query = select(dim_reseller.c.DimResellerID)
                    reseller_results = dim_conn.execute(reseller_query).fetchall()
                    
                    if not reseller_results:
                        # No resellers found, use unknown
                        reseller_ids = [unknown_reseller_id]
                    else:
                        # Use all actual resellers
                        reseller_ids = [r.DimResellerID for r in reseller_results if r.DimResellerID != unknown_reseller_id]
                        if not reseller_ids:
                            reseller_ids = [unknown_reseller_id]
                else:
                    # Not a reseller target
                    reseller_ids = [unknown_reseller_id]
                
                # Channel dimension lookup
                channel_id = unknown_channel_id
                if target.ChannelName:
                    channel_query = select(dim_channel.c.DimChannelID).where(
                        dim_channel.c.ChannelName == target.ChannelName
                    )
                    channel_result = dim_conn.execute(channel_query).first()
                    channel_id = channel_result.DimChannelID if channel_result else unknown_channel_id
                
                # Create target date ID (first day of year)
                year = target.Year if target.Year is not None else 1900
                target_date_id = int(f"{year}0101")
                
                # For each relevant store/reseller combination, create a target record
                for store_id in store_ids:
                    for reseller_id in reseller_ids:
                        # Skip irrelevant combinations (if store target, only unknown reseller, and vice versa)
                        if (target_name_upper == 'STORE' and reseller_id != unknown_reseller_id) or \
                           (target_name_upper == 'RESELLER' and store_id != unknown_store_id):
                            continue
                        
                        # Check if record already exists
                        existing = self.session.query(FactSRCSalesTarget).filter(
                            FactSRCSalesTarget.DimStoreID == store_id,
                            FactSRCSalesTarget.DimResellerID == reseller_id,
                            FactSRCSalesTarget.DimChannelID == channel_id,
                            FactSRCSalesTarget.DimTargetDateID == target_date_id
                        ).first()
                        
                        if not existing:
                            # Insert into fact table using ORM
                            self.session.add(FactSRCSalesTarget(
                                DimStoreID=store_id,
                                DimResellerID=reseller_id,
                                DimChannelID=channel_id,
                                DimTargetDateID=target_date_id,
                                SalesTargetAmount=target.TargetSalesAmount
                            ))
            
            # Commit all the inserts at once for better performance
            self.session.commit()
        
        # Get the count of rows loaded
        row_count = self.get_row_count('FACT_SRCSALESTARGET')
        print(f"Loaded {row_count} rows into Fact_SRCSalesTarget")
        return row_count

def load_fact_tables():
    """Load data from staging tables to fact tables using SQLAlchemy"""
    print(f"Step 5: Loading data from {STAGING_DB_NAME} to {DIMENSION_DB_NAME} fact tables")
    
    try:
        # Create a traditional Snowflake connection to create database and use it
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE
        )
        cursor = conn.cursor()
        
        # Make sure databases exist and are used
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DIMENSION_DB_NAME}")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {STAGING_DB_NAME}")
        cursor.execute(f"USE DATABASE {STAGING_DB_NAME}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")
        
        # Check if staging tables exist
        cursor.execute(f"SHOW TABLES IN {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}")
        staging_tables = cursor.fetchall()
        print(f"Staging tables found: {len(staging_tables)}")
        
        if len(staging_tables) == 0:
            print(f"⚠️ No staging tables found in {STAGING_DB_NAME}.{SNOWFLAKE_SCHEMA}")
            print("Please create staging tables first before running this script.")
            cursor.close()
            conn.close()
            return False
        
        # Also check if fact tables exist
        cursor.execute(f"USE DATABASE {DIMENSION_DB_NAME}")
        cursor.execute(f"SHOW TABLES LIKE 'FACT_%' IN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
        fact_tables = cursor.fetchall()
        
        if len(fact_tables) == 0:
            print(f"⚠️ No fact tables found in {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}")
            print("Please create fact tables first before running this script.")
            cursor.close()
            conn.close()
            return False
        
        # Close initial cursor
        cursor.close()
        conn.close()
        
        # Create engines for staging and dimension databases
        staging_engine = get_snowflake_engine(STAGING_DB_NAME)
        dim_engine = get_snowflake_engine(DIMENSION_DB_NAME)
        
        # Create and run loaders for each fact table
        loaders = [
            SalesActualLoader(staging_engine, dim_engine),
            ProductSalesTargetLoader(staging_engine, dim_engine),
            SRCSalesTargetLoader(staging_engine, dim_engine)
        ]
        
        # Load each fact table
        for loader in loaders:
            loader.load()
        
        # Display sample data from each fact table
        tables = ['FACT_SALESACTUAL', 'FACT_PRODUCTSALESTARGET', 'FACT_SRCSALESTARGET']
        
        # Create a traditional connection for sample data display
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE
        )
        cursor = conn.cursor()
        
        for table in tables:
            print(f"\nSample data from {table}:")
            cursor.execute(f"SELECT * FROM {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.{table} LIMIT 5")
            results = cursor.fetchall()
            headers = [column[0] for column in cursor.description]
            print(tabulate(results, headers=headers, tablefmt="grid"))
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        print(f"\n✅ Fact tables loaded successfully from staging tables")
        return True
    
    except Exception as e:
        print(f"\n❌ Error loading fact tables: {e}")
        return False

if __name__ == "__main__":
    load_fact_tables() 