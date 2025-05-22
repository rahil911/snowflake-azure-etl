#!/usr/bin/env python3
"""
Base loader class for dimension tables
Abstracts common loading functionality and handles Snowflake case sensitivity issues
"""
import snowflake.connector
from tabulate import tabulate
from sqlalchemy import create_engine, Table, MetaData, select, func, text
from sqlalchemy.orm import sessionmaker
from ..dim_config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, DIMENSION_DB_NAME,
    SNOWFLAKE_SCHEMA, STAGING_DB_NAME
)

def get_snowflake_engine(database):
    """Create SQLAlchemy engine for Snowflake"""
    return create_engine(
        f"snowflake://{SNOWFLAKE_USER}:{SNOWFLAKE_PASSWORD}@{SNOWFLAKE_ACCOUNT}/"
        f"{database}/{SNOWFLAKE_SCHEMA}?warehouse={SNOWFLAKE_WAREHOUSE}&role={SNOWFLAKE_ROLE}",
        connect_args={'session_parameters': {'QUOTED_IDENTIFIERS_IGNORE_CASE': True}},
        implicit_returning=True
    )

class DimensionLoader:
    """Base class for dimension loaders to abstract loading logic"""
    
    def __init__(self, staging_engine, dim_engine):
        """Initialize with database engines"""
        self.staging_engine = staging_engine
        self.dim_engine = dim_engine
        self.staging_metadata = MetaData()
        self.dimension_metadata = MetaData()
        
        # Create sessions
        Session = sessionmaker(bind=dim_engine, autoflush=True)
        self.session = Session()
    
    def ensure_unknown_record(self, model_class, unknown_record):
        """Ensure Unknown reference record exists"""
        try:
            pk_field = model_class.__mapper__.primary_key[0].name
            pk_val = getattr(unknown_record, pk_field)
            
            # Check if record exists using a new session
            with self.dim_engine.connect() as conn:
                table_name = model_class.__tablename__
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM \"{table_name}\" WHERE \"{pk_field}\" = :pk_val"), 
                    {"pk_val": pk_val}
                )
                count = result.scalar()
                
                if count == 0:
                    # Use a fresh session to avoid transaction conflicts
                    Session = sessionmaker(bind=self.dim_engine)
                    session = Session()
                    try:
                        session.add(unknown_record)
                        session.commit()
                        print(f"Added unknown record to {table_name}")
                    except Exception as e:
                        session.rollback()
                        print(f"Error adding unknown record to {table_name}: {e}")
                    finally:
                        session.close()
        except Exception as e:
            print(f"Error ensuring unknown record: {e}")
    
    def get_row_count(self, table_name):
        """Get row count for a dimension table"""
        with self.dim_engine.connect() as conn:
            result = conn.execute(text(
                f"SELECT COUNT(*) FROM \"{DIMENSION_DB_NAME}\".\"{SNOWFLAKE_SCHEMA}\".\"{table_name}\""
            ))
            return result.scalar()
    
    def get_sample_data(self, table_name, limit=5):
        """Get sample data from a dimension table"""
        with self.dim_engine.connect() as conn:
            result = conn.execute(text(
                f"SELECT * FROM \"{DIMENSION_DB_NAME}\".\"{SNOWFLAKE_SCHEMA}\".\"{table_name}\" LIMIT {limit}"
            ))
            rows = result.fetchall()
            headers = result.keys()
            return rows, headers
    
    def execute_text_query(self, query_text, params=None, engine=None):
        """Execute a text query and return results
        This helps avoid case sensitivity issues with Snowflake"""
        try:
            if engine is None:
                engine = self.staging_engine
                
            with engine.connect() as conn:
                if params:
                    result = conn.execute(text(query_text), params)
                else:
                    result = conn.execute(text(query_text))
                return result.fetchall()
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
    
    def lookup_location_id(self, address, city, country):
        """Look up location ID by address, city and country"""
        try:
            query = text("""
                SELECT "DimLocationID" 
                FROM "DIM_LOCATION" 
                WHERE UPPER("Address") = UPPER(:address) 
                AND UPPER("City") = UPPER(:city) 
                AND UPPER("Country") = UPPER(:country)
            """)
            params = {"address": address, "city": city, "country": country}
            with self.dim_engine.connect() as conn:
                result = conn.execute(query, params)
                row = result.fetchone()
                return row[0] if row else 1  # Return the Unknown location ID (1) if not found
        except Exception as e:
            print(f"Error looking up location: {e}")
            return 1  # Return the Unknown location ID (1) on error
    
    def create_fresh_session(self):
        """Create a fresh session and close the old one"""
        if self.session:
            try:
                self.session.close()
            except:
                pass
        
        Session = sessionmaker(bind=self.dim_engine)
        self.session = Session()
        return self.session

    def commit_records(self):
        """Safely commit records to the database"""
        try:
            self.session.commit() # Commit the transaction
            self.session.expire_all() # Expire all to refresh state after successful commit
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Error committing records: {e}")
            return False
    
    def add_record(self, record):
        """Add a record with error handling"""
        try:
            self.session.add(record)
            self.session.flush([record]) # Attempt to flush this specific record to get its PK
            return True
        except Exception as e:
            # If flush fails, print error. The broader transaction will be rolled back by commit_records if needed.
            print(f"DEBUG ERROR: Error flushing individual record (PK not populated?): {record} - {e}") 
            return False # Indicate failure for this record
    
    def close(self):
        """Close resources"""
        try:
            if self.session:
                self.session.close()
        except Exception as e:
            print(f"Error closing session: {e}")
            pass 