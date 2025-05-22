#!/usr/bin/env python3
"""
Test a single stage creation with explicit debugging
"""
import sys
sys.path.append('.')
from scripts.connection import get_snowflake_connection

def test_single_stage():
    """
    Test creating a single stage with extensive debugging.
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Create file format
        print("Creating CSV file format...")
        cursor.execute("""
        CREATE FILE FORMAT IF NOT EXISTS CSV_FORMAT
            TYPE = 'CSV'
            FIELD_DELIMITER = ','
            SKIP_HEADER = 1
            NULL_IF = ('NULL', 'null')
            EMPTY_FIELD_AS_NULL = TRUE
        """)
        
        # Show current environment
        cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE(), CURRENT_ROLE()")
        db, schema, wh, role = cursor.fetchone()
        print(f"Current environment: DB={db}, Schema={schema}, Warehouse={wh}, Role={role}")
        
        # Try creating a stage for 'channel' entity using different URL formats
        entity = 'channel'
        stage_name = f"TEST_{entity.upper()}_STAGE"
        
        # Format 1: Direct URL
        url1 = f"azure://sp72storage.blob.core.windows.net/{entity}.csv"
        print(f"\nTrying format 1: {url1}")
        try:
            sql = f"CREATE OR REPLACE STAGE {stage_name}_1 URL = '{url1}' FILE_FORMAT = CSV_FORMAT"
            print(f"Executing SQL: {sql}")
            cursor.execute(sql)
            print("Format 1 succeeded")
        except Exception as e:
            print(f"Format 1 failed: {str(e)}")
        
        # Format 2: Try different syntax
        url2 = f"'azure://sp72storage.blob.core.windows.net/{entity}.csv'"
        print(f"\nTrying format 2: {url2}")
        try:
            sql = f"CREATE OR REPLACE STAGE {stage_name}_2 URL = {url2} FILE_FORMAT = CSV_FORMAT"
            print(f"Executing SQL: {sql}")
            cursor.execute(sql)
            print("Format 2 succeeded")
        except Exception as e:
            print(f"Format 2 failed: {str(e)}")
        
        # Format 3: Try with container
        url3 = f"azure://sp72storage.blob.core.windows.net/{entity}.csv"
        print(f"\nTrying format 3: {url3}")
        try:
            sql = f"CREATE OR REPLACE STAGE {stage_name}_3 URL = '{url3}' FILE_FORMAT = CSV_FORMAT"
            print(f"Executing SQL: {sql}")
            cursor.execute(sql)
            print("Format 3 succeeded")
        except Exception as e:
            print(f"Format 3 failed: {str(e)}")
        
        # Format 4: Try with entire path enclosed in quotes
        url4 = f"'azure://sp72storage.blob.core.windows.net/{entity}.csv'"
        print(f"\nTrying format 4: {url4}")
        try:
            sql = f"CREATE OR REPLACE STAGE {stage_name}_4 URL = {url4} FILE_FORMAT = CSV_FORMAT"
            print(f"Executing SQL: {sql}")
            cursor.execute(sql)
            print("Format 4 succeeded")
        except Exception as e:
            print(f"Format 4 failed: {str(e)}")
        
        # Check what stages were created
        print("\nListing created stages:")
        cursor.execute("SHOW STAGES LIKE 'TEST\\_%'")
        stages = cursor.fetchall()
        if stages:
            for stage in stages:
                print(f"Stage: {stage[1]}")
                # Try listing files in the stage
                try:
                    cursor.execute(f"LIST @{stage[1]}")
                    files = cursor.fetchall()
                    if files:
                        print(f"  Files found: {len(files)}")
                    else:
                        print("  No files found")
                except Exception as e:
                    print(f"  Error listing files: {str(e)}")
        else:
            print("No test stages created")
        
        # Close connections
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_single_stage() 