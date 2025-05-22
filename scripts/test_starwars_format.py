#!/usr/bin/env python3
"""
Test creating stages using the exact STARWARSCHARACTERS format
"""
import sys
sys.path.append('.')
from scripts.connection import get_snowflake_connection

def test_starwars_format():
    """
    Test creating stages using the exact format from STARWARSCHARACTERS.
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # First, get the exact URL from STARWARSCHARACTERS
        cursor.execute("USE DATABASE IMT577_RMH_STARWARS_STAGING")
        cursor.execute("USE SCHEMA PUBLIC")
        cursor.execute("LIST @STARWARSCHARACTERS")
        starwars_files = cursor.fetchall()
        
        if not starwars_files:
            print("No files found in STARWARSCHARACTERS stage")
            return
        
        starwars_file_path = starwars_files[0][0]
        print(f"STARWARSCHARACTERS file path: {starwars_file_path}")
        
        # Extract the base URL (everything before the final filename)
        last_slash_index = starwars_file_path.rfind('/')
        base_url = starwars_file_path[:last_slash_index]
        print(f"Base URL: {base_url}")
        
        # Switch back to our database
        cursor.execute("USE DATABASE IMT577_DW_RAHIL_HARIHAR_STAGING")
        cursor.execute("USE SCHEMA PUBLIC")
        
        # Create file format
        cursor.execute("""
        CREATE FILE FORMAT IF NOT EXISTS CSV_FORMAT
            TYPE = 'CSV'
            FIELD_DELIMITER = ','
            SKIP_HEADER = 1
            NULL_IF = ('NULL', 'null')
            EMPTY_FIELD_AS_NULL = TRUE
        """)
        
        # Try to create stages for a few entities using the STARWARSCHARACTERS format
        entities = ['channel', 'product', 'store']
        
        for entity in entities:
            stage_name = f"STARWARS_FORMAT_{entity.upper()}_STAGE"
            file_path = f"{base_url}/{entity}.csv"
            
            print(f"\nCreating stage {stage_name} with path {file_path}...")
            try:
                sql = f"CREATE OR REPLACE STAGE {stage_name} URL = '{file_path}' FILE_FORMAT = CSV_FORMAT"
                cursor.execute(sql)
                print(f"Stage {stage_name} created successfully")
                
                # Try listing files
                print(f"Listing files in {stage_name}...")
                cursor.execute(f"LIST @{stage_name}")
                files = cursor.fetchall()
                
                if files:
                    print(f"Found {len(files)} files in stage {stage_name}:")
                    for file in files:
                        print(f"  - {file[0]}")
                else:
                    print(f"No files found in stage {stage_name}")
                    
            except Exception as e:
                print(f"Error with stage {stage_name}: {str(e)}")
        
        # Close connections
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_starwars_format() 