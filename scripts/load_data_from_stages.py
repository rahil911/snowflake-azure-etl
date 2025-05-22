#!/usr/bin/env python3
"""
Script to load data from stages into staging tables
"""
import sys
sys.path.append('.')
from scripts.connection import get_snowflake_connection

# List of entities to load data for
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

def load_data_from_stages():
    """
    Load data from stages into staging tables
    """
    try:
        # Connect to Snowflake
        print("Connecting to Snowflake...")
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Use our database
        cursor.execute("USE DATABASE IMT577_DW_RAHIL_HARIHAR_STAGING")
        cursor.execute("USE SCHEMA PUBLIC")
        
        # Process each entity
        successful_loads = []
        failed_loads = []
        
        for entity in entities:
            stage_name = f"{entity.upper()}_STAGE"
            table_name = f"STAGING_{entity.upper()}"
            
            print(f"\nLoading data from {stage_name} into {table_name}...")
            try:
                # First check if there are files in the stage
                cursor.execute(f"LIST @{stage_name}")
                files = cursor.fetchall()
                
                if not files:
                    print(f"No files found in stage {stage_name}. Skipping.")
                    failed_loads.append((entity, "No files found in stage"))
                    continue
                
                # Execute the COPY command
                cursor.execute(f"""
                COPY INTO {table_name}
                FROM @{stage_name}
                FILE_FORMAT = CSV_FORMAT
                ON_ERROR = 'CONTINUE'
                """)
                
                # Check how many rows were loaded
                result = cursor.fetchall()
                loaded_count = 0
                for row in result:
                    loaded_count += row[2] if row[2] is not None else 0
                
                if loaded_count > 0:
                    print(f"Successfully loaded {loaded_count} rows into {table_name}")
                    successful_loads.append((entity, loaded_count))
                else:
                    print(f"No rows loaded into {table_name}")
                    failed_loads.append((entity, "No rows loaded"))
                
            except Exception as e:
                print(f"Error loading data for {entity}: {str(e)}")
                failed_loads.append((entity, str(e)))
        
        # Print summary
        print("\nData Loading Summary:")
        print(f"Successful loads: {len(successful_loads)}")
        for entity, count in successful_loads:
            print(f"- {entity}: {count} rows")
        
        if failed_loads:
            print(f"\nFailed loads: {len(failed_loads)}")
            for entity, error in failed_loads:
                print(f"- {entity}: {error}")
        
        # Close connections
        cursor.close()
        conn.close()
        print("\nData loading process completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    load_data_from_stages() 