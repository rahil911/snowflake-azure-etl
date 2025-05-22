#!/usr/bin/env python3
"""
Step 4: Load Data
This script loads data from external stages to tables for all entities.
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

def load_data():
    """
    Load data from external stages to tables for all entities.
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Load data for each entity
        successful_loads = []
        failed_loads = []
        
        for entity in entities:
            stage_name = f"{entity.upper()}_STAGE"
            table_name = f"TABLE_{entity.upper()}"
            
            try:
                print(f"\nLoading data from {stage_name} to {table_name}...")
                
                # Use COPY INTO command to load data from stage to table
                cursor.execute(f"""
                COPY INTO {table_name}
                FROM @{stage_name}
                FILE_FORMAT = CSV_FORMAT
                ON_ERROR = 'CONTINUE'
                """)
                
                # Check if data was loaded
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    print(f"Successfully loaded {count} rows into {table_name}")
                    successful_loads.append((table_name, count))
                else:
                    print(f"No data loaded into {table_name}")
                    failed_loads.append((table_name, "No data loaded"))
                
            except Exception as e:
                print(f"Error loading data for {table_name}: {str(e)}")
                failed_loads.append((table_name, str(e)))
                
                # Try alternate approach - load using standard INSERT
                try:
                    print(f"Trying alternate loading approach for {table_name}...")
                    
                    # Query data from stage
                    cursor.execute(f"SELECT * FROM @{stage_name} (FILE_FORMAT => 'CSV_FORMAT')")
                    stage_data = cursor.fetchall()
                    
                    if stage_data:
                        print(f"Found {len(stage_data)} rows in stage {stage_name}")
                        # TODO: Would need to implement INSERT statements based on data structure
                        # For now, just report that we found data
                    else:
                        print(f"No data found in stage {stage_name}")
                        
                except Exception as e2:
                    print(f"Error with alternate loading approach for {table_name}: {str(e2)}")
        
        # Print summary
        print("\nSummary:")
        print(f"Successful loads: {len(successful_loads)}")
        for table, count in successful_loads:
            print(f"- {table}: {count} rows")
        
        if failed_loads:
            print(f"\nFailed loads: {len(failed_loads)}")
            for table, error in failed_loads:
                print(f"- {table}: {error}")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        print("\nData loading process completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    load_data() 