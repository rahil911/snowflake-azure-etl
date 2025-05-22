#!/usr/bin/env python3
"""
Step 3: Create External Stages
This script creates external stages pointing to Azure Blob Storage for all entities
using the exact URL format specified by the user.
"""
import sys
sys.path.append('.')
from scripts.connection import get_snowflake_connection

# List of entities to create stages for
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

def create_stages():
    """
    Create external stages for all entities.
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Create file format if it doesn't exist
        print("Creating CSV file format...")
        cursor.execute("""
        CREATE FILE FORMAT IF NOT EXISTS CSV_FORMAT
            TYPE = 'CSV'
            FIELD_DELIMITER = ','
            SKIP_HEADER = 1
            NULL_IF = ('NULL', 'null')
            EMPTY_FIELD_AS_NULL = TRUE
        """)
        
        # Create stages for each entity
        created_stages = []
        failed_stages = []
        
        for entity in entities:
            # Following the format from user specification
            stage_name = f"{entity.upper()}_STAGE"
            
            # Exact format specified by the user
            # @azure://sp72storage.blob.core.windows.net/starwarscharacters.csv
            azure_url = f"azure://sp72storage.blob.core.windows.net/{entity}.csv"
            
            try:
                print(f"\nCreating stage {stage_name}...")
                cursor.execute(f"""
                CREATE OR REPLACE STAGE {stage_name}
                URL = '{azure_url}'
                FILE_FORMAT = CSV_FORMAT
                """)
                
                created_stages.append(stage_name)
                print(f"Stage {stage_name} created successfully.")
                
                # Verify stage was created by listing files
                print(f"Listing files in {stage_name}...")
                cursor.execute(f"LIST @{stage_name}")
                files = cursor.fetchall()
                
                if files:
                    print(f"Found {len(files)} files in stage {stage_name}:")
                    for file in files:
                        print(f"  - File: {file[0]}, Size: {file[1]}, Last Modified: {file[2]}")
                else:
                    print(f"No files found in stage {stage_name}")
                
            except Exception as e:
                print(f"Error creating stage {stage_name}: {str(e)}")
                failed_stages.append((stage_name, str(e)))
        
        # Verify stages exist
        print("\nVerifying stages...")
        cursor.execute("SHOW STAGES")
        existing_stages = [row[1] for row in cursor.fetchall()]
        
        print("\nSummary:")
        print(f"Stages created: {len(created_stages)}")
        for stage in created_stages:
            status = "✅ Exists" if stage in existing_stages else "❌ Not found"
            print(f"- {stage}: {status}")
        
        if failed_stages:
            print(f"\nFailed stages: {len(failed_stages)}")
            for stage, error in failed_stages:
                print(f"- {stage}: {error}")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        print("\nStage creation process completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    create_stages() 