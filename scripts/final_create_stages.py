#!/usr/bin/env python3
"""
Final script to create stages using entity name as container pattern
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
    Create stage for each entity using entity name as container pattern.
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Use our database
        cursor.execute("USE DATABASE IMT577_DW_RAHIL_HARIHAR_STAGING")
        cursor.execute("USE SCHEMA PUBLIC")
        
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
        
        # Base Azure URL
        base_url = "azure://sp72storage.blob.core.windows.net"
        
        # Create stages for each entity
        created_stages = []
        
        for entity in entities:
            stage_name = f"{entity.upper()}_STAGE"
            # URL format: entity as the blob name only
            url = f"{base_url}/{entity}"
            
            try:
                print(f"\nCreating stage {stage_name}...")
                cursor.execute(f"""
                CREATE OR REPLACE STAGE {stage_name}
                URL = '{url}'
                FILE_FORMAT = CSV_FORMAT
                """)
                
                created_stages.append(stage_name)
                print(f"Stage {stage_name} created successfully.")
                
            except Exception as e:
                print(f"Error creating stage {stage_name}: {str(e)}")
        
        # Verify stages exist
        print("\nVerifying stages...")
        cursor.execute("SHOW STAGES")
        existing_stages = [row[1] for row in cursor.fetchall()]
        
        print("\nSummary:")
        print(f"Stages created: {len(created_stages)}")
        for stage_name in created_stages:
            status = "✅ Exists" if stage_name in existing_stages else "❌ Not found"
            print(f"- {stage_name}: {status}")
        
        # Close connections
        cursor.close()
        conn.close()
        print("\nStage creation process completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    create_stages() 