#!/usr/bin/env python3
"""
Script to create external stages in Snowflake pointing to Azure Blob Storage
"""
from . import config
from .connection import get_snowflake_connection

def create_stages():
    """
    Create stage for each entity in Azure Blob Storage
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Use our database
        cursor.execute(f"USE DATABASE {config.DATABASE_NAME}")
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
        base_url = f"azure://{config.AZURE_STORAGE_ACCOUNT}"
        
        # Create stages for each entity
        created_stages = []
        
        for entity in config.ENTITIES:
            stage_name = f"{entity.upper()}_STAGE"
            # Pattern: entity name as blob name
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