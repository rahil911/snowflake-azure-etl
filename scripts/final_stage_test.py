#!/usr/bin/env python3
"""
Final test for creating stages with different container names
"""
import sys
sys.path.append('.')
from scripts.connection import get_snowflake_connection

def test_container_patterns():
    """
    Test different container/folder patterns for stage creation.
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Use our database
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
        
        # Base Azure URL
        base_url = "azure://sp72storage.blob.core.windows.net"
        
        # Test with a few entities
        entities = ['channel', 'product']
        
        # Different container patterns to try
        patterns = [
            # Pattern 1: Entity name as container
            lambda entity: f"{base_url}/{entity}/{entity}.csv",
            # Pattern 2: Lowercase entity name as container
            lambda entity: f"{base_url}/{entity.lower()}/{entity}.csv",
            # Pattern 3: Using 'channel' as a common container for all
            lambda entity: f"{base_url}/channel/{entity}.csv",
            # Pattern 4: Direct file path
            lambda entity: f"{base_url}/{entity}.csv",
            # Pattern 5: Using 'staging' as a common container
            lambda entity: f"{base_url}/staging/{entity}.csv",
            # Pattern 6: Using the same pattern as STARWARSCHARACTERS
            lambda entity: f"{base_url}/starwarscharacters/{entity}.csv"
        ]
        
        # Try each pattern for each entity
        for entity in entities:
            for i, pattern_fn in enumerate(patterns, 1):
                try:
                    # Generate URL and stage name
                    url = pattern_fn(entity)
                    stage_name = f"FINAL_TEST_{entity.upper()}_P{i}_STAGE"
                    
                    print(f"\nTrying pattern {i} for {entity}: {url}")
                    sql = f"CREATE OR REPLACE STAGE {stage_name} URL = '{url}' FILE_FORMAT = CSV_FORMAT"
                    cursor.execute(sql)
                    print(f"Stage {stage_name} created successfully")
                    
                    # Try listing files
                    print(f"Listing files in {stage_name}...")
                    cursor.execute(f"LIST @{stage_name}")
                    files = cursor.fetchall()
                    
                    if files:
                        print(f"Success! Found {len(files)} files in stage {stage_name}:")
                        for file in files:
                            print(f"  - {file[0]}")
                    else:
                        print(f"No files found in stage {stage_name}")
                        
                except Exception as e:
                    print(f"Error with pattern {i} for {entity}: {str(e)}")
        
        # Close connections
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_container_patterns() 