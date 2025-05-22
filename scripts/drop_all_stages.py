#!/usr/bin/env python3
"""
Script to drop all existing stages in Snowflake
"""
import sys
sys.path.append('.')
from scripts.connection import get_snowflake_connection

def drop_all_stages():
    """
    Drop all existing stages in the database
    """
    try:
        # Connect to Snowflake
        print("Connecting to Snowflake...")
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Use our database
        cursor.execute("USE DATABASE IMT577_DW_RAHIL_HARIHAR_STAGING")
        cursor.execute("USE SCHEMA PUBLIC")
        
        # Get all existing stages
        print("Fetching all existing stages...")
        cursor.execute("SHOW STAGES")
        stages = cursor.fetchall()
        
        if not stages:
            print("No stages found to drop.")
            return
        
        # Drop each stage
        for stage in stages:
            stage_name = stage[1]
            print(f"Dropping stage: {stage_name}")
            cursor.execute(f"DROP STAGE IF EXISTS {stage_name}")
        
        print(f"\nSuccessfully dropped {len(stages)} stages.")
        
        # Close connections
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    drop_all_stages() 