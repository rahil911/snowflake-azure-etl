#!/usr/bin/env python3
"""
Check the URL format of the STARWARSCHARACTERS stage
"""
import sys
sys.path.append('.')
from scripts.connection import get_snowflake_connection

def check_starwars_url():
    """
    Check the URL format used for the STARWARSCHARACTERS stage.
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Change database to where STARWARSCHARACTERS stage is located
        cursor.execute("USE DATABASE IMT577_RMH_STARWARS_STAGING")
        cursor.execute("USE SCHEMA PUBLIC")
        
        # Get stage URL
        print("Checking STARWARSCHARACTERS stage URL format...")
        cursor.execute("DESC STAGE STARWARSCHARACTERS")
        
        # Find the URL property
        stage_props = cursor.fetchall()
        url = None
        for prop in stage_props:
            if prop[0] == 'STAGE_LOCATION':
                url = prop[1]
                break
        
        if url:
            print(f"STARWARSCHARACTERS stage URL: {url}")
        else:
            print("URL not found in stage properties")
        
        # Check other stages
        print("\nChecking STAGING_AREA stage...")
        cursor.execute("DESC STAGE STAGING_AREA")
        stage_props = cursor.fetchall()
        for prop in stage_props:
            if prop[0] == 'STAGE_LOCATION':
                print(f"STAGING_AREA stage URL: {prop[1]}")
                break
        
        # List files in the STARWARSCHARACTERS stage
        print("\nListing files in STARWARSCHARACTERS stage...")
        cursor.execute("LIST @STARWARSCHARACTERS")
        files = cursor.fetchall()
        if files:
            for file in files:
                print(f"File: {file[0]}")
        else:
            print("No files found")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_starwars_url() 