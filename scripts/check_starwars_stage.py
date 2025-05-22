#!/usr/bin/env python3
"""
Research: Check STARWARSCHARACTERS Stage Configuration
This script checks the configuration of the STARWARSCHARACTERS stage to understand 
how to properly create our own stages.
"""
import sys
sys.path.append('.')
from scripts.connection import get_snowflake_connection

def check_starwars_stage():
    """
    Check the configuration of the STARWARSCHARACTERS stage.
    """
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Change database to where STARWARSCHARACTERS stage is located
        cursor.execute("USE DATABASE IMT577_RMH_STARWARS_STAGING")
        cursor.execute("USE SCHEMA PUBLIC")
        
        # Check stage configuration
        print("Checking STARWARSCHARACTERS stage configuration...")
        cursor.execute("DESC STAGE STARWARSCHARACTERS")
        
        # Print stage details
        stage_details = cursor.fetchall()
        if stage_details:
            print("\nSTARWARSCHARACTERS Stage Configuration:")
            for detail in stage_details:
                print(f"{detail[0]}: {detail[1]}")
        else:
            print("No stage details found")
        
        # List files in the stage
        print("\nListing files in STARWARSCHARACTERS stage...")
        cursor.execute("LIST @STARWARSCHARACTERS")
        files = cursor.fetchall()
        if files:
            print("\nFiles in STARWARSCHARACTERS stage:")
            for file in files:
                print(f"File: {file[0]}, Size: {file[1]}, Last Modified: {file[2]}")
        else:
            print("No files found in the stage")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_starwars_stage() 