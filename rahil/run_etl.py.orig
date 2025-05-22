#!/usr/bin/env python3
"""
Main ETL runner script - executes all ETL steps in sequence
"""
import sys
import time
import subprocess
from . import config
from .create_database import create_database
from .create_stages import create_stages
from .create_tables import create_staging_tables
from .load_data import load_data_from_stages
from .view_sample_data import view_sample_data

def run_etl():
    """
    Run the entire ETL process from start to finish
    """
    print("=" * 80)
    print(f"Starting ETL process for {config.USER_NAME}")
    print(f"Database: {config.DATABASE_NAME}")
    print("=" * 80)
    
    try:
        # Step 0: Create database if not exists
        print("\nSTEP 0: Creating database if not exists...")
        create_database()
        time.sleep(1)
        
        # Step 1: Run database migrations
        print("\nSTEP 1: Applying migrations...")
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        time.sleep(1)

        # Step 2: Create stages
        print("\nSTEP 2: Creating external stages...")
        create_stages()
        time.sleep(1)

        # Step 3: Create tables
        print("\nSTEP 3: Creating staging tables...")
        create_staging_tables()
        time.sleep(1)

        # Step 4: Load data
        print("\nSTEP 4: Loading data from stages to tables...")
        load_data_from_stages()
        time.sleep(1)

        # Step 5: Display sample data
        print("\nSTEP 5: Displaying sample data from tables...")
        view_sample_data()
        
        print("\n" + "=" * 80)
        print("ETL process completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: ETL process failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_etl()) 