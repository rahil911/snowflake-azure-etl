#!/usr/bin/env python3
"""
Main ETL runner for dimensional model
"""
import os
import sys
import time
from datetime import datetime
from .create_dimension_database import create_dimension_database
from .create_dimension_tables import create_dimension_tables
from .load_dim_date import load_dim_date
from .load_dimension_tables import load_dimension_tables
from .create_fact_tables import create_fact_tables
from .load_fact_tables import load_fact_tables
from .dim_config import DIMENSION_DB_NAME
import subprocess

def run_dimensional_etl():
    """Run the dimensional model ETL process"""
    start_time = time.time()
    
    print("="*80)
    print(f"DIMENSIONAL MODEL ETL PROCESS FOR {DIMENSION_DB_NAME}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Step 0: Create Dimension Database
    if not create_dimension_database():
        print("\n❌ ETL process aborted due to error in step 0")
        sys.exit(1)

    # Apply migrations for dimension and fact tables
    print("\nSTEP 1: Applying migrations...")
    try:
        subprocess.run(["alembic", "upgrade", "head"], check=True)
    except Exception as e:
        print(f"WARNING: Alembic migration failed: {e}")
    
    # Step 1: Create Dimension Tables
    if not create_dimension_tables():
        print("\n❌ ETL process aborted due to error in step 1")
        sys.exit(1)
    
    # Step 2: Load Dim_Date Table
    if not load_dim_date():
        print("\n❌ ETL process aborted due to error in step 2")
        sys.exit(1)
    
    # Step 3: Load Dimension Tables
    if not load_dimension_tables():
        print("\n❌ ETL process aborted due to error in step 3")
        sys.exit(1)
    
    # Step 4: Create Fact Tables
    if not create_fact_tables():
        print("\n❌ ETL process aborted due to error in step 4")
        sys.exit(1)
    
    # Step 5: Load Fact Tables
    if not load_fact_tables():
        print("\n❌ ETL process aborted due to error in step 5")
        sys.exit(1)
    
    # ETL process completed successfully
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print(f"✅ DIMENSIONAL MODEL ETL PROCESS COMPLETED SUCCESSFULLY")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration:.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"dim_etl_run_{timestamp}.log")
    
    # Run ETL process
    run_dimensional_etl() 