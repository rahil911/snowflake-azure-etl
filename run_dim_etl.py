#!/usr/bin/env python3
"""
Runner script for dimensional ETL process
"""
import os
import sys
import time
from datetime import datetime

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the path to enable absolute imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import modules with absolute imports
try:
    from rahil.dim_config import DIMENSION_DB_NAME
    from rahil.create_dimension_database import create_dimension_database
    from rahil.create_dimension_tables import create_dimension_tables
    from rahil.load_dim_date import load_dim_date
    from rahil.load_dimension_tables import load_dimension_tables
    from rahil.create_fact_tables import create_fact_tables
    from rahil.load_fact_tables import load_fact_tables
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the root directory of the project.")
    sys.exit(1)

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
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"dim_etl_run_{timestamp}.log")
    
    # Run ETL process
    run_dimensional_etl() 