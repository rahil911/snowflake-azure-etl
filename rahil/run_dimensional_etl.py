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
        # Use python -m alembic to ensure it's found
        subprocess.run(["python3", "-m", "alembic", "upgrade", "head"], check=True)
        print("Alembic migrations applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Alembic migration failed with exit code {e.returncode}.")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        print("Check Alembic configuration and logs.")
        # Decide if this should be fatal; for now, it prints a warning and continues as before
        # but the user's suggestion was to make it fatal.
        # To make it fatal, uncomment the sys.exit(1) below or re-raise.
        # sys.exit(1)
    except FileNotFoundError:
        print("ERROR: 'python3' or 'alembic' command not found for migrations. Ensure they are installed and in PATH.")
        # sys.exit(1)
    except Exception as e:
        # General exception if Alembic fails for other reasons (e.g. misconfiguration in env.py)
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
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(logs_dir, f"dim_etl_run_{timestamp}.log")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    print(f"Dimensional ETL run starting. Logging to: {log_file_path}")

    try:
        with open(log_file_path, 'w') as log_f:
            sys.stdout = log_f
            sys.stderr = log_f
            
            # Run ETL process
            run_dimensional_etl() 

    except Exception as e:
        # If logging setup fails or main ETL fails, log to original stderr and also to file if possible
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Unhandled exception in ETL runner: {e}", file=sys.stderr)
        if 'log_f' in locals() and not log_f.closed:
            print(f"Unhandled exception in ETL runner: {e}", file=log_f)
        # Optionally, re-raise or sys.exit(1)
        sys.exit(1) # Ensure script exits with error status if something went wrong here
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Dimensional ETL run finished. Log saved to: {log_file_path}") 