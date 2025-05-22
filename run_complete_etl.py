#!/usr/bin/env python3
"""
Complete ETL runner - processes both staging and dimensional models
"""
import os
import sys
import time
import subprocess
from datetime import datetime

def run_complete_etl():
    """Run the complete ETL process - from Azure Blob Storage to Dimensional Model"""
    start_time = time.time()
    
    print("="*80)
    print("COMPLETE ETL PROCESS: AZURE BLOB STORAGE TO DIMENSIONAL MODEL")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Step 1: Run the staging ETL process
    print("\n📥 RUNNING STAGING ETL PROCESS")
    print("="*60)
    
    # Change directory to run staging ETL correctly using subprocess
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    staging_result = subprocess.run(
        [sys.executable, "-m", "rahil.run_etl"], 
        check=False
    )
    
    if staging_result.returncode != 0:
        print("\n❌ Staging ETL process failed! Aborting complete ETL process.")
        return staging_result.returncode
    
    print("\n✅ Staging ETL process completed successfully!")
    
    # Step 2: Run the dimensional ETL process
    print("\n📊 RUNNING DIMENSIONAL ETL PROCESS")
    print("="*60)
    
    # Run dimensional ETL using subprocess
    dim_etl_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_dim_etl.py')
    dim_result = subprocess.run([sys.executable, dim_etl_script], check=False)
    
    if dim_result.returncode != 0:
        print("\n❌ Dimensional ETL process failed!")
        return dim_result.returncode
    
    # Complete ETL process completed successfully
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("✅ COMPLETE ETL PROCESS FINISHED SUCCESSFULLY")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration:.2f} seconds")
    print("="*80)
    return 0

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"complete_etl_run_{timestamp}.log")
    
    # Run the complete ETL process
    sys.exit(run_complete_etl()) 