#!/usr/bin/env python3
"""
Script to create a sample log file to demonstrate the logging functionality
"""
import os
import sys
import datetime

def create_sample_log():
    """
    Creates a sample log file with timestamp in the logs directory
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a log file with the timestamp
    log_filename = os.path.join(logs_dir, f'etl_run_{timestamp}.log')
    
    # Write some sample content to the log file
    with open(log_filename, 'w') as f:
        f.write(f"Sample ETL Run Log - Generated on {datetime.datetime.now()}\n")
        f.write("="*80 + "\n")
        f.write("This is a sample log file to demonstrate where ETL logs are stored.\n")
        f.write("When you run the actual ETL process, detailed logs will be stored here.\n")
        f.write("="*80 + "\n")
        f.write("ETL Steps:\n")
        f.write("1. Database creation - Completed\n")
        f.write("2. External stage creation - Completed\n")
        f.write("3. Table creation - Completed\n")
        f.write("4. Data loading - Completed\n")
        f.write("5. Sample data display - Completed\n")
        f.write("="*80 + "\n")
        f.write("ETL process completed successfully!\n")
    
    print(f"Sample log file created: {log_filename}")
    return log_filename

if __name__ == "__main__":
    create_sample_log() 