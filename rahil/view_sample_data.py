#!/usr/bin/env python3
"""
Script to view sample data from all staging tables
"""
import sys
from tabulate import tabulate
from . import config
from .connection import get_snowflake_connection

def view_sample_data():
    """
    Fetch and display the top 5 rows from each staging table
    """
    print("\n" + "=" * 80)
    print(f"SAMPLE DATA FROM STAGING TABLES IN {config.DATABASE_NAME}")
    print("=" * 80)
    
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        for entity in config.ENTITIES:
            table_name = f"STAGING_{entity.upper()}"
            
            print(f"\n\n{'#' * 50}")
            print(f"### Table: {table_name}")
            print(f"{'#' * 50}")
            
            try:
                # Get column names
                cursor.execute(f"DESCRIBE TABLE {table_name}")
                columns = [row[0] for row in cursor.fetchall()]
                
                # Get top 5 rows
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                rows = cursor.fetchall()
                
                if rows:
                    # Create a nice formatted table display
                    print(f"\nShowing top 5 rows from {table_name}:\n")
                    print(tabulate(rows, headers=columns, tablefmt="pretty"))
                    print(f"\nTotal rows in table: ", end="")
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    print(f"{count:,}")
                else:
                    print(f"No data found in {table_name}")
            
            except Exception as e:
                print(f"Error retrieving data from {table_name}: {str(e)}")
        
        # Close connections
        cursor.close()
        conn.close()
        print("\n" + "=" * 80)
        print("Sample data retrieval completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    view_sample_data() 