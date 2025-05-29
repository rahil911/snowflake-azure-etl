#!/usr/bin/env python3
"""
Run the complete Views ETL process:
1. Create all secure pass-through views
2. Create all analytical views  
3. Verify all views with sample data
"""
import sys
from .dim_config import *
from .create_views import main as create_views_main
from .view_sample_views import main as view_sample_main

def main():
    """Main function to run the complete views ETL process"""
    
    print("\n" + "="*100)
    print("SECURE VIEWS ETL PIPELINE")
    print("="*100)
    print(f"Target Database: {DIMENSION_DB_NAME}")
    print(f"Target Schema: {SNOWFLAKE_SCHEMA}")
    print("="*100)
    
    try:
        print("\n🔧 STEP 1: Creating Secure Views...")
        print("-" * 50)
        create_views_main()
        
        print("\n🔍 STEP 2: Verifying Views and Sample Data...")
        print("-" * 50) 
        view_sample_main()
        
        print("\n" + "="*100)
        print("SECURE VIEWS ETL PIPELINE COMPLETED SUCCESSFULLY! ✅")
        print("="*100)
        print("\n📋 SUMMARY:")
        print("• Created 10 pass-through secure views for all dimension and fact tables")
        print("• Created 3 analytical secure views for business intelligence")
        print("• All views use explicit column lists (no SELECT *)")
        print("• All views are SECURE for sharing with professors/graders")
        print("• Views provide a data access layer protecting the warehouse")
        print("\n🎯 NEXT STEPS:")
        print("• Use these views in Tableau or other BI tools")
        print("• Share view access with professors and graders")
        print("• Build dashboards using the analytical views")
        print("• Query the views for data analysis and reporting")
        
    except Exception as e:
        print(f"\n❌ Error in Views ETL Pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 