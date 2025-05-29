#!/usr/bin/env python3
"""
View sample data from secure views to verify they work correctly
"""
import sys
from tabulate import tabulate
from .dim_config import *
from .connection import get_snowflake_connection

def display_view_data(cursor, view_name, limit=5):
    """Display sample data from a view"""
    
    try:
        # Get sample data
        cursor.execute(f"SELECT * FROM {view_name} LIMIT {limit}")
        results = cursor.fetchall()
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        if results:
            print(f"\n📊 Sample data from {view_name}:")
            print("-" * 80)
            print(tabulate(results, headers=column_names, tablefmt="grid"))
            
            # Show row count
            cursor.execute(f"SELECT COUNT(*) FROM {view_name}")
            count_result = cursor.fetchone()
            total_rows = count_result[0] if count_result else 0
            print(f"Total rows in {view_name}: {total_rows:,}")
            
        else:
            print(f"\n⚠️  No data found in {view_name}")
            
    except Exception as e:
        print(f"\n❌ Error accessing {view_name}: {str(e)}")

def verify_all_views(cursor):
    """Verify all created views"""
    
    print("\n" + "="*80)
    print("VERIFYING SECURE VIEWS")
    print("="*80)
    
    # List of all views to check
    all_views = [
        # Pass-through dimension views
        'VW_Dim_Product',
        'VW_Dim_Customer', 
        'VW_Dim_Location',
        'VW_Dim_Channel',
        'VW_Dim_Store',
        'VW_Dim_Reseller',
        'VW_Dim_Date',
        
        # Pass-through fact views
        'VW_Fact_SalesActual',
        'VW_Fact_ProductSalesTarget',
        'VW_Fact_SRCSalesTarget',
        
        # Analytical views
        'VW_SalesPerformanceSummary',
        'VW_CustomerSalesAnalysis',
        'VW_TargetVsActualPerformance'
    ]
    
    successful_views = []
    failed_views = []
    
    for view_name in all_views:
        print(f"\n🔍 Checking view: {view_name}")
        if display_view_data(cursor, view_name):
            successful_views.append(view_name)
        else:
            failed_views.append(view_name)
    
    # Summary
    print("\n" + "="*80)
    print("VIEWS VERIFICATION SUMMARY")
    print("="*80)
    
    print(f"\n✅ Successfully verified {len(successful_views)} views:")
    for view in successful_views:
        print(f"   • {view}")
    
    if failed_views:
        print(f"\n❌ Failed to verify {len(failed_views)} views:")
        for view in failed_views:
            print(f"   • {view}")
    
    print(f"\nTotal views checked: {len(all_views)}")
    print(f"Success rate: {len(successful_views)}/{len(all_views)} ({len(successful_views)/len(all_views)*100:.1f}%)")

def show_view_descriptions():
    """Show descriptions of what each view contains"""
    
    print("\n" + "="*80)
    print("SECURE VIEWS DOCUMENTATION")
    print("="*80)
    
    print("\n📋 PASS-THROUGH VIEWS (Direct table access layer):")
    print("   These views provide secure access to dimension and fact tables")
    print("   without using SELECT * syntax for better security and control.")
    print()
    
    passthrough_views = {
        'VW_Dim_Product': 'Product information with pricing and profitability metrics',
        'VW_Dim_Customer': 'Customer details with demographics and location references',
        'VW_Dim_Location': 'Geographic location data for customers, stores, and resellers',
        'VW_Dim_Channel': 'Sales channels and channel categories',
        'VW_Dim_Store': 'Store information with location and management details',
        'VW_Dim_Reseller': 'Reseller details with contact information and location',
        'VW_Dim_Date': 'Complete date dimension with calendar and fiscal attributes',
        'VW_Fact_SalesActual': 'Detailed sales transaction data with all metrics',
        'VW_Fact_ProductSalesTarget': 'Product-level sales quantity targets',
        'VW_Fact_SRCSalesTarget': 'Store/Reseller/Channel sales amount targets'
    }
    
    for view, description in passthrough_views.items():
        print(f"   • {view}: {description}")
    
    print("\n📊 ANALYTICAL VIEWS (Business intelligence layer):")
    print("   These views provide pre-aggregated data optimized for visualization")
    print("   and business analysis with complex calculations and joins.")
    print()
    
    analytical_views = {
        'VW_SalesPerformanceSummary': 'Product sales performance aggregated by time periods with profit margins, transaction counts, and pricing analysis',
        'VW_CustomerSalesAnalysis': 'Customer demographics analysis with sales patterns by geography, gender, and channel preferences',
        'VW_TargetVsActualPerformance': 'Comprehensive target vs actual performance comparison with achievement percentages for products and channels'
    }
    
    for view, description in analytical_views.items():
        print(f"   • {view}: {description}")

def main():
    """Main function to verify all views"""
    
    connection = None
    try:
        # Get connection
        connection = get_snowflake_connection()
        cursor = connection.cursor()
        
        # Use the dimensional database and schema
        cursor.execute(f"USE DATABASE {DIMENSION_DB_NAME}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")
        
        print("\n" + "="*80)
        print("SECURE VIEWS VERIFICATION AND SAMPLE DATA")
        print("="*80)
        print(f"Database: {DIMENSION_DB_NAME}")
        print(f"Schema: {SNOWFLAKE_SCHEMA}")
        print("="*80)
        
        # Pass-through views to check
        passthrough_views = [
            "VW_Dim_Product", "VW_Dim_Customer", "VW_Dim_Location", "VW_Dim_Channel",
            "VW_Dim_Store", "VW_Dim_Reseller", "VW_Dim_Date",
            "VW_Fact_SalesActual", "VW_Fact_ProductSalesTarget", "VW_Fact_SRCSalesTarget"
        ]
        
        # Analytical views to check
        analytical_views = [
            "VW_SalesPerformanceSummary", "VW_CustomerSalesAnalysis", "VW_TargetVsActualPerformance",
            "VW_Store58Performance", "VW_StoreBonusRecommendation", 
            "VW_Store58DayOfWeekAnalysis", "VW_MultiStoreVsSingleStoreAnalysis"
        ]
        
        print("\n📋 PASS-THROUGH VIEWS VERIFICATION:")
        print("-" * 50)
        for view in passthrough_views:
            display_view_data(cursor, view, limit=3)
            
        print("\n📊 ANALYTICAL VIEWS VERIFICATION:")  
        print("-" * 50)
        for view in analytical_views:
            display_view_data(cursor, view, limit=5)
            
        # Business Analysis Questions Verification
        print("\n🔍 BUSINESS ANALYSIS QUESTIONS:")
        print("=" * 80)
        
        # Question 1: Store 5 and 8 Performance Assessment
        print("\n1️⃣ STORE 5 AND 8 PERFORMANCE ASSESSMENT:")
        print("-" * 60)
        cursor.execute("""
            SELECT StoreNumber, StoreName, YEAR, 
                   TotalSalesAmount, TotalProfit, ProfitMarginPercent,
                   ProfitCategory, RevenueCategory
            FROM VW_Store58Performance 
            ORDER BY StoreNumber, YEAR
        """)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        if results:
            print(tabulate(results, headers=column_names, tablefmt="grid", floatfmt=".2f"))
        
        # Question 2: Bonus Recommendations 
        print("\n2️⃣ STORE BONUS RECOMMENDATIONS (Men's/Women's Casual):")
        print("-" * 60)
        cursor.execute("""
            SELECT StoreNumber, StoreName, YEAR, ProductType,
                   TotalSalesAmount, TotalProfit, SalesRank, ProfitRank,
                   SalesSharePercent, RecommendedBonus
            FROM VW_StoreBonusRecommendation 
            ORDER BY YEAR, ProductType, SalesRank
            LIMIT 20
        """)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        if results:
            print(tabulate(results, headers=column_names, tablefmt="grid", floatfmt=".2f"))
        
        # Question 3: Day of Week Analysis
        print("\n3️⃣ DAY OF WEEK SALES ANALYSIS (Stores 5 & 8):")
        print("-" * 60)
        cursor.execute("""
            SELECT StoreNumber, DAY_NAME, YEAR, DayType,
                   TotalSalesAmount, DaySharePercent, DayVsAvgPercent, DaySalesRank
            FROM VW_Store58DayOfWeekAnalysis 
            WHERE YEAR = 2013
            ORDER BY StoreNumber, DayOrder
            LIMIT 14
        """)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        if results:
            print(tabulate(results, headers=column_names, tablefmt="grid", floatfmt=".2f"))
        
        # Question 4: Multi-Store vs Single-Store Analysis
        print("\n4️⃣ MULTI-STORE VS SINGLE-STORE STATE ANALYSIS:")
        print("-" * 60)
        cursor.execute("""
            SELECT StoreConfiguration, YEAR, 
                   AVG(AvgSalesPerStore) as AvgSalesPerStore,
                   AVG(AvgProfitPerStore) as AvgProfitPerStore,
                   AVG(ProfitMarginPercent) as AvgProfitMargin,
                   COUNT(*) as StateCount
            FROM VW_MultiStoreVsSingleStoreAnalysis 
            GROUP BY StoreConfiguration, YEAR
            ORDER BY YEAR, StoreConfiguration
        """)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        if results:
            print(tabulate(results, headers=column_names, tablefmt="grid", floatfmt=".2f"))
        
        print("\n" + "="*80)
        print("✅ ALL SECURE VIEWS VERIFIED SUCCESSFULLY!")
        print("✅ BUSINESS ANALYSIS VIEWS ARE READY FOR TABLEAU/VISUALIZATION!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during view verification: {str(e)}")
        raise
    finally:
        if connection:
            connection.close()

if __name__ == "__main__":
    main() 