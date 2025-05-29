#!/usr/bin/env python3
"""
Create secure views for dimensional model
- Pass-through views for all dimension and fact tables
- Custom analytical views for data visualization support
"""
import sys
from .dim_config import *
from .connection import get_snowflake_connection

def create_passthrough_views(cursor):
    """Create pass-through secure views for all dimension and fact tables"""
    
    print("\n" + "="*80)
    print("CREATING PASS-THROUGH SECURE VIEWS")
    print("="*80)
    
    # Pass-through views for dimension tables
    dimension_views = {
        'VW_Dim_Product': {
            'table': 'Dim_Product',
            'columns': [
                'DimProductID', 'ProductID', 'ProductTypeID', 'ProductCategoryID',
                'ProductName', 'ProductType', 'ProductCategory', 'ProductRetailPrice',
                'ProductWholesalePrice', 'ProductCost', 'ProductRetailProfit',
                'ProductWholesaleUnitProfit', 'ProductProfitMarginUnitPercent'
            ]
        },
        'VW_Dim_Customer': {
            'table': 'Dim_Customer',
            'columns': [
                'DimCustomerID', 'CustomerID', 'DimLocationID', 'CustomerFullName',
                'CustomerFirstName', 'CustomerLastName', 'CustomerGender'
            ]
        },
        'VW_Dim_Location': {
            'table': 'Dim_Location',
            'columns': [
                'DimLocationID', 'Address', 'City', 'PostalCode', 'State_Province', 'Country'
            ]
        },
        'VW_Dim_Channel': {
            'table': 'Dim_Channel',
            'columns': [
                'DimChannelID', 'ChannelID', 'ChannelCategoryID', 'ChannelName', 'ChannelCategory'
            ]
        },
        'VW_Dim_Store': {
            'table': 'Dim_Store',
            'columns': [
                'DimStoreID', 'StoreID', 'DimLocationID', 'SourceStoreID',
                'StoreName', 'StoreNumber', 'StoreManager'
            ]
        },
        'VW_Dim_Reseller': {
            'table': 'Dim_Reseller',
            'columns': [
                'DimResellerID', 'ResellerID', 'DimLocationID', 'ResellerName',
                'ContactName', 'PhoneNumber', 'Email'
            ]
        },
        'VW_Dim_Date': {
            'table': 'Dim_Date',
            'columns': [
                'DATE_PKEY', 'DATE', 'FULL_DATE_DESC', 'DAY_NUM_IN_WEEK', 'DAY_NUM_IN_MONTH',
                'DAY_NUM_IN_YEAR', 'DAY_NAME', 'DAY_ABBREV', 'WEEKDAY_IND', 'US_HOLIDAY_IND',
                'MONTH_END_IND', 'WEEK_BEGIN_DATE_NKEY', 'WEEK_BEGIN_DATE', 'WEEK_END_DATE_NKEY',
                'WEEK_END_DATE', 'WEEK_NUM_IN_YEAR', 'MONTH_NAME', 'MONTH_ABBREV', 'MONTH_NUM_IN_YEAR',
                'YEARMONTH', 'QUARTER', 'YEARQUARTER', 'YEAR', 'FISCAL_WEEK_NUM', 'FISCAL_MONTH_NUM',
                'FISCAL_YEARMONTH', 'FISCAL_QUARTER', 'FISCAL_YEARQUARTER', 'FISCAL_HALFYEAR',
                'FISCAL_YEAR', 'SQL_TIMESTAMP', 'CURRENT_ROW_IND', 'EFFECTIVE_DATE', 'EXPIRATION_DATE'
            ]
        }
    }
    
    # Pass-through views for fact tables
    fact_views = {
        'VW_Fact_SalesActual': {
            'table': 'Fact_SalesActual',
            'columns': [
                'DimProductID', 'DimStoreID', 'DimResellerID', 'DimCustomerID', 'DimChannelID',
                'DimSaleDateID', 'DimLocationID', 'SalesHeaderID', 'SalesDetailID', 'SaleAmount',
                'SaleQuantity', 'SaleUnitPrice', 'SaleExtendedCost', 'SaleTotalProfit'
            ]
        },
        'VW_Fact_ProductSalesTarget': {
            'table': 'Fact_ProductSalesTarget',
            'columns': [
                'DimProductID', 'DimTargetDateID', 'ProductTargetSalesQuantity'
            ]
        },
        'VW_Fact_SRCSalesTarget': {
            'table': 'Fact_SRCSalesTarget',
            'columns': [
                'DimStoreID', 'DimResellerID', 'DimChannelID', 'DimTargetDateID', 'SalesTargetAmount'
            ]
        }
    }
    
    # Create dimension pass-through views
    for view_name, view_info in dimension_views.items():
        table_name = view_info['table']
        columns = ', '.join(view_info['columns'])
        
        view_sql = f"""
        CREATE OR REPLACE SECURE VIEW {view_name} AS
        SELECT 
            {columns}
        FROM {table_name};
        """
        
        try:
            cursor.execute(view_sql)
            print(f"✓ Created pass-through view: {view_name}")
        except Exception as e:
            print(f"✗ Error creating view {view_name}: {str(e)}")
    
    # Create fact pass-through views  
    for view_name, view_info in fact_views.items():
        table_name = view_info['table']
        columns = ', '.join(view_info['columns'])
        
        view_sql = f"""
        CREATE OR REPLACE SECURE VIEW {view_name} AS
        SELECT 
            {columns}
        FROM {table_name};
        """
        
        try:
            cursor.execute(view_sql)
            print(f"✓ Created pass-through view: {view_name}")
        except Exception as e:
            print(f"✗ Error creating view {view_name}: {str(e)}")

def create_analytical_views(cursor):
    """Create custom analytical secure views for data visualization support"""
    
    print("\n" + "="*80)
    print("CREATING CUSTOM ANALYTICAL SECURE VIEWS")
    print("="*80)
    
    # View 1: Sales Performance Summary by Product and Date
    sales_summary_view = """
    CREATE OR REPLACE SECURE VIEW VW_SalesPerformanceSummary AS
    SELECT 
        p.ProductName,
        p.ProductCategory,
        p.ProductType,
        d.YEAR,
        d.QUARTER,
        d.MONTH_NAME,
        d.YEARMONTH,
        COUNT(DISTINCT fs.SalesHeaderID) as TransactionCount,
        SUM(fs.SaleQuantity) as TotalQuantitySold,
        SUM(fs.SaleAmount) as TotalSalesAmount,
        SUM(fs.SaleTotalProfit) as TotalProfit,
        AVG(fs.SaleUnitPrice) as AvgUnitPrice,
        SUM(fs.SaleAmount) / NULLIF(SUM(fs.SaleQuantity), 0) as AvgSalePerUnit,
        SUM(fs.SaleTotalProfit) / NULLIF(SUM(fs.SaleAmount), 0) * 100 as ProfitMarginPercent,
        p.ProductRetailPrice,
        p.ProductCost,
        p.ProductProfitMarginUnitPercent
    FROM Fact_SalesActual fs
    INNER JOIN Dim_Product p ON fs.DimProductID = p.DimProductID
    INNER JOIN Dim_Date d ON fs.DimSaleDateID = d.DATE_PKEY
    GROUP BY 
        p.ProductName, p.ProductCategory, p.ProductType, p.ProductRetailPrice, 
        p.ProductCost, p.ProductProfitMarginUnitPercent,
        d.YEAR, d.QUARTER, d.MONTH_NAME, d.YEARMONTH
    """
    
    # View 2: Customer Demographics and Sales Analysis
    customer_analysis_view = """
    CREATE OR REPLACE SECURE VIEW VW_CustomerSalesAnalysis AS
    SELECT 
        c.CustomerGender,
        l.State_Province,
        l.Country,
        ch.ChannelName,
        ch.ChannelCategory,
        d.YEAR,
        d.QUARTER,
        COUNT(DISTINCT c.DimCustomerID) as UniqueCustomers,
        COUNT(DISTINCT fs.SalesHeaderID) as TotalTransactions,
        SUM(fs.SaleAmount) as TotalSalesAmount,
        SUM(fs.SaleQuantity) as TotalQuantity,
        SUM(fs.SaleTotalProfit) as TotalProfit,
        AVG(fs.SaleAmount) as AvgTransactionAmount,
        SUM(fs.SaleAmount) / NULLIF(COUNT(DISTINCT c.DimCustomerID), 0) as SalesPerCustomer,
        SUM(fs.SaleTotalProfit) / NULLIF(COUNT(DISTINCT c.DimCustomerID), 0) as ProfitPerCustomer
    FROM Fact_SalesActual fs
    INNER JOIN Dim_Customer c ON fs.DimCustomerID = c.DimCustomerID
    INNER JOIN Dim_Location l ON c.DimLocationID = l.DimLocationID
    INNER JOIN Dim_Channel ch ON fs.DimChannelID = ch.DimChannelID
    INNER JOIN Dim_Date d ON fs.DimSaleDateID = d.DATE_PKEY
    GROUP BY 
        c.CustomerGender, l.State_Province, l.Country, 
        ch.ChannelName, ch.ChannelCategory,
        d.YEAR, d.QUARTER
    """
    
    # View 3: Sales Target vs Actual Performance Analysis
    target_performance_view = """
    CREATE OR REPLACE SECURE VIEW VW_TargetVsActualPerformance AS
    SELECT 
        p.ProductName,
        p.ProductCategory,
        s.StoreName,
        sl.State_Province as StoreState,
        sl.Country as StoreCountry,
        r.ResellerName,
        rl.State_Province as ResellerState,
        c.ChannelName,
        td.YEAR as TargetYear,
        td.QUARTER as TargetQuarter,
        td.MONTH_NAME as TargetMonth,
        
        -- Product Sales Targets
        SUM(fps.ProductTargetSalesQuantity) as ProductSalesTarget,
        
        -- SRC Sales Targets  
        SUM(fst.SalesTargetAmount) as SRCSalesTarget,
        
        -- Actual Sales Performance
        SUM(CASE WHEN d.YEAR = td.YEAR AND d.QUARTER = td.QUARTER 
                 THEN fs.SaleQuantity ELSE 0 END) as ActualQuantitySold,
        SUM(CASE WHEN d.YEAR = td.YEAR AND d.QUARTER = td.QUARTER 
                 THEN fs.SaleAmount ELSE 0 END) as ActualSalesAmount,
        SUM(CASE WHEN d.YEAR = td.YEAR AND d.QUARTER = td.QUARTER 
                 THEN fs.SaleTotalProfit ELSE 0 END) as ActualProfit,
        
        -- Performance Ratios
        CASE WHEN SUM(fps.ProductTargetSalesQuantity) > 0 
             THEN SUM(CASE WHEN d.YEAR = td.YEAR AND d.QUARTER = td.QUARTER 
                           THEN fs.SaleQuantity ELSE 0 END) / SUM(fps.ProductTargetSalesQuantity) * 100
             ELSE 0 END as QuantityTargetAchievementPercent,
             
        CASE WHEN SUM(fst.SalesTargetAmount) > 0 
             THEN SUM(CASE WHEN d.YEAR = td.YEAR AND d.QUARTER = td.QUARTER 
                           THEN fs.SaleAmount ELSE 0 END) / SUM(fst.SalesTargetAmount) * 100
             ELSE 0 END as SalesTargetAchievementPercent
        
    FROM Dim_Date td
    LEFT JOIN Fact_ProductSalesTarget fps ON td.DATE_PKEY = fps.DimTargetDateID
    LEFT JOIN Fact_SRCSalesTarget fst ON td.DATE_PKEY = fst.DimTargetDateID
    LEFT JOIN Dim_Product p ON fps.DimProductID = p.DimProductID
    LEFT JOIN Dim_Store s ON fst.DimStoreID = s.DimStoreID
    LEFT JOIN Dim_Location sl ON s.DimLocationID = sl.DimLocationID
    LEFT JOIN Dim_Reseller r ON fst.DimResellerID = r.DimResellerID
    LEFT JOIN Dim_Location rl ON r.DimLocationID = rl.DimLocationID
    LEFT JOIN Dim_Channel c ON fst.DimChannelID = c.DimChannelID
    LEFT JOIN Fact_SalesActual fs ON (
        (fps.DimProductID = fs.DimProductID OR fps.DimProductID IS NULL) AND
        (fst.DimStoreID = fs.DimStoreID OR fst.DimStoreID IS NULL) AND
        (fst.DimResellerID = fs.DimResellerID OR fst.DimResellerID IS NULL) AND
        (fst.DimChannelID = fs.DimChannelID OR fst.DimChannelID IS NULL)
    )
    LEFT JOIN Dim_Date d ON fs.DimSaleDateID = d.DATE_PKEY
    WHERE td.YEAR >= 2013 AND td.DAY_NUM_IN_MONTH = 1  -- First day of each month for targets
    GROUP BY 
        p.ProductName, p.ProductCategory, s.StoreName, sl.State_Province, sl.Country,
        r.ResellerName, rl.State_Province, c.ChannelName,
        td.YEAR, td.QUARTER, td.MONTH_NAME
    HAVING SUM(fps.ProductTargetSalesQuantity) > 0 OR SUM(fst.SalesTargetAmount) > 0
    """
    
    # View 4: Store 5 and 8 Performance Assessment
    store58_performance_view = """
    CREATE OR REPLACE SECURE VIEW VW_Store58Performance AS
    SELECT 
        s.StoreNumber,
        s.StoreName,
        sl.State_Province as StoreState,
        d.YEAR,
        d.QUARTER,
        d.MONTH_NAME,
        COUNT(DISTINCT fs.SalesHeaderID) as TransactionCount,
        SUM(fs.SaleQuantity) as TotalQuantitySold,
        SUM(fs.SaleAmount) as TotalSalesAmount,
        SUM(fs.SaleTotalProfit) as TotalProfit,
        AVG(fs.SaleAmount) as AvgTransactionAmount,
        SUM(fs.SaleTotalProfit) / NULLIF(SUM(fs.SaleAmount), 0) * 100 as ProfitMarginPercent,
        
        -- Monthly averages for trend analysis
        SUM(fs.SaleAmount) / COUNT(DISTINCT d.MONTH_NUM_IN_YEAR) as AvgMonthlySales,
        SUM(fs.SaleTotalProfit) / COUNT(DISTINCT d.MONTH_NUM_IN_YEAR) as AvgMonthlyProfit,
        
        -- Performance indicators
        CASE 
            WHEN SUM(fs.SaleTotalProfit) / NULLIF(SUM(fs.SaleAmount), 0) * 100 > 40 THEN 'High Profit'
            WHEN SUM(fs.SaleTotalProfit) / NULLIF(SUM(fs.SaleAmount), 0) * 100 > 25 THEN 'Medium Profit'
            ELSE 'Low Profit'
        END as ProfitCategory,
        
        CASE 
            WHEN SUM(fs.SaleAmount) > 50000000 THEN 'High Revenue'
            WHEN SUM(fs.SaleAmount) > 25000000 THEN 'Medium Revenue'
            ELSE 'Low Revenue'
        END as RevenueCategory
        
    FROM Fact_SalesActual fs
    INNER JOIN Dim_Store s ON fs.DimStoreID = s.DimStoreID
    INNER JOIN Dim_Location sl ON s.DimLocationID = sl.DimLocationID
    INNER JOIN Dim_Date d ON fs.DimSaleDateID = d.DATE_PKEY
    WHERE s.StoreNumber IN ('5', '8')
    GROUP BY 
        s.StoreNumber, s.StoreName, sl.State_Province,
        d.YEAR, d.QUARTER, d.MONTH_NAME
    ORDER BY s.StoreNumber, d.YEAR, d.QUARTER
    """
    
    # View 5: Store Bonus Recommendation (Men's/Women's Casual)
    store_bonus_view = """
    CREATE OR REPLACE SECURE VIEW VW_StoreBonusRecommendation AS
    SELECT 
        s.StoreNumber,
        s.StoreName,
        sl.State_Province as StoreState,
        d.YEAR,
        p.ProductType,
        
        -- Sales Performance Metrics
        SUM(fs.SaleAmount) as TotalSalesAmount,
        SUM(fs.SaleTotalProfit) as TotalProfit,
        SUM(fs.SaleQuantity) as TotalQuantity,
        COUNT(DISTINCT fs.SalesHeaderID) as TransactionCount,
        
        -- Performance ratios
        SUM(fs.SaleTotalProfit) / NULLIF(SUM(fs.SaleAmount), 0) * 100 as ProfitMarginPercent,
        SUM(fs.SaleAmount) / COUNT(DISTINCT fs.SalesHeaderID) as AvgTransactionValue,
        
        -- Ranking within year for bonus allocation
        RANK() OVER (PARTITION BY d.YEAR, p.ProductType ORDER BY SUM(fs.SaleAmount) DESC) as SalesRank,
        RANK() OVER (PARTITION BY d.YEAR, p.ProductType ORDER BY SUM(fs.SaleTotalProfit) DESC) as ProfitRank,
        
        -- Share of total sales for bonus calculation
        SUM(fs.SaleAmount) / SUM(SUM(fs.SaleAmount)) OVER (PARTITION BY d.YEAR, p.ProductType) * 100 as SalesSharePercent,
        SUM(fs.SaleTotalProfit) / SUM(SUM(fs.SaleTotalProfit)) OVER (PARTITION BY d.YEAR, p.ProductType) * 100 as ProfitSharePercent,
        
        -- Bonus recommendations (base calculations)
        CASE d.YEAR
            WHEN 2013 THEN ROUND(500000 * (SUM(fs.SaleAmount) / SUM(SUM(fs.SaleAmount)) OVER (PARTITION BY d.YEAR, p.ProductType)) * 0.5 + 
                                 500000 * (SUM(fs.SaleTotalProfit) / SUM(SUM(fs.SaleTotalProfit)) OVER (PARTITION BY d.YEAR, p.ProductType)) * 0.5, 2)
            WHEN 2014 THEN ROUND(400000 * (SUM(fs.SaleAmount) / SUM(SUM(fs.SaleAmount)) OVER (PARTITION BY d.YEAR, p.ProductType)) * 0.5 + 
                                 400000 * (SUM(fs.SaleTotalProfit) / SUM(SUM(fs.SaleTotalProfit)) OVER (PARTITION BY d.YEAR, p.ProductType)) * 0.5, 2)
            ELSE 0
        END as RecommendedBonus
        
    FROM Fact_SalesActual fs
    INNER JOIN Dim_Store s ON fs.DimStoreID = s.DimStoreID
    INNER JOIN Dim_Location sl ON s.DimLocationID = sl.DimLocationID
    INNER JOIN Dim_Product p ON fs.DimProductID = p.DimProductID
    INNER JOIN Dim_Date d ON fs.DimSaleDateID = d.DATE_PKEY
    WHERE p.ProductType IN ('Men''s Casual', 'Women''s Casual')
      AND d.YEAR IN (2013, 2014)
      AND s.StoreNumber IS NOT NULL
      AND s.StoreNumber != 'Unknown'
    GROUP BY 
        s.StoreNumber, s.StoreName, sl.State_Province,
        d.YEAR, p.ProductType
    ORDER BY d.YEAR, p.ProductType, SalesRank
    """
    
    # View 6: Store 5 and 8 Day of Week Analysis  
    store58_dayofweek_view = """
    CREATE OR REPLACE SECURE VIEW VW_Store58DayOfWeekAnalysis AS
    SELECT 
        s.StoreNumber,
        s.StoreName,
        d.DAY_NAME,
        d.DAY_NUM_IN_WEEK,
        d.WEEKDAY_IND,
        d.YEAR,
        
        -- Sales metrics by day of week
        COUNT(DISTINCT fs.SalesHeaderID) as TransactionCount,
        SUM(fs.SaleAmount) as TotalSalesAmount,
        SUM(fs.SaleTotalProfit) as TotalProfit,
        SUM(fs.SaleQuantity) as TotalQuantity,
        AVG(fs.SaleAmount) as AvgTransactionAmount,
        
        -- Day of week performance indicators
        SUM(fs.SaleAmount) / SUM(SUM(fs.SaleAmount)) OVER (PARTITION BY s.StoreNumber, d.YEAR) * 100 as DaySharePercent,
        
        -- Comparative metrics
        AVG(SUM(fs.SaleAmount)) OVER (PARTITION BY s.StoreNumber, d.YEAR) as AvgDailySales,
        SUM(fs.SaleAmount) / AVG(SUM(fs.SaleAmount)) OVER (PARTITION BY s.StoreNumber, d.YEAR) * 100 as DayVsAvgPercent,
        
        -- Ranking days within store-year
        RANK() OVER (PARTITION BY s.StoreNumber, d.YEAR ORDER BY SUM(fs.SaleAmount) DESC) as DaySalesRank,
        
        -- Trend indicators
        CASE 
            WHEN d.DAY_NAME IN ('Saturday', 'Sunday') THEN 'Weekend'
            ELSE 'Weekday'
        END as DayType,
        
        CASE d.DAY_NAME
            WHEN 'Monday' THEN 1
            WHEN 'Tuesday' THEN 2
            WHEN 'Wednesday' THEN 3
            WHEN 'Thursday' THEN 4
            WHEN 'Friday' THEN 5
            WHEN 'Saturday' THEN 6
            WHEN 'Sunday' THEN 7
        END as DayOrder
        
    FROM Fact_SalesActual fs
    INNER JOIN Dim_Store s ON fs.DimStoreID = s.DimStoreID
    INNER JOIN Dim_Date d ON fs.DimSaleDateID = d.DATE_PKEY
    WHERE s.StoreNumber IN ('5', '8')
      AND d.YEAR IN (2013, 2014)
    GROUP BY 
        s.StoreNumber, s.StoreName, d.DAY_NAME, d.DAY_NUM_IN_WEEK, 
        d.WEEKDAY_IND, d.YEAR
    ORDER BY s.StoreNumber, d.YEAR, DayOrder
    """
    
    # View 7: Multi-Store vs Single-Store State Analysis
    multistore_analysis_view = """
    CREATE OR REPLACE SECURE VIEW VW_MultiStoreVsSingleStoreAnalysis AS
    WITH StoreCountByState AS (
        SELECT 
            l.State_Province,
            COUNT(DISTINCT s.DimStoreID) as StoreCount,
            CASE 
                WHEN COUNT(DISTINCT s.DimStoreID) > 1 THEN 'Multi-Store State'
                ELSE 'Single-Store State'
            END as StoreConfiguration
        FROM Dim_Store s
        INNER JOIN Dim_Location l ON s.DimLocationID = l.DimLocationID
        WHERE s.StoreNumber IS NOT NULL AND s.StoreNumber != 'Unknown'
        GROUP BY l.State_Province
    ),
    StatePerformance AS (
        SELECT 
            l.State_Province,
            scs.StoreConfiguration,
            scs.StoreCount,
            d.YEAR,
            
            -- Store-level metrics
            COUNT(DISTINCT s.DimStoreID) as ActiveStores,
            
            -- Sales performance metrics
            SUM(fs.SaleAmount) as TotalSalesAmount,
            SUM(fs.SaleTotalProfit) as TotalProfit,
            SUM(fs.SaleQuantity) as TotalQuantity,
            COUNT(DISTINCT fs.SalesHeaderID) as TotalTransactions,
            
            -- Per-store averages
            SUM(fs.SaleAmount) / COUNT(DISTINCT s.DimStoreID) as AvgSalesPerStore,
            SUM(fs.SaleTotalProfit) / COUNT(DISTINCT s.DimStoreID) as AvgProfitPerStore,
            COUNT(DISTINCT fs.SalesHeaderID) / COUNT(DISTINCT s.DimStoreID) as AvgTransactionsPerStore,
            
            -- Performance ratios
            SUM(fs.SaleTotalProfit) / NULLIF(SUM(fs.SaleAmount), 0) * 100 as ProfitMarginPercent,
            SUM(fs.SaleAmount) / NULLIF(COUNT(DISTINCT fs.SalesHeaderID), 0) as AvgTransactionValue
            
        FROM Fact_SalesActual fs
        INNER JOIN Dim_Store s ON fs.DimStoreID = s.DimStoreID
        INNER JOIN Dim_Location l ON s.DimLocationID = l.DimLocationID
        INNER JOIN Dim_Date d ON fs.DimSaleDateID = d.DATE_PKEY
        INNER JOIN StoreCountByState scs ON l.State_Province = scs.State_Province
        WHERE s.StoreNumber IS NOT NULL AND s.StoreNumber != 'Unknown'
          AND d.YEAR IN (2013, 2014)
        GROUP BY 
            l.State_Province, scs.StoreConfiguration, scs.StoreCount, d.YEAR
    )
    SELECT 
        sp.*,
        
        -- Comparative analysis
        AVG(CASE WHEN StoreConfiguration = 'Multi-Store State' THEN AvgSalesPerStore END) 
            OVER (PARTITION BY YEAR) as MultiStoreAvgSales,
        AVG(CASE WHEN StoreConfiguration = 'Single-Store State' THEN AvgSalesPerStore END) 
            OVER (PARTITION BY YEAR) as SingleStoreAvgSales,
            
        AVG(CASE WHEN StoreConfiguration = 'Multi-Store State' THEN AvgProfitPerStore END) 
            OVER (PARTITION BY YEAR) as MultiStoreAvgProfit,
        AVG(CASE WHEN StoreConfiguration = 'Single-Store State' THEN AvgProfitPerStore END) 
            OVER (PARTITION BY YEAR) as SingleStoreAvgProfit,
            
        -- Performance vs configuration average
        CASE StoreConfiguration
            WHEN 'Multi-Store State' THEN 
                AvgSalesPerStore / AVG(CASE WHEN StoreConfiguration = 'Multi-Store State' THEN AvgSalesPerStore END) 
                    OVER (PARTITION BY YEAR) * 100
            ELSE 
                AvgSalesPerStore / AVG(CASE WHEN StoreConfiguration = 'Single-Store State' THEN AvgSalesPerStore END) 
                    OVER (PARTITION BY YEAR) * 100
        END as SalesVsConfigAvgPercent
        
    FROM StatePerformance sp
    ORDER BY StoreConfiguration, YEAR, AvgSalesPerStore DESC
    """

    # Create the analytical views
    analytical_views = [
        ("VW_SalesPerformanceSummary", sales_summary_view),
        ("VW_CustomerSalesAnalysis", customer_analysis_view), 
        ("VW_TargetVsActualPerformance", target_performance_view),
        ("VW_Store58Performance", store58_performance_view),
        ("VW_StoreBonusRecommendation", store_bonus_view),
        ("VW_Store58DayOfWeekAnalysis", store58_dayofweek_view),
        ("VW_MultiStoreVsSingleStoreAnalysis", multistore_analysis_view)
    ]
    
    for view_name, view_sql in analytical_views:
        try:
            cursor.execute(view_sql)
            print(f"✓ Created analytical view: {view_name}")
        except Exception as e:
            print(f"✗ Error creating view {view_name}: {str(e)}")

def main():
    """Main function to create all secure views"""
    
    print("\n" + "="*80)
    print(f"CREATING SECURE VIEWS FOR: {DIMENSION_DB_NAME}")
    print("="*80)
    
    try:
        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Use the dimensional database
        cursor.execute(f"USE DATABASE {DIMENSION_DB_NAME}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")
        
        print(f"✓ Connected to database: {DIMENSION_DB_NAME}")
        print(f"✓ Using schema: {SNOWFLAKE_SCHEMA}")
        
        # Create pass-through views
        create_passthrough_views(cursor)
        
        # Create analytical views
        create_analytical_views(cursor)
        
        print("\n" + "="*80)
        print("SECURE VIEWS CREATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nPASS-THROUGH VIEWS CREATED:")
        print("• VW_Dim_Product, VW_Dim_Customer, VW_Dim_Location, VW_Dim_Channel")
        print("• VW_Dim_Store, VW_Dim_Reseller, VW_Dim_Date")
        print("• VW_Fact_SalesActual, VW_Fact_ProductSalesTarget, VW_Fact_SRCSalesTarget")
        print("\nANALYTICAL VIEWS CREATED:")
        print("• VW_SalesPerformanceSummary - Product sales performance by time periods")
        print("• VW_CustomerSalesAnalysis - Customer demographics and sales patterns")
        print("• VW_TargetVsActualPerformance - Sales targets vs actual performance comparison")
        print("• VW_Store58Performance - Store 5 and 8 performance assessment")
        print("• VW_StoreBonusRecommendation - Store bonus recommendation")
        print("• VW_Store58DayOfWeekAnalysis - Store 5 and 8 day of week analysis")
        print("• VW_MultiStoreVsSingleStoreAnalysis - Multi-store vs single-store state analysis")
        
    except Exception as e:
        print(f"\n✗ Error in main process: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 