# Secure Views Documentation

This document describes the secure views system created for the dimensional model, providing both pass-through views and analytical views for data access and visualization.

## Overview

The secure views system creates a data access layer between users and the dimensional model tables. This approach provides several benefits:

1. **Security**: All views are created as `SECURE VIEWS` for sharing with professors/graders
2. **Data Protection**: Views isolate the warehouse from direct table queries
3. **Change Insulation**: Downstream objects are protected from table structure changes
4. **Explicit Column Control**: No `SELECT *` usage for better security and performance
5. **Business Intelligence**: Pre-aggregated analytical views optimize visualization tools

## View Categories

### Pass-Through Views (10 views)

These are exact copies of dimension and fact tables using explicit column lists (no SELECT *):

**Dimension Views:**
- `VW_Dim_Product` - Product dimension pass-through
- `VW_Dim_Customer` - Customer dimension pass-through  
- `VW_Dim_Location` - Location dimension pass-through
- `VW_Dim_Channel` - Channel dimension pass-through
- `VW_Dim_Store` - Store dimension pass-through
- `VW_Dim_Reseller` - Reseller dimension pass-through
- `VW_DIM_DATE` - Date dimension pass-through

**Fact Views:**
- `VW_Fact_SalesActual` - Sales actual fact pass-through
- `VW_Fact_ProductSalesTarget` - Product sales target fact pass-through
- `VW_Fact_SRCSalesTarget` - SRC sales target fact pass-through

### Analytical Views (7 views)

Pre-aggregated views optimized for business intelligence and visualization:

**Standard Analytical Views:**
- `VW_SalesPerformanceSummary` - Sales performance by product, date, and metrics
- `VW_CustomerSalesAnalysis` - Customer demographics and sales patterns
- `VW_TargetVsActualPerformance` - Sales targets vs actual performance comparison

**Business Analysis Views (Custom for Specific Questions):**

#### 1. `VW_Store58Performance` - Store 5 and 8 Assessment
**Purpose:** Overall assessment of stores 5 and 8's sales performance
**Key Metrics:**
- Sales amounts, quantities, and profit margins
- Monthly averages and trend analysis
- Performance categorization (High/Medium/Low profit and revenue)
- Year-over-year and quarterly comparisons

**Business Questions Addressed:**
- How are stores 5 and 8 performing compared to targets?
- Will they meet their 2014 targets?
- Should either store be closed?
- What should be done to maximize store profits?

#### 2. `VW_StoreBonusRecommendation` - Bonus Allocation Analysis
**Purpose:** Recommend 2013 and 2014 bonus amounts based on Men's/Women's Casual sales
**Key Metrics:**
- Sales performance by store and product type
- Rankings and market share percentages
- Calculated bonus recommendations based on $500K (2013) and $400K (2014) pools
- Performance ratios weighted 50% sales, 50% profit

**Business Questions Addressed:**
- How should bonus pools be allocated across stores?
- Which stores perform best in Men's/Women's Casual categories?
- What are fair bonus amounts based on performance?

#### 3. `VW_Store58DayOfWeekAnalysis` - Day-of-Week Sales Trends
**Purpose:** Assess product sales by day of week at stores 5 and 8
**Key Metrics:**
- Sales by day of week with trend indicators
- Weekend vs weekday performance
- Day share percentages and rankings
- Performance vs daily averages

**Business Questions Addressed:**
- What sales trends can we learn from day-of-week patterns?
- Are there optimal staffing or promotion opportunities?
- How do weekend vs weekday sales compare?

#### 4. `VW_MultiStoreVsSingleStoreAnalysis` - Multi-Store State Performance
**Purpose:** Compare performance of multi-store states vs single-store states
**Key Metrics:**
- Store configuration analysis (Multi-Store vs Single-Store states)
- Per-store performance averages by configuration
- Comparative performance ratios
- State-level aggregations and insights

**Business Questions Addressed:**
- What can we learn about having multiple stores in a state?
- Do multi-store states perform better per store than single-store states?
- Are there economies of scale or market saturation effects?

## Usage Examples

### Running the Complete Views ETL

```bash
python -m rahil.run_views_etl
```

This will:
1. Create all 10 pass-through secure views
2. Create all 7 analytical secure views (including 4 business analysis views)
3. Verify all views with sample data display
4. Provide comprehensive logging and business insights

### Sample Business Analysis Queries

#### Store 5 and 8 Performance Assessment
```sql
SELECT StoreNumber, StoreName, YEAR, 
       TotalSalesAmount, TotalProfit, ProfitMarginPercent,
       ProfitCategory, RevenueCategory
FROM VW_Store58Performance 
ORDER BY StoreNumber, YEAR;
```

#### Bonus Recommendations
```sql
SELECT StoreNumber, StoreName, YEAR, ProductType,
       TotalSalesAmount, TotalProfit, SalesRank, 
       RecommendedBonus
FROM VW_StoreBonusRecommendation 
ORDER BY YEAR, ProductType, SalesRank;
```

#### Day of Week Analysis
```sql
SELECT StoreNumber, DAY_NAME, DayType,
       TotalSalesAmount, DaySharePercent, DaySalesRank
FROM VW_Store58DayOfWeekAnalysis 
WHERE YEAR = 2013
ORDER BY StoreNumber, DayOrder;
```

#### Multi-Store vs Single-Store Comparison
```sql
SELECT StoreConfiguration, YEAR, 
       AVG(AvgSalesPerStore) as AvgSalesPerStore,
       AVG(AvgProfitPerStore) as AvgProfitPerStore,
       COUNT(*) as StateCount
FROM VW_MultiStoreVsSingleStoreAnalysis 
GROUP BY StoreConfiguration, YEAR
ORDER BY YEAR, StoreConfiguration;
```

## Technical Implementation

### Security Features
- All views created with `CREATE SECURE VIEW` syntax
- Explicit column lists (no SELECT * usage)
- Row-level security considerations built-in
- Safe for sharing with external parties

### Performance Optimizations
- Pre-aggregated calculations reduce query complexity
- Strategic use of window functions for rankings and comparisons
- Optimized joins with appropriate WHERE clauses
- Indexed dimension keys for fast lookups

### Data Quality Safeguards
- COALESCE functions handle null values
- Default unknown values for missing dimension references
- Data type casting ensures consistent formats
- WHERE clauses filter invalid or test data

## Files Structure

```
rahil/
├── create_views.py           # Main view creation script
├── view_sample_views.py      # View verification and sampling
├── run_views_etl.py         # Complete ETL runner
└── connection.py            # Database connection utilities

VIEWS_README.md              # This documentation file
```

## Business Intelligence Integration

These views are optimized for use with:
- **Tableau** - Direct connection to secure views
- **Power BI** - ODBC/Native connector support  
- **Excel** - Pivot table and chart creation
- **SQL Clients** - Direct querying and analysis

The views provide a complete data access layer that supports both operational reporting and strategic business analysis while maintaining security and performance standards.

## Running the Views ETL

### Complete Views Pipeline

Run the entire views creation and verification process:
```bash
python -m rahil.run_views_etl
```

This will:
1. Create all 10 pass-through secure views
2. Create all 7 analytical secure views (including 4 business analysis views)
3. Verify all views with sample data display
4. Provide a summary of the process

### Individual Components

You can also run each component separately:

#### Create Views Only
```bash
python -m rahil.create_views
```

#### Verify Views Only
```bash
python -m rahil.view_sample_views
```

## View Structure Examples

### Pass-Through View Example

```sql
CREATE OR REPLACE SECURE VIEW VW_Dim_Product AS
SELECT 
    DimProductID,
    ProductID,
    ProductTypeID,
    ProductCategoryID,
    ProductName,
    ProductType,
    ProductCategory,
    ProductRetailPrice,
    ProductWholesalePrice,
    ProductCost,
    ProductRetailProfit,
    ProductWholesaleUnitProfit,
    ProductProfitMarginUnitPercent
FROM Dim_Product;
```

### Analytical View Example

```sql
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
    SUM(fs.SaleTotalProfit) / NULLIF(SUM(fs.SaleAmount), 0) * 100 as ProfitMarginPercent
FROM Fact_SalesActual fs
INNER JOIN Dim_Product p ON fs.DimProductID = p.DimProductID
INNER JOIN Dim_Date d ON fs.DimSaleDateID = d.DATE_PKEY
GROUP BY 
    p.ProductName, p.ProductCategory, p.ProductType,
    d.YEAR, d.QUARTER, d.MONTH_NAME, d.YEARMONTH;
```

## Usage in Business Intelligence Tools

### Tableau Integration

1. **Connect to Snowflake** using your credentials
2. **Select the dimensional database** (`IMT577_DW_{USER_NAME}_DIMENSION`)
3. **Use the secure views** instead of direct table access
4. **Drag and drop** pre-calculated metrics from analytical views
5. **Create relationships** between pass-through views if needed

### Sample Tableau Worksheets

#### Sales Performance Dashboard
- Use `VW_SalesPerformanceSummary`
- Create time series charts with `YEAR`, `QUARTER`, `MONTH_NAME`
- Show `TotalSalesAmount` and `TotalProfit` trends
- Filter by `ProductCategory` and `ProductType`

#### Customer Analysis Dashboard
- Use `VW_CustomerSalesAnalysis`
- Create geographic maps with `State_Province` and `Country`
- Show `SalesPerCustomer` by `CustomerGender`
- Analyze channel effectiveness with `ChannelName`

#### Performance Management Dashboard
- Use `VW_TargetVsActualPerformance`
- Create gauge charts for `QuantityTargetAchievementPercent`
- Show target vs actual comparisons over time
- Filter by product, store, and channel dimensions

## Security and Sharing

### SECURE VIEW Benefits

1. **Professor/Grader Access**: Views can be shared without exposing underlying table structures
2. **Data Privacy**: Sensitive columns can be excluded from views
3. **Access Control**: Views provide granular access control to specific data sets
4. **Audit Trail**: View usage can be monitored and logged

### Sharing Views

To share views with professors or graders:

1. **Grant database access**: `GRANT USAGE ON DATABASE IMT577_DW_{USER_NAME}_DIMENSION TO ROLE {PROFESSOR_ROLE}`
2. **Grant schema access**: `GRANT USAGE ON SCHEMA PUBLIC TO ROLE {PROFESSOR_ROLE}`
3. **Grant view access**: `GRANT SELECT ON ALL VIEWS IN SCHEMA PUBLIC TO ROLE {PROFESSOR_ROLE}`

## Maintenance and Updates

### Adding New Views

1. **Edit `create_views.py`** to add new view definitions
2. **Update `view_sample_views.py`** to include the new views in verification
3. **Run the pipeline** to create and verify the new views
4. **Update documentation** with new view descriptions

### Modifying Existing Views

1. **Update view definitions** in `create_views.py`
2. **Test changes** with sample queries
3. **Re-run the pipeline** to recreate the views
4. **Verify** that dependent objects still work correctly

## Best Practices

### View Design

1. **Explicit Columns**: Always list specific columns, never use `SELECT *`
2. **Meaningful Names**: Use descriptive view names with consistent prefixes
3. **Documentation**: Comment complex calculations and business logic
4. **Performance**: Consider indexing strategies for frequently used views

### Security

1. **SECURE VIEWS**: Always use `CREATE SECURE VIEW` for production views
2. **Column Selection**: Only include necessary columns in views
3. **Row-Level Security**: Implement filters for sensitive data
4. **Access Reviews**: Regularly review and audit view access permissions

### Analytics

1. **Pre-Aggregation**: Include common calculations in analytical views
2. **Null Handling**: Use `NULLIF()` to prevent division by zero errors
3. **Data Types**: Ensure consistent data types across joined tables
4. **Performance**: Monitor view query performance and optimize as needed

## Troubleshooting

### Common Issues

1. **View Creation Fails**: Check that all referenced tables exist and have data
2. **Permission Errors**: Ensure your role has CREATE VIEW privileges
3. **Column Mismatches**: Verify that column names match exactly between tables
4. **Performance Issues**: Review join conditions and consider adding filters

### Debugging Steps

1. **Check table existence**: Verify all referenced tables are available
2. **Test queries separately**: Run individual SELECT statements before creating views
3. **Review error messages**: Snowflake provides detailed error information
4. **Check data quality**: Ensure source tables have valid data for joins

## Future Enhancements

### Potential Additions

1. **Time Intelligence Views**: Period-over-period comparisons
2. **Ranking Views**: Top/bottom performers by various metrics
3. **Cohort Analysis Views**: Customer behavior over time
4. **Forecasting Views**: Trend analysis and projection calculations
5. **Executive Summary Views**: High-level KPIs and dashboard metrics

### Advanced Features

1. **Materialized Views**: For improved performance on large datasets
2. **Dynamic Views**: Parameter-driven views for flexible analysis
3. **Real-time Views**: Stream-based views for live data analysis
4. **ML Integration**: Views that incorporate machine learning predictions 