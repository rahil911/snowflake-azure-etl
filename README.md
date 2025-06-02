# Snowflake ETL Pipeline for Azure Blob Storage

This project provides a complete **ETL (Extract, Transform, Load)** solution for loading data from Azure Blob Storage into Snowflake, transforming it into a dimensional model, and creating secure views for data visualization and analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [ETL Pipeline](#etl-pipeline)
- [Dimensional Model](#dimensional-model)
- [Secure Views](#secure-views)
- [Business Intelligence](#business-intelligence)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Data Quality Features](#data-quality-features)

## 🎯 Overview

This comprehensive ETL solution automates the entire data warehousing process:

1. **Staging Layer**: Loads raw data from Azure Blob Storage into Snowflake staging tables
2. **Dimensional Model**: Transforms staging data into a star schema with dimension and fact tables
3. **Secure Views**: Creates secure views for data access, analysis, and sharing
4. **Business Intelligence**: Provides pre-built analytical views for visualization tools

### Key Features

- ✅ **Automated ETL Pipeline**: Complete staging to dimensional model transformation
- ✅ **Data Quality Assurance**: Unknown member handling, NULL protection, type safety
- ✅ **Secure Data Access**: All views created as SECURE VIEWS for academic sharing
- ✅ **Business Intelligence Ready**: Pre-aggregated analytical views for Tableau/BI tools
- ✅ **Comprehensive Logging**: Full process logging with timestamps and error handling
- ✅ **Flexible Configuration**: Environment-based configuration for multiple environments

## 📁 Project Structure

```
STAGING_ETL/
├── rahil/                           # Main ETL package
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Staging configuration
│   ├── dim_config.py               # Dimensional model configuration
│   ├── connection.py               # Snowflake connection handling
│   ├── create_database.py          # Database creation
│   ├── create_stages.py            # External stages setup
│   ├── create_tables.py            # Staging tables creation
│   ├── load_data.py                # Data loading from stages
│   ├── view_sample_data.py         # Sample data verification
│   ├── run_etl.py                  # Main staging ETL runner
│   ├── create_dimension_database.py # Dimensional database setup
│   ├── create_dimension_tables.py  # Dimension tables creation
│   ├── load_dim_date.py            # Date dimension loading
│   ├── load_dimension_tables.py    # Dimension data loading
│   ├── create_fact_tables.py       # Fact tables creation
│   ├── load_fact_tables.py         # Fact data loading
│   ├── run_dimensional_etl.py      # Dimensional model ETL runner
│   ├── create_views.py             # Secure views creation
│   ├── view_sample_views.py        # Views verification
│   ├── run_views_etl.py            # Views ETL runner
│   ├── example.env                 # Environment template
│   ├── .env                        # Your credentials (not in git)
│   └── logs/                       # ETL process logs
├── private_ddl/                     # SQL table definitions
│   ├── staging_*.sql               # Staging table definitions
│   ├── dim_*.sql                   # Dimension table definitions
│   ├── fact_*.sql                  # Fact table definitions
│   ├── load_*.sql                  # Data loading scripts
│   ├── DIM_DATE.sql                # Date dimension script
│   └── README.md                   # SQL conventions guide
├── config/                         # Configuration files
├── scripts/                        # Utility scripts
├── requirements.txt                # Python dependencies
├── setup.sh                       # Environment setup script
└── verify_sql.py                  # SQL validation utility
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Snowflake account with appropriate permissions
- Azure Blob Storage account
- Required Python packages (see requirements.txt)

### Installation

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd STAGING_ETL
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp rahil/example.env rahil/.env
   # Edit rahil/.env with your credentials
   ```

3. **Required environment variables:**
   ```bash
   # Snowflake Configuration
   SNOWFLAKE_ACCOUNT=your_account
   SNOWFLAKE_USER=your_username
   SNOWFLAKE_PASSWORD=your_password
   SNOWFLAKE_WAREHOUSE=your_warehouse
   SNOWFLAKE_ROLE=your_role
   
   # Azure Blob Storage
   AZURE_STORAGE_ACCOUNT=your_storage_account
   AZURE_SAS_TOKEN=your_sas_token
   
   # Database Configuration
   USER_NAME=your_user_name
   ENTITIES=channel,customer,product,salesdetail,salesheader,...
   ```

## 🔄 ETL Pipeline

### Stage 1: Staging ETL

Loads data from Azure Blob Storage into Snowflake staging tables:

```bash
# Run complete staging ETL
python3 -m rahil.run_etl
```

**Process:**
1. Creates staging database (`IMT577_DW_{USER_NAME}_STAGING`)
2. Sets up external stages for Azure Blob Storage
3. Creates staging tables from SQL definitions
4. Loads data from external stages
5. Displays sample data for verification

**Supported Entities:**
- channel, channelcategory, customer, product
- productcategory, producttype, reseller
- salesdetail, salesheader, store
- targetdatachannel, targetdataproduct

### Stage 2: Dimensional Model ETL

Transforms staging data into a star schema dimensional model:

```bash
# Run dimensional model ETL
python3 -m rahil.run_dimensional_etl
```

**Process:**
1. Creates dimensional database (`IMT577_DW_{USER_NAME}_DIMENSION`)
2. Creates dimension tables with unknown members
3. Loads comprehensive date dimension (730 days)
4. Transforms and loads dimension data from staging
5. Creates fact tables with proper relationships
6. Loads fact data with referential integrity

**Dimensional Model:**

**Dimension Tables:**
- `Dim_Product` - Product hierarchy and attributes
- `Dim_Customer` - Customer demographics and location
- `Dim_Location` - Geographic information (shared)
- `Dim_Channel` - Sales channels and categories
- `Dim_Store` - Store details and management
- `Dim_Reseller` - Reseller information and contacts
- `Dim_Date` - Calendar and fiscal date attributes

**Fact Tables:**
- `Fact_SalesActual` - Sales transactions with measures
- `Fact_ProductSalesTarget` - Product sales targets
- `Fact_SRCSalesTarget` - Store/Reseller/Channel targets

### Stage 3: Secure Views ETL

Creates secure views for data access and analysis:

```bash
# Run secure views ETL
python3 -m rahil.run_views_etl
```

**Process:**
1. Creates pass-through secure views for all tables
2. Creates analytical views for business intelligence
3. Provides sample data verification
4. Prepares business analysis insights

## 🔒 Secure Views

### Pass-Through Views (10 views)

Exact copies of dimension and fact tables using explicit column lists:

- `VW_Dim_Product`, `VW_Dim_Customer`, `VW_Dim_Location`
- `VW_Dim_Channel`, `VW_Dim_Store`, `VW_Dim_Reseller`, `VW_Dim_Date`
- `VW_Fact_SalesActual`, `VW_Fact_ProductSalesTarget`, `VW_Fact_SRCSalesTarget`

### Analytical Views (7 views)

Pre-aggregated views optimized for business intelligence:

1. **`VW_SalesPerformanceSummary`** - Product sales performance by time periods
   - Sales amounts, quantities, profit margins
   - Transaction counts and pricing analysis
   - Product hierarchy breakdowns

2. **`VW_CustomerSalesAnalysis`** - Customer demographics and sales patterns  
   - Sales by geography, gender, channel
   - Customer segmentation metrics
   - Channel preference analysis

3. **`VW_TargetVsActualPerformance`** - Sales targets vs actual comparison
   - Achievement percentages by product/store/channel
   - Performance gap analysis
   - Target attainment tracking

4. **`VW_Store58Performance`** - Store 5 and 8 performance assessment
5. **`VW_StoreBonusRecommendation`** - Store bonus recommendations
6. **`VW_Store58DayOfWeekAnalysis`** - Day of week sales patterns
7. **`VW_MultiStoreVsSingleStoreAnalysis`** - Multi vs single-store analysis

## 📊 Business Intelligence

### Key Business Questions Answered

1. **Store Performance Assessment** - Which stores (5 vs 8) perform better?
2. **Bonus Recommendations** - Which stores deserve bonuses for Men's/Women's Casual sales?
3. **Sales Patterns** - How do sales vary by day of week?
4. **Geographic Analysis** - Multi-store vs single-store state performance

### Integration with BI Tools

The secure views are designed for seamless integration with:
- **Tableau** - Use views as data sources for dashboards
- **Power BI** - Connect directly to Snowflake views
- **Looker** - Build models on top of analytical views
- **Excel** - Query views for ad-hoc analysis

### Sample Queries

```sql
-- Product performance analysis
SELECT * FROM VW_SalesPerformanceSummary 
WHERE YEAR = 2013 AND PRODUCTCATEGORY = 'Men''s Apparel';

-- Customer demographics
SELECT * FROM VW_CustomerSalesAnalysis
WHERE CUSTOMERGENDER != 'Unknown';

-- Target achievement
SELECT * FROM VW_TargetVsActualPerformance
WHERE QUANTITYTARGETACHIEVEMENTPERCENT > 0;
```

## ⚙️ Configuration

### Environment Variables

**Core Settings:**
```bash
USER_NAME=your_username              # Used in database naming
SNOWFLAKE_ACCOUNT=your_account       # Snowflake account identifier
SNOWFLAKE_WAREHOUSE=your_warehouse   # Compute warehouse
```

**Entity Configuration:**
```bash
# Process all entities
ENTITIES=channel,channelcategory,customer,product,productcategory,producttype,reseller,salesdetail,salesheader,store,targetdatachannel,targetdataproduct

# Process subset
ENTITIES=channel,customer,product,salesdetail,salesheader
```

### Azure Blob Storage Structure

Your storage should match entity names:
```
your_storage_account.blob.core.windows.net/
├── channel/
│   └── channel.csv
├── customer/
│   └── customer.csv
└── product/
    └── product.csv
```

## 🔧 Usage Examples

### Run Complete Pipeline

```bash
# Full ETL pipeline (staging + dimensional + views)
python3 -m rahil.run_etl
python3 -m rahil.run_dimensional_etl  
python3 -m rahil.run_views_etl
```

### Individual Components

```bash
# Staging only
python3 -m rahil.create_database
python3 -m rahil.create_stages
python3 -m rahil.create_tables
python3 -m rahil.load_data

# Dimensional model only
python3 -m rahil.create_dimension_database
python3 -m rahil.create_dimension_tables
python3 -m rahil.load_dimension_tables
python3 -m rahil.create_fact_tables
python3 -m rahil.load_fact_tables

# Views only
python3 -m rahil.create_views
python3 -m rahil.view_sample_views
```

### Data Verification

```bash
# View sample data
python3 -m rahil.view_sample_data
python3 -m rahil.view_sample_views

# Validate SQL files
python3 verify_sql.py
```

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify Snowflake credentials in `.env`
   - Check warehouse is running
   - Ensure role has required permissions

2. **Missing Data**
   - Verify Azure Blob Storage access
   - Check SAS token permissions
   - Confirm entity names match folder structure

3. **SQL Errors**
   - Validate SQL files in `private_ddl/`
   - Check column name consistency
   - Verify data type compatibility

### Debug Mode

Enable detailed logging by setting:
```bash
export PYTHON_LOG_LEVEL=DEBUG
```

### Log Files

ETL processes generate timestamped logs:
- `rahil/logs/etl_run_YYYYMMDD_HHMMSS.log`
- `rahil/logs/dimensional_etl_YYYYMMDD_HHMMSS.log`
- `rahil/logs/views_etl_YYYYMMDD_HHMMSS.log`

## 🛡️ Data Quality Features

### Unknown Member Handling
- Each dimension has "Unknown" records for missing references
- Prevents orphaned facts and broken relationships
- Maintains referential integrity in the dimensional model

### Data Type Protection
- UUID handling for CustomerID and ResellerID fields
- Proper type casting in JOIN conditions
- Consistent data type usage across tables

### NULL Value Protection
- Comprehensive COALESCE usage for default values
- NULL-safe transformations in dimension loading
- Protected aggregations in analytical views

### Referential Integrity
- Foreign key relationships maintained through lookups
- Unknown member references for missing dimension data
- Consistent surrogate key usage across fact tables

## 🤝 Contributing

1. Follow SQL naming conventions in `private_ddl/README.md`
2. Update environment template when adding new variables
3. Add logging for new ETL processes
4. Update this README when adding new features

## 📄 License

This project is for educational purposes in the IMT577 Data Warehousing course.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Verify environment configuration
4. Contact course instructors for academic support

---

**Happy Data Warehousing! 🎉** 