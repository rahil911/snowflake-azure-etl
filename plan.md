# Snowflake ETL Plan

## Overview
We need to create a data pipeline that:
1. Creates stages in Snowflake pointing to Azure Blob Storage for 12 different entities
2. Creates tables with the appropriate schemas for each entity
3. Loads data from the stages into the tables
4. Verifies the data loading by checking sample rows

## Entities
- channel
- channelcategory
- customer
- product
- productcategory
- producttype
- reseller
- salesdetail
- salesheader
- store
- targetdatachannel
- targetdataproduct

## Steps

### Step 1: Set up Connection Script
- Create a script that handles Snowflake connection
- Test the connection

### Step 2: Create Tables
- Create a script that creates tables with proper schemas from csv_schemas.md
- Test table creation

### Step 3: Create External Stages
- Create a script that creates external stages pointing to Azure Blob Storage
- Test stage creation with proper URL format
- Note: From the errors we're seeing, we need to be careful with the URL format

### Step 4: Load Data from Stages
- Create a script that loads data from stages to tables
- Test data loading

### Step 5: Verify Data
- Create a script that verifies data by checking sample rows
- Test data verification

## Notes
- We've seen issues with the Azure URL format
- Need to ensure the Azure URL is in the format Snowflake expects
- Looking at the STARWARSCHARACTERS example, we need to study exactly how that's set up 