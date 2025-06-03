# Private SQL Table Definitions

This directory contains SQL files with table definition statements that are used by the ETL scripts.

## Convention

Each file should contain a single CREATE TABLE statement for a table in one of the following categories:

- Staging tables (`staging_*.sql`)
- Dimension tables (`dim_*.sql`)
- Fact tables (`fact_*.sql`)

## Example

A typical SQL file should have the following format:

```sql
-- Example: staging_customer.sql
CREATE OR REPLACE TABLE STAGING_CUSTOMER (
  CUSTOMERID           VARCHAR,
  SUBSEGMENTID         INTEGER,
  FIRSTNAME            VARCHAR,
  LASTNAME             VARCHAR,
  GENDER               VARCHAR,
  -- ... other columns ...
  MODIFIEDBY           VARCHAR
);
```

## Usage

These SQL files are loaded at runtime by the ETL scripts. When you clone this repository, you will need to:

1. Create your own table definitions in this directory
2. Follow the naming conventions above
3. Ensure your SQL syntax is compatible with Snowflake

This directory is excluded from Git, so your specific table definitions won't be committed to the repository. 