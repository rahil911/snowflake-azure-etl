# ETL Demo Script

This demo script provides an end-to-end demonstration of the Snowflake ETL process. It shows how to initialize a project, run the ETL pipeline, and verify the results.

## What the Demo Does

1. **Gathers Credentials**: Prompts for Snowflake credentials (account, username, password, etc.)
2. **Creates Environment**: Sets up a temporary environment with the necessary configuration
3. **Initializes Project**: Uses the CLI scaffolder to create project structure
4. **Copies Sample Schemas**: Copies SQL schema files to the temporary environment
5. **Runs ETL Process**: Executes the complete ETL pipeline
6. **Shows Results**: Displays row counts for all tables loaded in Snowflake

## Running the Demo

To run the demo script:

```bash
python scripts/demo_etl.py
```

Or, if you've made it executable:

```bash
./scripts/demo_etl.py
```

## Requirements

- Python 3.9 or higher
- Snowflake account with appropriate permissions
- snowflake-connector-python installed (`pip install snowflake-connector-python`)
- All dependencies installed from requirements.txt

## Important Notes

- The demo creates a temporary directory for all files
- All credentials are only stored temporarily and deleted when the demo completes
- The script requires internet access to connect to Snowflake
- The demo uses schema evolution by default

## Example Output

```
================================================================================
SNOWFLAKE ETL DEMO SCRIPT
================================================================================

This script demonstrates the end-to-end ETL process.
It will create a temporary environment and run the ETL pipeline.

================================================================================
STEP 1: GATHER CREDENTIALS
================================================================================
To run the demo, we need your Snowflake credentials.
These will only be stored temporarily and will be deleted when the demo completes.
Snowflake account [your-account.snowflakecomputing.com]: my-account
Snowflake username [johndoe]: 
Snowflake password: 
Snowflake warehouse [COMPUTE_WH]: 
Snowflake role [ACCOUNTADMIN]: 
Azure Storage account [youraccountname.blob.core.windows.net]: myazure.blob.core.windows.net
Your name (for database naming) [johndoe]: 

Created temporary directory: /tmp/tmp_abcdef123

================================================================================
STEP 2: CREATE ENVIRONMENT FILE
================================================================================
Created temporary .env file at /tmp/tmp_abcdef123/.env

...

Database: IMT577_DW_JOHNDOE_STAGING
Tables: 12

Row counts:
--------------------------------------------------
Table Name                       Row Count
--------------------------------------------------
STAGING_CHANNEL                         10
STAGING_CHANNELCATEGORY                  3
STAGING_CUSTOMER                      1000
...

================================================================================
DEMO COMPLETED
================================================================================

Temporary directory will be deleted: /tmp/tmp_abcdef123
```

## Customization

You can modify this script to add additional steps or customize the demo process. Some ideas:
- Add data visualization of the loaded data
- Include performance metrics for each ETL step
- Generate a report with sample data from each table 