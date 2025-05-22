Got it. I’ll now put together a detailed, granular roadmap for refactoring your Snowflake ETL project into a fully reusable, user-friendly tool. It will use schema inference with advanced methods like ML and GenAI, support automation, and abstract away Snowflake complexities for non-technical users.

I'll create a very detailed list of 1-point AI-executable tasks in markdown format, outlining each step from your current code to the target architecture.


# Refactoring Snowflake-Azure ETL for Reusability and Automation

## Objectives and Overview

This refactoring project transforms the existing **Snowflake ETL Pipeline for Azure Blob Storage** into a **highly reusable, automated Snowflake ETL tool** suitable for non-technical users. The new tool will eliminate hard-coded schemas and Snowflake-specific details, replacing them with intelligent automation and modular design. Key goals include:

* **Automatic Schema Inference:** Leverage advanced techniques (statistical profiling, ML algorithms, and even generative AI) to accurately infer table schemas from incoming data with minimal human input.
* **Snowflake Configuration Abstraction:** Hide all Snowflake-specific setup (e.g. warehouse, roles, DDL details) behind the scenes. Users only provide high-level inputs (like data location and credentials), and the tool handles staging, DDL generation, and loading.
* **Dynamic Schema Generation:** Remove all fixed `CREATE TABLE` SQL definitions from code. Instead, the tool will dynamically generate Snowflake table schemas based on the inferred data structure. This can utilize Snowflake’s **INFER\_SCHEMA** capability for CSV/JSON, or custom inference logic for other formats.
* **Modular Architecture:** Separate the code into distinct, testable modules – for schema inference, data loading, transformations, etc. – so each part can be developed and maintained independently.
* **Automation & Ease of Use:** Enable one-click or scheduled pipeline runs with minimal user input. The user should simply point the tool at the data source and provide credentials, then the pipeline auto-runs end-to-end (from schema detection to load to validation).
* **Educational Safeguards:** Implement safeguards to avoid exposing any hard-coded assignment solutions from *IMT 577* (per the professor’s guidance). This includes removing course-specific constants (like the `IMT577` naming) and ensuring no sensitive solution logic is made public.

By achieving these objectives, the refactored tool will empower even non-technical users to ingest data into Snowflake seamlessly, while providing data engineers with a robust, extensible framework.

Task Breakdown is given in TASKS.md ->
You have to finish it all, and test it with dummy data creation in realtime... With all the testing capabilities and functions and features that we have, we make a dummy data and test all the functions and machine learning and everything that we have done. As you don't have internet, you will not be able to test it with real Azure. 


## Architecture and Module Design

The new ETL tool will follow a modular architecture, separating concerns to improve reusability and testability. Below is an outline of the proposed architecture, including the **folder structure**, **module responsibilities**, and how they interact:

```plaintext
snowflake_etl_tool/         # Top-level package for the ETL tool
├── __init__.py
├── config.py               # Loads YAML/ENV configuration into Python objects
├── connection.py           # Snowflake connection management (abstracts login details)
├── schema_inference.py     # SchemaInferer class or functions for data profiling & type detection
├── ddl_generator.py        # Functions to generate DDL SQL from inferred schema (or call Snowflake INFER_SCHEMA)
├── data_loader.py          # Functions to create stages and load data via COPY commands
├── transformer.py          # (Optional) Data transformation logic after loading, if any
├── pipeline_runner.py      # Coordinates the above modules to run the full pipeline
└── utils/                  # Utility functions (e.g., logging setup, common helpers)
    └── logger.py           # Configures logging format and output destinations

tests/                      # Test suite for all modules
├── test_schema_inference.py
├── test_ddl_generator.py
├── test_data_loader.py
├── test_pipeline_runner.py
└── ... (etc.)

examples/
├── example_config.yaml     # Example user configuration file
└── sample_data/            # (If allowed) small sample input files for trying out the tool

docs/                       # Documentation files (if separated from README)
└── ARCHITECTURE.md         # In-depth architecture notes for developers (optional external doc)
```

**Module Responsibilities and Interactions:**

* **Configuration Management (`config.py`):** This module is responsible for reading configuration from a YAML/JSON file and environment variables. It will likely use a library like `pyyaml` to parse YAML. The config defines things like list of data sources (with their paths, formats), target Snowflake database & schema (if not using defaults), and any user preferences (e.g. whether to enable ML/GenAI assistance, how much of the file to sample for inference, etc.). By centralizing configuration here, the rest of the code can remain Snowflake-agnostic – e.g., the user doesn’t need to modify code to change the warehouse or toggle a feature, they just edit the config. **All Snowflake-specific settings (account, warehouse, etc.) are abstracted into this config**, fulfilling the goal of not making the user deal with those in the code.

* **Connection Management (`connection.py`):** Handles connecting to Snowflake using the Python connector (or Snowpark). It reads credentials from config/env and creates a connection object. This module abstracts the details of Snowflake’s connector – the rest of the tool can call a generic method like `conn = get_snowflake_connection()` and not worry about the details. It can also include helper methods for executing queries and fetching results, so that other modules (like `ddl_generator` or `data_loader`) don’t directly call the connector’s API. For example, a method `execute_sql(query)` can wrap error handling and logging around the low-level calls. This abstraction not only simplifies other modules but also makes it easier to mock in tests (we can create a dummy connection object that simulates Snowflake responses).

* **Schema Inference (`schema_inference.py`):** This is a central piece of the new tool. It contains the logic to inspect data sources and deduce their structure. Likely a class `SchemaInferer` with methods like `infer_from_csv(file_path)` (and similar for JSON, etc.), or a universal `infer_schema(data_source)` that auto-detects format. Under the hood, it will implement the advanced techniques discussed: reading samples of the file, analyzing value patterns, and deciding data types. For large files, it might read in streaming mode or spawn parallel jobs for different chunks to speed up processing. This module might use pandas or Python’s CSV library for parsing, and incorporate rules or even an ML model for classification of column types (for example, a small decision tree or a pre-trained scikit-learn model that can classify a string of text as “likely date” vs “likely free text” based on regex features). We may also integrate the **`csv-schema-inference`** library or similar as a backend for robust type detection logic. The output of this module is an **abstract schema** – e.g. a Python list of `ColumnSchema` objects or a dict mapping column names to types and nullability. This abstraction layer ensures that our next module (DDL generation) doesn’t need to know how we arrived at the schema; it just consumes the result.

* **DDL Generation (`ddl_generator.py`):** This module takes the abstract schema from `schema_inference` and produces actual Snowflake SQL commands. It fulfills the requirement of **removing hardcoded schemas from code** – instead of static SQL strings, we programmatically build them. There are two possible modes here:

  1. **Python-Generated DDL:** The module iterates over the inferred columns and constructs a `CREATE TABLE` statement string (e.g., `CREATE TABLE myTable (col1 NUMBER, col2 VARCHAR(100), col3 DATE, ...);`). It will map our internal type labels (like “INTEGER”, “FLOAT”, “BOOLEAN”) to Snowflake SQL types (`NUMBER`, `FLOAT`, `BOOLEAN`, `VARCHAR`, etc.) appropriately. We will also include any decisions about lengths/precision (for example, defaulting to `VARCHAR(16777216)` or `TEXT` for string if length uncertain, or setting a specific length if we have one from profiling). This is also where we decide on the table name: it could be derived from the file name or a config entry.
  2. **Snowflake-Template DDL:** Alternatively, utilize Snowflake’s **USING TEMPLATE with INFER\_SCHEMA** feature to let Snowflake create the table for us. For example, the module can execute:

     ```sql
     CREATE OR REPLACE TABLE <table_name>
     USING TEMPLATE (
       SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
       FROM TABLE(
         INFER_SCHEMA(
           LOCATION=>'<stage_path>/<file>', 
           FILE_FORMAT=>'<format_name>'
         )
       )
     );
     ```

     This single command will query the file on the stage and create the table with the inferred schema. It’s very powerful and simplifies our job – essentially Snowflake itself is doing schema inference here. We’d use this especially if the file is already in a Snowflake stage. Since our pipeline will be copying from Azure Blob, we can either use *external tables* or stage the file temporarily. A strategy: use `CREATE STAGE` to point to the Azure container (which we do in data\_loader), then call `INFER_SCHEMA(LOCATION=>@stage/path)` on that.

     Both approaches have merit; we might even implement both and allow a config toggle (e.g. “use\_internal\_infer: true/false”). If using Snowflake’s inference, the `schema_inference.py` module might still parse data for validation but can rely on Snowflake’s result.

  Regardless of approach, `ddl_generator.py` ensures the **schema is created in Snowflake dynamically**. After running, the new table exists with columns ready for data. This module can also handle things like uppercasing or sanitizing column names (Snowflake tends to uppercase unquoted identifiers, and `INFER_SCHEMA` will return uppercase by default). We will ensure consistency (maybe choose to standardize on uppercase or lowercase and quote identifiers accordingly).

* **Data Loading (`data_loader.py`):** This module is responsible for moving the data from Azure into the newly created Snowflake tables. It will contain:

  * A function to **create external stages** in Snowflake. Given an Azure Blob container or file path and an associated Snowflake **storage integration** (likely configured manually beforehand for secure access), it executes `CREATE STAGE` pointing to that external location. In the original code, `create_stages.py` created 12 named stages. In the refactored version, we can do this more flexibly. For example, we might create one stage per data source *on the fly* and even immediately use it, then drop it if not needed persistently. Or create a single stage for the whole container and just use path filtering in COPY commands. For simplicity, perhaps maintain a mapping in memory: {data\_source\_name: stage\_name}.
  * A function to **load data via COPY**. For each table, call `COPY INTO <table> FROM <stage>/<path> FILE_FORMAT=(type=..., field_optionally_enclosed_by='"', etc.)` with appropriate options. We will definitely include `MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE` for CSV with header, so that columns load correctly even if order changes. If the data is JSON or Parquet, Snowflake can load it directly using the inferred schema (Parquet would already have schema, JSON might need SELECT with `$1:field` projections unless using infer). The loader should capture how many rows were loaded (the Snowflake connector can return row count or we can run `SELECT COUNT(*)` after load). It should also handle partial failures – e.g., if one file in a batch fails, log it and continue to next.
  * **Post-load Verification:** After loading each table, the module can fetch the top 5 rows (like the original `view_sample_data.py` did) and the total count, and log or print them. This provides immediate feedback that data landed correctly. Since the tool is for non-technical users, seeing a small preview of their data in Snowflake is reassuring. We can use the `tabulate` library (already in requirements) to format this output nicely.
  * **Schema Evolution Handling:** As noted, if COPY fails due to a data mismatch, this module should catch that exception (the Snowflake Python connector will throw an Error if the COPY command fails). Then it can analyze the error message. For example, Snowflake might say “Numeric value ‘ABC’ is not recognized” which means a supposedly numeric column got a non-numeric value. The tool can interpret this and decide to alter that column to TEXT/VARCHAR and retry. Or if it says column not found, it means the data file had an extra column – so we could add that column (perhaps using a new inference run on just that column’s data). This kind of **auto-validation and self-healing** makes the pipeline very robust. Snowflake’s own **schema evolution** feature can be enabled on COPY by `ALTER TABLE ... ADD COLUMN if new` setting, but since it’s new we might prefer handling it in Python for transparency.
  * This module will be heavily integrated with `connection.py` (to execute the SQL) and with `schema_inference.py` (for reacting to schema mismatches). It forms the **ETL “Load” step** primarily.

* **Transformation (`transformer.py`):** In a simple staging pipeline, this might not do much initially. However, to meet the modular design goal, we allocate this module for any transformations after data load. For example, if the end goal is to populate a star schema data warehouse, this module could contain SQL or logic to move data from staging tables into final dimension and fact tables, or to apply data quality rules. In our design, we ensure this is separate so that it can be swapped or extended. If the course or user wants to add a transformation (say, a derived column or a lookup from one table to enrich another), they can implement it here or add a new module under a similar interface. The pipeline will call something like `transformer.apply_all(conn)` after loading. If no transformations are configured, that function can simply log “No transformations to apply” and exit gracefully.

* **Pipeline Runner (`pipeline_runner.py`):** This is the orchestrator that glues everything together in the correct order. When the user runs the tool (via CLI or directly), this runner is invoked. It will:

  1. Initialize logging and read the configuration (from `config.py`).
  2. Establish the Snowflake connection (`connection.py`).
  3. Possibly create the target database & schema if not exist (calling a helper that executes `CREATE DATABASE IF NOT EXISTS ...`). Also create a Snowflake file format if needed (for CSV, define one global file format with `FIELD_OPTIONALLY_ENCLOSED_BY='"'`, `NULL_IF=''`, etc., so that we can reuse it for all stages).
  4. For each data source defined in config (or discovered in a container):

     * Call `schema_inference.infer_schema(source)` to get the schema.
     * Call `ddl_generator.create_table(schema, table_name)` to make the table in Snowflake.
     * Call `data_loader.create_stage_if_needed(source)` to ensure a stage is available for the file.
     * Call `data_loader.copy_into_table(table_name, source)` to load the data.
     * Call `data_loader.verify_load(table_name)` to log sample rows and row count.
  5. After all sources are loaded, call `transformer.apply_all()` for any post-load steps.
  6. Close the Snowflake connection and wrap up.

  The pipeline runner should handle exceptions at each step: for example, if schema inference fails for one file (perhaps an unrecognized format), log an error and skip that file rather than crash the whole pipeline. Similarly, if table creation fails (unlikely after inference, but e.g. if user lacks privileges), report it clearly. This robustness is key for automation – it should ideally never simply crash without explanation.

  The runner also implements the **automatic execution** logic: a user can schedule this runner or integrate it with cloud workflows. For instance, this module could be invoked by an Azure Function when a new blob arrives (if extended), or by Airflow/Dagster as a task. The design allows that because all crucial operations are encapsulated in functions that can be called independently as well.

* **Utility & Logging (`utils/logger.py`):** A small utility to configure logging levels, formats, and output files. For instance, we might set up a rotating file logger in `logs/etl.log` plus console output. Ensuring this is called early (in runner) so all modules use a consistent logger. This could also include any shared helper functions (e.g., a function to parse date strings in various formats, if used by inference logic, could live here to keep `schema_inference.py` focused on orchestration of type checks).

This modular architecture ensures **independent testability**: each piece can be tested with mocks for the others. For example, `schema_inference.py` can be tested without a real Snowflake connection by feeding it local data files and seeing if it produces the correct schema object. Similarly, `ddl_generator.py` can be tested by injecting a fake schema and verifying the SQL output (or that it calls the connector with the right command, if we design it to call connection methods directly). The pipeline runner logic can be tested with everything mocked (to simulate an end-to-end run quickly), and a full integration test can be run on a real Snowflake instance with a small dataset to validate the entire flow.

**File/Folder Structure Changes:** The structure outlined above differs from the original monolithic approach. Notably:

* We eliminate the separate scripts `create_database.py`, `create_stages.py`, `create_tables.py`, `load_data.py`, etc., in favor of cohesive modules. The original design separated by steps, which was good for the assignment, but our new design separates by function and reusability. For example, the original `create_tables.py` had all DDL statements; now `ddl_generator.py` handles that dynamically for any table.
* The `rahil/run_etl.py` script becomes `pipeline_runner.py` (and possibly a CLI entry point). Instead of a static sequence tied to 12 entities, it reads from config, so it’s generalized to N entities.
* We introduce a `tests/` directory to hold all our test code (this was not present before). This signals a production-grade project where testing is integrated.
* We keep a `logs/` directory for logs, as originally, but ensure the log naming is generic (not tied to `etl_run_<timestamp>` specifically, though that pattern is fine to continue). Possibly, each run could log to a timestamped file as before – that was a good feature to keep.
* We may remove the `DIMENSION_README.md` or other course-specific files if they exist, or move them to `docs/` if they are relevant for educational context but not for general use.

**Advanced Techniques Integration:** Within this architecture, the advanced techniques can be highlighted:

* The **Schema Inference module** is where ML/AI comes into play. We could incorporate a library or custom code that uses machine learning to improve type detection. For example, imagine a column full of values like “CA”, “NY”, “WA” – our basic inference might mark it as TEXT of length 2. An ML model or a lookup could recognize these as US state codes and maybe tag the column as a categorical feature. While Snowflake has no special type for that, in the future the tool could suggest creating a states dimension table, etc. We won’t go that far now, but it shows how an AI could add semantic understanding. Another case: free-form text vs categorical string – statistically, if a string column has few distinct values relative to rows, it might be categorical (maybe an ENUM type in other DBs). We can note such patterns. If generative AI is used, it might generate a human-readable description of the column (Snowflake’s new `GENERATE_COLUMN_DESCRIPTION` function is relevant – it uses an AI model to suggest descriptions for each column based on its name and data). Our tool could call that and store the descriptions (either logging them or even applying them as Snowflake column comments). This would greatly enhance the understandability of the warehouse.

* **YAML Templates** in this design act as an abstraction layer. Users can interact with YAML without touching Python code. For example, if a user wants to enforce a schema, they could provide a YAML like:

  ```yaml
  tables:
    - name: customers
      columns:
        - name: id
          type: INTEGER
          nullable: false
        - name: name
          type: VARCHAR(100)
          nullable: true
        # ... etc.
  ```

  The tool could read this and skip inference for that file, using the provided schema instead. Or the tool could output a similar YAML after inferring, as a record of what was decided. This approach aligns with practices in modern ETL tools where pipeline schemas and configs are declarative (for example, Airbyte and Singer taps often use JSON/YAML schema definitions).

* **Independently Replaceable Modules:** Because of this design, if tomorrow we wanted to switch out how schema inference works (say, use a different library or a cloud AI service), we can do so without affecting the rest. The interface would be the same (`infer_schema()` returns a schema object). The same goes for Snowflake connection – if we wanted to use the Snowpark API instead of the Snowflake connector for some reason, we could swap out `connection.py` and maybe slight adjustments in others, but the overall pipeline code stays intact. This is beneficial for long-term maintenance.

In summary, the architecture is **layered**:

1. **Config Layer** (user input and configuration),
2. **Inference & Generation Layer** (smart logic to figure out what to do),
3. **Execution Layer** (actually doing it in Snowflake and Azure), and
4. **Orchestration Layer** (coordinating the steps and handling flow control).

Each layer is modular. This makes the tool not only easier to test and extend, but also easier to explain. We can now proceed to how the tool operates under the hood in a typical run.

## How the Tool Works (Under the Hood)

To illustrate the inner workings, let's walk through the pipeline execution flow with the new system. This assumes a user has configured the tool and invoked a run (either via CLI or a scheduler):

1. **Initialization:** The pipeline runner starts, loading configurations and connecting to Snowflake. For example, if the user’s config specifies an Azure container and a list of files, the tool reads that into a list of `DataSource` objects (each might have attributes like `name`, `path`, `format`). It also reads Snowflake settings (account, DB, etc.) and uses `connection.py` to establish a connection. Logging is started at this point, so all subsequent steps produce log entries (both to console and `logs/` file).

2. **Target Environment Prep:** Before processing data, the tool ensures the Snowflake environment is ready. It checks if the specified **database** exists (by querying Snowflake). If not, it creates it (executing a `CREATE DATABASE` statement). Similarly for the **schema** and possibly a dedicated **Snowflake user/role** if part of config. All this is transparent to the user – they don’t need to manually set up anything in Snowflake. The connection context is set to use this database, schema, and a warehouse (which might be a default like `COMPUTE_WH` unless overridden).

3. **Loop Through Data Sources:** Now the core loop begins – for each data source (which could correspond to what was previously an “entity” like channel, customer, etc., but now it’s generic):

   * The tool logs which source it’s processing (e.g. “Processing file `customer_data.csv` ...”).
   * **Schema Inference:** It calls the Schema Inference module with the source info. Under the hood, this module might: open the file (if it’s a CSV on Azure, possibly stream it directly or use Azure SDK to fetch a sample), read the header to get column names, then sample the data. Suppose `customer_data.csv` has columns `ID, Name, Age, SignupDate`. The inference module sees header names and for each column inspects sample values:

     * **ID**: values look like integers “1001, 1002, 1003,...”. The module attempts to parse as int for each sampled value, finds all are valid integers, no decimals, etc. It will infer type = INTEGER, nullable = False (assuming no missing in sample).
     * **Name**: values are strings (e.g. “Alice”, “Bob”, maybe some longer names). They obviously can’t be numeric or date. The module infers type = TEXT/VARCHAR, and maybe determines a max length (say longest name in sample is 20 chars, it could choose VARCHAR(50) to be safe or just TEXT). Nullable might be True if any blank names found, otherwise False.
     * **Age**: values might be “34, 45, 29,...”. Similar to ID, but if there's any non-integer (like someone wrote “N/A”), the algorithm might catch that as a string outlier. If 99% are numeric and one is “N/A”, we might still choose INTEGER but mark nullable True (and treat “N/A” as null on load). Alternatively, convert “N/A” to null and still int. The tool’s strategy is configurable here – perhaps define in config that certain tokens represent null. In any case, type likely INTEGER, nullable possibly True due to the potential placeholder.
     * **SignupDate**: values like “2021-05-01”, “2021-08-15”, etc. The module will try date parsing on these strings. If they all parse successfully (and match a consistent format), it infers DATE. If some have time component like “2021-05-01 14:30:00”, then DATETIME/TIMESTAMP. If it fails to parse some, it might default to TEXT. But let’s assume a clean date column – type = DATE, nullable depending on blanks.
       After analysis, the SchemaInferer produces a schema representation, e.g. a list: `[("ID", "NUMBER", False), ("Name","TEXT",False), ("Age","NUMBER",False), ("SignupDate","DATE",False)]`. It may choose Snowflake-friendly types (`NUMBER` for integer, `TEXT` for string, etc.). It returns this to the pipeline runner. The runner logs “Inferred schema: ID (NUMBER), Name (TEXT), Age (NUMBER), SignupDate (DATE)”. If **GenAI** assistance is enabled, at this point the runner might call a function that uses an LLM to verify these choices. For instance, the LLM might confirm “ID should be integer, Name is text, etc.” – essentially double-checking. Or generate a neat comment like “SignupDate appears to be a date in YYYY-MM-DD format.” These could be stored for documentation but are optional.
   * **Table Creation:** Next, the pipeline runner calls the DDL generation to create a table for this data. Let’s call the table `CUSTOMER_DATA` (often based on file name, but configurable). The DDL module formats a CREATE TABLE SQL. Using our example, it might produce:

     ```sql
     CREATE OR REPLACE TABLE CUSTOMER_DATA (
       ID NUMBER(38,0) NOT NULL,
       Name VARCHAR(100),
       Age NUMBER(38,0),
       SignupDate DATE
     );
     ```

     It executes this on Snowflake via the connection. Because we removed all the hardcoded schemas, this creation is entirely dynamic. If instead we leverage Snowflake’s `USING TEMPLATE`: the runner could skip directly to:

     ```sql
     CREATE OR REPLACE TABLE CUSTOMER_DATA 
     USING TEMPLATE (
       SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
       FROM TABLE(INFER_SCHEMA(
         LOCATION=> '@AzureStage/path/to/customer_data.csv',
         FILE_FORMAT=> 'my_csv_format'
       ))
     );
     ```

     In this scenario, Snowflake reads the file and creates the table. The result should be similar, but one difference: Snowflake might create all text columns as VARCHAR if it can’t deduce numeric reliably. We trust Snowflake’s inference in straightforward cases (their algorithm likely also infers ints, dates, etc. now that CSV infer is GA). Either way, after this step the table exists. The runner logs “Created table CUSTOMER\_DATA with inferred schema.”
   * **Data Loading:** The runner now calls the loader to actually ingest the data. First, the loader ensures there’s an **external stage** for the file. Possibly earlier we created one stage for all files (e.g. `@AzureStage` pointing to the container). If not, it does it now: `CREATE STAGE AzureStage URL='azure://<account>.blob.core.windows.net/<container>' ...` using credentials from .env. Since the original project had 12 separate stages, we simplify by one stage plus path, or a stage per file if needed. Let’s assume one stage covering the whole container (the file path can then be specified in COPY).
     Now the loader executes `COPY INTO CUSTOMER_DATA FROM '@AzureStage/customer_data.csv' FILE_FORMAT=(FORMAT_NAME='my_csv_format') MATCH_BY_COLUMN_NAME=CASE_INSENSITIVE;`. The `FILE_FORMAT` “my\_csv\_format” would have been created (either in Snowflake or the tool) to specify delimiter, etc., presumably it’s a standard CSV with header. The `MATCH_BY_COLUMN_NAME` ensures that if the CSV columns are out of order or some are missing, Snowflake matches by name (and fills nulls for missing, etc.). The copy command reads the file and inserts data into the table. The loader captures how many rows were loaded (the Snowflake connector returns this or we can do `SELECT COUNT(*)` from the table after). It logs “Loaded 10,000 rows into CUSTOMER\_DATA.” If there's any **error** during copy, the loader will catch it:

     * If Snowflake’s `COPY` reports conversion errors (e.g., “Can’t parse 'N/A' as NUMBER for column AGE”), our logic kicks in. The loader will log a warning like “Data type mismatch on column AGE, attempting to adjust schema.” It then could do one of two things: (a) Alter the `AGE` column to TEXT (broadest type) and retry, or (b) treat “N/A” as null and retry the load with an instruction to skip conversion errors (`ON_ERROR` option in COPY). A smart approach is (b) if such errors are rare, or (a) if we anticipate more non-numeric data. The choice can be guided by a config flag like `strict_types: false` meaning be flexible. Let’s say we choose to alter schema for maximum integrity. The tool would run `ALTER TABLE CUSTOMER_DATA ALTER COLUMN AGE TYPE TEXT;` (or VARCHAR) and then redo the COPY. This time it succeeds. It would then note “Column AGE changed to TEXT due to incompatible data”. This is schema evolution in action. Snowflake actually has `COPY ... ON_ERROR=CONTINUE` which could load bad rows to a separate table, but for simplicity we handle it by schema change, ensuring *all data* gets loaded.
     * Another possible error: new columns in data that were not in schema. This could happen if a new CSV version adds a column “Status” for instance. Snowflake’s COPY would ignore extra data if `TRIM_SPACE` or similar, but if `MATCH_BY_COLUMN_NAME` is on, it might just not find a match and ignore that column’s data (or error if not handled). To catch this, we might proactively compare the file header to our table columns. If an extra column “Status” is found, the loader can log “New column 'Status' detected in data, adding to table schema.” It then does an `ALTER TABLE ADD COLUMN Status VARCHAR` (since we don’t know type, default to TEXT, or run inference on that column’s values separately). Then rerun COPY. This way, the pipeline doesn’t break when upstream adds a column – it auto-adapts, which is a big win for maintenance.
   * **Verification & Sample Output:** After successful load, the tool runs a quick verification. It will do a `SELECT COUNT(*)` and `SELECT * LIMIT 5` on `CUSTOMER_DATA`. Using `tabulate`, it prints a small table of the first 5 rows and prints "Total rows: X". This mirrors what the original `view_sample_data.py` did (displaying top 5 rows for verification). For non-technical users, this is the moment they see their data in Snowflake, confirming the pipeline worked. If any issues are visible (like a numeric column loaded as all NULLs due to an error), the user can notice it here. The tool also logs this info.
   * That completes one data source. The pipeline then moves to the next file/dataset and repeats (Inference -> Create table -> Copy -> Verify). This continues until all sources are processed.

4. **Post-Load Transformations:** Once all raw data files are loaded into staging tables, the pipeline can execute any transformation steps. Perhaps in the config, the user specified an SQL script to run that joins some of these tables or computes summary metrics. The `transformer.py` would handle this. For example, if the user wants to create a consolidated sales table from `salesheader` and `salesdetail` (from the original 577 project context), the transform module could run an SQL that selects from those staging tables into a new table (maybe applying some business logic). Since the question specifically asks to modularize transformations, we ensure this step exists, but it might be a no-op by default if no transforms are configured. Regardless, the architecture supports plugging in transforms without affecting earlier steps.

5. **Completion & Logging:** The pipeline runner now wraps up. It logs a summary: e.g. “5 tables created, 5 files loaded successfully, 0 errors. See logs/etl\_run\_20250521\_171500.log for details.” Then it closes the Snowflake connection and ends. The log file contains the full trace of what happened (including all SQL statements executed or at least the high-level actions). This is invaluable for debugging and for audit purposes. Non-technical users might not read the raw log, but an engineer can use it to trace any issues.

Throughout this process, the user did not have to intervene or provide any input beyond the initial configuration. The tool made decisions on their behalf using intelligent defaults and advanced logic:

* It *inferred schemas* accurately (relying on stats and ML rules to minimize mistakes). Snowflake’s own documentation emphasizes how automatic schema detection saves manual effort and reduces errors. We’ve implemented that philosophy here.
* It *handled Snowflake details* (like creating DB, defining file formats, choosing copy options) behind the scenes. The user doesn’t need to know Snowflake SQL at all.
* It even *handled unexpected data changes* (schema evolution) so that the pipeline would rarely require manual fixes if the input format changes over time.
* If generative AI was enabled, under the hood it provided an extra layer of validation (for instance, an LLM could have warned us that “Age” had an “N/A” which is text, reinforcing our detection to mark it nullable or text). Such guidance can improve accuracy, aligning with the goal of using GenAI for high accuracy in schema inference.

By understanding this under-the-hood flow, data engineers can trust that the system is doing the right steps in the right order. It’s essentially an automated data engineer that inspects data, writes DDL, loads files, and verifies outcomes – all tasks that would traditionally be done manually. The modular design means each of these steps is implemented in its own component, making it easier to maintain or upgrade in the future.

## End-User Onboarding and Usage

One of the primary aims is to make this tool accessible to non-technical users, meaning the setup and usage should be as straightforward as possible. This section provides a simple guide for end users and highlights the minimal instruction needed to run the new ETL tool.

### Getting Started Guide (for End Users)

**1. Installation:** Obtain the ETL tool by cloning the GitHub repository or installing via pip (if we package it). For example:

```bash
git clone https://github.com/your-org/snowflake-etl-tool.git  
cd snowflake-etl-tool  
pip install -r requirements.txt  
```

*(If released on PyPI, it would be `pip install snowflake-etl-tool`.)* The user does not need to install Snowflake or any database client separately; the required Python connector is included in requirements.

**2. Configuration:** Locate the `example.env` and `example_config.yaml` files provided. The user should copy these to create their own config:

```bash
cp example.env .env  
cp examples/example_config.yaml my_config.yaml  
```

Now, open `.env` in a text editor and fill in your Snowflake and Azure credentials:

```ini
# Snowflake credentials  
SNOWFLAKE_ACCOUNT=xyz12345.us-west-2  
SNOWFLAKE_USER=your.username  
SNOWFLAKE_PASSWORD=YourPasswordHere  
SNOWFLAKE_WAREHOUSE=COMPUTE_WH  
SNOWFLAKE_ROLE=ACCOUNTADMIN   # or a role with rights to create table & load data  
SNOWFLAKE_SCHEMA=PUBLIC       # target schema name (will be created if not exists)  

# Azure Blob Storage  
AZURE_STORAGE_ACCOUNT=yourstorageaccount.blob.core.windows.net  
AZURE_STORAGE_KEY=*** (if needed, or if using SAS/managed identity, provide those details)  
```

*(We ensure in docs to explain where to get these values: e.g. Snowflake account name from URL, etc.)*

Next, edit `my_config.yaml` to specify what data to load. For example:

```yaml
data_sources:
  - name: customer_data  
    path: customer_data.csv  
    file_format: csv  
  - name: sales_data  
    path: sales_2025-*.csv  
    file_format: csv  
target_database: AUTO_ETL_DB   # Snowflake database to use/create
target_schema: PUBLIC         # Snowflake schema to use/create
use_snowflake_infer: true     # whether to leverage Snowflake INFER_SCHEMA
```

In this example, we indicated two data sources. The second uses a wildcard pattern – the tool might support loading multiple files (e.g. all 2025 sales files) into one table. The config format will be documented clearly so users know how to list their files. Essentially, the user just needs to tell the tool “what to load” in a simple format.

**3. Running the Pipeline:** Once configured, the user runs the ETL pipeline. This can be as easy as:

```bash
python -m snowflake_etl_tool.pipeline_runner --config my_config.yaml  
```

If no config flag is given, the tool will default to a `config.yaml` or similar. The tool will begin execution, printing progress to the console. The user will see messages like:

```
Connecting to Snowflake...  
Creating database AUTO_ETL_DB (if not exists)...  
Processing data source: customer_data.csv  
- Inferring schema for customer_data.csv... [OK]  
- Creating table CUSTOMER_DATA... [OK]  
- Loading data into CUSTOMER_DATA... [OK] (10000 rows)  
- Sample rows from CUSTOMER_DATA: ... (it will display a small table)  
Processing data source: sales_2025-*.csv  
- Inferring schema for sales_2025-*.csv... [OK]  
- Creating table SALES_DATA... [OK]  
- Loading data into SALES_DATA... [OK] (120000 rows)  
- Sample rows from SALES_DATA: ...  
All data sources processed successfully.  
```

If the user provided an incorrect config or credentials, the tool will print a clear error (e.g. “Failed to connect to Snowflake: invalid credentials” or “File not found: sales\_2025-\*.csv”). The messaging is designed to be user-friendly. Non-technical users can understand what’s happening without diving into code.

**4. Verifying Results:** After the run, the data should be in Snowflake. The user (or their data analyst) can connect to Snowflake and query the new tables. We’ll have named the tables based on the provided names (e.g., `CUSTOMER_DATA`, `SALES_DATA` in the example). The log also shows a few sample rows as verification, so often that’s enough for the user to trust the data is loaded. Additionally, the log file in `logs/` contains all details. If something minor went wrong (like a column was auto-added), the log will mention it (e.g., “Note: Added missing column 'Status' to SALES\_DATA schema”). This transparency helps users trust the automation.

**5. Minimal Maintenance:** Since the tool infers schema each time, adding a new data file or a new column doesn’t require the user to update any code. For instance, if next month the user has a new `product_data.csv`, they just add an entry in `my_config.yaml` and re-run the pipeline. The tool will create a new table for it automatically. If a CSV’s structure changes (columns added/removed), the tool will adapt the Snowflake table accordingly. This means non-technical staff can continuously load evolving datasets without needing a developer to refactor pipelines each time. It’s largely “configure once, use many times.”

**6. Scheduling (Optional):** If the user wants this to run daily or whenever new files arrive, they have options. A simple approach: use a scheduled task or cron job to invoke the pipeline runner daily. Because the tool is idempotent (it can overwrite tables or load new data incrementally depending on config), running it regularly is fine. We will document how to set it up with a scheduler. For more advanced integration (like event-based triggers when a file is uploaded to Azure), it might require a bit of tech setup (Azure Event Grid + a trigger to run our script), which could be a future guide.

### Example Configuration Files

To further clarify usage, the repository will include example configuration files. One such example might be:

**`example_config.yaml`:**

```yaml
# Example configuration for the Snowflake ETL Tool
target_database: AUTO_ETL_DB              # Snowflake database name for staging tables
target_schema: PUBLIC                     # Snowflake schema name for staging tables
use_snowflake_infer: true                 # Use Snowflake's INFER_SCHEMA for DDL creation
file_format_options:                      # Options for the file format (CSV in this case)
  type: csv
  field_optionally_enclosed_by: '"'
  skip_header: 1
  null_if: ['\\N', 'NULL', '']            # treat these as null
data_sources:
  - name: customers
    path: data/customers.csv              # If relative path, the tool might assume it's in Azure container root
    file_format: csv
  - name: transactions
    path: data/transactions/*.parquet     # Support wildcard and different format
    file_format: parquet
    target_table: TXN_STAGE               # Optionally specify a custom table name
  - name: raw_json_events
    path: events/2025/05/21/**/*.json     # Maybe multiple JSON files in nested dirs
    file_format: json
    infer_schema_samples: 1000            # Override default sample size for inference for this source
```

This YAML demonstrates flexibility: we can handle CSV, Parquet, JSON differently, even use wildcards. The tool will interpret this and act accordingly. For Parquet, it might not need to infer much because Parquet has schema embedded; for JSON, it might use Snowflake infer or require specifying a column of type VARIANT to load the raw JSON. We will document any such nuances.

**`.env` file example:** (provided as `example.env` in repo)

```ini
# Snowflake connection settings
SNOWFLAKE_ACCOUNT=xxxxxx
SNOWFLAKE_USER=xxxxxx
SNOWFLAKE_PASSWORD=xxxxxx
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_ROLE=SYSADMIN
SNOWFLAKE_SCHEMA=PUBLIC

# Azure Storage settings
AZURE_STORAGE_ACCOUNT=youraccount.blob.core.windows.net
AZURE_STORAGE_SAS_TOKEN=?sv=...   # if using SAS token for auth (alternatively use AZURE_STORAGE_KEY)
# AZURE_STORAGE_KEY=...          # (not needed if SAS provided or using managed identity)
AZURE_CONTAINER=your-container-name
```

We clearly instruct the user to fill these with their values. The tool’s config can refer to these (for instance, the `path` in YAML might just be the container-relative path, and we know the container from env).

### User-Facing Safeguards and Tips

* We will include a note (especially for students using this in a course context) that while the tool automates the assignment, they should understand how it works. The professor’s guideline likely wants to ensure students don’t just run a tool blindly. Our documentation encourages learning: we might add commentary like “The schema inference uses advanced logic to determine data types – you can inspect `schema_inference.py` to see how it guesses types based on data patterns.” This way, the tool doubles as a learning aid rather than a black box.
* If a user attempts to use the tool on the exact course dataset, it will run, but because we removed any explicit answers (like exact schemas), it’s not handing them the solution on a platter – the tool is doing it algorithmically. This should satisfy the requirement of not exposing a static solution. Each student’s outcome might vary slightly (e.g. if a column is borderline between INT and FLOAT, maybe one implementation picks INT if no floats seen; if another student’s data had a float, it picks FLOAT). These differences mean the tool isn’t directly giving away a pre-fixed answer key.

### Maintenance and Support

For end users, we also provide guidance on how to troubleshoot and get support:

* If something fails, check the console output and the `logs/` file for errors. Common issues (with solutions) will be listed in a **Troubleshooting** section of the README, similar to the original one but updated. For example: “If you get a Snowflake connection timeout, verify your network access to Snowflake (you might need to be on VPN or allow your IP).” Or “If data isn’t loading, check that your Azure SAS token is valid and the container/path is correct.”
* We will encourage version control for config files (but not for .env containing secrets). If multiple users are collaborating, having the YAML in Git ensures everyone knows what data is being loaded and how.
* The tool will likely evolve, so we mention that advanced users can extend modules. But for a non-technical user, the promise is they won’t need to touch the code. All interactions are via config or input data changes.

In summary, onboarding a new user is boiled down to **“Configure and Run”** steps that are clearly documented. The heavy lifting (like figuring out schemas and writing SQL) is all done under the hood, so the user’s experience is one of simply pointing the tool at data and watching Snowflake get populated. This meets our goal of **minimal instruction onboarding** – essentially anyone who can edit a text file and run a Python command can use this tool to load data into Snowflake, without possessing Snowflake or SQL expertise.

## Conclusion and Next Steps

*(Optional summary)* The refactored Snowflake-Azure ETL project is now a flexible, intelligent tool that automates the ingestion of data into Snowflake. By completing the above tasks, we ensure the solution is generalized for various datasets, uses state-of-the-art schema detection to minimize errors, and provides a smooth experience for end users. Data engineers get a maintainable, extensible codebase with proper modularization and testing, while end users get a one-command pipeline that “just works.” This redesign not only accomplishes the immediate goals (1-6) but also lays a foundation for future enhancements like UI integrations, more AI-driven insights (e.g., data quality reports), and broader cloud compatibility (could extend to AWS S3 or GCP storage). The safeguards in place honor academic integrity guidelines, making the tool viable for educational use as well. With comprehensive documentation and a clear architecture, both developers and users can confidently adopt this tool for their Snowflake ETL needs.
