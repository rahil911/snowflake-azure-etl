## Task Breakdown ✔️

Below is a comprehensive checklist of refactoring tasks. Each task is a bite-sized, one-point effort, with sub-tasks as needed. These tasks cover everything from code changes and new features to documentation and testing:

* [ ] **Project Structure & Naming** – Restructure the repository for clarity and generality:

  * [ ] Rename the core package (e.g. from `rahil/` to a more generic `snowflake_etl_tool/`) to reflect a general-purpose tool.
  * [ ] Create a clear folder hierarchy (see “Architecture” below) separating configuration, core logic modules, and any example or sample resources.
  * [ ] Remove or replace any course-specific naming (e.g. `IMT577_DW` prefix) with generic names or configurable templates to avoid revealing assignment details.

* [ ] **Configuration Abstraction** – Abstract all Snowflake and environment configs into user-friendly config files and environment variables:

  * [ ] Use a single YAML/JSON **config file** for pipeline settings (e.g. input data locations, target database/schema names, etc.) so users don’t have to edit code.
  * [ ] Migrate Snowflake connection parameters (account, user, password, warehouse, role, etc.) to a config or `.env` file (continue using `.env` for secrets, pre-populated via `example.env`). Ensure these are loaded in a `config.py` or similar, but **not hard-coded** anywhere.
  * [ ] Implement logic to auto-create any needed Snowflake objects (database, schema, warehouse) based on config defaults if they aren’t present – **without user intervention**.
  * [ ] **Safeguard:** Do not expose any actual credentials or assignment data in the repository. (Double-check that `.env` stays in `.gitignore` and that example values are generic.)

* [ ] **Schema Inference Module** – Develop a new module to infer the schema of incoming data automatically:

  * [ ] **Data Sampling/Profiling:** Read a sample of each input dataset (e.g. first N rows or a random subset) to gather column statistics. Use statistical profiling to determine each column’s likely type (numeric, float, string, date, boolean, etc.). This may involve reading files in chunks to handle large sizes efficiently, possibly using parallel processing for speed.
  * [ ] **Type Detection Logic:** Implement robust rules or an ML model to infer data types: e.g. try to parse values as integers, floats, dates, booleans, etc., count success rates, and pick the best-fit type for each column. If a column’s values are mostly integers but some have decimals, infer it as FLOAT (using an “upgrade” rule). If most values look like dates, tag as DATE or DATETIME. Include detection of text length to set appropriate VARCHAR length (or use Snowflake `TEXT` which is unlimited).
  * [ ] **Nullable & Constraints:** Infer which columns can be NULL by checking if any empty/missing values exist in sample. Mark columns as NOT NULL if no nulls seen (with a safety margin). Consider basic constraints (e.g. if a column’s values all fall in a small set, we might flag it as categorical – for future optimization, though initial focus is on types).
  * [ ] **Leverage Snowflake INFER\_SCHEMA (Optional):** Where applicable, call Snowflake’s native `INFER_SCHEMA` function on staged files to get an initial schema suggestion. This is especially useful for CSV, JSON, Parquet, etc., and can directly provide column names and types. The module should compare Snowflake’s suggestion with our own inference for validation.
  * [ ] **GenAI Integration (Optional Mode):** If configured, use a Generative AI service to double-check or refine the inferred schema. For example, provide a snippet of data to an LLM and ask for the best column types or even a draft **CREATE TABLE** statement. Use this to catch edge cases or add semantic info. *(This can be an experimental feature for high accuracy: the AI might recognize patterns humans would, like zip code formats or encoded categories.)*
  * [ ] **Output Schema Definition:** Define a standard **schema representation** (e.g. a Python dict or a JSON/YAML structure) to hold the inferred schema. For example: a list of columns with name, type, nullable, etc. This structured schema will be used by other modules to generate DDL.
  * [ ] **Unit Tests – Schema Inference:** Write tests feeding known sample data into the inference module to ensure it guesses types correctly (e.g. a column of “123, 456, 789” -> INTEGER, a mixed “12.5, 7, 3.14” -> FLOAT, a date-formatted string -> DATE). Include edge cases like empty strings, “NULL” text, very large numbers, etc.

* [ ] **DDL Generation Module** – Remove hard-coded table schemas from `create_tables.py` and replace with dynamic DDL creation based on inferred schema:

  * [ ] **Generate `CREATE TABLE` SQL:** Use the schema definition from the previous step to programmatically build a Snowflake `CREATE TABLE` statement. This should assign each column name and data type as inferred. If using Snowflake’s `USING TEMPLATE` approach, the module can instead execute a `CREATE TABLE ... USING TEMPLATE ( SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*)) FROM TABLE(INFER_SCHEMA(...)) )` for convenience – however, ensure the resulting table matches our expectations (Snowflake might infer some columns as TEXT or VARIANT; the tool can decide to alter those if needed for accuracy).
  * [ ] **YAML/JSON Schema Templates:** Support outputting the inferred schema to a YAML/JSON template file. This file can serve as a **human-readable schema blueprint**. Advanced users or data engineers can review and edit this before creation, or save it for version control. The tool can also accept a user-provided schema file to override or pre-define certain columns (for example, if the user wants to specify that a column is `VARCHAR(50)` instead of the auto-inferred `TEXT`).
  * [ ] **Execute DDL in Snowflake:** Use the Snowflake connection module to run the `CREATE TABLE` statement on the target Snowflake database. Ensure the target database and schema exist (create them if not, as handled in a separate task). **Remove all existing SQL DDL strings from the codebase** – they should now be generated at runtime or moved to templates.
  * [ ] **Verify Table Structure:** After creation, query Snowflake metadata (`SHOW COLUMNS` or Information Schema) to verify the table was created with the expected columns and types. This acts as a safeguard that our inference matched the actual data structure.
  * [ ] **Unit Tests – DDL Generation:** If possible, mock the Snowflake connection and test that given a sample schema dict, the generated SQL string matches expected output. Also test the YAML export/import of schema.

* [ ] **Data Loading Module** – Refactor and generalize the data loading (formerly in `load_data.py`):

  * [ ] **Generalize External Stage Usage:** Instead of 12 fixed external stage names, dynamically determine the data source locations. For Azure Blob, if the user provides a container or list of blob paths, iterate through them. Create external stages for each data source programmatically (or reuse one stage and vary the path if appropriate). Abstract this in a `create_stage` function that takes a container/path and creates an external Snowflake stage (using the already configured Azure storage integration).
  * [ ] **COPY into Table:** For each dataset/table, execute Snowflake `COPY INTO <table>` from the external stage. Use `FILE_FORMAT` options appropriate to the data (e.g. if CSV, ensure header skipping or `MATCH_BY_COLUMN_NAME=CASE_INSENSITIVE` is set so that columns load by name). The COPY command should be constructed without user needing to edit anything.
  * [ ] **Automatic Pipeline Orchestration:** Implement an orchestrator (could be the `run_etl.py` refactored or a new `pipeline_runner.py`) that ties the steps together: for each data source, call schema inference -> create table -> load data. Make this loop handle any number of sources (not just 12). The orchestrator should log progress for each dataset and stop on critical errors but continue to next dataset if a non-fatal error occurs (robustness).
  * [ ] **Progress Logging:** Continue and enhance the logging system. Log each major action (schema inferred, table created, N rows loaded, etc.) to the console and to a file in `logs/`. Ensure that for non-technical users, log messages are clear (e.g. “✅ Loaded 10,000 rows into `CUSTOMER_TABLE` successfully.”).
  * [ ] **Error Handling & Schema Evolution:** Build in safeguards during loading: if a COPY into table fails due to a schema mismatch (e.g. a new column in data that wasn’t inferred, or a value conversion error), catch that error. The tool should then **auto-evolve** the schema if possible: for example, add the missing column (using Snowflake’s **schema evolution** capability or an ALTER TABLE), or adjust the column data type (e.g. widen a VARCHAR or switch an INT to FLOAT) and retry the load. This ensures the pipeline is resilient to upstream changes. Such schema evolution can be logged as warnings.
  * [ ] **Unit/Integration Tests – Loading:** Write tests for the data loading functions by simulating a small Snowflake environment. If direct Snowflake testing is not feasible in CI, abstract the Snowflake calls so they can be mocked (e.g. have a `SnowflakeClient` class with a method `copy_into(table, stage, ...)` that can be stubbed). Test that the COPY SQL is formed correctly for given inputs, and that the orchestrator calls the right sequence of actions.

* [ ] **Transformation Module (Future-proofing)** – Although the current pipeline only stages raw data, design the code to allow easy insertion of transformation steps:

  * [ ] Create a placeholder module (or section in config) for **data transformation** logic (e.g. merging staging tables into final tables, applying business rules). This module can load SQL scripts or use Snowflake queries to transform data. For now, it might simply log that “no transformations are configured.”
  * [ ] Ensure the pipeline orchestrator can invoke transformation step(s) after loading. For example, if a YAML config lists a transformation SQL file or a Python transformation plugin, the runner will execute it. Keeping this modular (perhaps via an interface or simply a function pointer) will allow swapping in different transformation implementations without altering core pipeline code.
  * [ ] **Testability:** If any sample transformation is included (e.g. a simple row count check or update), include tests for it. Otherwise, just ensure the pipeline doesn’t break if no transforms are present.

* [ ] **Automation & Execution** – Make the pipeline easy to run manually or automatically:

  * [ ] Provide a **command-line interface (CLI)** entry point (e.g. `python -m snowflake_etl_tool.run` or an installed console script) that a user can run. This should accept parameters like `--config config.yaml` so the user can specify their configuration file. Defaults should allow running with just `python run.py` if a default config is present.
  * [ ] Implement scheduling or triggering options. For example, allow the tool to be run on a schedule (document how to use Windows Task Scheduler or cron with the CLI). Alternatively, integrate with Snowflake **Tasks** and **Streams** or Azure Event Grid for new file detection, to trigger the pipeline when new data arrives (this can be a stretch goal; at minimum, design such that it’s possible to extend).
  * [ ] **Minimal User Input:** Document that the only things the user needs to do are: provide their credentials (in .env or config), and specify where the input data resides. The tool should not require any coding by the user. Emphasize that things like creating databases, defining stages, writing copy commands, etc., are all automated.
  * [ ] **End-to-End Test:** After implementation, perform an end-to-end test simulating a non-technical user: set up a fresh .env and config pointing to some sample data, run the pipeline, and verify it completes all steps (schema inferred, table created, data loaded, sample output shown). This ensures the “automated with minimal input” goal is met.

* [ ] **Course 577 Safeguards** – Special tasks to ensure no assignment solutions are leaked and misuse is prevented:

  * [ ] Remove the explicit list of 12 entities and their schemas (this is already achieved by dynamic schema inference). The tool should not contain any hard-coded answer or data specific to the course assignment.
  * [ ] Make the new default database/schema names generic (e.g. use a base name like `AUTO_ETL_DB` instead of `IMT577_DW_{USER}_STAGING`). If the course still needs the naming convention, allow it to be provided via config (so it’s not baked into code publicly).
  * [ ] Include a **warning or notice** in the README for students: e.g. “This tool is a generalized solution and should not be used to directly obtain assignment answers. It is intended for learning and professional use. Follow academic integrity policies.” This sets expectation that simply running the tool on provided assignment data is not the learning goal.
  * [ ] If any example data or queries from the course are included in this repo for demonstration, ensure they are simplified or changed so as not to directly solve graded parts. (Alternatively, host them privately or behind a login if needed.)
  * [ ] Possibly implement an **access control** in code: for instance, require a specific flag or config value to run the full pipeline on the course dataset – something that the professor can control but not students (this is optional and situation-dependent). The main point is to avoid “one-click cheating” scenarios using this tool.

* [ ] **Documentation & Onboarding** – Prepare thorough documentation for both engineers and end users:

  * [ ] **Architecture Documentation:** Write a section (in `README.md` or a separate `ARCHITECTURE.md`) for data engineers that explains how the tool is designed. Include a diagram or bullet workflow of the modules and their interactions (e.g. SchemaInferer -> DDLGenerator -> DataLoader -> etc.). Document the file/folder structure after refactor and the responsibility of each module.
  * [ ] **Under-the-Hood Guide:** In the documentation, provide a clear step-by-step explanation of what happens when the pipeline runs. For example: “Step 1: The tool reads the config and connects to Snowflake. Step 2: It scans the Azure Blob storage for data files. Step 3: For each file, it infers the schema (here’s how…) and creates a table in Snowflake. Step 4: Data is copied into Snowflake. Step 5: The tool verifies the load and either transforms data or logs the results.” This gives technical readers insight into the automation logic.
  * [ ] **User Onboarding Guide:** Create a **Getting Started** guide aimed at non-technical users. This should be a concise, bullet-point style set of instructions with minimal jargon. For example:

    1. *Install the tool* (e.g. “`pip install snowflake-etl-tool` or clone the repository”).
    2. *Configure your connection* (fill in `.env` or a config YAML with Snowflake and Azure details).
    3. *Provide data source info* (e.g. update the config with your container name or file paths).
    4. *Run the pipeline* (exact command to execute).
    5. *Monitor output* (where to see logs or any results, e.g. it might print sample loaded data or row counts).
    6. *Troubleshooting tips* (common issues like wrong credentials or data format problems and how to resolve them).
  * [ ] **Example Config Files:** Include example YAML/JSON configuration files in a `examples/` directory. For instance, a `example_config.yaml` demonstrating how to list multiple data sources with their paths and any custom options (like a specific file format or desired target table name). Also provide an example of the schema YAML output (if that feature is enabled), to illustrate what the auto-generated schema looks like.
  * [ ] **Inline Code Documentation:** Ensure each module and major function has docstrings/comments explaining its purpose and usage. This will help other developers (or course instructors) review and understand the implementation.
  * [ ] **Update README:** Replace the old README content (which was assignment-specific) with updated documentation reflecting the new tool. Preserve any useful general info (like how logging works), but remove references to the 12 entities or the course context unless needed.

* [ ] **Testing & QA** – Achieve good test coverage and quality assurance before release:

  * [ ] Write unit tests for each new module (as noted in previous tasks) and integration tests for the overall pipeline using small dummy datasets. Aim for **high coverage**, especially on the critical inference and DDL generation components.
  * [ ] Perform code reviews and perhaps static analysis to ensure code quality (PEP8 compliance, etc.). Since the tool may be used by others, ensure readability and maintainability.
  * [ ] Test with various input scenarios: a CSV with obvious types, a tricky CSV (e.g. numeric columns with a few stray text entries), a JSON file (if supported), large files vs. small files, etc. Verify the tool handles each gracefully (either by correct inference or by informative error messages).
  * [ ] Final professor review: Present the refactored code (or documentation) to the course professor to double-check that all “solution exposure” concerns are addressed. Get confirmation that the safeguards are sufficient.

Each of these tasks can be checked off upon completion, ensuring that the refactoring touches all required areas: from core functionality through to documentation and safeguards. By following this checklist, we can systematically transform the project into the desired state.
