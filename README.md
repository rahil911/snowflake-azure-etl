# Snowflake ETL Tool

Reusable command line tool for ingesting files from Azure Blob Storage (or the local filesystem) into Snowflake.

## Features

* Automatic schema inference from CSV samples
* Dynamic table creation and COPY loading
* Optional execution of transformation SQL scripts
* Schema export to YAML for review
* Configuration purely through `config.yaml` and `.env`

## Architecture

```
Config -> Connection -> SchemaInference -> DDLGenerator -> DataLoader -> Transformer
```

1. **Config** loads YAML and environment settings.
2. **Connection** manages Snowflake connections.
3. **SchemaInference** profiles CSV files to guess column types.
4. **DDLGenerator** builds and optionally exports `CREATE TABLE` statements.
5. **DataLoader** creates stages and loads data via `COPY`.
6. **Transformer** optionally runs post-load SQL.

## Getting Started

1. Install dependencies (optional in this environment):
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the examples and fill in your details:
   ```bash
   cp examples/example_config.yaml config.yaml
   cp examples/example.env .env
   ```
3. Edit the files with your Snowflake account info and data locations.
4. Run the pipeline:
   ```bash
   python -m snowflake_etl_tool.pipeline_runner --config config.yaml
   ```

## Academic Integrity Notice

This repository provides a generic automation framework. It should not be used to obtain assignment solutions without permission. Follow all course policies when applying it to class work.

## Tests

Run the unit tests with `pytest`:

```bash
python -m pytest -q
```
