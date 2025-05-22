from __future__ import annotations
"""Main pipeline orchestrator for Snowflake ETL."""

from .config import Config
from .connection import SnowflakeConnection
from .schema_inference import infer_schema_csv
from .ddl_generator import create_table_sql, export_schema_yaml
from .data_loader import create_stage, copy_into_table
from .transformer import run_transformations
from pathlib import Path


def run_pipeline(config_path: str = "config.yaml") -> None:
    """Run the ETL pipeline based on the provided config file."""
    config = Config(config_path)
    conn = SnowflakeConnection(config)
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {config.target_database}")
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {config.target_database}.{config.target_schema}")
    for source in config.data_sources:
        name = source['name']
        path = Path(source['path'])
        file_format = source.get('file_format', 'csv')
        table_name = source.get('target_table', name.upper())
        schema = infer_schema_csv(path)
        ddl = create_table_sql(table_name, schema)
        conn.execute(ddl)
        if config.export_schema_dir:
            export_schema_yaml(table_name, schema, config.export_schema_dir)
        stage_name = f"STAGE_{name}"
        create_stage(conn, stage_name, url=str(path.parent), file_format=file_format)
        copy_into_table(conn, table_name, stage_name)
    if config.transformations:
        run_transformations(conn, config.transformations)
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Snowflake ETL pipeline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
