"""DDL generation utilities for Snowflake."""

from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import yaml


def create_table_sql(table_name: str, schema: List[Dict]) -> str:
    """Generate CREATE TABLE DDL from schema list."""
    columns = []
    for col in schema:
        col_def = f"{col['name']} {col['type']}"
        if not col.get('nullable', True):
            col_def += " NOT NULL"
        columns.append(col_def)
    cols = ", ".join(columns)
    return f"CREATE OR REPLACE TABLE {table_name} ({cols});"


def export_schema_yaml(table_name: str, schema: List[Dict], output_dir: str) -> None:
    """Write schema definition to YAML file."""
    path = Path(output_dir) / f"{table_name.lower()}_schema.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(schema, f)
