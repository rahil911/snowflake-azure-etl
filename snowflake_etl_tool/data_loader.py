from __future__ import annotations
"""Utility helpers for creating stages and loading data."""

from typing import List, Dict
from .connection import SnowflakeConnection


def create_stage(conn: SnowflakeConnection, stage_name: str, url: str, file_format: str = 'csv') -> None:
    """Create or replace an external stage."""
    sql = f"CREATE OR REPLACE STAGE {stage_name} URL='{url}' FILE_FORMAT = (TYPE={file_format});"
    conn.execute(sql)


def copy_into_table(conn: SnowflakeConnection, table_name: str, stage_name: str) -> None:
    """Execute COPY INTO using the given stage."""
    sql = f"COPY INTO {table_name} FROM @{stage_name};"
    conn.execute(sql)
