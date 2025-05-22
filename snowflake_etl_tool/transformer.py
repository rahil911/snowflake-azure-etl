from __future__ import annotations
"""Optional transformation execution helpers."""

from pathlib import Path
from typing import List
from .connection import SnowflakeConnection


def run_transformations(conn: SnowflakeConnection, sql_files: List[str]) -> None:
    """Execute SQL transformation scripts in order."""
    for file_path in sql_files:
        path = Path(file_path)
        if not path.exists():
            continue
        with path.open() as f:
            sql = f.read()
        if sql.strip():
            conn.execute(sql)
