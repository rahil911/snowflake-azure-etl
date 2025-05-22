from __future__ import annotations
"""Snowflake connection wrapper for executing SQL commands."""

import snowflake.connector
from typing import Any
from .config import Config

class SnowflakeConnection:
    """Wrapper around snowflake.connector for easy mocking."""

    def __init__(self, config: Config):
        self.config = config
        self._conn = None

    def connect(self) -> None:
        self._conn = snowflake.connector.connect(
            account=self.config.snowflake_account,
            user=self.config.snowflake_user,
            password=self.config.snowflake_password,
            warehouse=self.config.snowflake_warehouse,
            role=self.config.snowflake_role,
            schema=self.config.target_schema,
            database=self.config.target_database,
        )

    def execute(self, sql: str) -> Any:
        if self._conn is None:
            self.connect()
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            return cur.fetchall()
        finally:
            cur.close()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
