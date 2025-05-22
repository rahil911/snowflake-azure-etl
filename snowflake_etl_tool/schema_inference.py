"""Simple schema inference for CSV sources."""

from __future__ import annotations
from typing import List, Dict
import pandas as pd


def infer_schema_csv(path: str, sample_rows: int = 1000) -> List[Dict]:
    """Infer schema for a CSV file using basic profiling."""
    df = pd.read_csv(path, nrows=sample_rows)
    schema: List[Dict] = []
    for col in df.columns:
        series = df[col]
        nullable = bool(series.isna().any())
        sample = series.dropna()
        col_type = "TEXT"
        if not sample.empty:
            if pd.api.types.is_integer_dtype(series):
                col_type = "INTEGER"
            elif pd.api.types.is_float_dtype(series):
                if (sample == sample.astype(int)).all():
                    col_type = "INTEGER"
                else:
                    col_type = "FLOAT"
            else:
                try:
                    sample.astype(int)
                    col_type = "INTEGER"
                except ValueError:
                    try:
                        sample.astype(float)
                        if (sample.astype(float) == sample.astype(int)).all():
                            col_type = "INTEGER"
                        else:
                            col_type = "FLOAT"
                    except ValueError:
                        if all(str(x).lower() in {"true", "false", "0", "1"} for x in sample.head(20)):
                            col_type = "BOOLEAN"
                        else:
                            try:
                                pd.to_datetime(sample, errors="raise")
                                col_type = "TIMESTAMP"
                            except Exception:
                                max_len = int(sample.astype(str).str.len().max())
                                col_type = f"VARCHAR({max_len})"
        schema.append({"name": col, "type": col_type, "nullable": nullable})
    return schema
