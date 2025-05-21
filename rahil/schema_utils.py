"""Utilities for working with SQLAlchemy schema definitions."""
from sqlalchemy.schema import CreateTable
from sqlalchemy import MetaData
from .schemas import Base


def get_create_statements() -> list[str]:
    """Return CREATE TABLE statements for all models."""
    metadata = Base.metadata
    return [str(CreateTable(table)) for table in metadata.sorted_tables]
