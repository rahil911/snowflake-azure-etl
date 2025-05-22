"""Dim_Date dimension table (simplified)."""
from sqlalchemy import Column, Integer, String, Date
from .. import DimensionBase

class DimDate(DimensionBase):
    """SQLAlchemy model for the Dim_Date table."""
    __tablename__ = "DIM_DATE"

    DimDateID = Column(Integer, primary_key=True, autoincrement=True)
    DateValue = Column(Date)
    DateLabel = Column(String(50))
