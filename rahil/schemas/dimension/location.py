"""Dim_Location dimension table."""
from sqlalchemy import Column, Integer, String
from . import DimensionBase

class DimLocation(DimensionBase):
    """SQLAlchemy model for the Dim_Location table."""
    __tablename__ = "DIM_LOCATION"

    DimLocationID = Column(Integer, primary_key=True, autoincrement=True)
    Address = Column(String(255))
    City = Column(String(255))
    PostalCode = Column(String(255))
    State_Province = Column(String(255))
    Country = Column(String(255))
