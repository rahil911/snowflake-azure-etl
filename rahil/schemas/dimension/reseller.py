"""Dim_Reseller dimension table."""
from sqlalchemy import Column, Integer, String
from .. import DimensionBase

class DimReseller(DimensionBase):
    """SQLAlchemy model for the Dim_Reseller table."""
    __tablename__ = "DIM_RESELLER"

    DimResellerID = Column(Integer, primary_key=True, autoincrement=True)
    ResellerID = Column(String(255))
    DimLocationID = Column(Integer)
    ResellerName = Column(String(255))
    ContactName = Column(String(255))
    PhoneNumber = Column(String(255))
    Email = Column(String(255))
