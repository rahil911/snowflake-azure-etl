"""Dim_Store dimension table."""
from sqlalchemy import Column, Integer, String
from . import DimensionBase

class DimStore(DimensionBase):
    """SQLAlchemy model for the Dim_Store table."""
    __tablename__ = "DIM_STORE"

    DimStoreID = Column(Integer, primary_key=True, autoincrement=True)
    StoreID = Column(Integer)
    DimLocationID = Column(Integer)
    SourceStoreID = Column(Integer)
    StoreName = Column(String(255))
    StoreNumber = Column(String(255))
    StoreManager = Column(String(255))
