"""Dim_Product dimension table."""
from sqlalchemy import Column, Integer, String, Float
from .. import DimensionBase

class DimProduct(DimensionBase):
    """SQLAlchemy model for the Dim_Product table."""
    __tablename__ = "DIM_PRODUCT"

    DimProductID = Column(Integer, primary_key=True, autoincrement=True)
    ProductID = Column(Integer)
    ProductTypeID = Column(Integer)
    ProductCategoryID = Column(Integer)
    ProductName = Column(String(255))
    ProductType = Column(String(255))
    ProductCategory = Column(String(255))
    ProductRetailPrice = Column(Float)
    ProductWholesalePrice = Column(Float)
    ProductCost = Column(Float)
    ProductRetailProfit = Column(Float)
    ProductWholesaleUnitProfit = Column(Float)
    ProductProfitMarginUnitPercent = Column(Float)
