"""Fact_ProductSalesTarget table."""
from sqlalchemy import Column, Integer
from . import FactBase

class FactProductSalesTarget(FactBase):
    """SQLAlchemy model for the Fact_ProductSalesTarget table."""
    __tablename__ = "FACT_PRODUCTSALESTARGET"

    DimProductID = Column(Integer, primary_key=True)
    DimTargetDateID = Column(Integer)
    ProductTargetSalesQuantity = Column(Integer)
