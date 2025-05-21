"""Fact_ProductSalesTarget table."""
from sqlalchemy import Column, Integer
from . import FactBase

class FactProductSalesTarget(FactBase):
    """SQLAlchemy model for the Fact_ProductSalesTarget table."""
    __tablename__ = "FACT_PRODUCTSALESTARGET"

    DimProductID = Column(Integer)
    DimTargetDateID = Column(Integer)
    ProductTargetSalesQuantity = Column(Integer)
