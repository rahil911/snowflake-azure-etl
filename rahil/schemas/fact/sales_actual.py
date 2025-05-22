"""Fact_SalesActual table."""
from sqlalchemy import Column, Integer, Float
from . import FactBase

class FactSalesActual(FactBase):
    """SQLAlchemy model for the Fact_SalesActual table."""
    __tablename__ = "FACT_SALESACTUAL"

    DimProductID = Column(Integer)
    DimStoreID = Column(Integer)
    DimResellerID = Column(Integer)
    DimCustomerID = Column(Integer)
    DimChannelID = Column(Integer)
    DimSaleDateID = Column(Integer)
    DimLocationID = Column(Integer)
    SalesHeaderID = Column(Integer)
    SalesDetailID = Column(Integer, primary_key=True)
    SaleAmount = Column(Float)
    SaleQuantity = Column(Integer)
    SaleUnitPrice = Column(Float)
    SaleExtendedCost = Column(Float)
    SaleTotalProfit = Column(Float)
