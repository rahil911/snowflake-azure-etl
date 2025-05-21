"""Fact_SRCSalesTarget table."""
from sqlalchemy import Column, Integer, Float
from . import FactBase

class FactSRCSalesTarget(FactBase):
    """SQLAlchemy model for the Fact_SRCSalesTarget table."""
    __tablename__ = "FACT_SRCSALESTARGET"

    DimStoreID = Column(Integer)
    DimResellerID = Column(Integer)
    DimChannelID = Column(Integer)
    DimTargetDateID = Column(Integer)
    SalesTargetAmount = Column(Float)
