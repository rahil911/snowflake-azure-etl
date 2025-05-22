"""Dim_Channel dimension table."""
from sqlalchemy import Column, Integer, String
from .. import DimensionBase

class DimChannel(DimensionBase):
    """SQLAlchemy model for the Dim_Channel table."""
    __tablename__ = "DIM_CHANNEL"

    DimChannelID = Column(Integer, primary_key=True, autoincrement=True)
    ChannelID = Column(Integer)
    ChannelCategoryID = Column(Integer)
    ChannelName = Column(String(255))
    ChannelCategory = Column(String(255))
