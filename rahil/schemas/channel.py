"""SQLAlchemy model for the STAGING_CHANNEL table."""
from sqlalchemy import Column, Integer, String
from . import Base

class Channel(Base):
    """Staging table storing channel information."""
    __tablename__ = "STAGING_CHANNEL"

    CHANNELID = Column(Integer, primary_key=True)
    CHANNELCATEGORYID = Column(Integer)
    CHANNEL = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
