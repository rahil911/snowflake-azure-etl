"""SQLAlchemy model for the STAGING_CHANNELCATEGORY table."""
from sqlalchemy import Column, Integer, String
from . import Base

class ChannelCategory(Base):
    """Staging table storing channel category information."""
    __tablename__ = "STAGING_CHANNELCATEGORY"

    CHANNELCATEGORYID = Column(Integer, primary_key=True)
    CHANNELCATEGORY = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
