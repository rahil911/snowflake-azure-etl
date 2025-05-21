"""SQLAlchemy model for the STAGING_TARGETDATACHANNEL table."""
from sqlalchemy import Column, Integer, String
from . import Base

class TargetDataChannel(Base):
    """Staging table storing target sales data by channel."""
    __tablename__ = "STAGING_TARGETDATACHANNEL"

    YEAR = Column(Integer, primary_key=True)
    CHANNELNAME = Column(String)
    TARGETNAME = Column(String)
    TARGETSALESAMOUNT = Column(Integer)
