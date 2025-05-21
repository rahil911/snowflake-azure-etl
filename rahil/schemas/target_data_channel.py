from sqlalchemy import Column, Integer, String
from . import Base

class TargetDataChannel(Base):
    __tablename__ = "STAGING_TARGETDATACHANNEL"

    YEAR = Column(Integer, primary_key=True)
    CHANNELNAME = Column(String)
    TARGETNAME = Column(String)
    TARGETSALESAMOUNT = Column(Integer)
