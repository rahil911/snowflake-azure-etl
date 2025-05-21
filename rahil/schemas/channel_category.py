from sqlalchemy import Column, Integer, String
from . import Base

class ChannelCategory(Base):
    __tablename__ = "STAGING_CHANNELCATEGORY"

    CHANNELCATEGORYID = Column(Integer, primary_key=True)
    CHANNELCATEGORY = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
