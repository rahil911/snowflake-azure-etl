from sqlalchemy import Column, Integer, String
from . import Base

class SalesHeader(Base):
    __tablename__ = "STAGING_SALESHEADER"

    SALESHEADERID = Column(Integer, primary_key=True)
    DATE = Column(String)
    CHANNELID = Column(Integer)
    STOREID = Column(Integer)
    CUSTOMERID = Column(String)
    RESELLERID = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
