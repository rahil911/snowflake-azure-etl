from sqlalchemy import Column, Integer, String
from . import Base

class Store(Base):
    __tablename__ = "STAGING_STORE"

    STOREID = Column(Integer, primary_key=True)
    SUBSEGMENTID = Column(Integer)
    STORENUMBER = Column(Integer)
    STOREMANAGER = Column(String)
    ADDRESS = Column(String)
    CITY = Column(String)
    STATEPROVINCE = Column(String)
    COUNTRY = Column(String)
    POSTALCODE = Column(Integer)
    PHONENUMBER = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
