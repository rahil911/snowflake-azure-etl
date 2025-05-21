from sqlalchemy import Column, Integer, String
from . import Base

class Reseller(Base):
    __tablename__ = "STAGING_RESELLER"

    RESELLERID = Column(String, primary_key=True)
    CONTACT = Column(String)
    EMAILADDRESS = Column(String)
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
    RESELLERNAME = Column(String)
