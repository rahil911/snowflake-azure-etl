"""SQLAlchemy model for the STAGING_CUSTOMER table."""
from sqlalchemy import Column, Integer, String
from . import Base

class Customer(Base):
    """Staging table storing customer information."""
    __tablename__ = "STAGING_CUSTOMER"

    CUSTOMERID = Column(String, primary_key=True)
    SUBSEGMENTID = Column(Integer)
    FIRSTNAME = Column(String)
    LASTNAME = Column(String)
    GENDER = Column(String)
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
