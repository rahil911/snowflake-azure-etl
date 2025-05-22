"""
SQLAlchemy model for the Reseller entity
"""
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from . import Base
from rahil import config

class Reseller(Base):
    """
    Reseller table model
    """
    __tablename__ = 'STAGING_RESELLER'
    __table_args__ = {'schema': config.SNOWFLAKE_SCHEMA}
    
    RESELLERID = Column(String, nullable=False, primary_key=True)
    CONTACT = Column(String, nullable=True)
    EMAILADDRESS = Column(String, nullable=True)
    ADDRESS = Column(String, nullable=True)
    CITY = Column(String, nullable=True)
    STATEPROVINCE = Column(String, nullable=True)
    COUNTRY = Column(String, nullable=True)
    POSTALCODE = Column(Integer, nullable=True)
    PHONENUMBER = Column(String, nullable=True)
    CREATEDDATE = Column(String, nullable=True)
    CREATEDBY = Column(String, nullable=True)
    MODIFIEDDATE = Column(String, nullable=True)
    MODIFIEDBY = Column(String, nullable=True)
    RESELLERNAME = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Reseller(RESELLERID={self.RESELLERID})>"
