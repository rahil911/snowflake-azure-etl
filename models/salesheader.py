"""
SQLAlchemy model for the Salesheader entity
"""
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from . import Base
from rahil import config

class Salesheader(Base):
    """
    Salesheader table model
    """
    __tablename__ = 'STAGING_SALESHEADER'
    __table_args__ = {'schema': config.SNOWFLAKE_SCHEMA}
    
    SALESHEADERID = Column(Integer, nullable=False, primary_key=True)
    DATE = Column(String, nullable=True)
    CHANNELID = Column(Integer, nullable=True)
    STOREID = Column(Integer, nullable=True)
    CUSTOMERID = Column(String, nullable=True)
    RESELLERID = Column(String, nullable=True)
    CREATEDDATE = Column(String, nullable=True)
    CREATEDBY = Column(String, nullable=True)
    MODIFIEDDATE = Column(String, nullable=True)
    MODIFIEDBY = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Salesheader(SALESHEADERID={self.SALESHEADERID})>"
