"""
SQLAlchemy model for the Targetdataproduct entity
"""
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from . import Base
from rahil import config

class Targetdataproduct(Base):
    """
    Targetdataproduct table model
    """
    __tablename__ = 'STAGING_TARGETDATAPRODUCT'
    __table_args__ = {'schema': config.SNOWFLAKE_SCHEMA}
    
    PRODUCTID = Column(Integer, nullable=False, primary_key=True)
    PRODUCT = Column(String, nullable=True)
    YEAR = Column(Integer, nullable=True)
    SALESQUANTITYTARGET = Column(Integer, nullable=True)
    
    def __repr__(self):
        return f"<Targetdataproduct(PRODUCTID={self.PRODUCTID})>"
