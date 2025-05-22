"""
SQLAlchemy model for the Producttype entity
"""
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from . import Base
from rahil import config

class Producttype(Base):
    """
    Producttype table model
    """
    __tablename__ = 'STAGING_PRODUCTTYPE'
    __table_args__ = {'schema': config.SNOWFLAKE_SCHEMA}
    
    PRODUCTTYPEID = Column(Integer, nullable=False, primary_key=True)
    PRODUCTCATEGORYID = Column(Integer, nullable=True)
    PRODUCTTYPE = Column(String, nullable=True)
    CREATEDDATE = Column(String, nullable=True)
    CREATEDBY = Column(String, nullable=True)
    MODIFIEDDATE = Column(String, nullable=True)
    MODIFIEDBY = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Producttype(PRODUCTTYPEID={self.PRODUCTTYPEID})>"
