"""
SQLAlchemy model for the Productcategory entity
"""
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from . import Base
from rahil import config

class Productcategory(Base):
    """
    Productcategory table model
    """
    __tablename__ = 'STAGING_PRODUCTCATEGORY'
    __table_args__ = {'schema': config.SNOWFLAKE_SCHEMA}
    
    PRODUCTCATEGORYID = Column(Integer, nullable=False, primary_key=True)
    PRODUCTCATEGORY = Column(String, nullable=True)
    CREATEDDATE = Column(String, nullable=True)
    CREATEDBY = Column(String, nullable=True)
    MODIFIEDDATE = Column(String, nullable=True)
    MODIFIEDBY = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Productcategory(PRODUCTCATEGORYID={self.PRODUCTCATEGORYID})>"
