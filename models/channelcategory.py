"""
SQLAlchemy model for the Channelcategory entity
"""
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from . import Base
from rahil import config

class Channelcategory(Base):
    """
    Channelcategory table model
    """
    __tablename__ = 'STAGING_CHANNELCATEGORY'
    __table_args__ = {'schema': config.SNOWFLAKE_SCHEMA}
    
    CHANNELCATEGORYID = Column(Integer, nullable=False, primary_key=True)
    CHANNELCATEGORY = Column(String, nullable=True)
    CREATEDDATE = Column(String, nullable=True)
    CREATEDBY = Column(String, nullable=True)
    MODIFIEDDATE = Column(String, nullable=True)
    MODIFIEDBY = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Channelcategory(CHANNELCATEGORYID={self.CHANNELCATEGORYID})>"
