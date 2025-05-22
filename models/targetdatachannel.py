"""
SQLAlchemy model for the Targetdatachannel entity
"""
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from . import Base
from rahil import config

class Targetdatachannel(Base):
    """
    Targetdatachannel table model
    """
    __tablename__ = 'STAGING_TARGETDATACHANNEL'
    __table_args__ = {'schema': config.SNOWFLAKE_SCHEMA}
    
    YEAR = Column(Integer, nullable=False, primary_key=True)
    CHANNELNAME = Column(String, nullable=True)
    TARGETNAME = Column(String, nullable=True)
    TARGETSALESAMOUNT = Column(Integer, nullable=True)
    
    def __repr__(self):
        return f"<Targetdatachannel(YEAR={self.YEAR})>"
