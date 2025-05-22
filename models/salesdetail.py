"""
SQLAlchemy model for the Salesdetail entity
"""
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from . import Base
from rahil import config

class Salesdetail(Base):
    """
    Salesdetail table model
    """
    __tablename__ = 'STAGING_SALESDETAIL'
    __table_args__ = {'schema': config.SNOWFLAKE_SCHEMA}
    
    SALESDETAILID = Column(Integer, nullable=False, primary_key=True)
    SALESHEADERID = Column(Integer, nullable=True)
    PRODUCTID = Column(Integer, nullable=True)
    SALESQUANTITY = Column(Integer, nullable=True)
    SALESAMOUNT = Column(String, nullable=True)
    CREATEDDATE = Column(String, nullable=True)
    CREATEDBY = Column(String, nullable=True)
    MODIFIEDDATE = Column(String, nullable=True)
    MODIFIEDBY = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Salesdetail(SALESDETAILID={self.SALESDETAILID})>"
