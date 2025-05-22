"""
SQLAlchemy model for the Product entity
"""
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from . import Base
from rahil import config

class Product(Base):
    """
    Product table model
    """
    __tablename__ = 'STAGING_PRODUCT'
    __table_args__ = {'schema': config.SNOWFLAKE_SCHEMA}
    
    PRODUCTID = Column(Integer, nullable=False, primary_key=True)
    PRODUCTTYPEID = Column(Integer, nullable=True)
    PRODUCT = Column(String, nullable=True)
    COLOR = Column(String, nullable=True)
    STYLE = Column(String, nullable=True)
    UNITOFMEASUREID = Column(Integer, nullable=True)
    WEIGHT = Column(String, nullable=True)
    PRICE = Column(String, nullable=True)
    COST = Column(String, nullable=True)
    CREATEDDATE = Column(String, nullable=True)
    CREATEDBY = Column(String, nullable=True)
    MODIFIEDDATE = Column(String, nullable=True)
    MODIFIEDBY = Column(String, nullable=True)
    WHOLESALEPRICE = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Product(PRODUCTID={self.PRODUCTID})>"
