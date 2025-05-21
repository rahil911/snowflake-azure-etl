from sqlalchemy import Column, Integer, String
from . import Base

class Product(Base):
    __tablename__ = "STAGING_PRODUCT"

    PRODUCTID = Column(Integer, primary_key=True)
    PRODUCTTYPEID = Column(Integer)
    PRODUCT = Column(String)
    COLOR = Column(String)
    STYLE = Column(String)
    UNITOFMEASUREID = Column(Integer)
    WEIGHT = Column(String)
    PRICE = Column(String)
    COST = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
    WHOLESALEPRICE = Column(String)
