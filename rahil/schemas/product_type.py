from sqlalchemy import Column, Integer, String
from . import Base

class ProductType(Base):
    __tablename__ = "STAGING_PRODUCTTYPE"

    PRODUCTTYPEID = Column(Integer, primary_key=True)
    PRODUCTCATEGORYID = Column(Integer)
    PRODUCTTYPE = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
