from sqlalchemy import Column, Integer, String
from . import Base

class ProductCategory(Base):
    __tablename__ = "STAGING_PRODUCTCATEGORY"

    PRODUCTCATEGORYID = Column(Integer, primary_key=True)
    PRODUCTCATEGORY = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
