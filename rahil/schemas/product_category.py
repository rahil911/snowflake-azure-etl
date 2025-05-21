"""SQLAlchemy model for the STAGING_PRODUCTCATEGORY table."""
from sqlalchemy import Column, Integer, String
from . import Base

class ProductCategory(Base):
    """Staging table storing product category information."""
    __tablename__ = "STAGING_PRODUCTCATEGORY"

    PRODUCTCATEGORYID = Column(Integer, primary_key=True)
    PRODUCTCATEGORY = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
