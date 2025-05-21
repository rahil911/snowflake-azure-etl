"""SQLAlchemy model for the STAGING_TARGETDATAPRODUCT table."""
from sqlalchemy import Column, Integer, String
from . import Base

class TargetDataProduct(Base):
    """Staging table storing target sales data by product."""
    __tablename__ = "STAGING_TARGETDATAPRODUCT"

    PRODUCTID = Column(Integer, primary_key=True)
    PRODUCT = Column(String)
    YEAR = Column(Integer)
    SALESQUANTITYTARGET = Column(Integer)
