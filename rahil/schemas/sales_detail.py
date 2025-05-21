"""SQLAlchemy model for the STAGING_SALESDETAIL table."""
from sqlalchemy import Column, Integer, String
from . import Base

class SalesDetail(Base):
    """Staging table storing sales detail records."""
    __tablename__ = "STAGING_SALESDETAIL"

    SALESDETAILID = Column(Integer, primary_key=True)
    SALESHEADERID = Column(Integer)
    PRODUCTID = Column(Integer)
    SALESQUANTITY = Column(Integer)
    SALESAMOUNT = Column(String)
    CREATEDDATE = Column(String)
    CREATEDBY = Column(String)
    MODIFIEDDATE = Column(String)
    MODIFIEDBY = Column(String)
