from sqlalchemy import Column, Integer, String
from . import Base

class SalesDetail(Base):
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
