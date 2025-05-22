"""Dim_Customer dimension table."""
from sqlalchemy import Column, Integer, String
from .. import DimensionBase

class DimCustomer(DimensionBase):
    """SQLAlchemy model for the Dim_Customer table."""
    __tablename__ = "DIM_CUSTOMER"

    DimCustomerID = Column(Integer, primary_key=True, autoincrement=True)
    CustomerID = Column(String(255))
    DimLocationID = Column(Integer)
    CustomerFullName = Column(String(255))
    CustomerFirstName = Column(String(255))
    CustomerLastName = Column(String(255))
    CustomerGender = Column(String(255))
