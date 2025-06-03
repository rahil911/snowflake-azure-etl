-- Fact_SalesActual table
CREATE OR REPLACE TABLE Fact_SalesActual (
    DimProductID INT,
    DimStoreID INT,
    DimResellerID INT,
    DimCustomerID INT,
    DimChannelID INT,
    DimSaleDateID INT,
    DimLocationID INT,
    SalesHeaderID INT,
    SalesDetailID INT,
    SaleAmount FLOAT,
    SaleQuantity INT,
    SaleUnitPrice FLOAT,
    SaleExtendedCost FLOAT,
    SaleTotalProfit FLOAT
) 