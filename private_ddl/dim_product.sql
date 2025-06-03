-- Dim_Product table
CREATE OR REPLACE TABLE Dim_Product (
    DimProductID INT IDENTITY(1,1) PRIMARY KEY,
    ProductID INT,
    ProductTypeID INT,
    ProductCategoryID INT,
    ProductName VARCHAR(255),
    ProductType VARCHAR(255),
    ProductCategory VARCHAR(255),
    ProductRetailPrice FLOAT,
    ProductWholesalePrice FLOAT,
    ProductCost FLOAT,
    ProductRetailProfit FLOAT,
    ProductWholesaleUnitProfit FLOAT,
    ProductProfitMarginUnitPercent FLOAT
) 