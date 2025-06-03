-- Dim_Reseller table
CREATE OR REPLACE TABLE Dim_Reseller (
    DimResellerID INT IDENTITY(1,1) PRIMARY KEY,
    ResellerID VARCHAR(255),
    DimLocationID INT,
    ResellerName VARCHAR(255),
    ContactName VARCHAR(255),
    PhoneNumber VARCHAR(255),
    Email VARCHAR(255)
) 