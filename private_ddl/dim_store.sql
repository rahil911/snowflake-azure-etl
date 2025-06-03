-- Dim_Store table
CREATE OR REPLACE TABLE Dim_Store (
    DimStoreID INT IDENTITY(1,1) PRIMARY KEY,
    StoreID INT,
    DimLocationID INT,
    SourceStoreID INT,
    StoreName VARCHAR(255),
    StoreNumber VARCHAR(255),
    StoreManager VARCHAR(255)
) 