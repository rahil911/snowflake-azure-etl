-- Fact_ProductSalesTarget table
CREATE OR REPLACE TABLE Fact_ProductSalesTarget (
    DimProductID INT,
    DimTargetDateID INT,
    ProductTargetSalesQuantity INT
) 