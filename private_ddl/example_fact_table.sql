-- Example template for fact tables
-- Replace with your actual column definitions
CREATE OR REPLACE TABLE Fact_Example (
    -- Foreign keys to dimension tables
    DimEntity1ID        INT,
    DimEntity2ID        INT,
    DimEntity3ID        INT,
    DimDateID           INT,
    
    -- Source system keys
    SourceSystem1ID     INT,
    SourceSystem2ID     INT,
    
    -- Measures
    Amount              FLOAT,
    Quantity            INT,
    UnitPrice           FLOAT,
    ExtendedCost        FLOAT,
    TotalProfit         FLOAT
); 