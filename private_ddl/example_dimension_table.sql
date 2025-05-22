-- Example template for dimension tables
-- Replace with your actual column definitions
CREATE OR REPLACE TABLE Dim_Example (
    -- Surrogate key
    DimExampleID        INT IDENTITY(1,1) PRIMARY KEY,
    
    -- Business keys
    SourceID            INT,
    
    -- Foreign keys to other dimensions
    DimForeignKeyID     INT,
    
    -- Descriptive attributes
    ExampleName         VARCHAR(255),
    ExampleType         VARCHAR(255),
    ExampleCategory     VARCHAR(255),
    
    -- Measures or calculated fields
    ExampleMetric1      FLOAT,
    ExampleMetric2      FLOAT
); 