-- Example template for staging tables
-- Replace with your actual column definitions
CREATE OR REPLACE TABLE STAGING_EXAMPLE (
    -- Primary key
    ID                  INTEGER,
    
    -- Reference keys
    FOREIGN_KEY_ID      INTEGER,
    
    -- Descriptive fields
    NAME                VARCHAR,
    DESCRIPTION         VARCHAR,
    
    -- Numeric values
    AMOUNT              FLOAT,
    QUANTITY            INTEGER,
    
    -- Dates and timestamps
    DATE                DATE,
    
    -- Audit fields
    CREATEDDATE         VARCHAR,
    CREATEDBY           VARCHAR,
    MODIFIEDDATE        VARCHAR,
    MODIFIEDBY          VARCHAR
); 