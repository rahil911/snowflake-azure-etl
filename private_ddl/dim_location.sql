-- Dim_Location table
CREATE OR REPLACE TABLE Dim_Location (
    DimLocationID INT IDENTITY(1,1) PRIMARY KEY,
    Address VARCHAR(255),
    City VARCHAR(255),
    PostalCode VARCHAR(255),
    State_Province VARCHAR(255),
    Country VARCHAR(255)
) 