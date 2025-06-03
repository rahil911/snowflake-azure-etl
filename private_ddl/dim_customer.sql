-- Dim_Customer table
CREATE OR REPLACE TABLE Dim_Customer (
    DimCustomerID INT IDENTITY(1,1) PRIMARY KEY,
    CustomerID VARCHAR(255),
    DimLocationID INT,
    CustomerFullName VARCHAR(255),
    CustomerFirstName VARCHAR(255),
    CustomerLastName VARCHAR(255),
    CustomerGender VARCHAR(255)
) 