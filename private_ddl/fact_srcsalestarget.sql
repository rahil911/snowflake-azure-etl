-- Fact_SRCSalesTarget table (Store/Reseller/Channel Sales Target)
CREATE OR REPLACE TABLE Fact_SRCSalesTarget (
    DimStoreID INT,
    DimResellerID INT,
    DimChannelID INT,
    DimTargetDateID INT,
    SalesTargetAmount FLOAT
) 