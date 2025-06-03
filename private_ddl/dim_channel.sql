-- Dim_Channel table
CREATE OR REPLACE TABLE Dim_Channel (
    DimChannelID INT IDENTITY(1,1) PRIMARY KEY,
    ChannelID INT,
    ChannelCategoryID INT,
    ChannelName VARCHAR(255),
    ChannelCategory VARCHAR(255)
) 