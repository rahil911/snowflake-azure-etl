-- Load Fact_SRCSalesTarget table (Store/Reseller/Channel Sales Target)
INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_SRCSalesTarget (
    DimStoreID, DimResellerID, DimChannelID, DimTargetDateID, SalesTargetAmount
)
SELECT 
    1 as DimStoreID,      -- Default to Unknown Store (no store mapping available)
    1 as DimResellerID,   -- Default to Unknown Reseller (no reseller mapping available)
    COALESCE(dc.DimChannelID, 1) as DimChannelID,   -- Map by channel name
    COALESCE(dd.DATE_PKEY, 20130101) as DimTargetDateID, -- Default to base date
    COALESCE(CAST(tdc.TargetSalesAmount AS FLOAT), 0) as SalesTargetAmount
FROM {staging_db}.{SNOWFLAKE_SCHEMA}.STAGING_TARGETDATACHANNEL tdc
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Channel dc ON tdc.ChannelName = dc.ChannelName
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.DIM_DATE dd ON dd.YEAR = tdc.YEAR AND dd.MONTH_NUM_IN_YEAR = 1 AND dd.DAY_NUM_IN_MONTH = 1
WHERE tdc.ChannelName IS NOT NULL
  AND tdc.TargetSalesAmount IS NOT NULL 