-- Load Fact_ProductSalesTarget table
INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_ProductSalesTarget (
    DimProductID, DimTargetDateID, ProductTargetSalesQuantity
)
SELECT 
    COALESCE(dp.DimProductID, 1) as DimProductID,  -- Default to Unknown Product
    COALESCE(dd.DATE_PKEY, 20130101) as DimTargetDateID, -- Default to base date
    COALESCE(CAST(tdp.SalesQuantityTarget AS INT), 0) as ProductTargetSalesQuantity
FROM {staging_db}.{SNOWFLAKE_SCHEMA}.STAGING_TARGETDATAPRODUCT tdp
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Product dp ON tdp.ProductID = dp.ProductID
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.DIM_DATE dd ON dd.YEAR = tdp.YEAR AND dd.MONTH_NUM_IN_YEAR = 1 AND dd.DAY_NUM_IN_MONTH = 1
WHERE tdp.ProductID IS NOT NULL
  AND tdp.SalesQuantityTarget IS NOT NULL 