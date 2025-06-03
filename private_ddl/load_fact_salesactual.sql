-- Load Fact_SalesActual table
INSERT INTO {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Fact_SalesActual (
    DimProductID, DimStoreID, DimResellerID, DimCustomerID, DimChannelID, 
    DimSaleDateID, DimLocationID, SalesHeaderID, SalesDetailID, 
    SaleAmount, SaleQuantity, SaleUnitPrice, SaleExtendedCost, SaleTotalProfit
)
SELECT 
    COALESCE(dp.DimProductID, 1) as DimProductID,  -- Default to Unknown Product
    COALESCE(ds.DimStoreID, 1) as DimStoreID,      -- Default to Unknown Store  
    COALESCE(dr.DimResellerID, 1) as DimResellerID, -- Default to Unknown Reseller
    COALESCE(dc.DimCustomerID, 1) as DimCustomerID, -- Default to Unknown Customer
    COALESCE(dch.DimChannelID, 1) as DimChannelID,  -- Default to Unknown Channel
    COALESCE(dd.DATE_PKEY, 20130101) as DimSaleDateID, -- Default to base date
    1 as DimLocationID, -- Default to Unknown Location
    sh.SalesHeaderID,
    sd.SalesDetailID,
    COALESCE(CAST(sd.SalesAmount AS FLOAT), 0) as SaleAmount,
    COALESCE(CAST(sd.SalesQuantity AS INT), 0) as SaleQuantity,
    COALESCE(CAST(sd.SalesAmount AS FLOAT), 0) / NULLIF(COALESCE(CAST(sd.SalesQuantity AS INT), 1), 0) as SaleUnitPrice,
    COALESCE(CAST(sd.SalesQuantity AS INT), 0) * COALESCE(CAST(dp.ProductCost AS FLOAT), 0) as SaleExtendedCost,
    COALESCE(CAST(sd.SalesAmount AS FLOAT), 0) - (COALESCE(CAST(sd.SalesQuantity AS INT), 0) * COALESCE(CAST(dp.ProductCost AS FLOAT), 0)) as SaleTotalProfit
FROM {staging_db}.{SNOWFLAKE_SCHEMA}.STAGING_SALESDETAIL sd
JOIN {staging_db}.{SNOWFLAKE_SCHEMA}.STAGING_SALESHEADER sh ON sd.SalesHeaderID = sh.SalesHeaderID
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Product dp ON sd.ProductID = dp.ProductID
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Store ds ON sh.StoreID = ds.StoreID
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Reseller dr ON sh.ResellerID = dr.ResellerID
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Customer dc ON sh.CustomerID = dc.CustomerID
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.Dim_Channel dch ON sh.ChannelID = dch.ChannelID
LEFT JOIN {DIMENSION_DB_NAME}.{SNOWFLAKE_SCHEMA}.DIM_DATE dd ON TO_CHAR(TRY_TO_DATE(sh.DATE, 'YYYY-MM-DD'), 'YYYYMMDD') = CAST(dd.DATE_PKEY AS VARCHAR)
WHERE sd.SalesDetailID IS NOT NULL
  AND sh.SalesHeaderID IS NOT NULL 