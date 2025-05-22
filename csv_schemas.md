# CSV Schemas

## Channel.csv
- `ChannelID`
- `ChannelCategoryID`
- `Channel`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`

## ChannelCategory.csv
- `ChannelCategoryID`
- `ChannelCategory`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`

## Customer.csv
- `CustomerID`
- `SubSegmentID`
- `FirstName`
- `LastName`
- `Gender`
- `EmailAddress`
- `Address`
- `City`
- `StateProvince`
- `Country`
- `PostalCode`
- `PhoneNumber`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`

## Product.csv
- `ProductID`
- `ProductTypeID`
- `Product`
- `Color`
- `Style`
- `UnitofMeasureID`
- `Weight`
- `Price`
- `Cost`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`
- `WholesalePrice`

## ProductCategory.csv
- `ProductCategoryID`
- `ProductCategory`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`

## ProductType.csv
- `ProductTypeID`
- `ProductCategoryID`
- `ProductType`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`

## Reseller.csv
- `ResellerID`
- `Contact`
- `EmailAddress`
- `Address`
- `City`
- `StateProvince`
- `Country`
- `PostalCode`
- `PhoneNumber`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`
- `ResellerName`

## SalesDetail.csv
- `SalesDetailID`
- `SalesHeaderID`
- `ProductID`
- `SalesQuantity`
- `SalesAmount`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`

## SalesHeader_New.csv
- `SalesHeaderID`
- `Date`
- `ChannelID`
- `StoreID`
- `CustomerID`
- `ResellerID`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`

## Store.csv
- `StoreID`
- `SubSegmentID`
- `StoreNumber`
- `StoreManager`
- `Address`
- `City`
- `StateProvince`
- `Country`
- `PostalCode`
- `PhoneNumber`
- `CreatedDate`
- `CreatedBy`
- `ModifiedDate`
- `ModifiedBy`

## Target Data - Channel Reseller and Store.csv
- `Year`
- `ChannelName`
- `TargetName`
- `TargetSalesAmount`

## Target Data - Product.csv
- `ProductID`
- `Product`
- `Year`
- `SalesQuantityTarget`

