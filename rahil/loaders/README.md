# Dimension Table Loaders

This module contains loader classes that are responsible for loading data from staging tables to dimension tables.
Each loader is implemented in a separate file and follows a common interface defined in the `DimensionLoader` base class.

## Key Features

- Separation of loader logic from schema definitions
- Consistent handling of Snowflake case sensitivity issues
- Proper connection management to avoid resource leaks
- Error handling with detailed logging
- Use of SQLAlchemy ORM to leverage dimension model definitions

## Loaders

- **LocationLoader**: Loads location data from customer, store and reseller tables
- **ChannelLoader**: Loads channel data from channel and channel category tables
- **CustomerLoader**: Loads customer data and links to locations
- **ResellerLoader**: Loads reseller data and links to locations
- **StoreLoader**: Loads store data and links to locations
- **ProductLoader**: Loads product data and calculates profit metrics 