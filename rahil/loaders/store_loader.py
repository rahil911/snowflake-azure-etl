#!/usr/bin/env python3
"""
StoreLoader implementation for the Dim_Store dimension
"""
from sqlalchemy import Table, text
from ..schemas.dimension.store import DimStore
from ..schemas.dimension.location import DimLocation
from ..schemas.dimension_loader import DimensionLoader

class StoreLoader(DimensionLoader):
    """Loader for the Store dimension"""
    
    def load(self):
        """Load data from staging tables to the Store dimension"""
        print("\nLoading Dim_Store table...")
        
        # Ensure Unknown store exists
        unknown_store = DimStore(
            DimStoreID=1,
            StoreID=-1,
            DimLocationID=1,
            SourceStoreID=-1,
            StoreName='Unknown Store',
            StoreNumber='Unknown',
            StoreManager='Unknown'
        )
        print("DEBUG: Ensuring Unknown store record exists...")
        self.ensure_unknown_record(DimStore, unknown_store)
        
        # Reflect staging tables
        print("DEBUG: Reflecting tables...")
        staging_store = Table('STAGING_STORE', self.staging_metadata, autoload_with=self.staging_engine)
        location_table = Table('DIM_LOCATION', self.dimension_metadata, autoload_with=self.dim_engine)
        
        # Get column names for logging
        store_columns = [col.name for col in staging_store.columns]
        location_columns = [col.name for col in location_table.columns]
        print(f"Store table columns: {store_columns}")
        print(f"Location table columns: {location_columns}")
        
        # Get all stores using text query to avoid case sensitivity
        print("DEBUG: Fetching all stores...")
        query = """
            SELECT 
                storeid,
                COALESCE(storenumber, 'Unknown') as storenumber,
                COALESCE(storemanager, 'Unknown') as storemanager,
                COALESCE(address, 'Unknown') as address,
                COALESCE(city, 'Unknown') as city,
                COALESCE(country, 'Unknown') as country
            FROM STAGING_STORE
        """
        stores = self.execute_text_query(query)
        print(f"DEBUG: Found {len(stores)} stores")
        
        # Create a fresh session
        self.create_fresh_session()
        
        # Process each store
        print("DEBUG: Processing stores...")
        stores_added = 0
        
        for store in stores:
            try:
                # Extract store attributes
                store_id = int(store.storeid) if hasattr(store, 'storeid') else int(store[0])
                
                # Check if store already exists
                existing_query = text("""
                    SELECT COUNT(*) FROM "DIM_STORE" 
                    WHERE "StoreID" = :store_id
                """)
                
                with self.dim_engine.connect() as conn:
                    result = conn.execute(existing_query, {"store_id": store_id})
                    if result.scalar() > 0:
                        continue  # Skip if store already exists
                
                # Get store details
                store_number = store.storenumber if hasattr(store, 'storenumber') else store[1]
                store_manager = store.storemanager if hasattr(store, 'storemanager') else store[2]
                address = store.address if hasattr(store, 'address') else store[3]
                city = store.city if hasattr(store, 'city') else store[4]
                country = store.country if hasattr(store, 'country') else store[5]
                
                # Generate store name
                store_name = f"Store {store_number}" if store_number else "Unknown Store"
                
                # Look up location ID using the helper method
                location_id = self.lookup_location_id(address, city, country)
                
                # Create new store record
                new_store = DimStore(
                    StoreID=store_id,
                    DimLocationID=location_id,
                    SourceStoreID=store_id,  # Use the same ID as source for simplicity
                    StoreName=store_name,
                    StoreNumber=store_number if store_number else "Unknown",
                    StoreManager=store_manager if store_manager else "Unknown"
                )
                
                if self.add_record(new_store):
                    stores_added += 1
                
                # Commit in small batches to avoid transaction issues
                if stores_added % 10 == 0:
                    self.commit_records()
                    self.create_fresh_session()
                    
            except Exception as e:
                print(f"DEBUG ERROR: Error adding store {store_id if 'store_id' in locals() else 'Unknown'}: {e}")
                self.create_fresh_session()  # Create a fresh session after error
                continue
        
        # Commit any remaining records
        print(f"DEBUG: Committing {stores_added} stores")
        self.commit_records()
        
        # Get final counts
        store_count = self.get_row_count('DIM_STORE')
        print(f"Loaded {store_count} stores into Dim_Store")
        return store_count 