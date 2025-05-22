#!/usr/bin/env python3
"""
LocationLoader implementation for the Dim_Location dimension
"""
from sqlalchemy import Table, text, func
from ..schemas.dimension.location import DimLocation
from ..schemas.dimension_loader import DimensionLoader

class LocationLoader(DimensionLoader):
    """Loader for the Location dimension"""
    
    def load(self):
        """Load data from staging tables to the Location dimension"""
        print("\nLoading Dim_Location table...")
        
        # Ensure Unknown location exists
        unknown_loc = DimLocation(
            DimLocationID=1, 
            Address='Unknown', 
            City='Unknown', 
            PostalCode='Unknown', 
            State_Province='Unknown', 
            Country='Unknown'
        )
        print("DEBUG: Ensuring Unknown location record exists...")
        self.ensure_unknown_record(DimLocation, unknown_loc)
        
        # Reflect staging tables
        print("DEBUG: Reflecting staging tables...")
        staging_customer = Table('STAGING_CUSTOMER', self.staging_metadata, autoload_with=self.staging_engine)
        staging_store = Table('STAGING_STORE', self.staging_metadata, autoload_with=self.staging_engine)
        staging_reseller = Table('STAGING_RESELLER', self.staging_metadata, autoload_with=self.staging_engine)
        
        # Collect all unique locations from various sources
        print("DEBUG: Collecting unique locations from staging tables...")
        
        # Get customer locations
        customer_query = """
            SELECT DISTINCT 
                COALESCE(address, 'Unknown') as address,
                COALESCE(city, 'Unknown') as city, 
                COALESCE(postalcode, 'Unknown') as postalcode,
                COALESCE(stateprovince, 'Unknown') as state_province,
                COALESCE(country, 'Unknown') as country
            FROM STAGING_CUSTOMER
            WHERE address IS NOT NULL AND city IS NOT NULL AND country IS NOT NULL
        """
        customer_locations = self.execute_text_query(customer_query)
        print(f"DEBUG: Found {len(customer_locations)} customer locations")
        
        # Get store locations  
        store_query = """
            SELECT DISTINCT 
                COALESCE(address, 'Unknown') as address,
                COALESCE(city, 'Unknown') as city, 
                COALESCE(postalcode, 'Unknown') as postalcode,
                COALESCE(stateprovince, 'Unknown') as state_province,
                COALESCE(country, 'Unknown') as country
            FROM STAGING_STORE
            WHERE address IS NOT NULL AND city IS NOT NULL AND country IS NOT NULL
        """
        store_locations = self.execute_text_query(store_query)
        print(f"DEBUG: Found {len(store_locations)} store locations")
        
        # Get reseller locations
        reseller_query = """
            SELECT DISTINCT 
                COALESCE(address, 'Unknown') as address,
                COALESCE(city, 'Unknown') as city, 
                COALESCE(postalcode, 'Unknown') as postalcode,
                COALESCE(stateprovince, 'Unknown') as state_province,
                COALESCE(country, 'Unknown') as country
            FROM STAGING_RESELLER
            WHERE address IS NOT NULL AND city IS NOT NULL AND country IS NOT NULL
        """
        reseller_locations = self.execute_text_query(reseller_query)
        print(f"DEBUG: Found {len(reseller_locations)} reseller locations")
        
        # Combine all locations, tracking uniqueness by (address, city, country)
        print("DEBUG: Combining unique locations...")
        unique_locations = {}
        
        # Helper function to extract location attributes
        def extract_location(location):
            address = location.address if hasattr(location, 'address') else location[0]
            city = location.city if hasattr(location, 'city') else location[1]
            postal_code = location.postalcode if hasattr(location, 'postalcode') else location[2]
            state_province = location.state_province if hasattr(location, 'state_province') else location[3]
            country = location.country if hasattr(location, 'country') else location[4]
            
            # Ensure we have non-NULL values
            if not address:
                address = 'Unknown'
            if not city:
                city = 'Unknown'
            if not postal_code:
                postal_code = 'Unknown'
            if not state_province:
                state_province = 'Unknown'
            if not country:
                country = 'Unknown'
                
            return (address, city, postal_code, state_province, country)
        
        # Process customer locations
        for location in customer_locations:
            address, city, postal_code, state_province, country = extract_location(location)
            key = (address, city, country)
            unique_locations[key] = (address, city, postal_code, state_province, country)
        
        # Process store locations
        for location in store_locations:
            address, city, postal_code, state_province, country = extract_location(location)
            key = (address, city, country)
            unique_locations[key] = (address, city, postal_code, state_province, country)
        
        # Process reseller locations
        for location in reseller_locations:
            address, city, postal_code, state_province, country = extract_location(location)
            key = (address, city, country)
            unique_locations[key] = (address, city, postal_code, state_province, country)
        
        print(f"DEBUG: Found {len(unique_locations)} total unique locations")
        
        # Insert locations into DimLocation
        print("DEBUG: Inserting locations into DimLocation table...")
        
        # Create a fresh session for this load operation if needed, or rely on the one from init
        # self.create_fresh_session() # Already done in base loader, or manage transactions carefully
        
        new_locations = 0
        processed_count = 0

        for (address, city, country_key), (full_address, full_city, postal_code, state_province, full_country) in unique_locations.items():
            processed_count += 1
            try:
                if full_address == 'Unknown' and full_city == 'Unknown' and full_country == 'Unknown':
                    continue
                
                if not full_address or not full_city or not full_country:
                    print(f"Skipping invalid location data: addr={full_address}, city={full_city}, country={full_country}")
                    continue

                # Check if location already exists using ORM query
                existing_location = self.session.query(DimLocation).filter(
                    func.upper(DimLocation.Address) == func.upper(full_address),
                    func.upper(DimLocation.City) == func.upper(full_city),
                    func.upper(DimLocation.Country) == func.upper(full_country)
                ).first()

                if existing_location:
                    continue

                new_location_record = DimLocation(
                    Address=full_address,
                    City=full_city,
                    PostalCode=postal_code if postal_code else 'Unknown',
                    State_Province=state_province if state_province else 'Unknown',
                    Country=full_country
                )
                
                if self.add_record(new_location_record): # Using DimensionLoader's add_record
                    new_locations += 1
                else:
                    # Error already printed by add_record, a rollback might have occurred
                    # We might need to break or handle this more gracefully
                    print(f"Failed to add location: {full_address}, {full_city}, {full_country}. Check previous errors.")
                    # If add_record created a new session, this loop's subsequent existing_location checks might be on an old session.
                    # This part needs careful transaction management.
                    # For now, we assume add_record handles session state or we commit periodically.

                # Commit periodically or at the end
                if processed_count % 50 == 0: # Commit every 50 records
                    print(f"DEBUG: Committing batch of locations ({processed_count}/{len(unique_locations)} processed)")
                    if not self.commit_records():
                        print("Failed to commit batch of locations. Aborting this loader.")
                        # Decide on error handling: break, return, or raise
                        return self.get_row_count('DIM_LOCATION') # Return current count

            except Exception as e:
                print(f"DEBUG ERROR: General error processing location ({full_address}, {full_city}, {full_country}): {e}")
                self.session.rollback() # Rollback on general error
                self.create_fresh_session() # Get a new session to continue if possible

        if not self.commit_records(): # Commit any remaining records
             print("Failed to commit final batch of locations.")

        print(f"DEBUG: Added {new_locations} new locations")
        
        location_count = self.get_row_count('DIM_LOCATION')
        print(f"Loaded {location_count} locations into Dim_Location")
        return location_count 