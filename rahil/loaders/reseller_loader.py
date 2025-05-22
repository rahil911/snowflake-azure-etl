#!/usr/bin/env python3
"""
ResellerLoader implementation for the Dim_Reseller dimension
"""
from sqlalchemy import Table, text
from ..schemas.dimension.reseller import DimReseller
from ..schemas.dimension.location import DimLocation
from ..schemas.dimension_loader import DimensionLoader

class ResellerLoader(DimensionLoader):
    """Loader for the Reseller dimension"""
    
    def load(self):
        """Load data from staging tables to the Reseller dimension"""
        print("\nLoading Dim_Reseller table...")
        
        # Ensure Unknown reseller exists
        unknown_reseller = DimReseller(
            DimResellerID=1,
            ResellerID='UNKNOWN',
            DimLocationID=1,
            ResellerName='Unknown Reseller',
            ContactName='Unknown',
            PhoneNumber='Unknown',
            Email='Unknown'
        )
        print("DEBUG: Ensuring Unknown reseller record exists...")
        self.ensure_unknown_record(DimReseller, unknown_reseller)
        
        # Reflect staging tables
        print("DEBUG: Reflecting tables...")
        staging_reseller = Table('STAGING_RESELLER', self.staging_metadata, autoload_with=self.staging_engine)
        location_table = Table('DIM_LOCATION', self.dimension_metadata, autoload_with=self.dim_engine)
        
        # Get column names for logging
        reseller_columns = [col.name for col in staging_reseller.columns]
        location_columns = [col.name for col in location_table.columns]
        print(f"Reseller table columns: {reseller_columns}")
        print(f"Location table columns: {location_columns}")
        
        # Get all resellers using text query to avoid case sensitivity
        print("DEBUG: Building reseller query...")
        query = """
            SELECT 
                resellerid, 
                COALESCE(resellername, 'Unknown Reseller') as resellername,
                COALESCE(contact, 'Unknown') as contact, 
                COALESCE(phonenumber, 'Unknown') as phonenumber,
                COALESCE(emailaddress, 'Unknown') as emailaddress,
                COALESCE(address, 'Unknown') as address,
                COALESCE(city, 'Unknown') as city,
                COALESCE(country, 'Unknown') as country
            FROM STAGING_RESELLER
        """
        resellers = self.execute_text_query(query)
        print(f"DEBUG: Found {len(resellers)} resellers")
        
        # Create a fresh session
        self.create_fresh_session()
        
        # Process each reseller
        print("DEBUG: Processing resellers...")
        resellers_added = 0
        
        for reseller in resellers:
            try:
                # Extract reseller attributes safely
                reseller_id = reseller.resellerid if hasattr(reseller, 'resellerid') else reseller[0]
                
                # Check if reseller already exists
                existing_query = text("""
                    SELECT COUNT(*) FROM "DIM_RESELLER" 
                    WHERE "ResellerID" = :reseller_id
                """)
                
                with self.dim_engine.connect() as conn:
                    result = conn.execute(existing_query, {"reseller_id": reseller_id})
                    if result.scalar() > 0:
                        continue  # Skip if reseller already exists
                
                # Get reseller details
                reseller_name = reseller.resellername if hasattr(reseller, 'resellername') else reseller[1]
                contact_name = reseller.contact if hasattr(reseller, 'contact') else reseller[2]
                phone_number = reseller.phonenumber if hasattr(reseller, 'phonenumber') else reseller[3]
                email = reseller.emailaddress if hasattr(reseller, 'emailaddress') else reseller[4]
                address = reseller.address if hasattr(reseller, 'address') else reseller[5]
                city = reseller.city if hasattr(reseller, 'city') else reseller[6]
                country = reseller.country if hasattr(reseller, 'country') else reseller[7]
                
                # Look up location ID using the helper method
                location_id = self.lookup_location_id(address, city, country)
                
                # Create new reseller record
                new_reseller = DimReseller(
                    ResellerID=reseller_id,
                    DimLocationID=location_id,
                    ResellerName=reseller_name if reseller_name else 'Unknown Reseller',
                    ContactName=contact_name if contact_name else 'Unknown',
                    PhoneNumber=phone_number if phone_number else 'Unknown',
                    Email=email if email else 'Unknown'
                )
                
                if self.add_record(new_reseller):
                    resellers_added += 1
                
                # Commit in small batches to avoid transaction issues
                if resellers_added % 10 == 0:
                    self.commit_records()
                    self.create_fresh_session()
                    
            except Exception as e:
                print(f"DEBUG ERROR: Error adding reseller {reseller_id if 'reseller_id' in locals() else 'Unknown'}: {e}")
                self.create_fresh_session()  # Create a fresh session after error
                continue
        
        # Commit any remaining records
        print(f"DEBUG: Committing {resellers_added} resellers")
        self.commit_records()
        
        # Get final counts
        reseller_count = self.get_row_count('DIM_RESELLER')
        print(f"Loaded {reseller_count} resellers into Dim_Reseller")
        return reseller_count 