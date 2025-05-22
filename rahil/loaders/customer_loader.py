#!/usr/bin/env python3
"""
CustomerLoader implementation for the Dim_Customer dimension
"""
from sqlalchemy import Table, text
from ..schemas.dimension.customer import DimCustomer
from ..schemas.dimension.location import DimLocation
from ..schemas.dimension_loader import DimensionLoader

class CustomerLoader(DimensionLoader):
    """Loader for the Customer dimension"""
    
    def load(self):
        """Load data from staging tables to the Customer dimension"""
        print("\nLoading Dim_Customer table...")
        
        # Ensure Unknown customer exists
        unknown_customer = DimCustomer(
            DimCustomerID=1,
            CustomerID='UNKNOWN',
            DimLocationID=1,
            CustomerFullName='Unknown Customer',
            CustomerFirstName='Unknown',
            CustomerLastName='Unknown',
            CustomerGender='Unknown'
        )
        print("DEBUG: Ensuring Unknown customer record exists...")
        self.ensure_unknown_record(DimCustomer, unknown_customer)
        
        # Reflect staging tables
        print("DEBUG: Reflecting tables...")
        staging_customer = Table('STAGING_CUSTOMER', self.staging_metadata, autoload_with=self.staging_engine)
        location_table = Table('DIM_LOCATION', self.dimension_metadata, autoload_with=self.dim_engine)
        
        # Get column names for logging
        customer_columns = [col.name for col in staging_customer.columns]
        location_columns = [col.name for col in location_table.columns]
        print(f"Customer table columns: {customer_columns}")
        print(f"Location table columns: {location_columns}")
        
        # Get all customers using text query to avoid case sensitivity
        print("DEBUG: Fetching all customers...")
        customers = self.execute_text_query("""
            SELECT 
                customerid, 
                COALESCE(firstname, 'Unknown') as firstname, 
                COALESCE(lastname, 'Unknown') as lastname, 
                COALESCE(gender, 'Unknown') as gender,
                COALESCE(address, 'Unknown') as address,
                COALESCE(city, 'Unknown') as city,
                COALESCE(country, 'Unknown') as country
            FROM STAGING_CUSTOMER
        """)
        print(f"DEBUG: Found {len(customers)} customers")
        
        # Create a fresh session for adding customers
        self.create_fresh_session()
        
        # Process each customer
        print("DEBUG: Processing customers...")
        customers_added = 0
        
        for customer in customers:
            try:
                # Check if customer already exists
                customer_id = customer.customerid if hasattr(customer, 'customerid') else customer[0]
                existing_query = text("""
                    SELECT COUNT(*) FROM "DIM_CUSTOMER" 
                    WHERE "CustomerID" = :customer_id
                """)
                
                with self.dim_engine.connect() as conn:
                    result = conn.execute(existing_query, {"customer_id": customer_id})
                    if result.scalar() > 0:
                        continue  # Skip if customer already exists
                
                # Look up location ID - use location lookup function
                address = customer.address if hasattr(customer, 'address') else customer[4]
                city = customer.city if hasattr(customer, 'city') else customer[5]
                country = customer.country if hasattr(customer, 'country') else customer[6]
                location_id = self.lookup_location_id(address, city, country)
                
                # Create name components
                firstname = customer.firstname if hasattr(customer, 'firstname') else customer[1]
                lastname = customer.lastname if hasattr(customer, 'lastname') else customer[2]
                fullname = f"{firstname} {lastname}".strip()
                if fullname == "":
                    fullname = "Unknown Customer"
                    
                gender = customer.gender if hasattr(customer, 'gender') else customer[3]
                if not gender:
                    gender = "Unknown"
                
                # Add new customer
                new_customer = DimCustomer(
                    CustomerID=customer_id,
                    DimLocationID=location_id,
                    CustomerFullName=fullname,
                    CustomerFirstName=firstname if firstname else "Unknown",
                    CustomerLastName=lastname if lastname else "Unknown",
                    CustomerGender=gender
                )
                
                if self.add_record(new_customer):
                    customers_added += 1
                
                # Commit in small batches to avoid transaction issues
                if customers_added % 10 == 0:
                    self.commit_records()
                    
            except Exception as e:
                print(f"DEBUG ERROR: Error adding customer {customer_id if 'customer_id' in locals() else 'Unknown'}: {e}")
                self.create_fresh_session()  # Create a fresh session after error
                continue
        
        # Commit any remaining records
        print(f"DEBUG: Committing {customers_added} customers")
        self.commit_records()
        
        # Get final counts
        customer_count = self.get_row_count('DIM_CUSTOMER')
        print(f"Loaded {customer_count} customers into Dim_Customer")
        return customer_count 