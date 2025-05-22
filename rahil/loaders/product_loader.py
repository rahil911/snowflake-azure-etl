#!/usr/bin/env python3
"""
ProductLoader implementation for the Dim_Product dimension
"""
from sqlalchemy import Table, text
from ..schemas.dimension.product import DimProduct
from ..schemas.dimension_loader import DimensionLoader

class ProductLoader(DimensionLoader):
    """Loader for the Product dimension"""
    
    def load(self):
        """Load data from staging tables to the Product dimension"""
        print("\nLoading Dim_Product table...")
        
        # Ensure Unknown product exists
        unknown_product = DimProduct(
            DimProductID=1,
            ProductID=-1,
            ProductTypeID=-1,
            ProductCategoryID=-1,
            ProductName='Unknown Product',
            ProductType='Unknown',
            ProductCategory='Unknown',
            ProductRetailPrice=0,
            ProductWholesalePrice=0,
            ProductCost=0,
            ProductRetailProfit=0,
            ProductWholesaleUnitProfit=0,
            ProductProfitMarginUnitPercent=0
        )
        print("DEBUG: Ensuring Unknown product record exists...")
        self.ensure_unknown_record(DimProduct, unknown_product)
        
        # Reflect tables
        print("DEBUG: Reflecting tables...")
        staging_product = Table('STAGING_PRODUCT', self.staging_metadata, autoload_with=self.staging_engine)
        staging_product_type = Table('STAGING_PRODUCTTYPE', self.staging_metadata, autoload_with=self.staging_engine)
        staging_product_category = Table('STAGING_PRODUCTCATEGORY', self.staging_metadata, autoload_with=self.staging_engine)
        
        print(f"Product table columns: {[c.name for c in staging_product.columns]}")
        print(f"Product type table columns: {[c.name for c in staging_product_type.columns]}")
        print(f"Product category table columns: {[c.name for c in staging_product_category.columns]}")
        
        # Get products data using text query
        print("DEBUG: Fetching all products...")
        products = self.execute_text_query("""
            SELECT 
                p.productid AS ProductID,
                p.producttypeid AS ProductTypeID,
                pt.productcategoryid AS ProductCategoryID,
                p.product AS ProductName,
                pt.producttype AS ProductType,
                pc.productcategory AS ProductCategory,
                COALESCE(p.price, 0) AS Price,
                COALESCE(p.wholesaleprice, 0) AS WholesalePrice,
                COALESCE(p.cost, 0) AS Cost
            FROM STAGING_PRODUCT p
            JOIN STAGING_PRODUCTTYPE pt ON p.producttypeid = pt.producttypeid
            JOIN STAGING_PRODUCTCATEGORY pc ON pt.productcategoryid = pc.productcategoryid
        """)
        
        print(f"DEBUG: Found {len(products)} products")
        products_added = 0
        
        # Process each product
        print("DEBUG: Processing products...")
        for product in products:
            try:
                # Extract product attributes safely
                product_id = product.ProductID if hasattr(product, 'ProductID') else product[0]
                product_type_id = product.ProductTypeID if hasattr(product, 'ProductTypeID') else product[1]
                product_category_id = product.ProductCategoryID if hasattr(product, 'ProductCategoryID') else product[2]
                product_name = product.ProductName if hasattr(product, 'ProductName') else product[3]
                product_type = product.ProductType if hasattr(product, 'ProductType') else product[4]
                product_category = product.ProductCategory if hasattr(product, 'ProductCategory') else product[5]
                product_price = float(product.Price if hasattr(product, 'Price') else product[6])
                product_wholesale_price = float(product.WholesalePrice if hasattr(product, 'WholesalePrice') else product[7])
                product_cost = float(product.Cost if hasattr(product, 'Cost') else product[8])
                
                # Calculate profit metrics
                retail_profit = product_price - product_cost
                wholesale_profit = product_wholesale_price - product_cost
                
                # Calculate profit margin percent
                profit_margin_percent = 0
                if product_price > 0:
                    profit_margin_percent = (retail_profit / product_price) * 100
                
                # Check if product exists
                existing = self.session.query(DimProduct).filter(
                    DimProduct.ProductID == product_id
                ).first()
                
                if not existing:
                    self.session.add(DimProduct(
                        ProductID=product_id,
                        ProductTypeID=product_type_id,
                        ProductCategoryID=product_category_id,
                        ProductName=product_name or 'Unknown',
                        ProductType=product_type or 'Unknown',
                        ProductCategory=product_category or 'Unknown',
                        ProductRetailPrice=product_price,
                        ProductWholesalePrice=product_wholesale_price,
                        ProductCost=product_cost,
                        ProductRetailProfit=retail_profit,
                        ProductWholesaleUnitProfit=wholesale_profit,
                        ProductProfitMarginUnitPercent=profit_margin_percent
                    ))
                    products_added += 1
            except Exception as e:
                print(f"DEBUG ERROR: Error adding product {getattr(product, 'ProductID', 'Unknown')}: {e}")
        
        print(f"DEBUG: Committing {products_added} products")
        self.commit_records()
        
        product_count = self.get_row_count('DIM_PRODUCT')
        print(f"Loaded {product_count} products into Dim_Product")
        return product_count 