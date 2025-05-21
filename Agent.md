<!--
# Agent Prompting Instructions
- Think step-by-step using the plan tasks as a guideline.
- Use appropriate code edit tools for file modifications.
- Write clear and concise commit messages.
- Validate tasks before and after execution.
- Ask clarifying questions when tasks are unclear.
-->


# Schema Management Enhancement Plan

## 1. Setup Alembic Integration ✅
- [x] Install Alembic and its dependencies (`pip install alembic sqlalchemy`)
- [x] Initialize Alembic in project structure (`alembic init migrations`)
- [x] Configure Alembic environment (`alembic.ini` and `env.py`)
- [x] Update `.gitignore` to exclude user-specific schema files

## 2. Create Schema Definition Structure ✅
- [x] Create a `schemas` directory to store table definitions
- [x] Create a base schema definition template with SQLAlchemy models
- [x] Move table schemas from `create_tables.py` into separate Python modules:
  - [x] Extract Channel schema to `schemas/channel.py`
  - [x] Extract ChannelCategory schema to `schemas/channel_category.py`
  - [x] Extract Customer schema to `schemas/customer.py`
  - [x] Extract Product schema to `schemas/product.py`
  - [x] Extract ProductCategory schema to `schemas/product_category.py`
  - [x] Extract ProductType schema to `schemas/product_type.py`
  - [x] Extract Reseller schema to `schemas/reseller.py`
  - [x] Extract SalesDetail schema to `schemas/sales_detail.py`
  - [x] Extract SalesHeader schema to `schemas/sales_header.py`
  - [x] Extract Store schema to `schemas/store.py`
  - [x] Extract TargetDataChannel schema to `schemas/target_data_channel.py`
  - [x] Extract TargetDataProduct schema to `schemas/target_data_product.py`

## 3. Create SQLAlchemy Model Definitions ✅
- [x] Define SQLAlchemy Base class in `schemas/__init__.py`
- [x] For each entity, create SQLAlchemy model with:
  - [x] Table name matching existing staging table names
  - [x] Column definitions with appropriate data types
  - [x] Primary key and foreign key relationships
  - [x] Indexes and constraints as needed

## 4. Update Table Creation Process ✅
- [x] Modify `create_tables.py` to use the SQLAlchemy models instead of hardcoded SQL
- [x] Create utility to generate table creation SQL from SQLAlchemy models
- [x] Implement option to use either SQLAlchemy or native Snowflake SQL

## 5. Implement Schema Versioning ✅
- [x] Create initial Alembic migration (`alembic revision --autogenerate -m "initial"`)
- [x] Update `run_etl.py` to run Alembic migrations as part of the process
- [x] Add command-line option to upgrade or downgrade schema versions

## 6. Create Schema Customization Mechanism ✅
- [x] Add support for user-specific schema overrides in `.schema_customizations/`
- [x] Implement mechanism to merge base schemas with user customizations
- [x] Update `.gitignore` to exclude the `.schema_customizations/` directory

## 7. Update Dimension Model ETL
- [ ] Create dimension schema directory structure
  - [ ] Create `schemas/dimension` directory for dimension tables
  - [ ] Create `schemas/fact` directory for fact tables
  - [ ] Update `schemas/__init__.py` to import dimension and fact models
- [ ] Extract dimension table definitions to SQLAlchemy models
  - [ ] Extract Dim_Date schema to `schemas/dimension/date.py`
  - [ ] Extract Dim_Product schema to `schemas/dimension/product.py`
  - [ ] Extract Dim_Store schema to `schemas/dimension/store.py`
  - [ ] Extract Dim_Reseller schema to `schemas/dimension/reseller.py`
  - [ ] Extract Dim_Location schema to `schemas/dimension/location.py`
  - [ ] Extract Dim_Customer schema to `schemas/dimension/customer.py`
  - [ ] Extract Dim_Channel schema to `schemas/dimension/channel.py`
- [ ] Extract fact table definitions to SQLAlchemy models
  - [ ] Extract Fact_SalesActual schema to `schemas/fact/sales_actual.py`
  - [ ] Extract Fact_ProductSalesTarget schema to `schemas/fact/product_sales_target.py`
  - [ ] Extract Fact_SRCSalesTarget schema to `schemas/fact/src_sales_target.py`
- [ ] Update dimension model creation process
  - [ ] Refactor `create_dimension_tables.py` to use SQLAlchemy models
  - [ ] Refactor `create_fact_tables.py` to use SQLAlchemy models
  - [ ] Create seed data module to handle "Unknown" member insertion
  - [ ] Implement relationships between fact and dimension tables
- [ ] Create migrations for dimension and fact tables
  - [ ] Create Alembic migration for dimension tables
  - [ ] Create Alembic migration for fact tables
  - [ ] Update `run_dimensional_etl.py` to use migrations for table creation

## 8. Documentation Updates
- [ ] Document the schema definition approach in unified README.md
  - [ ] Merge README.md and DIMENSION_README.md
  - [ ] Document how to use SQLAlchemy for table definitions
  - [ ] Explain the schema customization process for students
  - [ ] Add Alembic command examples for schema management
- [ ] Create a SCHEMA_CUSTOMIZATION.md guide with examples
  - [ ] Explain how to override existing models
  - [ ] Show how to add new tables/columns
  - [ ] Provide examples for common customizations
  - [ ] Document testing and migration workflow
- [ ] Update in-code documentation
  - [ ] Add docstrings to SQLAlchemy models
  - [ ] Document schema organization and design patterns
  - [ ] Update comments in ETL scripts to reference models

## 9. Testing Framework
- [ ] Create tests for schema validation
- [ ] Implement test for migration process
- [ ] Add tests for schema customization mechanism
- [ ] Create integration test for end-to-end schema setup

## 10. Migration Script for Existing Projects
- [ ] Create a utility script to convert existing hardcoded schemas to SQLAlchemy models
- [ ] Add documentation on how to migrate from the old approach to the new one
- [ ] Include examples of common schema modifications 