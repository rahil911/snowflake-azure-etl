<!--
# Agent Prompting Instructions
- Think step-by-step using the plan tasks as a guideline.
- Use appropriate code edit tools for file modifications.
- Write clear and concise commit messages.
- Validate tasks before and after execution.
- Ask clarifying questions when tasks are unclear.
-->


# Schema Management Enhancement Plan

## 1. Setup Alembic Integration
- [ ] Install Alembic and its dependencies (`pip install alembic sqlalchemy`)
- [ ] Initialize Alembic in project structure (`alembic init migrations`)
- [ ] Configure Alembic environment (`alembic.ini` and `env.py`)
- [ ] Update `.gitignore` to exclude user-specific schema files

## 2. Create Schema Definition Structure
- [ ] Create a `schemas` directory to store table definitions
- [ ] Create a base schema definition template with SQLAlchemy models
- [ ] Move table schemas from `create_tables.py` into separate Python modules:
  - [ ] Extract Channel schema to `schemas/channel.py`
  - [ ] Extract ChannelCategory schema to `schemas/channel_category.py`
  - [ ] Extract Customer schema to `schemas/customer.py`
  - [ ] Extract Product schema to `schemas/product.py`
  - [ ] Extract ProductCategory schema to `schemas/product_category.py`
  - [ ] Extract ProductType schema to `schemas/product_type.py`
  - [ ] Extract Reseller schema to `schemas/reseller.py`
  - [ ] Extract SalesDetail schema to `schemas/sales_detail.py`
  - [ ] Extract SalesHeader schema to `schemas/sales_header.py`
  - [ ] Extract Store schema to `schemas/store.py`
  - [ ] Extract TargetDataChannel schema to `schemas/target_data_channel.py`
  - [ ] Extract TargetDataProduct schema to `schemas/target_data_product.py`

## 3. Create SQLAlchemy Model Definitions
- [ ] Define SQLAlchemy Base class in `schemas/__init__.py`
- [ ] For each entity, create SQLAlchemy model with:
  - [ ] Table name matching existing staging table names
  - [ ] Column definitions with appropriate data types
  - [ ] Primary key and foreign key relationships
  - [ ] Indexes and constraints as needed

## 4. Update Table Creation Process
- [ ] Modify `create_tables.py` to use the SQLAlchemy models instead of hardcoded SQL
- [ ] Create utility to generate table creation SQL from SQLAlchemy models
- [ ] Implement option to use either SQLAlchemy or native Snowflake SQL

## 5. Implement Schema Versioning
- [ ] Create initial Alembic migration (`alembic revision --autogenerate -m "initial"`)
- [ ] Update `run_etl.py` to run Alembic migrations as part of the process
- [ ] Add command-line option to upgrade or downgrade schema versions

## 6. Create Schema Customization Mechanism
- [ ] Add support for user-specific schema overrides in `.schema_customizations/`
- [ ] Implement mechanism to merge base schemas with user customizations
- [ ] Update `.gitignore` to exclude the `.schema_customizations/` directory

## 7. Update Dimension Model ETL
- [ ] Update dimension table creation to use SQLAlchemy models
- [ ] Modify fact table creation to reference dimension SQLAlchemy models
- [ ] Ensure proper relationships between fact and dimension tables

## 8. Documentation Updates
- [ ] Document the new schema definition approach in README.md
- [ ] Create a SCHEMA_CUSTOMIZATION.md guide with examples
- [ ] Update existing documentation to reference the new approach
- [ ] Add Alembic migration commands to documentation

## 9. Testing Framework
- [ ] Create tests for schema validation
- [ ] Implement test for migration process
- [ ] Add tests for schema customization mechanism
- [ ] Create integration test for end-to-end schema setup

## 10. Migration Script for Existing Projects
- [ ] Create a utility script to convert existing hardcoded schemas to SQLAlchemy models
- [ ] Add documentation on how to migrate from the old approach to the new one
- [ ] Include examples of common schema modifications 