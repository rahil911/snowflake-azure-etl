<!--
# Automation Prompting Instructions
- Think step-by-step and list every task needed for zero-touch.
- Include unchecked boxes for tasks.
- Provide command-line examples for adding and running a new entity.
- Highlight all refactoring and integration steps required.
-->

# Zero-Touch ETL Automation Plan

This document outlines everything required to support **zero-touch** onboarding of new CSV entities into the ETL pipeline. After implementation, adding a new entity will be as simple as running a single command.

## 1. Container Discovery
- [ ] Implement Azure Blob Storage container listing using Azure SDK.
- [ ] Filter and identify containers to treat as new entities.

## 2. Schema Inference
- [ ] Sample a configurable number of rows (e.g., 500) from each CSV file.
- [ ] Heuristically detect column types:
  - Attempt Integer → Float → Date/DateTime → Boolean → String(length).
  - Determine nullability based on presence of blanks or mixed types.
- [ ] Handle edge cases and fallback to `String` for inconsistent or empty columns.

## 3. Model Generation
- [ ] Create a `generated_schemas/` directory for auto-generated model files.
- [ ] Build a template (e.g., Jinja2) to render SQLAlchemy models:
  ```jinja
  from sqlalchemy import Column, {{ types|join(", ") }}
  from . import Base

  class {{EntityClassName}}(Base):
      __tablename__ = 'STAGING_{{ENTITY_NAME_UPPER}}'
  {% for col, typ, nullable in columns %}
      {{col}} = Column({{typ}}, nullable={{nullable}})
  {% endfor %}
  ```
- [ ] Save generated files as `generated_schemas/{{entity_name}}.py`.
- [ ] Update `rahil/schemas/__init__.py` or `migrations/env.py` to import `generated_schemas/` models.

## 4. Alembic Integration
- [ ] Modify `migrations/env.py` to automatically import all files in `generated_schemas/`.
- [ ] Automate migration commands:
  ```bash
  alembic revision --autogenerate -m "Add {{ENTITY}} staging table"
  alembic upgrade head
  ```
- [ ] Stamp the existing database on the first run to avoid recreating existing tables:
  ```bash
  alembic stamp head
  ```

## 5. ETL Orchestration
- [ ] Extend `rahil/run_etl.py` to include:
  1. Container discovery step
  2. Schema inference step
  3. Model generation step
  4. Alembic migration step
  5. Existing staging pipeline steps
- [ ] Extend `rahil/run_dimensional_etl.py` similarly to handle dimensions and facts.

## 6. Error Handling & Logging
- [ ] Wrap inference and generation logic in `try/except`, logging schema warnings.
- [ ] Configure Snowflake `COPY INTO` to use `ON_ERROR = 'CONTINUE'` or similar options.
- [ ] Persist detailed logs for:
  - Schema inference results
  - Generated model files
  - Migration diffs and errors
  - Data load summaries and errors

## 7. Customization & Overrides
- [ ] Maintain a `.schema_customizations/` directory (in `.gitignore`) for manual overrides.
- [ ] If a customization file exists for an entity, use it instead of generated models.

## 8. CLI Usage
After implementation, onboarding a new entity `my_new_table` will be as simple as:
```bash
python scripts/auto_etl.py my_new_table
```
This single command will:
1. Discover and sample `my_new_table` container in Azure Blob Storage
2. Infer schema and generate SQLAlchemy model
3. Autogenerate + apply Alembic migrations
4. Execute the full staging and dimensional ETL pipelines

## 9. Testing & Validation
- [ ] Write unit tests for type inference functions.
- [ ] Create integration tests to verify:
  - Generated models match actual CSV schemas
  - Migrations succeed without errors
  - Staging and dimensional pipelines load expected row counts

## 10. Documentation Updates
- [ ] Add zero-touch usage instructions to `README.md`.
- [ ] Document `scripts/auto_etl.py` arguments and options.
- [ ] Provide examples of customization and rollback workflows.

---

With these tasks completed, any user can simply point at a new container and watch the pipeline auto-discover, generate, migrate, and load end-to-end—no manual SQL or schema definitions required. 