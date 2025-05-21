# Schema Customization Guide

This guide explains how to override the default SQLAlchemy models provided in `rahil/schemas`.

1. **Create a customization directory**
   ```bash
   mkdir -p .schema_customizations
   ```

2. **Override an existing model**
   Create a Python file in `.schema_customizations/` with a class that has the same `__tablename__` as the model you want to replace.

3. **Add new columns or tables**
   Define additional fields on your custom model or create entirely new models. Migrations can then be generated with Alembic:
   ```bash
   alembic revision --autogenerate -m "custom change"
   alembic upgrade head
   ```

4. **Testing workflow**
   Use `alembic upgrade head` to apply schema changes and run `python -m compileall rahil` to ensure your code compiles.
