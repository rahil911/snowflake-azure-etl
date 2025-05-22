import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# This line allows finding the 'rahil' module
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

from rahil.dim_config import (
    SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT,
    DIMENSION_DB_NAME, SNOWFLAKE_SCHEMA, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE
)
from rahil.schemas import DimensionBase # Your DimensionBase

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.attributes.get('configure_logger', True) and config.config_file_name:
    fileConfig(config.config_file_name)

# Construct the Snowflake URL
snowflake_url = (
    f"snowflake://{SNOWFLAKE_USER}:{SNOWFLAKE_PASSWORD}@"
    f"{SNOWFLAKE_ACCOUNT}/{DIMENSION_DB_NAME}/{SNOWFLAKE_SCHEMA}"
    f"?warehouse={SNOWFLAKE_WAREHOUSE}&role={SNOWFLAKE_ROLE}"
)
# Set the sqlalchemy.url in the config object for Alembic
# This ensures it's available if Alembic tries to read it from config directly
config.set_main_option('sqlalchemy.url', snowflake_url)


# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = DimensionBase.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url") # This will now get the snowflake_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schema in compare for autogenerate
        compare_type=True, 
        include_schemas=True, # Important for Snowflake if using specific schema
        version_table_schema=SNOWFLAKE_SCHEMA # Ensure Alembic version table is in the correct schema
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Get the engine from the config (which now has the correct URL)
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            # Include schema in compare for autogenerate
            compare_type=True,
            include_schemas=True, # Important for Snowflake if using specific schema
            version_table_schema=SNOWFLAKE_SCHEMA # Ensure Alembic version table is in the correct schema
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
