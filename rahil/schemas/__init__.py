"""SQLAlchemy models for staging, dimension, and fact tables."""
from sqlalchemy.orm import declarative_base

# Separate bases so ETL scripts can create only the required tables
StagingBase = declarative_base()
DimensionBase = declarative_base()
FactBase = declarative_base()

# Backwards compatibility
Base = StagingBase

# Import models so that Base.metadata is populated
from . import channel  # noqa: E402
from . import channel_category  # noqa: E402
from . import customer  # noqa: E402
from . import product  # noqa: E402
from . import product_category  # noqa: E402
from . import product_type  # noqa: E402
from . import reseller  # noqa: E402
from . import store  # noqa: E402
from . import sales_detail  # noqa: E402
from . import sales_header  # noqa: E402
from . import target_data_channel  # noqa: E402
from . import target_data_product  # noqa: E402

# Dimension and fact models
from .dimension import (  # noqa: E402
    channel as dim_channel,
    customer as dim_customer,
    date as dim_date,
    location as dim_location,
    product as dim_product,
    reseller as dim_reseller,
    store as dim_store,
)
from .fact import (  # noqa: E402
    sales_actual as fact_sales_actual,
    product_sales_target as fact_product_sales_target,
    src_sales_target as fact_src_sales_target,
)

__all__ = [
    "Base",
    "StagingBase",
    "DimensionBase",
    "FactBase",
    "channel",
    "channel_category",
    "customer",
    "product",
    "product_category",
    "product_type",
    "reseller",
    "store",
    "sales_detail",
    "sales_header",
    "target_data_channel",
    "target_data_product",
    "dim_channel",
    "dim_customer",
    "dim_date",
    "dim_location",
    "dim_product",
    "dim_reseller",
    "dim_store",
    "fact_sales_actual",
    "fact_product_sales_target",
    "fact_src_sales_target",
]
