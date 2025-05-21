"""SQLAlchemy models for staging tables."""
from sqlalchemy.orm import declarative_base

Base = declarative_base()

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

__all__ = [
    "Base",
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
]
