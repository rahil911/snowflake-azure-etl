"""SQLAlchemy models for dimension tables."""
from sqlalchemy.orm import declarative_base

DimensionBase = declarative_base()

# Import models so that metadata is populated
try:
    from . import date  # noqa: E402
    from . import product  # noqa: E402
    from . import store  # noqa: E402
    from . import reseller  # noqa: E402
    from . import location  # noqa: E402
    from . import customer  # noqa: E402
    from . import channel  # noqa: E402
except ImportError:
    pass

__all__ = [
    "DimensionBase",
    "date",
    "product",
    "store",
    "reseller",
    "location",
    "customer",
    "channel",
]
