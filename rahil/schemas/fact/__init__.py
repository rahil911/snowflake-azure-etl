"""SQLAlchemy models for fact tables."""
from sqlalchemy.orm import declarative_base

FactBase = declarative_base()

try:
    from . import sales_actual  # noqa: E402
    from . import product_sales_target  # noqa: E402
    from . import src_sales_target  # noqa: E402
except ImportError:
    pass

__all__ = [
    "FactBase",
    "sales_actual",
    "product_sales_target",
    "src_sales_target",
]
