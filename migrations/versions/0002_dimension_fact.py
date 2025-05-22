"""add dimension and fact tables"""
from alembic import op
import sqlalchemy as sa
from rahil.schemas import DimensionBase
from rahil.schemas.fact import FactBase
from sqlalchemy.schema import CreateTable

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

def upgrade() -> None:
    conn = op.get_bind()
    for table in DimensionBase.metadata.sorted_tables:
        conn.execute(sa.text(str(CreateTable(table))))
    for table in FactBase.metadata.sorted_tables:
        conn.execute(sa.text(str(CreateTable(table))))


def downgrade() -> None:
    conn = op.get_bind()
    for table in reversed(FactBase.metadata.sorted_tables):
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {table.name}"))
    for table in reversed(DimensionBase.metadata.sorted_tables):
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {table.name}"))
