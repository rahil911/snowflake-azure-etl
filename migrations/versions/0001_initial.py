"""initial

Revision ID: 0001
Revises: 
Create Date: 2024-05-23
"""

from alembic import op
import sqlalchemy as sa
from rahil.schemas import Base
from sqlalchemy.schema import CreateTable

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    conn = op.get_bind()
    for table in Base.metadata.sorted_tables:
        conn.execute(sa.text(str(CreateTable(table))))


def downgrade() -> None:
    conn = op.get_bind()
    for table in reversed(Base.metadata.sorted_tables):
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {table.name}"))
