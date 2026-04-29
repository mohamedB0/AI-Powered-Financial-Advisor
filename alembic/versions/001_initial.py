"""initial migration - create financial tables

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'stock_prices',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('ticker', sa.String(length=10), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('open', sa.Float(), nullable=True),
        sa.Column('high', sa.Float(), nullable=True),
        sa.Column('low', sa.Float(), nullable=True),
        sa.Column('close', sa.Float(), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_stock_prices_ticker', 'stock_prices', ['ticker'])
    op.create_index('ix_stock_prices_date', 'stock_prices', ['date'])
    op.create_index('uq_price_ticker_date', 'stock_prices', ['ticker', 'date'], unique=True)
    
    op.create_table(
        'financial_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('ticker', sa.String(length=10), nullable=True),
        sa.Column('company', sa.String(length=100), nullable=True),
        sa.Column('metric_name', sa.String(length=50), nullable=False),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('period', sa.String(length=10), nullable=True),
        sa.Column('report_date', sa.Date(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_financial_metrics_ticker', 'financial_metrics', ['ticker'])
    op.create_index('ix_financial_metrics_company', 'financial_metrics', ['company'])
    op.create_index('uq_metric_ticker_name_period', 'financial_metrics', ['ticker', 'metric_name', 'period'], unique=True)
    
    op.create_table(
        'macro_indicators',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('country', sa.String(length=100), nullable=True),
        sa.Column('indicator_name', sa.String(length=80), nullable=False),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('unit', sa.String(length=50), nullable=True),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('source', sa.String(length=120), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_macro_indicators_country', 'macro_indicators', ['country'])
    op.create_index('ix_macro_indicators_indicator', 'macro_indicators', ['indicator_name'])
    op.create_index('uq_macro_name_date_country', 'macro_indicators', ['indicator_name', 'date', 'country'], unique=True)


def downgrade() -> None:
    op.drop_table('macro_indicators')
    op.drop_table('financial_metrics')
    op.drop_table('stock_prices')
