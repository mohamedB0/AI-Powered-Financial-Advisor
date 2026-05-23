"""Migrate users table for Google OAuth authentication

Revision ID: 004_google_oauth_migration
Revises: 003_add_conversations_messages
Create Date: 2025-05-19

Changes:
  - Make hashed_password nullable (Google users don't have passwords)
  - Add google_sub column (Google unique user ID)
  - Add picture column (Google profile picture URL)
  - Add index on google_sub for fast lookups
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '004_google_oauth_migration'
down_revision: Union[str, None] = '003_add_conversations_messages'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Make hashed_password nullable (Google users won't have one)
    op.alter_column('users', 'hashed_password',
                    existing_type=sa.String(length=255),
                    nullable=True)

    # Add Google-specific columns
    op.add_column('users', sa.Column('google_sub', sa.String(length=255), nullable=True))
    op.add_column('users', sa.Column('picture', sa.String(length=500), nullable=True))

    # Index for fast Google sub lookups
    op.create_index('ix_users_google_sub', 'users', ['google_sub'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_users_google_sub', table_name='users')
    op.drop_column('users', 'picture')
    op.drop_column('users', 'google_sub')
    op.alter_column('users', 'hashed_password',
                    existing_type=sa.String(length=255),
                    nullable=False)
