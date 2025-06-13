"""add v3 trace schema columns

Revision ID: cbc13b556ace
Revises: 5b0e9adcef9c
Create Date: 2025-01-13 14:20:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = "cbc13b556ace"
down_revision = "6953534de441"
branch_labels = None
depends_on = None


def upgrade():
    """
    Migrate trace_info table from V2 to unified V3 schema.
    Simple column renames and additions - no data transformation needed.
    """
    connection = op.get_bind()
    
    # Step 1: Add new V3-only columns  
    op.add_column("trace_info", sa.Column("request_preview", sa.Text, nullable=True))
    op.add_column("trace_info", sa.Column("response_preview", sa.Text, nullable=True))
    op.add_column("trace_info", sa.Column("client_request_id", sa.String(255), nullable=True))
    
    # Step 2: Populate client_request_id from existing request_id
    connection.execute(
        text("UPDATE trace_info SET client_request_id = request_id")
    )
    
    # Step 3: Rename columns (V2 -> V3 mapping)
    with op.batch_alter_table("trace_info") as batch_op:
        # Rename V2 columns to V3 names, keeping same data
        batch_op.alter_column("request_id", new_column_name="trace_id", type_=sa.String(255))
        batch_op.alter_column("timestamp_ms", new_column_name="request_time")  
        batch_op.alter_column("execution_time_ms", new_column_name="execution_duration")
        batch_op.alter_column("status", new_column_name="state")
    
    # Step 4: Update foreign key columns in related tables (simple renames)
    with op.batch_alter_table("trace_tags") as batch_op:
        batch_op.alter_column("request_id", new_column_name="trace_id", type_=sa.String(255))
        
    with op.batch_alter_table("trace_request_metadata") as batch_op:
        batch_op.alter_column("request_id", new_column_name="trace_id", type_=sa.String(255))
    
    # Step 5: Update constraints and indexes using batch mode for SQLite compatibility
    with op.batch_alter_table("trace_info") as batch_op:
        # Drop old constraints/indexes
        batch_op.drop_constraint("trace_info_pk", type_="primary")
        batch_op.drop_index("index_trace_info_experiment_id_timestamp_ms")
        
        # Create new constraints/indexes with V3 column names
        batch_op.create_primary_key("trace_info_pk", ["trace_id"])
        batch_op.create_index("index_trace_info_experiment_id_request_time", ["experiment_id", "request_time"])


def downgrade():
    """
    Revert unified V3 schema back to V2 schema.
    Simple column renames - V3-specific data will be lost.
    """
    
    # Step 1: Drop V3 constraints and indexes using batch mode
    with op.batch_alter_table("trace_info") as batch_op:
        batch_op.drop_constraint("trace_info_pk", type_="primary")
        batch_op.drop_index("index_trace_info_experiment_id_request_time")
    
    # Step 2: Rename columns back (V3 -> V2 mapping)
    with op.batch_alter_table("trace_info") as batch_op:
        # Rename V3 columns back to V2 names
        batch_op.alter_column("trace_id", new_column_name="request_id", type_=sa.String(50))
        batch_op.alter_column("request_time", new_column_name="timestamp_ms")
        batch_op.alter_column("execution_duration", new_column_name="execution_time_ms")
        batch_op.alter_column("state", new_column_name="status")
    
    # Step 3: Rename foreign key columns in related tables
    with op.batch_alter_table("trace_tags") as batch_op:
        batch_op.alter_column("trace_id", new_column_name="request_id", type_=sa.String(50))
        
    with op.batch_alter_table("trace_request_metadata") as batch_op:
        batch_op.alter_column("trace_id", new_column_name="request_id", type_=sa.String(50))
    
    # Step 4: Drop V3-only columns
    op.drop_column("trace_info", "client_request_id")
    op.drop_column("trace_info", "response_preview")
    op.drop_column("trace_info", "request_preview")
    
    # Step 5: Restore V2 primary key and indexes using batch mode  
    with op.batch_alter_table("trace_info") as batch_op:
        batch_op.create_primary_key("trace_info_pk", ["request_id"])
        batch_op.create_index("index_trace_info_experiment_id_timestamp_ms", ["experiment_id", "timestamp_ms"])