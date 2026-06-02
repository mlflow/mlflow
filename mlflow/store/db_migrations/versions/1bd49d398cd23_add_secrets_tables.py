"""add secrets tables

Revision ID: 1bd49d398cd23
Revises: bf29a5ff90ea
Create Date: 2025-11-20 12:22:19.451124
"""

import time

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.

revision = "1bd49d398cd23"
down_revision = "bf29a5ff90ea"
branch_labels = None
depends_on = None

SQLITE_TRIGGER = """
CREATE TRIGGER prevent_secrets_aad_mutation
BEFORE UPDATE ON secrets
FOR EACH ROW
WHEN OLD.secret_id != NEW.secret_id OR OLD.secret_name != NEW.secret_name
BEGIN
SELECT RAISE(ABORT, 'secret_id and secret_name are immutable');
END;
"""

POSTGRESQL_FUNCTION = """
CREATE OR REPLACE FUNCTION prevent_secrets_aad_mutation()
RETURNS TRIGGER AS $$
BEGIN
IF OLD.secret_id != NEW.secret_id OR OLD.secret_name != NEW.secret_name THEN
RAISE EXCEPTION 'secret_id and secret_name are immutable';
END IF;
RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

POSTGRESQL_TRIGGER = """
CREATE TRIGGER prevent_secrets_aad_mutation
BEFORE UPDATE ON secrets
FOR EACH ROW
EXECUTE FUNCTION prevent_secrets_aad_mutation();
"""

MYSQL_TRIGGER = """
CREATE TRIGGER prevent_secrets_aad_mutation
BEFORE UPDATE ON secrets
FOR EACH ROW
BEGIN
IF OLD.secret_id != NEW.secret_id OR OLD.secret_name != NEW.secret_name THEN
SIGNAL SQLSTATE '45000'
SET MESSAGE_TEXT = 'secret_id and secret_name are immutable';
END IF;
END;
"""

MSSQL_TRIGGER = """
CREATE TRIGGER prevent_secrets_aad_mutation
ON secrets
AFTER UPDATE
AS
BEGIN
SET NOCOUNT ON;

```
IF EXISTS (
    SELECT 1
    FROM inserted i
    INNER JOIN deleted d
    ON i.secret_id = d.secret_id
    WHERE i.secret_id != d.secret_id
       OR i.secret_name != d.secret_name
)
BEGIN
    RAISERROR(
        'secret_id and secret_name are immutable',
        16,
        1
    );
    ROLLBACK TRANSACTION;
END
```

END;
"""

def _table_exists(table_name):
bind = op.get_bind()
inspector = inspect(bind)
return table_name in inspector.get_table_names()

def _index_exists(table_name, index_name):
bind = op.get_bind()
inspector = inspect(bind)

```
indexes = inspector.get_indexes(table_name)
return any(index["name"] == index_name for index in indexes)
```

def _trigger_exists(trigger_name):
bind = op.get_bind()
dialect = bind.engine.dialect.name

```
if dialect == "mysql":
    result = bind.execute(
        sa.text(
            """
            SELECT TRIGGER_NAME
            FROM information_schema.TRIGGERS
            WHERE TRIGGER_SCHEMA = DATABASE()
            AND TRIGGER_NAME = :trigger_name
            """
        ),
        {"trigger_name": trigger_name},
    ).fetchone()

    return result is not None

return False
```

def _create_immutability_trigger():
bind = op.get_bind()
dialect = bind.engine.dialect.name

```
if _trigger_exists("prevent_secrets_aad_mutation"):
    return

if dialect == "sqlite":
    op.execute(SQLITE_TRIGGER)

elif dialect == "postgresql":
    op.execute(POSTGRESQL_FUNCTION)
    op.execute(POSTGRESQL_TRIGGER)

elif dialect == "mysql":
    op.execute(MYSQL_TRIGGER)

elif dialect == "mssql":
    op.execute(MSSQL_TRIGGER)
```

def _drop_immutability_trigger():
bind = op.get_bind()
dialect = bind.engine.dialect.name

```
if dialect == "sqlite":
    op.execute("DROP TRIGGER IF EXISTS prevent_secrets_aad_mutation")

elif dialect == "postgresql":
    op.execute(
        "DROP TRIGGER IF EXISTS prevent_secrets_aad_mutation ON secrets"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS prevent_secrets_aad_mutation()"
    )

elif dialect == "mysql":
    op.execute(
        "DROP TRIGGER IF EXISTS prevent_secrets_aad_mutation"
    )

elif dialect == "mssql":
    op.execute(
        """
        IF EXISTS (
            SELECT *
            FROM sys.triggers
            WHERE name = 'prevent_secrets_aad_mutation'
        )
        DROP TRIGGER prevent_secrets_aad_mutation
        """
    )
```

def upgrade():
# ------------------------------------------------------------------
# secrets
# ------------------------------------------------------------------

```
if not _table_exists("secrets"):
    op.create_table(
        "secrets",
        sa.Column("secret_id", sa.String(length=36), nullable=False),
        sa.Column("secret_name", sa.String(length=255), nullable=False),
        sa.Column("encrypted_value", sa.LargeBinary(), nullable=False),
        sa.Column("wrapped_dek", sa.LargeBinary(), nullable=False),
        sa.Column("kek_version", sa.Integer(), nullable=False, default=1),
        sa.Column("masked_value", sa.String(length=500), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("auth_config", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column(
            "last_updated_by",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "secret_id",
            name="secrets_pk",
        ),
    )

if not _index_exists("secrets", "unique_secret_name"):
    with op.batch_alter_table("secrets") as batch_op:
        batch_op.create_index(
            "unique_secret_name",
            ["secret_name"],
            unique=True,
        )

# ------------------------------------------------------------------
# endpoints
# ------------------------------------------------------------------

if not _table_exists("endpoints"):
    op.create_table(
        "endpoints",
        sa.Column("endpoint_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column(
            "last_updated_by",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "endpoint_id",
            name="endpoints_pk",
        ),
    )

if not _index_exists("endpoints", "unique_endpoint_name"):
    with op.batch_alter_table("endpoints") as batch_op:
        batch_op.create_index(
            "unique_endpoint_name",
            ["name"],
            unique=True,
        )

# ------------------------------------------------------------------
# Repeat same idempotent pattern for:
# - model_definitions
# - endpoint_model_mappings
# - endpoint_bindings
# - endpoint_tags
# ------------------------------------------------------------------

_create_immutability_trigger()
```

def downgrade():
_drop_immutability_trigger()

```
for table_name in [
    "endpoint_tags",
    "endpoint_bindings",
    "endpoint_model_mappings",
    "model_definitions",
    "endpoints",
    "secrets",
]:
    if _table_exists(table_name):
        op.drop_table(table_name)
```
