"""add secrets tables

Create Date: 2025-11-20 12:22:19.451124

"""

import time

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "1bd49d398cd23"
down_revision = "bf29a5ff90ea"
branch_labels = None
depends_on = None


# Trigger SQL for each database dialect to enforce immutability of secret_id and secret_name.
# These fields are used as AAD (Additional Authenticated Data) in AES-GCM encryption.
# If modified, decryption will fail. This is enforced at the database level to prevent
# any code path from accidentally allowing mutation.

SQLITE_TRIGGER = """
CREATE TRIGGER prevent_secrets_aad_mutation
BEFORE UPDATE ON secrets
FOR EACH ROW
WHEN OLD.secret_id != NEW.secret_id OR OLD.secret_name != NEW.secret_name
BEGIN
    SELECT RAISE(ABORT, 'secret_id and secret_name are immutable (used as AAD in encryption)');
END;
"""

POSTGRESQL_FUNCTION = """
CREATE OR REPLACE FUNCTION prevent_secrets_aad_mutation()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.secret_id != NEW.secret_id OR OLD.secret_name != NEW.secret_name THEN
        RAISE EXCEPTION 'secret_id and secret_name are immutable (used as AAD in encryption)';
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
        SET MESSAGE_TEXT = 'secret_id and secret_name are immutable (used as AAD in encryption)';
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
    IF EXISTS (
        SELECT 1 FROM inserted i
        INNER JOIN deleted d ON i.secret_id = d.secret_id
        WHERE i.secret_id != d.secret_id OR i.secret_name != d.secret_name
    )
    BEGIN
        RAISERROR('secret_id and secret_name are immutable (used as AAD in encryption)', 16, 1);
        ROLLBACK TRANSACTION;
    END
END;
"""


def _create_immutability_trigger():
    bind = op.get_bind()
    dialect = bind.engine.dialect.name

    if dialect == "sqlite":
        op.execute(SQLITE_TRIGGER)
    elif dialect == "postgresql":
        op.execute(POSTGRESQL_FUNCTION)
        op.execute(POSTGRESQL_TRIGGER)
    elif dialect == "mysql":
        op.execute(MYSQL_TRIGGER)
    elif dialect == "mssql":
        op.execute(MSSQL_TRIGGER)


def _drop_immutability_trigger():
    bind = op.get_bind()
    dialect = bind.engine.dialect.name

    if dialect == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS prevent_secrets_aad_mutation;")
    elif dialect == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS prevent_secrets_aad_mutation ON secrets;")
        op.execute("DROP FUNCTION IF EXISTS prevent_secrets_aad_mutation();")
    elif dialect == "mysql":
        op.execute("DROP TRIGGER IF EXISTS prevent_secrets_aad_mutation;")
    elif dialect == "mssql":
        op.execute(
            "IF EXISTS (SELECT * FROM sys.triggers WHERE name = 'prevent_secrets_aad_mutation') "
            "DROP TRIGGER prevent_secrets_aad_mutation;"
        )


def upgrade():
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
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("secret_id", name="secrets_pk"),
    )
    with op.batch_alter_table("secrets", schema=None) as batch_op:
        batch_op.create_index("unique_secret_name", ["secret_name"], unique=True)

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
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("endpoint_id", name="endpoints_pk"),
    )
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.create_index("unique_endpoint_name", ["name"], unique=True)

    op.create_table(
        "model_definitions",
        sa.Column("model_definition_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("secret_id", sa.String(length=36), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=256), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["secret_id"],
            ["secrets.secret_id"],
            name="fk_model_definitions_secret_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("model_definition_id", name="model_definitions_pk"),
    )
    with op.batch_alter_table("model_definitions", schema=None) as batch_op:
        batch_op.create_index("unique_model_definition_name", ["name"], unique=True)
        batch_op.create_index("index_model_definitions_secret_id", ["secret_id"], unique=False)
        batch_op.create_index("index_model_definitions_provider", ["provider"], unique=False)

    op.create_table(
        "endpoint_model_mappings",
        sa.Column("mapping_id", sa.String(length=36), nullable=False),
        sa.Column("endpoint_id", sa.String(length=36), nullable=False),
        sa.Column("model_definition_id", sa.String(length=36), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, default=1.0),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["endpoint_id"],
            ["endpoints.endpoint_id"],
            name="fk_endpoint_model_mappings_endpoint_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["model_definition_id"],
            ["model_definitions.model_definition_id"],
            name="fk_endpoint_model_mappings_model_definition_id",
        ),
        sa.PrimaryKeyConstraint("mapping_id", name="endpoint_model_mappings_pk"),
    )
    with op.batch_alter_table("endpoint_model_mappings", schema=None) as batch_op:
        batch_op.create_index(
            "index_endpoint_model_mappings_endpoint_id", ["endpoint_id"], unique=False
        )
        batch_op.create_index(
            "index_endpoint_model_mappings_model_definition_id",
            ["model_definition_id"],
            unique=False,
        )
        batch_op.create_index(
            "unique_endpoint_model_mapping",
            ["endpoint_id", "model_definition_id"],
            unique=True,
        )

    op.create_table(
        "endpoint_bindings",
        sa.Column("endpoint_id", sa.String(length=36), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=False),
        sa.Column("resource_id", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["endpoint_id"],
            ["endpoints.endpoint_id"],
            name="fk_endpoint_bindings_endpoint_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "endpoint_id", "resource_type", "resource_id", name="endpoint_bindings_pk"
        ),
    )

    op.create_table(
        "endpoint_tags",
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.String(length=5000), nullable=True),
        sa.Column("endpoint_id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(
            ["endpoint_id"],
            ["endpoints.endpoint_id"],
            name="fk_endpoint_tags_endpoint_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("key", "endpoint_id", name="endpoint_tag_pk"),
    )
    with op.batch_alter_table("endpoint_tags", schema=None) as batch_op:
        batch_op.create_index("index_endpoint_tags_endpoint_id", ["endpoint_id"], unique=False)

    _create_immutability_trigger()


def downgrade():
    _drop_immutability_trigger()
    op.drop_table("endpoint_tags")
    op.drop_table("endpoint_bindings")
    op.drop_table("endpoint_model_mappings")
    op.drop_table("model_definitions")
    op.drop_table("endpoints")
    op.drop_table("secrets")
