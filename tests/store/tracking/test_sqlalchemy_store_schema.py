"""Tests verifying that the SQLAlchemyStore generates the expected database schema"""

import os
import sqlite3

import pytest
import sqlalchemy
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory

import mlflow.db
from mlflow.exceptions import MlflowException
from mlflow.store.db.base_sql_model import Base
from mlflow.store.db.utils import _get_alembic_config, _verify_schema
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

from tests.integration.utils import invoke_cli_runner
from tests.store.dump_schema import dump_db_schema


def _assert_schema_files_equal(generated_schema_file, expected_schema_file):
    """
    Assert equivalence of two SQL schema dump files consisting of CREATE TABLE statements delimited
    by double-newlines, allowing for the reordering of individual lines within each CREATE TABLE
    statement to account for differences in schema-dumping across platforms & Python versions.
    """
    # Extract "CREATE TABLE" statement chunks from both files, assuming tables are listed in the
    # same order across files
    with open(generated_schema_file) as generated_schema_handle:
        generated_schema_table_chunks = generated_schema_handle.read().split("\n\n")
    with open(expected_schema_file) as expected_schema_handle:
        expected_schema_table_chunks = expected_schema_handle.read().split("\n\n")
    # Compare the two files table-by-table. We assume each CREATE TABLE statement is valid and
    # so sort the lines within the statements before comparing them.
    for generated_schema_table, expected_schema_table in zip(
        generated_schema_table_chunks, expected_schema_table_chunks
    ):
        generated_lines = [x.strip() for x in sorted(generated_schema_table.split("\n"))]
        expected_lines = [x.strip() for x in sorted(expected_schema_table.split("\n"))]
        assert generated_lines == expected_lines, (
            "Generated schema did not match expected schema. Generated schema had table "
            f"definition:\n{generated_schema_table}\nExpected schema had table definition:"
            f"\n{expected_schema_table}\nIf you intended to make schema changes, run "
            f"'python tests/store/dump_schema.py {expected_schema_file}' from your checkout"
            " of MLflow to update the schema snapshot."
        )


@pytest.fixture
def expected_schema_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(current_dir, os.pardir, os.pardir, "resources", "db", "latest_schema.sql")
    )


@pytest.fixture
def db_url(tmp_path):
    db_file = tmp_path.joinpath("db_file")
    return f"sqlite:///{db_file}"


def test_sqlalchemystore_idempotently_generates_up_to_date_schema(
    tmp_path, db_url, expected_schema_file
):
    generated_schema_file = tmp_path.joinpath("generated-schema.sql")
    # Repeatedly initialize a SQLAlchemyStore against the same DB URL. Initialization should
    # succeed and the schema should be the same.
    for _ in range(3):
        SqlAlchemyStore(db_url, tmp_path.joinpath("ARTIFACTS").as_uri())
        dump_db_schema(db_url, dst_file=generated_schema_file)
        _assert_schema_files_equal(generated_schema_file, expected_schema_file)


def test_running_migrations_generates_expected_schema(tmp_path, expected_schema_file, db_url):
    """Test that migrating an existing database generates the desired schema."""
    engine = sqlalchemy.create_engine(db_url)
    InitialBase.metadata.create_all(engine)
    invoke_cli_runner(mlflow.db.commands, ["upgrade", db_url])
    generated_schema_file = tmp_path.joinpath("generated-schema.sql")
    dump_db_schema(db_url, generated_schema_file)
    _assert_schema_files_equal(generated_schema_file, expected_schema_file)


def test_sqlalchemy_store_detects_schema_mismatch(db_url):
    def _assert_invalid_schema(engine):
        with pytest.raises(MlflowException, match="Detected out-of-date database schema."):
            _verify_schema(engine)

    # Initialize an empty database & verify that we detect a schema mismatch
    engine = sqlalchemy.create_engine(db_url)
    _assert_invalid_schema(engine)
    # Create legacy tables, verify schema is still out of date
    InitialBase.metadata.create_all(engine)
    _assert_invalid_schema(engine)
    # Run each migration. Until the last one, schema should be out of date
    config = _get_alembic_config(db_url)
    script = ScriptDirectory.from_config(config)
    revisions = list(script.walk_revisions())
    revisions.reverse()
    for rev in revisions[:-1]:
        command.upgrade(config, rev.revision)
        _assert_invalid_schema(engine)
    # Run migrations, schema verification should now pass
    invoke_cli_runner(mlflow.db.commands, ["upgrade", db_url])
    _verify_schema(engine)


def test_store_generated_schema_matches_base(tmp_path, db_url):
    # Create a SQLAlchemyStore against tmpfile, directly verify that tmpfile contains a
    # database with a valid schema
    SqlAlchemyStore(db_url, tmp_path.joinpath("ARTIFACTS").as_uri())
    engine = sqlalchemy.create_engine(db_url)
    mc = MigrationContext.configure(engine.connect(), opts={"compare_type": False})
    diff = compare_metadata(mc, Base.metadata)
    # `diff` contains several `remove_index` operations because `Base.metadata` does not contain
    # index metadata but `mc` does. Note this doesn't mean the MLflow database is missing indexes
    # as tested in `test_create_index_on_run_uuid`.
    diff = [d for d in diff if (d[0] not in ["remove_index", "add_index", "add_fk"])]
    assert len(diff) == 0, (
        "if this test is failing after writing a DB migration, please make sure you've "
        "updated the ORM definitions in `mlflow/store/tracking/dbmodels/models.py`."
    )


def test_create_index_on_run_uuid(tmp_path, db_url):
    # Test for mlflow/store/db_migrations/versions/bd07f7e963c5_create_index_on_run_uuid.py
    SqlAlchemyStore(db_url, tmp_path.joinpath("ARTIFACTS").as_uri())
    with sqlite3.connect(db_url[len("sqlite:///") :]) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        all_index_names = [r[0] for r in cursor.fetchall()]
        run_uuid_index_names = {
            "index_params_run_uuid",
            "index_metrics_run_uuid",
            "index_latest_metrics_run_uuid",
            "index_tags_run_uuid",
        }
        assert run_uuid_index_names.issubset(all_index_names)


def test_index_for_dataset_tables(tmp_path, db_url):
    # Test for
    # mlflow/store/db_migrations/versions/7f2a7d5fae7d_add_datasets_inputs_input_tags_tables.py
    SqlAlchemyStore(db_url, tmp_path.joinpath("ARTIFACTS").as_uri())
    with sqlite3.connect(db_url[len("sqlite:///") :]) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        all_index_names = [r[0] for r in cursor.fetchall()]
        new_index_names = {
            "index_datasets_experiment_id_dataset_source_type",
            "index_inputs_input_uuid",
            "index_inputs_destination_type_destination_id_source_type",
        }
        assert new_index_names.issubset(all_index_names)
