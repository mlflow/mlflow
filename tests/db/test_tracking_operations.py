import sqlite3
import uuid
from unittest import mock

import pytest
import sqlalchemy.dialects.sqlite.pysqlite

import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_TRACKING_URI

pytestmark = pytest.mark.notrackingurimock


class Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input, params=None):
        pass


def start_run_and_log_data():
    with mlflow.start_run():
        mlflow.log_param("p", "param")
        mlflow.log_metric("m", 1.0)
        mlflow.set_tag("t", "tag")
        mlflow.pyfunc.log_model(
            artifact_path="model", python_model=Model(), registered_model_name="model"
        )


def test_search_runs():
    start_run_and_log_data()
    runs = mlflow.search_runs(experiment_ids=["0"], order_by=["param.start_time DESC"])
    mlflow.get_run(runs["run_id"][0])


def test_set_run_status_to_killed():
    """
    This test ensures the following migration scripts work correctly:
    - cfd24bdc0731_update_run_status_constraint_with_killed.py
    - 0a8213491aaa_drop_duplicate_killed_constraint.py
    """
    with mlflow.start_run() as run:
        pass
    client = MlflowClient()
    client.set_terminated(run_id=run.info.run_id, status="KILLED")


@mock.patch("mlflow.store.db.utils._logger.exception")
def test_database_operational_error(exception, monkeypatch):
    # This test is specifically designed to force errors with SQLite. Skip it if
    # using a non-SQLite backend.
    if not MLFLOW_TRACKING_URI.get().startswith("sqlite"):
        pytest.skip("Only works on SQLite")

    # This test patches parts of SQLAlchemy and sqlite3.dbapi to simulate a
    # SQLAlchemy OperationalError. PEP 249 describes OperationalError as:
    #
    # > Exception raised for errors that are related to the database’s operation
    # > and not necessarily under the control of the programmer, e.g. an
    # > unexpected disconnect occurs, the data source name is not found, a
    # > transaction could not be processed, a memory allocation error occurred
    # > during processing, etc.
    #
    # These errors are typically transient and can be resolved by retrying the
    # operation, hence MLflow has different handling for them as compared to
    # the more generic exception type, SQLAlchemyError.
    #
    # This is particularly important for REST clients, where
    # TEMPORARILY_UNAVAILABLE triggers MLflow REST clients to retry the request,
    # whereas BAD_REQUEST does not.
    api_module = None
    old_connect = None

    # Depending on the version of SQLAlchemy, the function we need to patch is
    # either called "dbapi" (sqlalchemy<2.0) or "import_dbapi"
    # (sqlalchemy>=2.0).
    for dialect_attr in ["dbapi", "import_dbapi"]:
        if hasattr(sqlalchemy.dialects.sqlite.pysqlite.SQLiteDialect_pysqlite, dialect_attr):
            break
    else:
        raise AssertionError("Could not find dbapi attribute on SQLiteDialect_pysqlite")

    old_dbapi = getattr(sqlalchemy.dialects.sqlite.pysqlite.SQLiteDialect_pysqlite, dialect_attr)

    class ConnectionWrapper:
        """Wraps a sqlite3.Connection object."""

        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def cursor(self):
            """Return a wrapped SQLite cursor."""
            return CursorWrapper(self.conn.cursor())

    class CursorWrapper:
        """Wraps a sqlite3.Cursor object."""

        def __init__(self, cursor):
            self.cursor = cursor

        def __getattr__(self, name):
            return getattr(self.cursor, name)

        def execute(self, *args, **kwargs):
            """Wraps execute(), simulating sporadic OperationalErrors."""
            if (
                len(args) >= 2
                and "test_database_operational_error_1667938883_param" in args[1]
                and "test_database_operational_error_1667938883_value" in args[1]
            ):
                # Simulate a database error
                raise sqlite3.OperationalError("test")
            return self.cursor.execute(*args, **kwargs)

    def connect(*args, **kwargs):
        """Wraps sqlite3.dbapi.connect(), returning a wrapped connection."""
        global connect_counter
        conn = old_connect(*args, **kwargs)
        return ConnectionWrapper(conn)

    def dbapi(*args, **kwargs):
        """Wraps SQLiteDialect_pysqlite.dbapi(), returning patched dbapi."""
        nonlocal api_module, old_connect
        if api_module is None:
            # Only patch the first time dbapi() is called, to avoid recursion.
            api_module = old_dbapi(*args, **kwargs)
            old_connect = api_module.connect
            monkeypatch.setattr(api_module, "connect", connect)
        return api_module

    monkeypatch.setattr(
        sqlalchemy.dialects.sqlite.pysqlite.SQLiteDialect_pysqlite, dialect_attr, dbapi
    )

    # Create and use a unique tracking URI for this test. This avoids an issue
    # where an earlier test has already created and cached a SQLAlchemy engine
    # (i.e. database connections), preventing our error-throwing monkeypatches
    # from being called.
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, f"{MLFLOW_TRACKING_URI.get()}-{uuid.uuid4().hex}")
    with pytest.raises(mlflow.MlflowException, match=r"sqlite3\.OperationalError"):
        with mlflow.start_run():
            # This statement will fail with an OperationalError.
            mlflow.log_param(
                "test_database_operational_error_1667938883_param",
                "test_database_operational_error_1667938883_value",
            )
    # Verify that the error handling was executed.
    assert any(
        "SQLAlchemy database error" in str(call) and "sqlite3.OperationalError" in str(call)
        for call in exception.mock_calls
    )
