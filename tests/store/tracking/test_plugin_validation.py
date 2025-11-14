import subprocess
import sys

from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


def test_sqlalchemy_store_import_does_not_cause_circular_import():
    """
    Regression test for circular import issue (https://github.com/mlflow/mlflow/issues/18386).

    Store plugins that inherit from SqlAlchemyStore need to import it at module level, which
    triggers imports of EvaluationDataset. The EvaluationDataset class in turn imports from
    tracking service utilities, which can create a circular dependency if not handled carefully.

    This test verifies that basic imports work without circular dependency errors. The circular
    import is broken by using lazy imports within EvaluationDataset's methods rather than at
    module level.
    """
    code = """
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.entities import EvaluationDataset
"""

    subprocess.check_call([sys.executable, "-c", code], timeout=20)


def test_plugin_entrypoint_registration_does_not_fail():
    """
    Regression test for plugin loading issue (https://github.com/mlflow/mlflow/issues/18386).

    When MLflow discovers and loads store plugins via entrypoints, it imports the plugin module
    which typically defines a class inheriting from SqlAlchemyStore. This import chain needs to
    work without ImportError.

    This test simulates the entrypoint.load() process during plugin registration to ensure the
    plugin module can be imported successfully.
    """
    code = """
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

class CustomTrackingStore(SqlAlchemyStore):
    pass

"""

    subprocess.check_call([sys.executable, "-c", code], timeout=20)


def test_plugin_can_create_dataset_without_name_error(tmp_path):
    """
    Regression test for plugin runtime usage (https://github.com/mlflow/mlflow/issues/18386).

    Store plugins that inherit from SqlAlchemyStore need to be able to call methods like
    create_dataset() which instantiate EvaluationDataset at runtime.

    This test ensures that after a plugin loads, it can actually use store methods that reference
    EvaluationDataset. This catches the actual runtime failure that users experienced, where the
    plugin would load successfully but fail when trying to perform dataset operations.
    """
    # Pre-initialize the database to avoid expensive migrations in subprocess
    db_path = tmp_path / "test.db"
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()

    # Initialize database with SqlAlchemyStore (runs migrations)
    store = SqlAlchemyStore(f"sqlite:///{db_path}", str(artifact_path))
    store.engine.dispose()  # Close connection to allow subprocess to use the database

    # Now run the test code in subprocess with the pre-initialized database
    code = f"""
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

class PluginStore(SqlAlchemyStore):
    pass

db_path = r"{db_path}"
artifact_path = r"{artifact_path}"
store = PluginStore(f"sqlite:///{{db_path}}", artifact_path)

dataset = store.create_dataset("test_dataset", tags={{"key": "value"}}, experiment_ids=[])

assert dataset is not None
assert dataset.name == "test_dataset"
"""

    subprocess.check_call([sys.executable, "-c", code], timeout=20)


def test_evaluation_dataset_not_in_entities_all():
    """
    Regression test for circular import issue (https://github.com/mlflow/mlflow/issues/18386).

    EvaluationDataset must be excluded from mlflow.entities.__all__ to prevent wildcard imports
    from triggering circular dependencies. When store plugins are loaded via entrypoints, any
    code that uses "from mlflow.entities import *" would pull in EvaluationDataset, which has
    dependencies that create import cycles with the store infrastructure.

    This test ensures EvaluationDataset remains importable directly but isn't exposed through
    wildcard imports, allowing plugins to safely inherit from store classes without encountering
    circular import issues during initialization.
    """
    import mlflow.entities

    assert "EvaluationDataset" not in mlflow.entities.__all__
