from click.testing import CliRunner
from mock import mock
import tempfile
import mlflow
import sys
from mlflow.store.cli import log_artifact

def test_store_cli_log_artifact():
    """
    Test that the store CLI doesn't import SQLAlchemy or Alembic
    """
    artifact_src_dir = tempfile.mkdtemp()
    _, filepath = tempfile.mkstemp(dir=artifact_src_dir)
    with open(filepath, "w") as handle:
        handle.write("test")
    run = mlflow.start_run()
    run_id = run.info.run_id
    # artifact_uri = mlflow.get_artifact_uri()
    # run_artifact_dir = local_file_uri_to_path(artifact_uri)
    mlflow.end_run()
    CliRunner().invoke(log_artifact,
                        ["--local-file", filepath,
                         "--run-id", run_id])
    assert("sqlalchemy" not in sys.modules)
    assert("alembic" not in sys.modules)
    # included when log_artifact is invoked
    assert("logging" in sys.modules)

