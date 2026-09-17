import pytest

from mlflow.environment_variables import MLFLOW_ALLOW_FILE_STORE
from mlflow.store.tracking.file_store import FileStore


def test_file_store_does_not_implement_skill_registry(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ALLOW_FILE_STORE.name, "true")
    store = FileStore(str(tmp_path / "mlruns"))

    with pytest.raises(NotImplementedError, match="FileStore"):
        store.get_skill("reviewer")
