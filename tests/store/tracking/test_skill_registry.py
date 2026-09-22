import pytest

from mlflow.environment_variables import MLFLOW_ALLOW_FILE_STORE
from mlflow.store.tracking.file_store import FileStore


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        ("create_skill", ("reviewer",), {}),
        ("get_skill", ("reviewer",), {}),
        ("update_skill", ("reviewer",), {}),
        ("search_skills", (), {}),
        ("create_skill_version", ("reviewer",), {}),
        ("get_skill_version", ("reviewer", 1), {}),
    ],
)
def test_file_store_does_not_implement_skill_registry(
    tmp_path, monkeypatch, method_name, args, kwargs
):
    monkeypatch.setenv(MLFLOW_ALLOW_FILE_STORE.name, "true")
    store = FileStore(str(tmp_path / "mlruns"))

    with pytest.raises(NotImplementedError, match="FileStore"):
        getattr(store, method_name)(*args, **kwargs)
