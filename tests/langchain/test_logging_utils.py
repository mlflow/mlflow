import cloudpickle
import pytest

from mlflow.exceptions import MlflowException
from mlflow.langchain.utils.logging import _load_from_pickle


def test_load_from_pickle_disallows_pickle_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "obj.pkl"
    with open(path, "wb") as f:
        cloudpickle.dump({"key": "value"}, f)

    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "false")
    with pytest.raises(MlflowException, match="MLFLOW_ALLOW_PICKLE_DESERIALIZATION"):
        _load_from_pickle(path)


def test_load_from_pickle_allows_pickle_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "obj.pkl"
    with open(path, "wb") as f:
        cloudpickle.dump({"key": "value"}, f)

    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")
    assert _load_from_pickle(path) == {"key": "value"}
