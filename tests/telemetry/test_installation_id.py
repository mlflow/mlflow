import json
import uuid
from unittest import mock

import pytest

import mlflow
from mlflow.telemetry.client import get_telemetry_client, set_telemetry_client
from mlflow.telemetry.installation_id import get_or_create_installation_id
from mlflow.utils.os import is_windows
from mlflow.version import VERSION


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # macos/linux
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)  # macos/linux with custom location
    monkeypatch.setenv("APPDATA", str(tmp_path))  # windows
    return tmp_path


@pytest.fixture(autouse=True)
def clear_installation_id_cache():
    mlflow.telemetry.installation_id._INSTALLATION_ID_CACHE = None


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def test_installation_id_persisted_and_reused(tmp_home):
    first = get_or_create_installation_id()
    assert _is_uuid(first)

    base_path = tmp_home if is_windows() else tmp_home / ".config"
    path = base_path / "mlflow" / "telemetry.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("installation_id") == first
    assert data.get("schema_version") == 1
    assert data.get("created_version") == VERSION
    assert data.get("created_at") is not None

    # Second call returns the same value without changing the file
    second = get_or_create_installation_id()
    assert second == first


def test_installation_id_saved_to_xdg_config_dir_if_set(monkeypatch, tmp_home):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_home))
    first = get_or_create_installation_id()
    assert _is_uuid(first)
    path = tmp_home / "mlflow" / "telemetry.json"
    assert path.exists()


def test_installation_id_corrupted_file(tmp_home):
    # If the file is corrupted, installation ID should be recreated
    base_path = tmp_home if is_windows() else tmp_home / ".config"
    dir_path = base_path / "mlflow"
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "telemetry.json"
    path.write_text("invalid JSON", encoding="utf-8")
    third = get_or_create_installation_id()
    assert _is_uuid(third)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("installation_id") == third


@pytest.mark.parametrize("env_var", ["MLFLOW_DISABLE_TELEMETRY", "DO_NOT_TRACK"])
def test_installation_id_not_created_when_telemetry_disabled(monkeypatch, tmp_home, env_var):
    monkeypatch.setenv(env_var, "true")
    # This env var is set to True in conftest.py and force enable telemetry
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_MLFLOW_TESTING_TELEMETRY", False)
    set_telemetry_client()
    assert not (tmp_home / ".config" / "mlflow" / "telemetry.json").exists()
    assert get_telemetry_client() is None


def test_get_or_create_installation_id_should_not_raise():
    with mock.patch(
        "mlflow.telemetry.installation_id._load_installation_id_from_disk",
        side_effect=Exception("test"),
    ) as mocked:
        assert get_or_create_installation_id() is None
        mocked.assert_called_once()
