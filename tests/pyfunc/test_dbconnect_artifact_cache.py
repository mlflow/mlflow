import os
from unittest import mock

from mlflow.pyfunc.dbconnect_artifact_cache import DBConnectArtifactCache


def test_get_unpacked_artifact_dir_without_session(monkeypatch, tmp_path):
    cache = DBConnectArtifactCache(spark=object())
    cache_key = "archive"
    archive_file = "archive.tar.gz"
    cache._cache[cache_key] = archive_file

    monkeypatch.delenv("DB_SESSION_UUID", raising=False)
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    result = cache.get_unpacked_artifact_dir(cache_key)

    assert result == os.path.join(str(tmp_path), archive_file)


def test_get_unpacked_artifact_dir_multi_driver_with_driver_id(monkeypatch):
    session_id = "session-123"
    driver_id = "driver-456"

    monkeypatch.setenv("DB_SESSION_UUID", session_id)
    monkeypatch.setenv("AETHER_MULTI_DRIVER_ENABLED", "true")
    monkeypatch.setenv("AETHER_MULTI_DRIVER_NOTEBOOK_LIBRARY_ENABLED", "true")
    monkeypatch.setenv("DRIVER_ID", driver_id)

    cache = DBConnectArtifactCache(spark=object())
    cache_key = "archive"
    archive_file = "archive.tar.gz"
    cache._cache[cache_key] = archive_file

    relative_path = os.path.join("artifacts", session_id, "archives", archive_file)
    multi_driver_root = "/local_disk0/.ephemeral_nfs_multi_driver"
    expected_path = os.path.join(multi_driver_root, driver_id, relative_path)

    with (
        mock.patch("os.path.exists", side_effect=lambda path: path == expected_path),
        mock.patch("os.path.isdir", side_effect=lambda path: path == multi_driver_root),
    ):
        result = cache.get_unpacked_artifact_dir(cache_key)

    assert result == expected_path


def test_get_unpacked_artifact_dir_falls_back_if_directory_does_not_exist(monkeypatch):
    session_id = "session-123"

    monkeypatch.setenv("DB_SESSION_UUID", session_id)
    monkeypatch.setenv("AETHER_MULTI_DRIVER_ENABLED", "false")
    monkeypatch.setenv("AETHER_MULTI_DRIVER_NOTEBOOK_LIBRARY_ENABLED", "false")
    monkeypatch.setenv("DRIVER_ID", "nonexistent-driver")

    cache = DBConnectArtifactCache(spark=object())
    cache_key = "archive"
    archive_file = "archive.tar.gz"
    cache._cache[cache_key] = archive_file

    relative_path = os.path.join("artifacts", session_id, "archives", archive_file)
    single_driver_root = "/local_disk0/.ephemeral_nfs"
    expected_path = os.path.join(single_driver_root, relative_path)

    with mock.patch("os.path.exists", return_value=False):
        result = cache.get_unpacked_artifact_dir(cache_key)

    assert result == expected_path


def test_get_unpacked_artifact_dir_falls_back_if_driver_id_is_not_set(monkeypatch):
    session_id = "session-123"

    monkeypatch.setenv("DB_SESSION_UUID", session_id)
    monkeypatch.setenv("AETHER_MULTI_DRIVER_ENABLED", "false")
    monkeypatch.setenv("AETHER_MULTI_DRIVER_NOTEBOOK_LIBRARY_ENABLED", "false")

    cache = DBConnectArtifactCache(spark=object())
    cache_key = "archive"
    archive_file = "archive.tar.gz"
    cache._cache[cache_key] = archive_file

    relative_path = os.path.join("artifacts", session_id, "archives", archive_file)
    single_driver_root = "/local_disk0/.ephemeral_nfs"
    expected_path = os.path.join(single_driver_root, relative_path)

    with mock.patch("os.path.exists", return_value=False):
        result = cache.get_unpacked_artifact_dir(cache_key)

    assert result == expected_path
