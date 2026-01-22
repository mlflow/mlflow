import logging
import posixpath
import time
from unittest import mock

import pytest

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import (
    ArtifactRepository,
    _sanitize_path_component_for_windows,
    _sanitize_path_for_windows,
)
from mlflow.utils.file_utils import TempDir
from mlflow.utils.os import is_windows

from tests.utils.test_logging_utils import logger, reset_logging_level  # noqa F401

_MOCK_ERROR = "MOCK ERROR"
_MODEL_FILE = "modelfile"
_MODEL_DIR = "model"
_PARENT_DIR = "12345"
_PARENT_MODEL_DIR = _PARENT_DIR + "/" + _MODEL_DIR
_PARENT_MODEL_FILE = _PARENT_MODEL_DIR + "/" + _MODEL_FILE
_EMPTY_DIR = "emptydir"
_DUMMY_FILE_SIZE = 123
_EMPTY_FILE_SIZE = 0


class ArtifactRepositoryImpl(ArtifactRepository):
    def log_artifact(self, local_file, artifact_path=None):
        raise NotImplementedError()

    def log_artifacts(self, local_dir, artifact_path=None):
        raise NotImplementedError()

    def list_artifacts(self, path):
        raise NotImplementedError()

    def _download_file(self, remote_file_path, local_path):
        assert remote_file_path.endswith(_MODEL_FILE)


class SlowArtifactRepositoryImpl(ArtifactRepository):
    """Implementation of ArtifactRepository which simulates large artifact download."""

    def log_artifact(self, local_file, artifact_path=None):
        raise NotImplementedError()

    def log_artifacts(self, local_dir, artifact_path=None):
        raise NotImplementedError()

    def list_artifacts(self, path):
        raise NotImplementedError()

    def _download_file(self, remote_file_path, local_path):
        # Sleep in order to simulate a longer-running asynchronous download
        time.sleep(2)
        assert remote_file_path.endswith(_MODEL_FILE)


class FailureArtifactRepositoryImpl(ArtifactRepository):
    """Implementation of ArtifactRepository which simulates download failures."""

    def log_artifact(self, local_file, artifact_path=None):
        raise NotImplementedError()

    def log_artifacts(self, local_dir, artifact_path=None):
        raise NotImplementedError()

    def list_artifacts(self, path):
        raise NotImplementedError()

    def _download_file(self, remote_file_path, local_path):
        raise MlflowException(_MOCK_ERROR)


@pytest.mark.parametrize(
    ("base_uri", "download_arg", "list_return_val"),
    [
        (_PARENT_MODEL_DIR, "", [_MODEL_FILE]),
        (_PARENT_MODEL_DIR, "", [".", _MODEL_FILE]),
        (_PARENT_DIR, _MODEL_DIR, [_MODEL_DIR + "/" + _MODEL_FILE]),
        (_PARENT_DIR, _MODEL_DIR, [_MODEL_DIR, _MODEL_DIR + "/" + _MODEL_FILE]),
        ("", _PARENT_MODEL_DIR, [_PARENT_MODEL_FILE]),
        ("", _PARENT_MODEL_DIR, [_PARENT_MODEL_DIR, _PARENT_MODEL_FILE]),
    ],
)
def test_download_artifacts_does_not_infinitely_loop(base_uri, download_arg, list_return_val):
    def list_artifacts(path):
        fullpath = posixpath.join(base_uri, path)
        if fullpath.endswith(_MODEL_DIR) or fullpath.endswith(_MODEL_DIR + "/"):
            return [FileInfo(item, False, _DUMMY_FILE_SIZE) for item in list_return_val]
        elif fullpath.endswith(_PARENT_MODEL_DIR) or fullpath.endswith(_PARENT_MODEL_DIR + "/"):
            return [FileInfo(posixpath.join(path, _MODEL_DIR), True, _EMPTY_FILE_SIZE)]
        else:
            return []

    with mock.patch.object(ArtifactRepositoryImpl, "list_artifacts") as list_artifacts_mock:
        list_artifacts_mock.side_effect = list_artifacts
        repo = ArtifactRepositoryImpl(base_uri)
        repo.download_artifacts(download_arg)


def test_download_artifacts_download_file():
    with mock.patch.object(ArtifactRepositoryImpl, "list_artifacts", return_value=[]):
        repo = ArtifactRepositoryImpl(_PARENT_DIR)
        repo.download_artifacts(_MODEL_FILE)


def test_download_artifacts_empty_root_returns_dst_path(tmp_path):
    with (
        mock.patch.object(ArtifactRepositoryImpl, "list_artifacts", return_value=[]),
        mock.patch.object(ArtifactRepositoryImpl, "_download_file") as download_mock,
    ):
        repo = ArtifactRepositoryImpl(_PARENT_DIR)
        result = repo.download_artifacts("", dst_path=tmp_path)

        assert result == str(tmp_path)
        download_mock.assert_not_called()


def test_download_artifacts_dst_path_does_not_exist(tmp_path):
    repo = ArtifactRepositoryImpl(_PARENT_DIR)
    dst_path = tmp_path.joinpath("does_not_exist")
    with pytest.raises(
        MlflowException, match="The destination path for downloaded artifacts does not exist"
    ):
        repo.download_artifacts(_MODEL_DIR, dst_path)


def test_download_artifacts_dst_path_is_file(tmp_path):
    repo = ArtifactRepositoryImpl(_PARENT_DIR)
    dst_path = tmp_path.joinpath("file")
    dst_path.touch()
    with pytest.raises(
        MlflowException, match="The destination path for downloaded artifacts must be a directory"
    ):
        repo.download_artifacts(_MODEL_DIR, dst_path)


@pytest.mark.parametrize(
    ("base_uri", "download_arg", "list_return_val"),
    [
        (
            "",
            _PARENT_MODEL_DIR,
            [_PARENT_MODEL_DIR, _PARENT_MODEL_FILE, _PARENT_MODEL_DIR + "/" + _EMPTY_DIR],
        )
    ],
)
def test_download_artifacts_handles_empty_dir(base_uri, download_arg, list_return_val):
    def list_artifacts(path):
        if path.endswith(_MODEL_DIR):
            return [
                FileInfo(item, item.endswith(_EMPTY_DIR), _DUMMY_FILE_SIZE)
                for item in list_return_val
            ]
        elif path.endswith(_PARENT_DIR) or path.endswith(_PARENT_DIR + "/"):
            return [FileInfo(_PARENT_MODEL_DIR, True, _EMPTY_FILE_SIZE)]
        else:
            return []

    with mock.patch.object(ArtifactRepositoryImpl, "list_artifacts") as list_artifacts_mock:
        list_artifacts_mock.side_effect = list_artifacts
        repo = ArtifactRepositoryImpl(base_uri)
        with TempDir() as tmp:
            repo.download_artifacts(download_arg, dst_path=tmp.path())


@pytest.mark.parametrize(
    ("base_uri", "download_arg", "list_return_val"),
    [
        (_PARENT_MODEL_DIR, "", [_MODEL_FILE]),
        (_PARENT_MODEL_DIR, "", [".", _MODEL_FILE]),
        (_PARENT_DIR, _MODEL_DIR, [_MODEL_DIR + "/" + _MODEL_FILE]),
        (_PARENT_DIR, _MODEL_DIR, [_MODEL_DIR, _MODEL_DIR + "/" + _MODEL_FILE]),
        ("", _PARENT_MODEL_DIR, [_PARENT_MODEL_FILE]),
        ("", _PARENT_MODEL_DIR, [_PARENT_MODEL_DIR, _PARENT_MODEL_FILE]),
    ],
)
def test_download_artifacts_awaits_download_completion(base_uri, download_arg, list_return_val):
    """
    Verifies that all asynchronous artifact downloads are joined before `download_artifacts()`
    returns a result to the caller
    """

    def list_artifacts(path):
        fullpath = posixpath.join(base_uri, path)
        if fullpath.endswith(_MODEL_DIR) or fullpath.endswith(_MODEL_DIR + "/"):
            return [FileInfo(item, False, _DUMMY_FILE_SIZE) for item in list_return_val]
        elif fullpath.endswith(_PARENT_MODEL_DIR) or fullpath.endswith(_PARENT_MODEL_DIR + "/"):
            return [FileInfo(posixpath.join(path, _MODEL_DIR), True, _EMPTY_FILE_SIZE)]
        else:
            return []

    with mock.patch.object(SlowArtifactRepositoryImpl, "list_artifacts") as list_artifacts_mock:
        list_artifacts_mock.side_effect = list_artifacts
        repo = SlowArtifactRepositoryImpl(base_uri)
        repo.download_artifacts(download_arg)


@pytest.mark.parametrize(
    ("base_uri", "download_arg", "list_return_val"),
    [
        (_PARENT_MODEL_DIR, "", [_MODEL_FILE]),
    ],
)
def test_download_artifacts_provides_failure_info(base_uri, download_arg, list_return_val):
    def list_artifacts(path):
        fullpath = posixpath.join(base_uri, path)
        if fullpath.endswith(_MODEL_DIR) or fullpath.endswith(_MODEL_DIR + "/"):
            return [FileInfo(item, False, _DUMMY_FILE_SIZE) for item in list_return_val]
        else:
            return []

    with mock.patch.object(FailureArtifactRepositoryImpl, "list_artifacts") as list_artifacts_mock:
        list_artifacts_mock.side_effect = list_artifacts
        repo = FailureArtifactRepositoryImpl(base_uri)
        match = r"The following failures occurred while downloading one or more artifacts."
        with pytest.raises(MlflowException, match=match) as exc:
            repo.download_artifacts(download_arg)

        err_msg = str(exc.value)
        assert _MODEL_FILE in err_msg
        assert _MOCK_ERROR in err_msg


@pytest.mark.parametrize("debug", [True, False])
def test_download_artifacts_provides_traceback_info(debug, reset_logging_level):
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    def list_artifacts(path):
        fullpath = posixpath.join(_PARENT_MODEL_DIR, path)
        if fullpath.endswith(_MODEL_DIR) or fullpath.endswith(_MODEL_DIR + "/"):
            return [FileInfo(item, False, _DUMMY_FILE_SIZE) for item in [_MODEL_FILE]]
        else:
            return []

    with mock.patch.object(FailureArtifactRepositoryImpl, "list_artifacts") as list_artifacts_mock:
        list_artifacts_mock.side_effect = list_artifacts
        repo = FailureArtifactRepositoryImpl(_PARENT_MODEL_DIR)
        try:
            repo.download_artifacts("")
        except MlflowException as exc:
            err_msg = str(exc.message)
            if debug:
                assert "Traceback" in err_msg
            else:
                assert "Traceback" not in err_msg


@pytest.mark.parametrize(
    ("input_path", "expected_output"),
    [
        ("simple/path", "simple/path"),
        ("path/with:colon", "path/with_colon"),
        ("path/with<bracket", "path/with_bracket"),
        ("path/with>bracket", "path/with_bracket"),
        ('path/with"quote', "path/with_quote"),
        ("path/with|pipe", "path/with_pipe"),
        ("path/with?question", "path/with_question"),
        ("path/with*asterisk", "path/with_asterisk"),
        ("multiple:invalid<chars>in|path?", "multiple_invalid_chars_in_path_"),
        ("Subagent BonzaiSubAgent: Admin Area", "Subagent BonzaiSubAgent_ Admin Area"),
        # Test paths with spaces (should be preserved)
        ("path /with spaces", "path /with spaces"),
        ("leading / space", "leading / space"),
        ("trailing/ space", "trailing/ space"),
        ("spans/Component: Name /Other", "spans/Component_ Name /Other"),
    ],
)
def test_sanitize_path_for_windows(input_path, expected_output):
    result = _sanitize_path_for_windows(input_path)
    if is_windows():
        assert result == expected_output
    else:
        assert result == input_path


def test_sanitize_path_component_for_windows():
    if is_windows():
        assert _sanitize_path_component_for_windows("file:name") == "file_name"
        assert _sanitize_path_component_for_windows("simple") == "simple"
        assert _sanitize_path_component_for_windows("multi<>:|?*chars") == "multi______chars"
    else:
        assert _sanitize_path_component_for_windows("file:name") == "file:name"


def test_create_download_destination_sanitizes_windows_paths():
    class TestRepo(ArtifactRepository):
        def log_artifact(self, local_file, artifact_path=None):
            pass

        def log_artifacts(self, local_dir, artifact_path=None):
            pass

        def list_artifacts(self, path):
            pass

        def _download_file(self, remote_file_path, local_path):
            pass

    with TempDir() as tmp:
        repo = TestRepo("test://")
        artifact_path = "spans/Subagent placeholder name: placeholder name"
        result = repo._create_download_destination(artifact_path, tmp.path())

        if is_windows():
            # Verify the path uses backslashes and has sanitized the colon
            assert ":" not in result.split(tmp.path())[1]  # Check only the artifact part
            assert "\\" in result or "/" not in result  # Should use backslashes on Windows
        # The file should always be created successfully regardless of platform
        assert result.startswith(tmp.path())


def test_create_download_destination_prevents_path_traversal():
    """Test that path traversal attacks are prevented."""

    class TestRepo(ArtifactRepository):
        def log_artifact(self, local_file, artifact_path=None):
            pass

        def log_artifacts(self, local_dir, artifact_path=None):
            pass

        def list_artifacts(self, path):
            pass

        def _download_file(self, remote_file_path, local_path):
            pass

    with TempDir() as tmp:
        repo = TestRepo("test://")

        # Test various path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "../../etc/passwd",
            "../sensitive",
            "subdir/../../etc/passwd",
        ]

        for malicious_path in malicious_paths:
            with pytest.raises(
                MlflowException,
                match="Invalid artifact path.*",
            ):
                repo._create_download_destination(malicious_path, tmp.path())
