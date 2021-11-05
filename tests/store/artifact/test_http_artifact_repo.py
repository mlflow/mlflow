import os
import posixpath
from unittest import mock

import pytest

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_artifact_uri_factory(scheme):
    repo = get_artifact_repository(f"{scheme}://test.com")
    assert isinstance(repo, HttpArtifactRepository)


class MockResponse:
    def __init__(self, data, status_code):
        self.data = data
        self.status_code = status_code

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("request failed")


class MockStreamResponse(MockResponse):
    def iter_content(self, chunk_size):  # pylint: disable=unused-argument
        yield self.data.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class FileObjectMatcher:
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode

    def __eq__(self, other):
        return self.name == other.name and self.mode == other.mode


@pytest.fixture
def http_artifact_repo():
    artifact_uri = "http://test.com/api/2.0/mlflow-artifacts/artifacts"
    return HttpArtifactRepository(artifact_uri)


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_artifact(http_artifact_repo, tmpdir, artifact_path):
    tmp_path = tmpdir.join("a.txt")
    tmp_path.write("0")
    with mock.patch("requests.Session.put", return_value=MockResponse({}, 200)) as mock_put:
        http_artifact_repo.log_artifact(tmp_path, artifact_path)
        paths = (artifact_path,) if artifact_path else ()
        expected_url = posixpath.join(http_artifact_repo.artifact_uri, *paths, tmp_path.basename)
        mock_put.assert_called_once_with(
            expected_url, data=FileObjectMatcher(tmp_path, "rb"), timeout=mock.ANY
        )

    with mock.patch("requests.Session.put", return_value=MockResponse({}, 400)) as mock_put:
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo.log_artifact(tmp_path, artifact_path)


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_artifacts(http_artifact_repo, tmpdir, artifact_path):
    tmp_path_a = tmpdir.join("a.txt")
    tmp_path_b = tmpdir.mkdir("dir").join("b.txt")
    tmp_path_a.write("0")
    tmp_path_b.write("1")

    with mock.patch("requests.Session.put", return_value=MockResponse({}, 200)) as mock_put:
        http_artifact_repo.log_artifacts(tmpdir, artifact_path)
        paths = (artifact_path,) if artifact_path else ()
        expected_url_1 = posixpath.join(
            http_artifact_repo.artifact_uri, *paths, tmp_path_a.basename
        )
        expected_url_2 = posixpath.join(
            http_artifact_repo.artifact_uri, *paths, "dir", tmp_path_b.basename
        )
        calls = [(args[0], kwargs["data"]) for args, kwargs in mock_put.call_args_list]
        assert calls == [
            (expected_url_1, FileObjectMatcher(tmp_path_a, "rb")),
            (expected_url_2, FileObjectMatcher(tmp_path_b, "rb")),
        ]

    with mock.patch("requests.Session.put", return_value=MockResponse({}, 400)) as mock_put:
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo.log_artifacts(tmpdir, artifact_path)


def test_list_artifacts(http_artifact_repo):
    with mock.patch("requests.Session.get", return_value=MockResponse({}, 200)) as mock_get:
        assert http_artifact_repo.list_artifacts() == []
        mock_get.assert_called_once_with(
            http_artifact_repo.artifact_uri, params={"path": ""}, timeout=mock.ANY
        )

    with mock.patch(
        "requests.Session.get",
        return_value=MockResponse(
            {
                "files": [
                    {"path": "1.txt", "is_dir": False, "file_size": 1},
                    {"path": "dir", "is_dir": True},
                ]
            },
            200,
        ),
    ) as mock_get:
        assert [a.path for a in http_artifact_repo.list_artifacts()] == ["1.txt", "dir"]

    with mock.patch(
        "requests.Session.get",
        return_value=MockResponse(
            {
                "files": [
                    {"path": "1.txt", "is_dir": False, "file_size": 1},
                    {"path": "dir", "is_dir": True},
                ]
            },
            200,
        ),
    ) as mock_get:
        assert [a.path for a in http_artifact_repo.list_artifacts(path="path")] == [
            "path/1.txt",
            "path/dir",
        ]

    with mock.patch("requests.Session.get", return_value=MockResponse({}, 400)) as mock_get:
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo.list_artifacts()


def read_file(path):
    with open(path) as f:
        return f.read()


@pytest.mark.parametrize("remote_file_path", ["a.txt", "dir/b.xtx"])
def test_download_file(http_artifact_repo, tmpdir, remote_file_path):
    with mock.patch(
        "requests.Session.get", return_value=MockStreamResponse("data", 200)
    ) as mock_get:
        tmp_path = tmpdir.join(posixpath.basename(remote_file_path))
        http_artifact_repo._download_file(remote_file_path, tmp_path)
        expected_url = posixpath.join(http_artifact_repo.artifact_uri, remote_file_path)
        mock_get.assert_called_once_with(expected_url, stream=True, timeout=mock.ANY)
        with open(tmp_path) as f:
            assert f.read() == "data"

    with mock.patch(
        "requests.Session.get", return_value=MockStreamResponse("data", 400)
    ) as mock_get:
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo._download_file(remote_file_path, tmp_path)


def test_download_artifacts(http_artifact_repo, tmpdir):
    # This test simulates downloading artifacts in the following structure:
    # ---------
    # - a.txt
    # - dir
    #   - b.txt
    # ---------
    side_effect = [
        # Response for `list_experiments("")` called by `_is_directory("")`
        MockResponse(
            {
                "files": [
                    {"path": "a.txt", "is_dir": False, "file_size": 6},
                    {"path": "dir", "is_dir": True},
                ]
            },
            200,
        ),
        # Response for `list_experiments("")`
        MockResponse(
            {
                "files": [
                    {"path": "a.txt", "is_dir": False, "file_size": 6},
                    {"path": "dir", "is_dir": True},
                ]
            },
            200,
        ),
        # Response for `_download_file("a.txt")`
        MockStreamResponse("data_a", 200),
        # Response for `list_experiments("dir")`
        MockResponse({"files": [{"path": "b.txt", "is_dir": False, "file_size": 1}]}, 200),
        # Response for `_download_file("dir/b.txt")`
        MockStreamResponse("data_b", 200),
    ]
    with mock.patch("requests.Session.get", side_effect=side_effect):
        http_artifact_repo.download_artifacts("", tmpdir)
        paths = [os.path.join(root, f) for root, _, files in os.walk(tmpdir) for f in files]
        assert [os.path.relpath(p, tmpdir) for p in paths] == [
            "a.txt",
            os.path.join("dir", "b.txt"),
        ]
        assert read_file(paths[0]) == "data_a"
        assert read_file(paths[1]) == "data_b"
