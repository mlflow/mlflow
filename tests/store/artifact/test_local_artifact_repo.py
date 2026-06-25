import io
import json
import os
import pathlib
import posixpath
import stat
from unittest import mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.span import Span, SpanAttributeKey
from mlflow.entities.trace_data import TraceData
from mlflow.exceptions import MlflowException, MlflowTraceDataCorrupted, MlflowTraceDataNotFound
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.tracing.otel.otel_archival import TRACE_ARCHIVAL_FILENAME, spans_to_traces_data_pb
from mlflow.tracing.utils import build_otel_context
from mlflow.utils.file_utils import TempDir


@pytest.fixture
def local_artifact_root(tmp_path):
    return str(tmp_path)


@pytest.fixture
def local_artifact_repo(local_artifact_root):
    from mlflow.utils.file_utils import path_to_local_file_uri

    return LocalArtifactRepository(artifact_uri=path_to_local_file_uri(local_artifact_root))


def test_list_artifacts(local_artifact_repo, local_artifact_root):
    assert len(local_artifact_repo.list_artifacts()) == 0

    artifact_rel_path = "artifact"
    artifact_path = os.path.join(local_artifact_root, artifact_rel_path)
    with open(artifact_path, "w") as f:
        f.write("artifact")
    artifacts_list = local_artifact_repo.list_artifacts()
    assert len(artifacts_list) == 1
    assert artifacts_list[0].path == artifact_rel_path


def test_log_artifacts(local_artifact_repo, local_artifact_root):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir() as src_dir:
        artifact_src_path = src_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path)

    artifacts_list = local_artifact_repo.list_artifacts()
    assert len(artifacts_list) == 1
    assert artifacts_list[0].path == artifact_rel_path

    artifact_dst_path = os.path.join(local_artifact_root, artifact_rel_path)
    assert os.path.exists(artifact_dst_path)
    assert artifact_dst_path != artifact_src_path
    with open(artifact_dst_path) as f:
        assert f.read() == artifact_text


def test_log_artifact_preserves_source_file_mtime(local_artifact_repo, local_artifact_root):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    requested_mtime_ns = 1_700_000_000_123_456_789

    with TempDir() as src_dir:
        artifact_src_path = src_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        os.utime(artifact_src_path, ns=(requested_mtime_ns, requested_mtime_ns))
        expected_mtime_ns = os.stat(artifact_src_path).st_mtime_ns
        local_artifact_repo.log_artifact(artifact_src_path)

    artifact_dst_path = os.path.join(local_artifact_root, artifact_rel_path)
    assert os.stat(artifact_dst_path).st_mtime_ns == expected_mtime_ns


@pytest.mark.parametrize("dst_path", [None, "dest"])
def test_download_artifacts(local_artifact_repo, dst_path):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    empty_dir_path = "empty_dir"
    with TempDir(chdr=True) as local_dir:
        if dst_path:
            os.mkdir(dst_path)
        artifact_src_path = local_dir.path(artifact_rel_path)
        os.mkdir(local_dir.path(empty_dir_path))
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifacts(local_dir.path())
        result = local_artifact_repo.download_artifacts(
            artifact_path=artifact_rel_path, dst_path=dst_path
        )
        with open(result) as f:
            assert f.read() == artifact_text
        result = local_artifact_repo.download_artifacts(artifact_path="", dst_path=dst_path)
        empty_dir_dst_path = os.path.join(result, empty_dir_path)
        assert os.path.isdir(empty_dir_dst_path)
        assert len(os.listdir(empty_dir_dst_path)) == 0


def test_download_artifacts_does_not_copy(local_artifact_repo):
    """
    The LocalArtifactRepository.download_artifact function should not copy the artifact if
    the ``dst_path`` argument is None.
    """
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir(chdr=True) as local_dir:
        artifact_src_path = local_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path)
        dst_path = local_artifact_repo.download_artifacts(artifact_path=artifact_rel_path)
        with open(dst_path) as f:
            assert f.read() == artifact_text
        assert dst_path.startswith(local_artifact_repo.artifact_dir), (
            "downloaded artifact is not in local_artifact_repo.artifact_dir root"
        )


def test_get_local_path_returns_absolute_path(local_artifact_repo):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir(chdr=True) as local_dir:
        artifact_src_path = local_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path)

        local_path = local_artifact_repo.get_local_path(artifact_rel_path)

    assert os.path.abspath(local_path) == local_path
    assert local_path.startswith(local_artifact_repo.artifact_dir)
    with open(local_path) as f:
        assert f.read() == artifact_text


def test_download_artifacts_returns_absolute_paths(local_artifact_repo):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir(chdr=True) as local_dir:
        artifact_src_path = local_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path)

        for dst_dir in ["dst1", local_dir.path("dst2"), None]:
            if dst_dir is not None:
                os.makedirs(dst_dir)
            dst_path = local_artifact_repo.download_artifacts(
                artifact_path=artifact_rel_path, dst_path=dst_dir
            )
            if dst_dir is not None:
                # If dst_dir isn't none, assert we're actually downloading to dst_dir.
                assert dst_path.startswith(os.path.abspath(dst_dir))
            assert dst_path == os.path.abspath(dst_path)


@pytest.mark.parametrize("repo_subdir_path", ["aaa", "aaa/bbb", "aaa/bbb/ccc/ddd"])
def test_artifacts_are_logged_to_and_downloaded_from_repo_subdirectory_successfully(
    local_artifact_repo, repo_subdir_path
):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir(chdr=True) as local_dir:
        artifact_src_path = local_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path, artifact_path=repo_subdir_path)

    downloaded_subdir = local_artifact_repo.download_artifacts(repo_subdir_path)
    assert os.path.isdir(downloaded_subdir)
    subdir_contents = os.listdir(downloaded_subdir)
    assert len(subdir_contents) == 1
    assert artifact_rel_path in subdir_contents
    with open(os.path.join(downloaded_subdir, artifact_rel_path)) as f:
        assert f.read() == artifact_text

    downloaded_file = local_artifact_repo.download_artifacts(
        posixpath.join(repo_subdir_path, artifact_rel_path)
    )
    with open(downloaded_file) as f:
        assert f.read() == artifact_text


def test_log_artifact_throws_exception_for_invalid_artifact_paths(local_artifact_repo):
    with TempDir() as local_dir:
        for bad_artifact_path in ["/", "//", "/tmp", "/bad_path", ".", "../terrible_path"]:
            with pytest.raises(MlflowException, match="Invalid artifact path"):
                local_artifact_repo.log_artifact(local_dir.path(), bad_artifact_path)


def test_logging_directory_of_artifacts_produces_expected_repo_contents(local_artifact_repo):
    with TempDir() as local_dir:
        os.mkdir(local_dir.path("subdir"))
        os.mkdir(local_dir.path("subdir", "nested"))
        with open(local_dir.path("subdir", "a.txt"), "w") as f:
            f.write("A")
        with open(local_dir.path("subdir", "b.txt"), "w") as f:
            f.write("B")
        with open(local_dir.path("subdir", "nested", "c.txt"), "w") as f:
            f.write("C")
        local_artifact_repo.log_artifacts(local_dir.path("subdir"))
        with open(local_artifact_repo.download_artifacts("a.txt")) as f:
            assert f.read() == "A"
        with open(local_artifact_repo.download_artifacts("b.txt")) as f:
            assert f.read() == "B"
        with open(local_artifact_repo.download_artifacts("nested/c.txt")) as f:
            assert f.read() == "C"


def test_hidden_files_are_logged_correctly(local_artifact_repo):
    with TempDir() as local_dir:
        hidden_file = local_dir.path(".mystery")
        with open(hidden_file, "w") as f:
            f.write("42")
        local_artifact_repo.log_artifact(hidden_file)
        with open(local_artifact_repo.download_artifacts(".mystery")) as f:
            assert f.read() == "42"


def test_log_artifact_is_noop_when_source_matches_destination(local_artifact_repo, local_artifact_root):
    artifact_path = os.path.join(local_artifact_root, "test.txt")
    with open(artifact_path, "w") as f:
        f.write("artifact")

    with mock.patch.object(local_artifact_repo, "_write_to_destination_path") as write_mock:
        local_artifact_repo.log_artifact(artifact_path)

    write_mock.assert_not_called()


def test_log_artifact_rejects_internal_temp_file_prefix(local_artifact_repo):
    with TempDir() as local_dir:
        local_path = local_dir.path(".artifact.uploading.mock")
        with open(local_path, "w") as f:
            f.write("artifact")

        with pytest.raises(MlflowException, match="Artifact names starting with"):
            local_artifact_repo.log_artifact(local_path)


def test_log_artifact_from_stream_writes_file_atomically(local_artifact_repo):
    artifact_contents = b"hello world!"

    local_artifact_repo.log_artifact_from_stream(
        io.BytesIO(artifact_contents),
        "test.txt",
        artifact_path="nested",
    )

    artifact_dir = pathlib.Path(local_artifact_repo.artifact_dir) / "nested"
    assert (artifact_dir / "test.txt").read_bytes() == artifact_contents
    assert list(artifact_dir.glob(".artifact.uploading.*")) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only permission semantics")
def test_log_artifact_from_stream_preserves_default_file_mode(local_artifact_repo):
    artifact_contents = b"hello world!"
    artifact_dir = pathlib.Path(local_artifact_repo.artifact_dir) / "nested"
    artifact_dir.mkdir()

    reference_path = artifact_dir / "reference.txt"
    with open(reference_path, "wb"):
        pass
    expected_mode = stat.S_IMODE(reference_path.stat().st_mode)
    reference_path.unlink()

    local_artifact_repo.log_artifact_from_stream(
        io.BytesIO(artifact_contents),
        "test.txt",
        artifact_path="nested",
    )

    final_path = artifact_dir / "test.txt"
    assert stat.S_IMODE(final_path.stat().st_mode) == expected_mode


def test_log_artifact_from_stream_cleans_up_temp_file_on_failure(local_artifact_repo):
    artifact_dir = pathlib.Path(local_artifact_repo.artifact_dir) / "nested"
    artifact_dir.mkdir()
    final_path = artifact_dir / "test.txt"
    final_path.write_bytes(b"existing-content")

    class FailingStream:
        def __init__(self):
            self._chunks = iter([b"new-content", RuntimeError("stream failed")])

        def read(self, _size: int) -> bytes:
            chunk = next(self._chunks, b"")
            if isinstance(chunk, Exception):
                raise chunk
            return chunk

    with pytest.raises(RuntimeError, match="stream failed"):
        local_artifact_repo.log_artifact_from_stream(
            FailingStream(),
            "test.txt",
            artifact_path="nested",
        )

    assert final_path.read_bytes() == b"existing-content"
    assert list(artifact_dir.glob(".artifact.uploading.*")) == []


def test_log_artifact_from_stream_closes_fd_when_fdopen_fails(local_artifact_repo):
    artifact_dir = pathlib.Path(local_artifact_repo.artifact_dir) / "nested"
    artifact_dir.mkdir()
    temp_artifact_path = artifact_dir / ".artifact.uploading.mock"
    temp_artifact_path.touch()

    with (
        mock.patch(
            "mlflow.store.artifact.local_artifact_repo.tempfile.mkstemp",
            return_value=(123, str(temp_artifact_path)),
        ),
        mock.patch(
            "mlflow.store.artifact.local_artifact_repo.os.fdopen",
            side_effect=OSError("fdopen failed"),
        ),
        mock.patch("mlflow.store.artifact.local_artifact_repo.os.close") as close_mock,
        mock.patch("mlflow.store.artifact.local_artifact_repo.os.remove") as remove_mock,
    ):
        with pytest.raises(OSError, match="fdopen failed"):
            local_artifact_repo.log_artifact_from_stream(
                io.BytesIO(b"hello world!"),
                "test.txt",
                artifact_path="nested",
            )

    close_mock.assert_called_once_with(123)
    remove_mock.assert_called_once_with(str(temp_artifact_path))


def test_delete_artifacts_folder(local_artifact_repo):
    with TempDir() as local_dir:
        os.mkdir(local_dir.path("subdir"))
        os.mkdir(local_dir.path("subdir", "nested"))
        with open(local_dir.path("subdir", "a.txt"), "w") as f:
            f.write("A")
        with open(local_dir.path("subdir", "b.txt"), "w") as f:
            f.write("B")
        with open(local_dir.path("subdir", "nested", "c.txt"), "w") as f:
            f.write("C")
        local_artifact_repo.log_artifacts(local_dir.path("subdir"))
        assert os.path.exists(os.path.join(local_artifact_repo._artifact_dir, "nested"))
        assert os.path.exists(os.path.join(local_artifact_repo._artifact_dir, "a.txt"))
        assert os.path.exists(os.path.join(local_artifact_repo._artifact_dir, "b.txt"))
        local_artifact_repo.delete_artifacts()
        assert not os.path.exists(os.path.join(local_artifact_repo._artifact_dir))


def test_delete_artifacts_files(local_artifact_repo, tmp_path):
    subdir = tmp_path / "subdir"
    nested = subdir / "nested"
    subdir.mkdir()
    nested.mkdir()

    (subdir / "a.txt").write_text("A")
    (subdir / "b.txt").write_text("B")
    (nested / "c.txt").write_text("C")

    local_artifact_repo.log_artifacts(str(subdir))
    artifact_dir = pathlib.Path(local_artifact_repo._artifact_dir)
    assert (artifact_dir / "nested").exists()
    assert (artifact_dir / "a.txt").exists()
    assert (artifact_dir / "b.txt").exists()

    local_artifact_repo.delete_artifacts(artifact_path="nested/c.txt")
    local_artifact_repo.delete_artifacts(artifact_path="b.txt")

    assert not (artifact_dir / "nested" / "c.txt").exists()
    assert not (artifact_dir / "b.txt").exists()
    assert (artifact_dir / "a.txt").exists()


def test_delete_artifacts_with_nonexistent_path_succeeds(local_artifact_repo):
    local_artifact_repo.delete_artifacts("nonexistent")


def test_download_artifacts_invalid_remote_file_path(local_artifact_repo):
    with pytest.raises(MlflowException, match="Invalid path"):
        local_artifact_repo.download_artifacts("/absolute/path/to/file")


def test_trace_data(local_artifact_repo):
    with pytest.raises(MlflowTraceDataNotFound, match=r"Trace data not found for path="):
        local_artifact_repo.download_trace_data()
    local_artifact_repo.upload_trace_data("invalid data")
    with pytest.raises(MlflowTraceDataCorrupted, match=r"Trace data is corrupted for path="):
        local_artifact_repo.download_trace_data()

    mock_trace_data = {"spans": [], "request": {"test": 1}, "response": {"test": 2}}
    local_artifact_repo.upload_trace_data(json.dumps(mock_trace_data))
    assert local_artifact_repo.download_trace_data() == mock_trace_data


def _make_span() -> Span:
    otel_span = OTelReadableSpan(
        name="test-span",
        context=build_otel_context(1, 10),
        start_time=1_000_000,
        end_time=2_000_000,
        attributes={
            SpanAttributeKey.REQUEST_ID: json.dumps("tr-abc123"),
            SpanAttributeKey.INPUTS: json.dumps({"q": "hello"}),
            SpanAttributeKey.OUTPUTS: json.dumps({"a": "world"}),
            SpanAttributeKey.SPAN_TYPE: json.dumps("UNKNOWN"),
        },
    )
    return Span(otel_span)


def test_archived_trace_data_errors(local_artifact_repo):
    assert (
        local_artifact_repo.download_archived_trace_data().to_dict()
        == TraceData(spans=[]).to_dict()
    )

    trace_pb_path = pathlib.Path(local_artifact_repo.artifact_dir, TRACE_ARCHIVAL_FILENAME)
    trace_pb_path.write_bytes(b"")
    with pytest.raises(MlflowTraceDataCorrupted, match=r"Trace data is corrupted for path="):
        local_artifact_repo.download_archived_trace_data()


def test_upload_archived_trace_data_rejects_empty_spans(local_artifact_repo):
    with pytest.raises(MlflowException, match="at least one span"):
        local_artifact_repo.upload_archived_trace_data(TraceData(spans=[]))


def test_trace_data_artifact_repo(local_artifact_repo):
    trace_data = TraceData(spans=[_make_span()]).to_dict()

    local_artifact_repo.upload_trace_data(json.dumps(trace_data))

    restored = local_artifact_repo.download_trace_data()
    assert restored == trace_data


def test_upload_archived_trace_data_rejects_non_trace_data(local_artifact_repo):
    with pytest.raises(MlflowException, match="Archived trace data must be a TraceData object."):
        local_artifact_repo.upload_archived_trace_data("not-trace-data")


def test_archived_trace_data_with_trace_data_object(local_artifact_repo):
    trace_data = TraceData(spans=[_make_span()])

    local_artifact_repo.upload_archived_trace_data(trace_data)

    restored = local_artifact_repo.download_archived_trace_data()
    assert restored.to_dict() == trace_data.to_dict()


def test_upload_archived_trace_data_bytes(local_artifact_repo):
    trace_data = TraceData(spans=[_make_span()])

    local_artifact_repo.upload_archived_trace_data_bytes(spans_to_traces_data_pb(trace_data.spans))

    restored = local_artifact_repo.download_archived_trace_data()
    assert restored.to_dict() == trace_data.to_dict()


@pytest.fixture
def external_secret_dir(tmp_path):
    secret_dir = tmp_path.parent / "secrets_outside"
    secret_dir.mkdir(exist_ok=True)
    secret_file = secret_dir / "secret.txt"
    secret_file.touch()
    return secret_dir


def _execute_operation(local_artifact_repo, operation, access_path, tmp_path):
    if operation == "download_artifacts":
        local_artifact_repo.download_artifacts(access_path)
    elif operation == "get_local_path":
        local_artifact_repo.get_local_path(access_path)
    elif operation == "list_artifacts":
        local_artifact_repo.list_artifacts(access_path)
    elif operation == "_download_file":
        dst_path = tmp_path / "downloaded.txt"
        local_artifact_repo._download_file(access_path, str(dst_path))


@pytest.mark.parametrize(
    ("symlink_name", "access_path", "operation"),
    [
        ("leak", "leak/secret.txt", "download_artifacts"),
        ("leak", "leak/secret.txt", "get_local_path"),
        ("leak", "leak", "list_artifacts"),
        ("leak", "leak/secret.txt", "_download_file"),
        ("parent_link", "parent_link/secret.txt", "download_artifacts"),
    ],
)
def test_symlink_path_traversal_blocked(
    local_artifact_repo, external_secret_dir, tmp_path, symlink_name, access_path, operation
):
    artifact_dir = pathlib.Path(local_artifact_repo.artifact_dir)
    symlink_path = artifact_dir / symlink_name
    symlink_path.symlink_to(external_secret_dir)

    with pytest.raises(MlflowException, match="resolved path is outside the artifact directory"):
        _execute_operation(local_artifact_repo, operation, access_path, tmp_path)


def test_nested_symlink_traversal_blocked(local_artifact_repo, external_secret_dir):
    artifact_dir = pathlib.Path(local_artifact_repo.artifact_dir)
    nested_dir = artifact_dir / "nested"
    nested_dir.mkdir()
    symlink_path = nested_dir / "leak"
    symlink_path.symlink_to(external_secret_dir)

    with pytest.raises(MlflowException, match="resolved path is outside the artifact directory"):
        local_artifact_repo.download_artifacts("nested/leak/secret.txt")


@pytest.mark.parametrize(
    ("setup_type", "access_path", "expected_content"),
    [
        ("file", "artifact_link.txt", "LEGITIMATE_CONTENT"),
        ("subdir", "link_to_subdir/file.txt", "CONTENT"),
    ],
)
def test_symlink_within_artifact_dir_allowed(
    local_artifact_repo, setup_type, access_path, expected_content
):
    artifact_dir = pathlib.Path(local_artifact_repo.artifact_dir)

    if setup_type == "file":
        real_file = artifact_dir / "real_artifact.txt"
        real_file.write_text(expected_content)
        symlink_path = artifact_dir / "artifact_link.txt"
        symlink_path.symlink_to(real_file)
    elif setup_type == "subdir":
        subdir = artifact_dir / "subdir"
        subdir.mkdir()
        real_file = subdir / "file.txt"
        real_file.write_text(expected_content)
        symlink_path = artifact_dir / "link_to_subdir"
        symlink_path.symlink_to(subdir)

    result = local_artifact_repo.download_artifacts(access_path)
    with open(result) as f:
        assert f.read() == expected_content

    if setup_type == "subdir":
        artifacts = local_artifact_repo.list_artifacts("link_to_subdir")
        assert len(artifacts) == 1
        assert artifacts[0].path == "link_to_subdir/file.txt"


def test_list_artifacts_on_file_returns_empty(local_artifact_repo, local_artifact_root):
    artifact_path = os.path.join(local_artifact_root, "file.txt")
    with open(artifact_path, "w") as f:
        f.write("data")
    assert local_artifact_repo.list_artifacts("file.txt") == []


@pytest.mark.parametrize("dst_path_provided", [True, False])
def test_download_artifacts_nonexistent_raises_resource_does_not_exist(
    local_artifact_repo, tmp_path, dst_path_provided
):
    dst = str(tmp_path) if dst_path_provided else None
    with pytest.raises(MlflowException, match="No such artifact") as exc_info:
        local_artifact_repo.download_artifacts("nonexistent.txt", dst_path=dst)
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_get_local_path_nonexistent_raises_resource_does_not_exist(local_artifact_repo):
    with pytest.raises(MlflowException, match="No such artifact") as exc_info:
        local_artifact_repo.get_local_path("nonexistent.txt")
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"
