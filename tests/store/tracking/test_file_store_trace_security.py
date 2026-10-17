import os
import subprocess
import uuid
from pathlib import Path
from unittest import mock

import pytest

from mlflow.entities import TraceInfo, TraceLocation, TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.file_utils import local_file_uri_to_path


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return FileStore(str(tmp_path / "mlruns"))


def _trace(trace_id, experiment_id="0"):
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id(experiment_id),
        request_time=1,
        state=TraceState.OK,
        trace_metadata={"metadata": "original"},
        tags={"tag": "original"},
    )


def _tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def _symlink(link, target):
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (NotImplementedError, OSError) as exc:
        if os.name == "nt" and target.is_dir():
            try:
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                    capture_output=True,
                    check=True,
                )
                return
            except subprocess.CalledProcessError as junction_exc:
                pytest.skip(f"Filesystem links and junctions are unavailable: {junction_exc}")
        pytest.skip(f"Filesystem links are unavailable: {exc}")


@pytest.mark.parametrize(
    "trace_id",
    [
        "../../../escaped-trace-dir",
        r"..\..\..\escaped-trace-dir",
        "a/b",
        r"a\b",
        ".",
        "..",
        "/tmp/escaped",
        r"C:\escaped",
        "C:escaped",
        r"\\server\share\escaped",
        "%2e%2e%2fescaped",
        "%252e%252e%252fescaped",
        "trace#fragment",
        "trace?query",
        "trace\n",
        "trace\x00",
        "NUL.txt",
        "",
        None,
        123,
    ],
)
def test_invalid_trace_id_has_no_side_effects(store, tmp_path, trace_id):
    trace = _trace(trace_id)
    before = _tree(tmp_path)
    with pytest.raises(MlflowException, match="Invalid FileStore trace ID") as exc:
        store.start_trace(trace)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert _tree(tmp_path) == before
    assert trace.tags == {"tag": "original"}


@pytest.mark.parametrize(
    "trace_id",
    [
        f"tr-{uuid.uuid4().hex}",
        uuid.uuid4().hex,
        "trace.v1",
        "_custom-id",
        "-legacy",
        ".hidden-trace",
        "trace with spaces",
        "追跡-α",
    ],
)
def test_safe_custom_trace_lifecycle(store, trace_id):
    trace = store.start_trace(_trace(trace_id))
    trace_dir = Path(store.root_directory) / "0" / "traces" / trace_id
    assert (trace_dir / "trace_info.yaml").is_file()
    artifact_path = Path(local_file_uri_to_path(trace.tags["mlflow.artifactLocation"]))
    assert artifact_path == trace_dir / "artifacts"
    assert store.get_trace_info(trace_id) == trace
    traces, _ = store.search_traces(["0"])
    assert [t.trace_id for t in traces] == [trace_id]
    store.set_trace_tag(trace_id, "key", "updated")
    assert store.get_trace_info(trace_id).tags["key"] == "updated"
    store.delete_trace_tag(trace_id, "key")
    assert "key" not in store.get_trace_info(trace_id).tags
    assert store.delete_traces("0", trace_ids=[trace_id, "missing-trace"]) == 1
    assert not trace_dir.exists()


def test_legacy_v2_trace_lifecycle(store):
    trace = store.deprecated_start_trace_v2("0", 1, {"metadata": "original"}, {})
    assert len(trace.request_id) == 32
    updated = store.deprecated_end_trace_v2(
        trace.request_id, 2, TraceStatus.OK, {"more": "metadata"}, {"tag": "value"}
    )
    assert updated.execution_time_ms == 1
    assert store.get_trace_info(trace.request_id).tags["tag"] == "value"
    assert store.delete_traces("0", max_timestamp_millis=2) == 1


@pytest.mark.parametrize("operation", ["lookup", "set_tag", "delete_tag", "artifact", "delete"])
def test_invalid_id_is_rejected_by_related_operations(store, tmp_path, operation):
    before = _tree(tmp_path)
    bad_id = "../../../escaped-trace-dir"
    operations = {
        "lookup": lambda: store.get_trace_info(bad_id),
        "set_tag": lambda: store.set_trace_tag(bad_id, "key", "value"),
        "delete_tag": lambda: store.delete_trace_tag(bad_id, "key"),
        "artifact": lambda: store._get_traces_artifact_dir("0", bad_id),
        "delete": lambda: store.delete_traces("0", trace_ids=[bad_id]),
    }
    with pytest.raises(MlflowException, match="Invalid FileStore trace ID") as exc:
        operations[operation]()
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert _tree(tmp_path) == before


def test_delete_validates_entire_id_list_before_deleting(store, tmp_path):
    store.start_trace(_trace("valid-trace"))
    before = _tree(tmp_path)
    with pytest.raises(MlflowException, match="Invalid FileStore trace ID") as exc:
        store.delete_traces("0", trace_ids=["valid-trace", "../../../escaped-trace-dir"])
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("boundary", ["traces", "trace"])
@pytest.mark.parametrize("operation", ["create", "lookup", "tag", "delete", "timestamp", "search"])
def test_outside_directory_links_are_rejected(store, tmp_path, boundary, operation):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("unchanged")
    traces_dir = Path(store.root_directory) / "0" / "traces"
    if boundary == "traces":
        _symlink(traces_dir, outside)
    else:
        traces_dir.mkdir()
        _symlink(traces_dir / "linked-trace", outside)

    operations = {
        "create": lambda: store.start_trace(_trace("linked-trace")),
        "lookup": lambda: store.get_trace_info("linked-trace"),
        "tag": lambda: store.set_trace_tag("linked-trace", "tag", "changed"),
        "delete": lambda: store.delete_traces("0", trace_ids=["linked-trace"]),
        "timestamp": lambda: store.delete_traces("0", max_timestamp_millis=2),
        "search": lambda: store.search_traces(["0"]),
    }
    with (
        mock.patch.object(
            store, "_get_trace_info_from_dir", wraps=store._get_trace_info_from_dir
        ) as read_info,
        pytest.raises(MlflowException, match="Invalid path") as exc,
    ):
        operations[operation]()
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    read_info.assert_not_called()
    assert sentinel.read_text() == "unchanged"
    assert list(outside.iterdir()) == [sentinel]


def test_timestamp_delete_validates_all_paths_before_deleting(store, tmp_path):
    store.start_trace(_trace("valid-trace"))
    outside = tmp_path / "outside"
    outside.mkdir()
    traces_dir = Path(store.root_directory) / "0" / "traces"
    _symlink(traces_dir / "linked-trace", outside)
    with pytest.raises(MlflowException, match="Invalid path"):
        store.delete_traces("0", max_timestamp_millis=2)
    assert (traces_dir / "valid-trace" / "trace_info.yaml").is_file()


@pytest.mark.parametrize("subfolder", ["tags", "request_metadata", "assessments"])
def test_trace_reads_reject_outside_subfolder_links(store, tmp_path, subfolder):
    store.start_trace(_trace("valid-trace"))
    trace_dir = Path(store.root_directory) / "0" / "traces" / "valid-trace"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("unchanged")
    folder = trace_dir / subfolder
    if folder.exists():
        for child in folder.iterdir():
            child.unlink()
        folder.rmdir()
    _symlink(folder, outside)
    with pytest.raises(MlflowException, match="Invalid path") as exc:
        store.get_trace_info("valid-trace")
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert sentinel.read_text() == "unchanged"


def test_tag_updates_reject_outside_file_links(store, tmp_path):
    store.start_trace(_trace("valid-trace"))
    outside = tmp_path / "sentinel"
    outside.write_text("unchanged")
    tag_path = Path(store.root_directory) / "0" / "traces" / "valid-trace" / "tags" / "tag"
    tag_path.unlink()
    _symlink(tag_path, outside)
    with pytest.raises(MlflowException, match="Invalid path"):
        store.set_trace_tag("valid-trace", "tag", "changed")
    with pytest.raises(MlflowException, match="Invalid path"):
        store.delete_trace_tag("valid-trace", "tag")
    assert outside.read_text() == "unchanged"
    assert os.path.islink(tag_path)


def test_legacy_v2_creation_rejects_outside_traces_link(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink(Path(store.root_directory) / "0" / "traces", outside)
    with pytest.raises(MlflowException, match="Invalid path"):
        store.deprecated_start_trace_v2("0", 1, {}, {})
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("operation", ["read", "create"])
def test_trace_info_rejects_outside_file_link(store, tmp_path, operation):
    store.start_trace(_trace("valid-trace"))
    info_path = Path(store.root_directory) / "0" / "traces" / "valid-trace" / "trace_info.yaml"
    outside = tmp_path / "sentinel.yaml"
    outside.write_bytes(info_path.read_bytes())
    before = outside.read_bytes()
    info_path.unlink()
    _symlink(info_path, outside)
    action = (
        (lambda: store.get_trace_info("valid-trace"))
        if operation == "read"
        else (lambda: store.start_trace(_trace("valid-trace")))
    )
    with pytest.raises(MlflowException, match="Invalid path"):
        action()
    assert outside.read_bytes() == before
