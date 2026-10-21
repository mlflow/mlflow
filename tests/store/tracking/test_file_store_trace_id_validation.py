import os
from pathlib import Path

import pytest

from mlflow.entities import TraceInfo, TraceLocation, TraceState
from mlflow.environment_variables import MLFLOW_ALLOW_FILE_STORE
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.time import get_current_time_millis

# Templates for trace ids that resolve outside <experiment>/traces: an absolute path, a
# dot-dot path to the same target, the traces folder itself, its parent, and a child
# followed by "..".
_ESCAPING_TRACE_ID_TEMPLATES = [
    "{target}",
    "{target_relpath}",
    "..",
    ".",
    "",
    "tr-seed/..",
]


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FileStore:
    monkeypatch.setenv(MLFLOW_ALLOW_FILE_STORE.name, "true")
    return FileStore(str(tmp_path / "mlruns"))


@pytest.fixture
def victim_dir(tmp_path: Path) -> Path:
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "data.txt").write_text("data")
    return victim


def _trace_info(experiment_id: str, trace_id: str) -> TraceInfo:
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        state=TraceState.OK,
    )


def _traces_dir(tmp_path: Path, experiment_id: str) -> Path:
    return tmp_path / "mlruns" / experiment_id / FileStore.TRACES_FOLDER_NAME


def _escaping_trace_id(template: str, traces_dir: Path, target: Path) -> str:
    return template.format(target=target, target_relpath=os.path.relpath(target, traces_dir))


@pytest.mark.parametrize("template", _ESCAPING_TRACE_ID_TEMPLATES)
def test_delete_traces_rejects_trace_ids_that_escape_the_traces_dir(
    store: FileStore, tmp_path: Path, victim_dir: Path, template: str
):
    exp_id = store.create_experiment("exp")
    store.start_trace(_trace_info(exp_id, "tr-seed"))
    traces_dir = _traces_dir(tmp_path, exp_id)
    trace_id = _escaping_trace_id(template, traces_dir, victim_dir)

    with pytest.raises(MlflowException, match="parameter 'trace_id'") as exc_info:
        store.delete_traces(exp_id, trace_ids=[trace_id])

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert (victim_dir / "data.txt").is_file()
    assert (traces_dir / "tr-seed").is_dir()


def test_delete_traces_validates_every_trace_id_before_deleting_any(
    store: FileStore, tmp_path: Path, victim_dir: Path
):
    exp_id = store.create_experiment("exp")
    store.start_trace(_trace_info(exp_id, "tr-seed"))

    with pytest.raises(MlflowException, match="parameter 'trace_id'"):
        store.delete_traces(exp_id, trace_ids=["tr-seed", str(victim_dir)])

    assert (_traces_dir(tmp_path, exp_id) / "tr-seed").is_dir()
    assert (victim_dir / "data.txt").is_file()


@pytest.mark.parametrize("template", _ESCAPING_TRACE_ID_TEMPLATES)
def test_start_trace_rejects_trace_ids_that_escape_the_traces_dir(
    store: FileStore, tmp_path: Path, template: str
):
    exp_id = store.create_experiment("exp")
    traces_dir = _traces_dir(tmp_path, exp_id)
    outside = tmp_path / "outside"
    trace_id = _escaping_trace_id(template, traces_dir, outside)

    with pytest.raises(MlflowException, match="parameter 'trace_id'") as exc_info:
        store.start_trace(_trace_info(exp_id, trace_id))

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert not outside.exists()
    assert not traces_dir.exists()


def test_start_and_delete_trace_with_a_plain_trace_id(store: FileStore, tmp_path: Path):
    exp_id = store.create_experiment("exp")
    store.start_trace(_trace_info(exp_id, "tr-abc123"))
    trace_dir = _traces_dir(tmp_path, exp_id) / "tr-abc123"
    assert trace_dir.is_dir()

    assert store.delete_traces(exp_id, trace_ids=["tr-abc123"]) == 1
    assert not trace_dir.exists()
