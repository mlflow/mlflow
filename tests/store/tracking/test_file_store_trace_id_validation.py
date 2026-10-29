import os
from pathlib import Path

import pytest

from mlflow.entities import (
    AssessmentSource,
    AssessmentSourceType,
    Feedback,
    TraceInfo,
    TraceLocation,
    TraceState,
)
from mlflow.environment_variables import MLFLOW_ALLOW_FILE_STORE
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.time import get_current_time_millis

# Templates for trace ids that resolve outside <experiment>/traces: an absolute path, a
# dot-dot path to the same target, the traces folder itself, its parent, a child followed
# by "..", and names that Windows normalizes to "." or ".." by stripping trailing spaces
# and periods.
_ESCAPING_TRACE_ID_TEMPLATES = [
    "{target}",
    "{target_relpath}",
    "..",
    ".",
    "",
    "tr-seed/..",
    ".. ",
    ". ",
    "...",
    " ..",
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


@pytest.fixture
def victim_assessment(tmp_path: Path) -> Path:
    victim = tmp_path / "victim.yaml"
    victim.write_text("data")
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


@pytest.mark.parametrize("trace_id", ["tr-seed.", "tr-seed "])
def test_trace_ids_with_trailing_periods_or_spaces_are_rejected(store: FileStore, trace_id: str):
    exp_id = store.create_experiment("exp")

    with pytest.raises(MlflowException, match="parameter 'trace_id'"):
        store.start_trace(_trace_info(exp_id, trace_id))
    with pytest.raises(MlflowException, match="parameter 'trace_id'"):
        store.delete_traces(exp_id, trace_ids=[trace_id])


def test_start_and_delete_trace_with_a_plain_trace_id(store: FileStore, tmp_path: Path):
    exp_id = store.create_experiment("exp")
    store.start_trace(_trace_info(exp_id, "tr-abc123"))
    trace_dir = _traces_dir(tmp_path, exp_id) / "tr-abc123"
    assert trace_dir.is_dir()

    assert store.delete_traces(exp_id, trace_ids=["tr-abc123"]) == 1
    assert not trace_dir.exists()


def _feedback(trace_id: str, **kwargs) -> Feedback:
    return Feedback(
        trace_id=trace_id,
        name="correctness",
        value=True,
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
        **kwargs,
    )


@pytest.mark.parametrize(
    "trace_id",
    [
        "%2e%2e%2f%2e%2e%2foutside",
        "tr-a?b",
        "tr-a#b",
        "..\t",
        "..\n",
        "tr-a\x00b",
        "tr-a\x1fb",
        "tr-a\x7fb",
    ],
)
def test_trace_ids_with_uri_special_or_control_characters_are_rejected(
    store: FileStore, tmp_path: Path, trace_id: str
):
    exp_id = store.create_experiment("exp")

    with pytest.raises(MlflowException, match="parameter 'trace_id'") as exc_info:
        store.start_trace(_trace_info(exp_id, trace_id))
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert not _traces_dir(tmp_path, exp_id).exists()

    with pytest.raises(MlflowException, match="parameter 'trace_id'"):
        store.delete_traces(exp_id, trace_ids=[trace_id])


@pytest.mark.parametrize("template", ["{target}", "{target_relpath}"])
def test_assessment_ids_that_escape_the_assessments_dir_are_rejected(
    store: FileStore, tmp_path: Path, victim_assessment: Path, template: str
):
    exp_id = store.create_experiment("exp")
    store.start_trace(_trace_info(exp_id, "tr-seed"))
    store.create_assessment(_feedback("tr-seed"))
    assessments_dir = _traces_dir(tmp_path, exp_id) / "tr-seed" / FileStore.ASSESSMENTS_FOLDER_NAME
    assessment_id = _escaping_trace_id(template, assessments_dir, victim_assessment.with_suffix(""))

    with pytest.raises(MlflowException, match="parameter 'assessment_id'") as exc_info:
        store.get_assessment("tr-seed", assessment_id)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    with pytest.raises(MlflowException, match="parameter 'assessment_id'"):
        store.delete_assessment("tr-seed", assessment_id)
    with pytest.raises(MlflowException, match="parameter 'assessment_id'"):
        store.create_assessment(_feedback("tr-seed", overrides=assessment_id))

    assert victim_assessment.read_text() == "data"


def test_create_get_and_delete_an_assessment_with_a_generated_id(store: FileStore):
    exp_id = store.create_experiment("exp")
    store.start_trace(_trace_info(exp_id, "tr-seed"))

    created = store.create_assessment(_feedback("tr-seed"))
    assert created.assessment_id.startswith("a-")
    assert store.get_assessment("tr-seed", created.assessment_id).value is True
    assert [a.assessment_id for a in store.get_trace_info("tr-seed").assessments] == [
        created.assessment_id
    ]

    store.delete_assessment("tr-seed", created.assessment_id)
    assert store.get_trace_info("tr-seed").assessments == []
