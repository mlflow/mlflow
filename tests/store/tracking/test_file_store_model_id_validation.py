from pathlib import Path

import pytest

from mlflow.entities import LoggedModelOutput, Metric
from mlflow.environment_variables import MLFLOW_ALLOW_FILE_STORE
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.file_store import FileStore

_ESCAPING_MODEL_ID_TEMPLATES = ["{target}", "../../../../{name}", "..", ".", "", ".. "]


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FileStore:
    monkeypatch.setenv(MLFLOW_ALLOW_FILE_STORE.name, "true")
    return FileStore(str(tmp_path / "mlruns"))


def _create_run_id(store: FileStore) -> str:
    experiment_id = store.create_experiment("exp")
    run = store.create_run(experiment_id, user_id="user", start_time=0, tags=[], run_name="run")
    return run.info.run_id


@pytest.mark.parametrize("template", _ESCAPING_MODEL_ID_TEMPLATES)
def test_log_outputs_rejects_model_ids_that_escape_the_outputs_dir(store, tmp_path, template):
    target = tmp_path / "escaped"
    model_id = template.format(target=target, name=target.name)
    run_id = _create_run_id(store)

    with pytest.raises(MlflowException, match="must be a single path component"):
        store.log_outputs(run_id, [LoggedModelOutput(model_id=model_id, step=0)])

    assert not target.exists()


def test_log_outputs_accepts_plain_model_id(store):
    run_id = _create_run_id(store)

    store.log_outputs(run_id, [LoggedModelOutput(model_id="m-123", step=0)])

    assert [o.model_id for o in store.get_run(run_id).outputs.model_outputs] == ["m-123"]


@pytest.mark.parametrize("template", _ESCAPING_MODEL_ID_TEMPLATES)
def test_log_metric_rejects_model_ids_that_escape_the_models_dir(store, tmp_path, template):
    target = tmp_path / "escaped"
    model_id = template.format(target=target, name=target.name)
    run_id = _create_run_id(store)
    metric = Metric("m", 1.0, 0, 0, model_id=model_id)

    with pytest.raises(MlflowException, match="must be a single path component"):
        store.log_metric(run_id, metric)

    assert not target.exists()
