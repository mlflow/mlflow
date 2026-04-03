import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

import mlflow
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.utils.mlflow_tags import MLFLOW_MODEL_IS_EXTERNAL


class DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, model_input):
        return len(model_input) * [0]


class TraceModel(mlflow.pyfunc.PythonModel):
    @mlflow.trace
    def predict(self, model_input):
        return len(model_input) * [0]


def test_model_id_tracking():
    model = TraceModel()
    model.predict([1, 2, 3])
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert TraceMetadataKey.MODEL_ID not in trace.info.request_metadata

    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(name="my_model", python_model=model)
        # Log another model to ensure that the model ID is correctly associated with the first model
        mlflow.pyfunc.log_model(name="another_model", python_model=model)

    model = mlflow.pyfunc.load_model(info.model_uri)
    model.predict([4, 5, 6])

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == info.model_id


def test_model_id_tracking_evaluate():
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(name="my_model", python_model=TraceModel())

    mlflow.evaluate(model=info.model_uri, data=[[1, 2, 3]], model_type="regressor", targets=[1])
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == info.model_id


def test_model_id_tracking_thread_safety():
    models = []
    for _ in range(5):
        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(
                name="my_model",
                python_model=TraceModel(),
                pip_requirements=[],  # to skip dependency inference
            )
            model = mlflow.pyfunc.load_model(info.model_uri)
            models.append(model)

    def predict(idx, model) -> None:
        model.predict([idx])

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = [executor.submit(predict, idx, model) for idx, model in enumerate(models)]
        for f in futures:
            f.result()

    traces = mlflow.search_traces(return_type="list")
    assert len(traces) == len(models)
    for trace in traces:
        trace_inputs = trace.info.request_metadata["mlflow.traceInputs"]
        index = json.loads(trace_inputs)["model_input"][0]
        model_id = trace.info.request_metadata["mlflow.modelId"]
        assert model_id == models[index].model_id


def test_run_params_are_logged_to_model():
    with mlflow.start_run():
        mlflow.log_params({"a": 1})
        mlflow.pyfunc.log_model(name="my_model", python_model=DummyModel())

    model = mlflow.last_logged_model()
    assert model.params == {"a": "1"}


def test_run_metrics_are_logged_to_model():
    with mlflow.start_run():
        mlflow.log_metrics({"a": 1, "b": 2})
        mlflow.pyfunc.log_model(name="my_model", python_model=DummyModel())

    model = mlflow.last_logged_model()
    assert [(m.key, m.value) for m in model.metrics] == [("a", 1), ("b", 2)]


def test_log_model_finalizes_existing_pending_model():
    model = mlflow.initialize_logged_model(name="testmodel")
    assert model.status == LoggedModelStatus.PENDING
    mlflow.pyfunc.log_model(python_model=DummyModel(), model_id=model.model_id)
    updated_model = mlflow.get_logged_model(model.model_id)
    assert updated_model.status == LoggedModelStatus.READY


def test_log_model_permits_logging_to_ready_model(tmp_path):
    # Create a non-external model and finalize it to READY status
    model = mlflow.initialize_logged_model(name="testmodel")
    model = mlflow.finalize_logged_model(model.model_id, LoggedModelStatus.READY)
    assert model.status == LoggedModelStatus.READY
    assert model.tags.get(MLFLOW_MODEL_IS_EXTERNAL, "false").lower() == "false"

    # Verify we can log to the READY model
    mlflow.pyfunc.log_model(python_model=DummyModel(), model_id=model.model_id)

    # Verify the model can be loaded
    mlflow.pyfunc.load_model(f"models:/{model.model_id}")

    # Verify the model artifacts were updated
    dst_dir = os.path.join(tmp_path, "dst")
    mlflow.artifacts.download_artifacts(f"models:/{model.model_id}", dst_path=dst_dir)
    mlflow_model = Model.load(os.path.join(dst_dir, "MLmodel"))
    assert mlflow_model.flavors.get("python_function") is not None


def test_log_model_permits_logging_model_artifacts_to_external_models(tmp_path):
    model = mlflow.create_external_model(name="testmodel")
    assert model.status == LoggedModelStatus.READY
    assert model.tags.get(MLFLOW_MODEL_IS_EXTERNAL) == "true"
    dst_dir_1 = os.path.join(tmp_path, "dst_1")
    mlflow.artifacts.download_artifacts(f"models:/{model.model_id}", dst_path=dst_dir_1)
    mlflow_model: Model = Model.load(os.path.join(dst_dir_1, "MLmodel"))

    model_info = mlflow.pyfunc.log_model(python_model=DummyModel(), model_id=model.model_id)

    # Verify that the model can now be loaded and is no longer tagged as external
    mlflow.pyfunc.load_model(model_info.model_uri)
    assert MLFLOW_MODEL_IS_EXTERNAL not in mlflow.get_logged_model(model.model_id).tags
    dst_dir_2 = os.path.join(tmp_path, "dst_2")
    mlflow.artifacts.download_artifacts(f"models:/{model.model_id}", dst_path=dst_dir_2)
    mlflow_model = Model.load(os.path.join(dst_dir_2, "MLmodel"))
    assert MLFLOW_MODEL_IS_EXTERNAL not in (mlflow_model.metadata or {})


def test_external_logged_model_cannot_be_loaded_with_pyfunc():
    model = mlflow.create_external_model(name="testmodel")
    with pytest.raises(
        MlflowException,
        match="This model's artifacts are external.*cannot be loaded",
    ):
        mlflow.pyfunc.load_model(f"models:/{model.model_id}")
