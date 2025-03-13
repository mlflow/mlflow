import json
from concurrent.futures import ThreadPoolExecutor

import mlflow
from mlflow.tracing.constant import TraceMetadataKey


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
    trace = mlflow.get_last_active_trace()
    assert TraceMetadataKey.MODEL_ID not in trace.info.request_metadata

    with mlflow.start_run():
        info = mlflow.pyfunc.log_model("my_model", python_model=model)
        # Log another model to ensure that the model ID is correctly associated with the first model
        mlflow.pyfunc.log_model("another_model", python_model=model)

    model = mlflow.pyfunc.load_model(info.model_uri)
    model.predict([4, 5, 6])

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == info.model_id


def test_model_id_tracking_evaluate():
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model("my_model", python_model=TraceModel())

    mlflow.evaluate(model=info.model_uri, data=[[1, 2, 3]], model_type="regressor", targets=[1])
    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == info.model_id


def test_model_id_tracking_thread_safety():
    models = []
    for _ in range(5):
        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(
                "my_model",
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

    traces = mlflow.search_traces()
    assert len(traces) == len(models)
    for trace in mlflow.search_traces()["trace"]:
        trace_inputs = trace.info.request_metadata["mlflow.traceInputs"]
        index = json.loads(trace_inputs)["model_input"][0]
        model_id = trace.info.request_metadata["mlflow.modelId"]
        assert model_id == models[index].model_id


def test_logged_model_params_and_tags_are_logged_to_run():
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "my_model", python_model=DummyModel(), params={"a": 1}, tags={"b": 2}
        )

    run = mlflow.last_active_run()
    assert run.data.params["a"] == "1"
    assert run.data.tags["b"] == "2"
