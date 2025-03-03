import mlflow
from mlflow.tracing.constant import TraceMetadataKey


def test_model_id_tracking():
    class TraceModel(mlflow.pyfunc.PythonModel):
        @mlflow.trace
        def predict(self, model_input: list[int]) -> list[int]:
            return model_input

    model = TraceModel()
    assert model.predict([1, 2, 3]) == [1, 2, 3]
    trace = mlflow.get_last_active_trace()
    assert TraceMetadataKey.MODEL_ID not in trace.info.request_metadata

    with mlflow.start_run():
        info = mlflow.pyfunc.log_model("my_model", python_model=TraceModel())

    model = mlflow.pyfunc.load_model(info.model_uri)
    assert model.predict([4, 5, 6]) == [4, 5, 6]

    trace = mlflow.get_last_active_trace()
    assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == model.model_id
