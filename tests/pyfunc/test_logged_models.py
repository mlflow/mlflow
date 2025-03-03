import mlflow
from mlflow.tracing.constant import TraceMetadataKey


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
