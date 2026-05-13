import mlflow

_SAMPLE_TRACE = {
    "info": {
        "request_id": "2e72d64369624e6888324462b62dc120",
        "experiment_id": "0",
        "timestamp_ms": 1726145090860,
        "execution_time_ms": 162,
        "status": "OK",
        "request_metadata": {
            "mlflow.trace_schema.version": "2",
            "mlflow.traceInputs": '{"x": 1}',
            "mlflow.traceOutputs": '{"prediction": 1}',
        },
        "tags": {
            "fruit": "apple",
            "food": "pizza",
        },
    },
    "data": {
        "spans": [
            {
                "name": "remote",
                "context": {
                    "span_id": "0x337af925d6629c01",
                    "trace_id": "0x05e82d1fc4486f3986fae6dd7b5352b1",
                },
                "parent_id": None,
                "start_time": 1726145091022155863,
                "end_time": 1726145091022572053,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": '"2e72d64369624e6888324462b62dc120"',
                    "mlflow.spanType": '"UNKNOWN"',
                    "mlflow.spanInputs": '{"x": 1}',
                    "mlflow.spanOutputs": '{"prediction": 1}',
                },
                "events": [
                    {"name": "event", "timestamp": 1726145091022287, "attributes": {"foo": "bar"}}
                ],
            },
        ],
        "request": '{"x": 1}',
        "response": '{"prediction": 1}',
    },
}


class Model(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        mlflow.add_trace(_SAMPLE_TRACE)
        return 1


mlflow.models.set_model(Model())
