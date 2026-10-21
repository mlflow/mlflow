from pathlib import Path

import pytest

from mlflow.entities import TraceInfo, TraceLocation, TraceState
from mlflow.protos.service_pb2 import StartTraceV3, Trace
from mlflow.server import app, handlers
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.proto_json_utils import message_to_json


@pytest.mark.parametrize(
    "trace_id",
    ["../../../escaped-trace-dir", r"..\..\..\escaped-trace-dir", "%2e%2e%2fescaped", "NUL.txt"],
)
def test_start_trace_rejects_unsafe_file_store_ids(tmp_path, monkeypatch, trace_id):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.delenv("MLFLOW_SERVER_ARTIFACTS_ONLY", raising=False)
    store = FileStore(str(tmp_path / "mlruns"))
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: store)

    def request_body(trace_id):
        info = TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id("0"),
            request_time=1,
            state=TraceState.OK,
            trace_metadata={"metadata": "value"},
            tags={"tag": "value"},
        )
        return message_to_json(StartTraceV3(trace=Trace(trace_info=info.to_proto())))

    with app.test_client() as client:
        response = client.post(
            "/api/3.0/mlflow/traces",
            data=request_body(trace_id),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
        assert not (tmp_path / "escaped-trace-dir").exists()
        assert not (Path(store.root_directory) / "0" / "traces").exists()

        response = client.post(
            "/api/3.0/mlflow/traces",
            data=request_body("_custom.trace-v1"),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert store.get_trace_info("_custom.trace-v1").tags["tag"] == "value"
