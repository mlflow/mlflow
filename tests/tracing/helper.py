import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import Event, ReadableSpan

import mlflow
from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.ml_package_versions import FLAVOR_TO_MODULE_NAME
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.export.inference_table import pop_trace
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.provider import _get_tracer
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS, get_autolog_function
from mlflow.utils.autologging_utils.safety import revert_patches
from mlflow.version import IS_TRACING_SDK_ONLY


def create_mock_otel_span(
    trace_id: int,
    span_id: int,
    name: str = "test_span",
    parent_id: int | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
):
    """
    Create a mock OpenTelemetry span for testing purposes.

    OpenTelemetry doesn't allow creating a span outside of a tracer. So here we create a mock span
    that extends ReadableSpan (data object) and exposes the necessary attributes for testing.
    """

    @dataclass
    class _MockSpanContext:
        trace_id: str
        span_id: str
        trace_flags: trace_api.TraceFlags = trace_api.TraceFlags(1)
        trace_state: trace_api.TraceState = field(default_factory=trace_api.TraceState)

    class _MockOTelSpan(trace_api.Span, ReadableSpan):
        def __init__(
            self,
            name,
            context,
            parent,
            start_time=None,
            end_time=None,
            status=trace_api.Status(trace_api.StatusCode.UNSET),
        ):
            self._name = name
            self._parent = parent
            self._context = context
            self._start_time = start_time if start_time is not None else int(time.time() * 1e9)
            self._end_time = end_time
            self._status = status
            self._attributes = {}
            self._events = []

        # NB: The following methods are defined as abstract method in the Span class.
        def set_attributes(self, attributes):
            self._attributes.update(attributes)

        def set_attribute(self, key, value):
            self._attributes[key] = value

        def set_status(self, status):
            self._status = status

        def add_event(self, name, attributes=None, timestamp=None):
            self._events.append(Event(name, attributes, timestamp))

        def get_span_context(self):
            return self._context

        def is_recording(self):
            return self._end_time is None

        def update_name(self, name):
            self.name = name

        def end(self, end_time_ns=None):
            pass

        def record_exception():
            pass

    return _MockOTelSpan(
        name=name,
        context=_MockSpanContext(trace_id, span_id),
        parent=_MockSpanContext(trace_id, parent_id) if parent_id else None,
        start_time=start_time,
        end_time=end_time,
    )


def create_trace(request_id) -> Trace:
    return Trace(info=create_test_trace_info(request_id), data=TraceData())


def create_test_trace_info(
    trace_id,
    experiment_id="test",
    request_time=0,
    execution_duration=1,
    state=TraceState.OK,
    trace_metadata=None,
    tags=None,
):
    # Add schema version to metadata if not provided, to match real trace creation behavior
    final_metadata = trace_metadata or {}
    if TRACE_SCHEMA_VERSION_KEY not in final_metadata:
        final_metadata = final_metadata.copy()
        final_metadata[TRACE_SCHEMA_VERSION_KEY] = str(TRACE_SCHEMA_VERSION)

    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id(experiment_id),
        request_time=request_time,
        execution_duration=execution_duration,
        state=state,
        trace_metadata=final_metadata,
        tags=tags or {},
    )


def get_traces(experiment_id=None) -> list[Trace]:
    # Get all traces from the backend
    return TracingClient().search_traces(
        experiment_ids=[experiment_id or _get_experiment_id()],
    )


def purge_traces(experiment_id=None):
    if len(get_traces(experiment_id)) == 0:
        return

    # Delete all traces from the backend
    TracingClient().delete_traces(
        experiment_id=experiment_id or _get_experiment_id(),
        max_traces=1000,
        max_timestamp_millis=int(time.time() * 1000),
    )


def get_tracer_tracking_uri() -> str | None:
    """Get current tracking URI configured as the trace export destination."""
    from opentelemetry import trace

    tracer = _get_tracer(__name__)
    if isinstance(tracer, trace.ProxyTracer):
        tracer = tracer._tracer
    span_processor = tracer.span_processor._span_processors[0]

    if isinstance(span_processor, MlflowV3SpanProcessor):
        return span_processor.span_exporter._client.tracking_uri


@pytest.fixture
def reset_autolog_state():
    """Reset autologging state to avoid interference between tests"""
    yield

    for flavor in FLAVOR_TO_MODULE_NAME:
        # 1. Remove post-import hooks (registered by global mlflow.autolog() function)
        mlflow.utils.import_hooks._post_import_hooks.pop(flavor, None)

    for flavor in AUTOLOGGING_INTEGRATIONS.keys():
        # 2. Disable autologging for the flavor. This is necessary because some autologging
        #    update global settings (e.g. callbacks) and we need to revert them.
        try:
            if autolog := get_autolog_function(flavor):
                autolog(disable=True)
        except ImportError:
            pass

        # 3. Revert any patches applied by autologging
        revert_patches(flavor)

    AUTOLOGGING_INTEGRATIONS.clear()


def score_in_model_serving(model_uri: str, model_input: dict[str, Any]):
    """
    A helper function to emulate model prediction inside a Databricks model serving environment.

    This is highly simplified version, but captures important aspects for testing tracing:
      1. Setting env vars that users set for enable tracing in model serving
      2. Load the model in a background thread
    """
    from mlflow.pyfunc.context import Context, set_prediction_context

    with mock.patch.dict(
        "os.environ",
        os.environ | {"IS_IN_DB_MODEL_SERVING_ENV": "true", "ENABLE_MLFLOW_TRACING": "true"},
        clear=True,
    ):
        # Reset tracing setup to start fresh w/ model serving environment
        mlflow.tracing.reset()

        def _load_model():
            return mlflow.pyfunc.load_model(model_uri)

        with ThreadPoolExecutor(max_workers=1) as executor:
            model = executor.submit(_load_model).result()

        # Score the model
        request_id = uuid.uuid4().hex
        with set_prediction_context(Context(request_id=request_id)):
            predictions = model.predict(model_input)

        trace = pop_trace(request_id)
        return (request_id, predictions, trace)


def skip_when_testing_trace_sdk(f):
    # Decorator to Skip the test if only mlflow-tracing package is installed and
    # not the full mlflow package.
    msg = "Skipping test because it requires mlflow or mlflow-skinny to be installed."
    skip_decorator = pytest.mark.skipif(IS_TRACING_SDK_ONLY, reason=msg)
    return skip_decorator(f)


def skip_module_when_testing_trace_sdk():
    """Skip the entire module if only mlflow-tracing package is installed"""
    if IS_TRACING_SDK_ONLY:
        pytest.skip(
            "Skipping test because it requires mlflow or mlflow-skinny to be installed.",
            allow_module_level=True,
        )


V2_TRACE_DICT = {
    "info": {
        "request_id": "58f4e27101304034b15c512b603bf1b2",
        "experiment_id": "0",
        "timestamp_ms": 100,
        "execution_time_ms": 200,
        "status": "OK",
        "request_metadata": {
            "mlflow.trace_schema.version": "2",
            "mlflow.traceInputs": '{"x": 2, "y": 5}',
            "mlflow.traceOutputs": "8",
        },
        "tags": {
            "mlflow.source.name": "test",
            "mlflow.source.type": "LOCAL",
            "mlflow.traceName": "predict",
            "mlflow.artifactLocation": "/path/to/artifact",
        },
        "assessments": [],
    },
    "data": {
        "spans": [
            {
                "name": "predict",
                "context": {
                    "span_id": "0d48a6670588966b",
                    "trace_id": "63076d0c1b90f1df0970f897dc428bd6",
                },
                "parent_id": None,
                "start_time": 100,
                "end_time": 200,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": '"58f4e27101304034b15c512b603bf1b2"',
                    "mlflow.spanType": '"UNKNOWN"',
                    "mlflow.spanFunctionName": '"predict"',
                    "mlflow.spanInputs": '{"x": 2, "y": 5}',
                    "mlflow.spanOutputs": "8",
                },
                "events": [],
            },
            {
                "name": "add_one_with_custom_name",
                "context": {
                    "span_id": "6fc32f36ef591f60",
                    "trace_id": "63076d0c1b90f1df0970f897dc428bd6",
                },
                "parent_id": "0d48a6670588966b",
                "start_time": 300,
                "end_time": 400,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": '"58f4e27101304034b15c512b603bf1b2"',
                    "mlflow.spanType": '"LLM"',
                    "delta": "1",
                    "metadata": '{"foo": "bar"}',
                    "datetime": '"2025-04-29 08:37:06.772253"',
                    "mlflow.spanFunctionName": '"add_one"',
                    "mlflow.spanInputs": '{"z": 7}',
                    "mlflow.spanOutputs": "8",
                },
                "events": [],
            },
        ],
        "request": '{"x": 2, "y": 5}',
        "response": "8",
    },
}
