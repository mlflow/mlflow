import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
from unittest import mock

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import Event, ReadableSpan

import mlflow
from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.ml_package_versions import FLAVOR_TO_MODULE_NAME
from mlflow.tracing.export.inference_table import pop_trace
from mlflow.tracing.processor.mlflow import MlflowSpanProcessor
from mlflow.tracing.provider import _get_tracer
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS, get_autolog_function
from mlflow.utils.autologging_utils.safety import revert_patches


def create_mock_otel_span(
    trace_id: int,
    span_id: int,
    name: str = "test_span",
    parent_id: Optional[int] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
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

        def end(self, end_time=None):
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
    request_id,
    experiment_id="test",
    timestamp_ms=0,
    execution_time_ms=1,
    status=TraceStatus.OK,
    request_metadata=None,
    tags=None,
):
    return TraceInfo(
        request_id=request_id,
        experiment_id=experiment_id,
        timestamp_ms=timestamp_ms,
        execution_time_ms=execution_time_ms,
        status=status,
        request_metadata=request_metadata or {},
        tags=tags or {},
    )


def get_traces(experiment_id=DEFAULT_EXPERIMENT_ID) -> list[Trace]:
    # Get all traces from the backend
    return mlflow.MlflowClient().search_traces(experiment_ids=[experiment_id])


def purge_traces(experiment_id=DEFAULT_EXPERIMENT_ID):
    # Delete all traces from the backend
    mlflow.tracking.MlflowClient().delete_traces(
        experiment_id=experiment_id, max_traces=1000, max_timestamp_millis=0
    )


def get_tracer_tracking_uri() -> Optional[str]:
    """Get current tracking URI configured as the trace export destination."""
    from opentelemetry import trace

    tracer = _get_tracer(__name__)
    if isinstance(tracer, trace.ProxyTracer):
        tracer = tracer._tracer
    span_processor = tracer.span_processor._span_processors[0]

    if isinstance(span_processor, MlflowSpanProcessor):
        return span_processor._client.tracking_uri


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


def score_in_model_serving(model_uri: str, model_input: dict):
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
