import logging
from time import time_ns
from typing import Any, Dict, Optional

from opentelemetry import trace as trace_api

from mlflow.tracing.types.model import Event, Span, SpanContext, SpanType, Status, StatusCode

_logger = logging.getLogger(__name__)


class MLflowSpanWrapper:
    """
    A wrapper around OpenTelemetry's Span object to provide MLflow-specific functionality.

    This class is passed to the exporter class to be processed on behalf of the OpenTelemetry's
    Span object, so need to implement the same interfaces as the original Span.
    """

    def __init__(self, span: trace_api.Span, span_type: SpanType = SpanType.UNKNOWN):
        self._span = span
        self._span_type = span_type
        self._inputs = None
        self._outputs = None

    @property
    def trace_id(self):
        return self._span.context.trace_id

    @property
    def name(self):
        return self._span.name

    @property
    def start_time(self):
        return self._span._start_time

    @property
    def end_time(self):
        return self._span._end_time

    @property
    def context(self):
        return self._span.get_span_context()

    @property
    def parent_span_id(self):
        if self._span.parent is None:
            return None
        return self._span.parent.span_id

    @property
    def status(self) -> Status:
        return Status(self._span.status.status_code, self._span.status.description)

    def set_inputs(self, inputs: Dict[str, Any]):
        self._inputs = inputs

    def set_outputs(self, outputs: Dict[str, Any]):
        self._outputs = outputs

    def set_attributes(self, attributes: Dict[str, Any]):
        try:
            self._span.set_attributes(attributes)
        except Exception as e:
            _logger.warning(f"Failed to set attributes {attributes} on span {self.name}: {e}")

    def set_attribute(self, key: str, value: Any):
        self._span.set_attribute(key, value)

    def set_status(self, status_code: StatusCode, description: str = ""):
        self._span.set_status(status_code, description)

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ):
        self._span.add_event(name, attributes, timestamp)


    def _end(self):
        # NB: In OpenTelemetry, status code remains UNSET if not explicitly set
        # by the user. However, there is not way to set the status when using
        # @mlflow.trace decorator. Therefore, we just automatically set the status
        # to OK if it is not ERROR.
        if self.status.status_code != StatusCode.ERROR:
            self.set_status(StatusCode.OK)

        # Mimic the OTel's span end hook to pass this wrapper to processor/exporter
        # Ref: https://github.com/open-telemetry/opentelemetry-python/blob/216411f03a3a067177a0b927b668a87a60cf8797/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L909
        with self._span._lock:
            if self._span._start_time is None:
                _logger.warning("Calling end() on a not started span. Ignoring.")
                return
            if self._span._end_time is not None:
                _logger.warning("Calling end() on an ended span. Ignoring.")
                return

            self._span._end_time = time_ns()

        self._span._span_processor.on_end(self)


    def _to_mlflow_span(self):
        """
        Create an MLflow Span object from this wrapper and the original Span object.
        """
        return Span(
            name=self._span.name,
            context=SpanContext(
                trace_id=self.trace_id,
                span_id=self._span.get_span_context().span_id,
            ),
            parent_span_id=self.parent_span_id,
            span_type=self._span_type,
            status=self.status,
            start_time=self._span._start_time,
            end_time=self._span._end_time,
            inputs=self._inputs,
            outputs=self._outputs,
            attributes=self._span.attributes,
            events=[
                Event(event.name, event.timestamp, event.attributes) for event in self._span.events
            ],
        )


class NoOpMLflowSpanWrapper:
    """
    No-op implementation of all MLflowSpanWrapper.

    This instance should be returned from the mlflow.start_span context manager when span
    creation fails. This class should have exactly the same interface as MLflowSpanWrapper
    so that user's setter calls do not raise runtime errors.

    E.g.
    ```
    with mlflow.start_span("span_name") as span:
        # Even if the span creation fails, the following calls should pass.
        span.set_inputs({"x": 1})
        ...
    ```

    """

    @property
    def trace_id(self):
        return None

    @property
    def name(self):
        return None

    @property
    def start_time(self):
        return None

    @property
    def end_time(self):
        return None

    @property
    def context(self):
        return None

    @property
    def parent_span_id(self):
        return None

    @property
    def status(self):
        return None

    def set_inputs(self, inputs: Dict[str, Any]):
        pass

    def set_outputs(self, outputs: Dict[str, Any]):
        pass

    def set_attributes(self, attributes: Dict[str, Any]):
        pass

    def set_attribute(self, key: str, value: Any):
        pass

    def set_status(self, status_code: StatusCode, description: str = ""):
        pass

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ):
        pass

    def _end(self):
        pass
