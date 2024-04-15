import json
import logging
from typing import Any, Dict, Optional, Union

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from mlflow.entities import Span, SpanContext, SpanEvent, SpanStatus, SpanType, TraceStatus
from mlflow.tracing.types.constant import SpanAttributeKey
from mlflow.tracing.utils import TraceJSONEncoder

_logger = logging.getLogger(__name__)


class MlflowSpanWrapper:
    """
    A wrapper around OpenTelemetry's Span object to provide MLflow-specific functionality.

    This class should expose the (subset of) same APIs as the OpenTelemetry's
    Span object, so there should be no difference from the user's perspective.
    The wrapper is only used for the span creation and mutation at runtime, and
    will be converted to the immutable :py:class:`Span <mlflow.entities.Span>` object once
    the span is ended, before being sent to the logging client.
    """

    def __init__(self, span: trace_api.Span, request_id: str, span_type: str = SpanType.UNKNOWN):
        self._span = span
        self._set_attribute(SpanAttributeKey.REQUEST_ID, request_id, serialize=False)
        self._set_attribute(SpanAttributeKey.SPAN_TYPE, span_type, serialize=False)

    @property
    def request_id(self) -> str:
        """
        The request ID of the span, a unique identifier for the trace it belongs to.
        Request ID is equivalent to the trace ID in OpenTelemetry, but generated
        differently by the tracing backend.
        """
        return self._span._attributes.get(SpanAttributeKey.REQUEST_ID, None)

    @property
    def span_id(self) -> str:
        """The ID of the span. This is only unique within a trace."""
        return self._span.get_span_context().span_id

    @property
    def name(self) -> str:
        """The name of the span."""
        return self._span.name

    @property
    def start_time_ns(self) -> int:
        """The start time of the span in nanosecond."""
        return self._span._start_time

    @property
    def end_time_ns(self) -> Optional[int]:
        """The end time of the span in nanosecond."""
        return self._span._end_time

    @property
    def parent_id(self) -> str:
        """The span ID of the parent span."""
        if self._span.parent is None:
            return None
        return self._span.parent.span_id

    @property
    def status(self) -> SpanStatus:
        """The status of the span."""
        return SpanStatus.from_otel_status(self._span.status)

    @property
    def inputs(self) -> Any:
        """The input values of the span."""
        input_serialized = self._span._attributes.get(SpanAttributeKey.INPUTS)
        return json.loads(input_serialized) if input_serialized else None

    @property
    def outputs(self) -> Any:
        """The output values of the span."""
        output_serialized = self._span._attributes.get(SpanAttributeKey.OUTPUTS)
        return json.loads(output_serialized) if output_serialized else None

    def set_inputs(self, inputs: Any):
        """Set the input values to the span."""
        self._set_attribute(SpanAttributeKey.INPUTS, inputs, serialize=True)

    def set_outputs(self, outputs: Any):
        """Set the output values to the span."""
        self._set_attribute(SpanAttributeKey.OUTPUTS, outputs, serialize=True)

    def set_attributes(self, attributes: Dict[str, Any]):
        """
        Set the attributes to the span. The attributes must be a dictionary of key-value pairs.
        This method is additive, i.e. it will add new attributes to the existing ones. If an
        attribute with the same key already exists, it will be overwritten.
        """
        if not isinstance(attributes, dict):
            _logger.warning(
                f"Attributes must be a dictionary, but got {type(attributes)}. Skipping."
            )
            return

        for key, value in attributes.items():
            self.set_attribute(key, value)

    def set_attribute(self, key: str, value: Any):
        """Set a single attribute to the span."""
        self._set_attribute(key, value)

    def _set_attribute(self, key: str, value: Any, serialize=True):
        if not isinstance(key, str):
            _logger.warning(f"Attribute key must be a string, but got {type(key)}. Skipping.")
            return

        # NB: OpenTelemetry attribute can store not only string but also a few primitives like
        #   int, float, bool, and list of them. However, we serialize all into JSON string here
        #   for the simplicity in deserialization process.
        if serialize:
            value = json.dumps(value, cls=TraceJSONEncoder)
        self._span.set_attribute(key, value)

    def get_attribute(self, key: str) -> Optional[AttributeValue]:
        """
        Get a single attribute value from the span.

        Args:
            key: The key of the attribute to get.

        Returns:
            The value of the attribute if it exists, otherwise None.
        """
        return self._span.attributes.get(key, None)

    def set_status(self, status: Union[SpanStatus, str]):
        """
        Set the status of the span.

        Args:
            status: The status of the span. This can be a
                :py:class:`SpanStatus <mlflow.entities.SpanStatus>` object or a string representing
                of the status code defined in :py:class:`TraceStatus <mlflow.entities.TraceStatus>`
                e.g. ``"OK"``, ``"ERROR"``.
        """
        if isinstance(status, str):
            status = SpanStatus(status)

        # NB: We need to set the OpenTelemetry native StatusCode, because span's set_status
        #     method only accepts a StatusCode enum in their definition.
        #     https://github.com/open-telemetry/opentelemetry-python/blob/8ed71b15fb8fc9534529da8ce4a21e686248a8f3/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L949
        #     Working around this is possible, but requires some hack to handle automatic status
        #     propagation mechanism, so here we just use the native object that meets our
        #     current requirements at least. Nevertheless, declaring the new class extending
        #     the OpenTelemetry Status class so users code doesn't have to import the OTel's
        #     StatusCode object, which makes future migration easier.
        self._span.set_status(status.to_otel_status())

    def add_event(self, event: SpanEvent):
        """
        Add an event to the span.

        Args:
            event: The event to add to the span. This should be a
                :py:class:`SpanEvent <mlflow.entities.SpanEvent>` object.
        """
        self._span.add_event(event.name, event.attributes, event.timestamp)

    def end(self):
        """
        End the span. This is a thin wrapper around the OpenTelemetry's end method but just
        to handle the status update.

        This method should not be called directly by the user, only by called via fluent APIs
        context exit or by MlflowClient APIs.

        :meta private:
        """
        # NB: In OpenTelemetry, status code remains UNSET if not explicitly set
        # by the user. However, there is not way to set the status when using
        # @mlflow.trace decorator. Therefore, we just automatically set the status
        # to OK if it is not ERROR.
        if self.status.status_code != TraceStatus.ERROR:
            self.set_status(SpanStatus(TraceStatus.OK))

        self._span.end()

    def to_mlflow_span(self) -> Span:
        """
        Create an MLflow Span object from this wrapper and the original Span object.

        :meta private:
        """
        return Span(
            name=self._span.name,
            context=SpanContext(
                trace_id=self._span.get_span_context().trace_id,
                span_id=self.span_id,
            ),
            parent_id=self.parent_id,
            status_code=self.status.status_code.value,
            status_message=self.status.description,
            start_time=self.start_time_ns,
            end_time=self.end_time_ns,
            # Convert from MappingProxyType class to a simple dict
            attributes=dict(self._span.attributes),
            events=[
                SpanEvent(
                    name=event.name,
                    timestamp=event.timestamp,
                    # Convert from OpenTelemetry's BoundedAttributes class to a simple dict
                    attributes=dict(event.attributes),
                )
                for event in self._span.events
            ],
        )


class NoOpMlflowSpanWrapper:
    """
    No-op implementation of all MlflowSpanWrapper.

    This instance should be returned from the mlflow.start_span context manager when span
    creation fails. This class should have exactly the same interface as MlflowSpanWrapper
    so that user's setter calls do not raise runtime errors.

    E.g.

    .. code-block:: python

        with mlflow.start_span("span_name") as span:
            # Even if the span creation fails, the following calls should pass.
            span.set_inputs({"x": 1})
            # Do something

    """

    @property
    def request_id(self):
        return None

    @property
    def id(self):
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
    def parent_id(self):
        return None

    @property
    def status(self):
        return None

    @property
    def inputs(self):
        return None

    @property
    def outputs(self):
        return None

    def set_inputs(self, inputs: Dict[str, Any]):
        pass

    def set_outputs(self, outputs: Dict[str, Any]):
        pass

    def set_attributes(self, attributes: Dict[str, Any]):
        pass

    def set_attribute(self, key: str, value: Any):
        pass

    def _set_attribute(self, key: str, value: Any, serialize=True):
        pass

    def set_status(self, status: SpanStatus):
        pass

    def add_event(self, event: SpanEvent):
        pass

    def end(self):
        pass
