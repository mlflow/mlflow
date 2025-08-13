import base64
import json
import logging
from functools import lru_cache
from typing import Any, Union

from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import Event as OTelEvent
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from opentelemetry.trace import Span as OTelSpan

import mlflow
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.databricks_trace_server_pb2 import Span as ProtoSpan
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import (
    TraceJSONEncoder,
    build_otel_context,
    decode_id,
    encode_span_id,
    encode_trace_id,
)
from mlflow.tracing.utils.processor import apply_span_processors

_logger = logging.getLogger(__name__)


# Not using enum as we want to allow custom span type string.
class SpanType:
    """
    Predefined set of span types.
    """

    LLM = "LLM"
    CHAIN = "CHAIN"
    AGENT = "AGENT"
    TOOL = "TOOL"
    CHAT_MODEL = "CHAT_MODEL"
    RETRIEVER = "RETRIEVER"
    PARSER = "PARSER"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    MEMORY = "MEMORY"
    UNKNOWN = "UNKNOWN"


def create_mlflow_span(
    otel_span: Any, trace_id: str, span_type: str | None = None
) -> Union["Span", "LiveSpan", "NoOpSpan"]:
    """
    Factory function to create a span object.

    When creating a MLflow span object from the OpenTelemetry span, the factory function
    should always be used to ensure the correct span object is created.
    """
    if not otel_span or isinstance(otel_span, NonRecordingSpan):
        return NoOpSpan()

    if isinstance(otel_span, OTelSpan):
        return LiveSpan(otel_span, trace_id, span_type)

    if isinstance(otel_span, OTelReadableSpan):
        return Span(otel_span)

    raise MlflowException(
        "The `otel_span` argument must be an instance of one of valid "
        f"OpenTelemetry span classes, but got {type(otel_span)}.",
        INVALID_PARAMETER_VALUE,
    )


class Span:
    """
    A span object. A span represents a unit of work or operation and is the building
    block of Traces.

    This Span class represents immutable span data that is already finished and persisted.
    The "live" span that is being created and updated during the application runtime is
    represented by the :py:class:`LiveSpan <mlflow.entities.LiveSpan>` subclass.
    """

    def __init__(self, otel_span: OTelReadableSpan):
        if not isinstance(otel_span, OTelReadableSpan):
            raise MlflowException(
                "The `otel_span` argument for the Span class must be an instance of ReadableSpan, "
                f"but got {type(otel_span)}.",
                INVALID_PARAMETER_VALUE,
            )

        self._span = otel_span
        # Since the span is immutable, we can cache the attributes to avoid the redundant
        # deserialization of the attribute values.
        self._attributes = _CachedSpanAttributesRegistry(otel_span)

    @property
    @lru_cache(maxsize=1)
    def trace_id(self) -> str:
        """The trace ID of the span, a unique identifier for the trace it belongs to."""
        return self.get_attribute(SpanAttributeKey.REQUEST_ID)

    @property
    def request_id(self) -> str:
        """Deprecated. Use `trace_id` instead."""
        return self.trace_id

    @property
    def span_id(self) -> str:
        """The ID of the span. This is only unique within a trace."""
        return encode_span_id(self._span.context.span_id)

    @property
    def name(self) -> str:
        """The name of the span."""
        return self._span.name

    @property
    def start_time_ns(self) -> int:
        """The start time of the span in nanosecond."""
        return self._span._start_time

    @property
    def end_time_ns(self) -> int | None:
        """The end time of the span in nanosecond."""
        return self._span._end_time

    @property
    def parent_id(self) -> str | None:
        """The span ID of the parent span."""
        if self._span.parent is None:
            return None
        return encode_span_id(self._span.parent.span_id)

    @property
    def status(self) -> SpanStatus:
        """The status of the span."""
        return SpanStatus.from_otel_status(self._span.status)

    @property
    def inputs(self) -> Any:
        """The input values of the span."""
        return self.get_attribute(SpanAttributeKey.INPUTS)

    @property
    def outputs(self) -> Any:
        """The output values of the span."""
        return self.get_attribute(SpanAttributeKey.OUTPUTS)

    @property
    def span_type(self) -> str:
        """The type of the span."""
        return self.get_attribute(SpanAttributeKey.SPAN_TYPE)

    @property
    def _trace_id(self) -> str:
        """
        The OpenTelemetry trace ID of the span. Note that this should not be exposed to
        the user, instead, use trace_id property as an unique identifier for a trace.
        """
        return encode_trace_id(self._span.context.trace_id)

    @property
    def attributes(self) -> dict[str, Any]:
        """
        Get all attributes of the span.

        Returns:
            A dictionary of all attributes of the span.
        """
        return self._attributes.get_all()

    @property
    def events(self) -> list[SpanEvent]:
        """
        Get all events of the span.

        Returns:
            A list of all events of the span.
        """
        return [
            SpanEvent(
                name=event.name,
                timestamp=event.timestamp,
                # Convert from OpenTelemetry's BoundedAttributes class to a simple dict
                # to avoid the serialization issue due to having a lock object.
                attributes=dict(event.attributes),
            )
            for event in self._span.events
        ]

    def __repr__(self):
        return (
            f"{type(self).__name__}(name={self.name!r}, trace_id={self.trace_id!r}, "
            f"span_id={self.span_id!r}, parent_id={self.parent_id!r})"
        )

    def get_attribute(self, key: str) -> Any | None:
        """
        Get a single attribute value from the span.

        Args:
            key: The key of the attribute to get.

        Returns:
            The value of the attribute if it exists, otherwise None.
        """
        return self._attributes.get(key)

    def to_dict(self) -> dict[str, Any]:
        d = MessageToDict(
            self.to_proto(),
            preserving_proto_field_name=True,
        )
        # Casting fields types as MessageToDict convert everything to string
        d["start_time_unix_nano"] = self.start_time_ns
        d["end_time_unix_nano"] = self.end_time_ns
        for i, event in enumerate(d.get("events", [])):
            event["time_unix_nano"] = self.events[i].timestamp
            event["attributes"] = self.events[i].attributes
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Span":
        """Create a Span object from the given dictionary."""
        try:
            # Try to deserialize the span using the v3 schema
            request_id = data.get("attributes", {}).get(SpanAttributeKey.REQUEST_ID)
            if not request_id:
                raise MlflowException(
                    f"The {SpanAttributeKey.REQUEST_ID} attribute is empty or missing.",
                    INVALID_PARAMETER_VALUE,
                )

            if Span._is_span_v2_schema(data):
                return cls.from_dict_v2(data)

            trace_id = _decode_id_from_byte(data["trace_id"])
            span_id = _decode_id_from_byte(data["span_id"])
            # Parent ID always exists in proto (empty string) even if the span is a root span.
            parent_id = (
                _decode_id_from_byte(data["parent_span_id"]) if data["parent_span_id"] else None
            )

            end_time_ns = data.get("end_time_unix_nano")
            end_time_ns = int(end_time_ns) if end_time_ns else None

            otel_span = OTelReadableSpan(
                name=data["name"],
                context=build_otel_context(trace_id, span_id),
                parent=build_otel_context(trace_id, parent_id) if parent_id else None,
                start_time=int(data["start_time_unix_nano"]),
                end_time=end_time_ns,
                attributes=data["attributes"],
                status=SpanStatus(
                    status_code=SpanStatusCode.from_proto_status_code(data["status"]["code"]),
                    description=data["status"].get("message"),
                ).to_otel_status(),
                # Setting an empty resource explicitly. Otherwise OTel create a new Resource by
                # Resource.create(), which introduces a significant overhead in some environments.
                # https://github.com/mlflow/mlflow/issues/15625
                resource=_OTelResource.get_empty(),
                events=[
                    OTelEvent(
                        name=event["name"],
                        timestamp=int(event["time_unix_nano"]),
                        attributes=event.get("attributes", {}),
                    )
                    for event in data.get("events", [])
                ],
            )
            return cls(otel_span)
        except Exception as e:
            raise MlflowException(
                "Failed to create a Span object from the given dictionary",
                INVALID_PARAMETER_VALUE,
            ) from e

    @staticmethod
    def _is_span_v2_schema(data: dict[str, Any]) -> bool:
        return "context" in data

    @classmethod
    def from_dict_v2(cls, data: dict[str, Any]) -> "Span":
        """Create a Span object from the given dictionary in v2 schema."""
        trace_id = decode_id(data["context"]["trace_id"])
        span_id = decode_id(data["context"]["span_id"])
        parent_id = decode_id(data["parent_id"]) if data["parent_id"] else None

        otel_span = OTelReadableSpan(
            name=data["name"],
            context=build_otel_context(trace_id, span_id),
            parent=build_otel_context(trace_id, parent_id) if parent_id else None,
            start_time=data["start_time"],
            end_time=data["end_time"],
            attributes=data["attributes"],
            status=SpanStatus(data["status_code"], data["status_message"]).to_otel_status(),
            # Setting an empty resource explicitly. Otherwise OTel create a new Resource by
            # Resource.create(), which introduces a significant overhead in some environments.
            # https://github.com/mlflow/mlflow/issues/15625
            resource=_OTelResource.get_empty(),
            events=[
                OTelEvent(
                    name=event["name"],
                    timestamp=event["timestamp"],
                    attributes=event["attributes"],
                )
                for event in data["events"]
            ],
        )
        return cls(otel_span)

    def to_proto(self):
        """Convert into OTLP compatible proto object to sent to the Databricks Trace Server."""
        otel_status = self._span.status
        status = ProtoSpan.Status(
            code=otel_status.status_code.value,
            message=otel_status.description,
        )
        parent = _encode_span_id_to_byte(self._span.parent.span_id) if self._span.parent else b""

        # NB: This is a workaround that some DBX internal code pass float timestamp
        start_time_unix_nano = int(self._span.start_time) if self._span.start_time else None
        end_time_unix_nano = int(self._span.end_time) if self._span.end_time else None

        return ProtoSpan(
            trace_id=_encode_trace_id_to_byte(self._span.context.trace_id),
            span_id=_encode_span_id_to_byte(self._span.context.span_id),
            trace_state=self._span.context.trace_state or "",
            parent_span_id=parent,
            name=self.name,
            start_time_unix_nano=start_time_unix_nano,
            end_time_unix_nano=end_time_unix_nano,
            events=[event.to_proto() for event in self.events],
            status=status,
            attributes={k: ParseDict(v, Value()) for k, v in self._span.attributes.items()},
        )


def _encode_span_id_to_byte(span_id: int | None) -> bytes:
    # https://github.com/open-telemetry/opentelemetry-python/blob/e01fa0c77a7be0af77d008a888c2b6a707b05c3d/exporter/opentelemetry-exporter-otlp-proto-common/src/opentelemetry/exporter/otlp/proto/common/_internal/__init__.py#L131
    return span_id.to_bytes(length=8, byteorder="big", signed=False)


def _encode_trace_id_to_byte(trace_id: int) -> bytes:
    # https://github.com/open-telemetry/opentelemetry-python/blob/e01fa0c77a7be0af77d008a888c2b6a707b05c3d/exporter/opentelemetry-exporter-otlp-proto-common/src/opentelemetry/exporter/otlp/proto/common/_internal/__init__.py#L135
    return trace_id.to_bytes(length=16, byteorder="big", signed=False)


def _decode_id_from_byte(trace_or_span_id_b64: str) -> int:
    # Decoding the base64 encoded trace or span ID to bytes and then converting it to int.
    bytes = base64.b64decode(trace_or_span_id_b64)
    return int.from_bytes(bytes, byteorder="big", signed=False)


class LiveSpan(Span):
    """
    A "live" version of the :py:class:`Span <mlflow.entities.Span>` class.

    The live spans are those being created and updated during the application runtime.
    When users start a new span using the tracing APIs within their code, this live span
    object is returned to get and set the span attributes, status, events, and etc.
    """

    def __init__(
        self,
        otel_span: OTelSpan,
        trace_id: str,
        span_type: str = SpanType.UNKNOWN,
    ):
        """
        The `otel_span` argument takes an instance of OpenTelemetry Span class, which is
        indeed a subclass of ReadableSpan. Thanks to this, the getter methods of the Span
        class can be reused without any modification.

        Note that the constructor doesn't call the super().__init__ method, because the Span
        initialization logic is a bit different from the immutable span.
        """
        if not isinstance(otel_span, OTelReadableSpan):
            raise MlflowException(
                "The `otel_span` argument for the LiveSpan class must be an instance of "
                f"trace.Span, but got {type(otel_span)}.",
                INVALID_PARAMETER_VALUE,
            )

        self._span = otel_span
        self._attributes = _SpanAttributesRegistry(otel_span)
        self._attributes.set(SpanAttributeKey.REQUEST_ID, trace_id)
        self._attributes.set(SpanAttributeKey.SPAN_TYPE, span_type)

    def set_span_type(self, span_type: str):
        """Set the type of the span."""
        self.set_attribute(SpanAttributeKey.SPAN_TYPE, span_type)

    def set_inputs(self, inputs: Any):
        """Set the input values to the span."""
        self.set_attribute(SpanAttributeKey.INPUTS, inputs)

    def set_outputs(self, outputs: Any):
        """Set the output values to the span."""
        self.set_attribute(SpanAttributeKey.OUTPUTS, outputs)

    def set_attributes(self, attributes: dict[str, Any]):
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
        self._attributes.set(key, value)

    def set_status(self, status: SpanStatusCode | str):
        """
        Set the status of the span.

        Args:
            status: The status of the span. This can be a
                :py:class:`SpanStatus <mlflow.entities.SpanStatus>` object or a string representing
                of the status code defined in
                :py:class:`SpanStatusCode <mlflow.entities.SpanStatusCode>`
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

    def record_exception(self, exception: str | Exception):
        """
        Record an exception on the span, adding an exception event and setting span status to ERROR.

        Args:
            exception: The exception to record. Can be an Exception instance or a string
                describing the exception.
        """
        if isinstance(exception, Exception):
            self.add_event(SpanEvent.from_exception(exception))
        elif isinstance(exception, str):
            self.add_event(SpanEvent.from_exception(Exception(exception)))
        else:
            raise MlflowException(
                "The `exception` parameter must be an Exception instance or a string.",
                INVALID_PARAMETER_VALUE,
            )

        self.set_status(
            SpanStatus(
                status_code=SpanStatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def end(
        self,
        outputs: Any | None = None,
        attributes: dict[str, Any] | None = None,
        status: SpanStatus | str | None = None,
        end_time_ns: int | None = None,
    ):
        """
        End the span.

        outputs: Outputs to set on the span.
        attributes: A dictionary of attributes to set on the span. If the span already has
            attributes, the new attributes will be merged with the existing ones. If the same
            key already exists, the new value will overwrite the old one.
        status: The status of the span. This can be a
            :py:class:`SpanStatus <mlflow.entities.SpanStatus>` object or a string
            representing the status code defined in
            :py:class:`SpanStatusCode <mlflow.entities.SpanStatusCode>`
            e.g. ``"OK"``, ``"ERROR"``. The default status is OK.
        end_time_ns: The end time of the span in nano seconds since the UNIX epoch.
            If not provided, the current time will be used.

        :meta private:
        """
        try:
            self.set_attributes(attributes or {})
            if outputs is not None:
                self.set_outputs(outputs)
            if status is not None:
                self.set_status(status)

            # NB: In OpenTelemetry, status code remains UNSET if not explicitly set
            # by the user. However, there is not way to set the status when using
            # @mlflow.trace decorator. Therefore, we just automatically set the status
            # to OK if it is not ERROR.
            if self.status.status_code != SpanStatusCode.ERROR:
                self.set_status(SpanStatus(SpanStatusCode.OK))

            # Apply span processors
            apply_span_processors(self)

            self._span.end(end_time=end_time_ns)

        except Exception as e:
            _logger.warning(
                f"Failed to end span {self.span_id}: {e}. "
                "For full traceback, set logging level to debug.",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )

    def from_dict(cls, data: dict[str, Any]) -> "Span":
        raise NotImplementedError("The `from_dict` method is not supported for the LiveSpan class.")

    def to_immutable_span(self) -> "Span":
        """
        Downcast the live span object to the immutable span.

        :meta private:
        """
        # All state of the live span is already persisted in the OpenTelemetry span object.
        return Span(self._span)

    @classmethod
    def from_immutable_span(
        cls,
        span: Span,
        parent_span_id: str | None = None,
        trace_id: str | None = None,
        experiment_id: str | None = None,
        otel_trace_id: str | None = None,
        end_trace: bool = True,
    ) -> "LiveSpan":
        """
        Create a new LiveSpan object from the given immutable span by
        cloning the underlying OpenTelemetry span within current context.

        This is particularly useful when we merging a remote trace into the current trace.
        We cannot merge the remote trace directly, because it is already stored as an immutable
        span, meaning that we cannot update metadata like trace ID, parent span ID,
        which are necessary for merging the trace.

        Args:
            span: The immutable span object to clone.
            parent_span_id: The parent span ID of the new span.
                If it is None, the span will be created as a root span.
            trace_id: The trace ID to be set on the new span. Specify this if you want to
                create the new span with a particular trace ID.
            experiment_id: The experiment ID to be set on the new span. If not specified, the
                experiment ID will be set to the current experiment ID.
            otel_trace_id: The OpenTelemetry trace ID of the new span in hex encoded format.
                If not specified, the newly generated trace ID will be used.
            end_trace: Whether to end the trace after cloning the span. Default is True.

        Returns:
            The new LiveSpan object with the same state as the original span.

        :meta private:
        """
        from mlflow.tracing.trace_manager import InMemoryTraceManager

        trace_manager = InMemoryTraceManager.get_instance()
        parent_span = trace_manager.get_span_from_id(trace_id, parent_span_id)

        # Create a new span with the same name, parent, and start time
        otel_span = mlflow.tracing.provider.start_detached_span(
            name=span.name,
            parent=parent_span._span if parent_span else None,
            start_time_ns=span.start_time_ns,
            experiment_id=experiment_id,
        )
        # The latter one from attributes is the newly generated trace ID by the span processor.
        trace_id = trace_id or json.loads(otel_span.attributes.get(SpanAttributeKey.REQUEST_ID))
        clone_span = LiveSpan(otel_span, trace_id, span.span_type)

        # Copy all the attributes, inputs, outputs, and events from the original span
        clone_span.set_status(span.status)
        clone_span.set_attributes(
            {k: v for k, v in span.attributes.items() if k != SpanAttributeKey.REQUEST_ID}
        )
        if span.inputs:
            clone_span.set_inputs(span.inputs)
        if span.outputs:
            clone_span.set_outputs(span.outputs)
        for event in span.events:
            clone_span.add_event(event)

        # Update trace ID and span ID
        context = span._span.get_span_context()
        clone_span._span._context = SpanContext(
            # Override otel_trace_id if provided, otherwise use the new trace ID
            trace_id=decode_id(otel_trace_id) if otel_trace_id else otel_span.context.trace_id,
            # Re-use same span ID as their ID space is local to the trace
            span_id=context.span_id,
            is_remote=context.is_remote,
            # Override trace flag as if it is sampled within current context.
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        if end_trace:
            clone_span.end(end_time_ns=span.end_time_ns)

        return clone_span


NO_OP_SPAN_TRACE_ID = "MLFLOW_NO_OP_SPAN_TRACE_ID"


class NoOpSpan(Span):
    """
    No-op implementation of the Span interface.

    This instance should be returned from the mlflow.start_span context manager when span
    creation fails. This class should have exactly the same interface as the Span so that
    user's setter calls do not raise runtime errors.

    E.g.

    .. code-block:: python

        with mlflow.start_span("span_name") as span:
            # Even if the span creation fails, the following calls should pass.
            span.set_inputs({"x": 1})
            # Do something

    """

    def __init__(self):
        self._span = NonRecordingSpan(context=None)
        self._attributes = {}

    @property
    def trace_id(self):
        """
        No-op span returns a special trace ID to distinguish it from the real spans.
        """
        return NO_OP_SPAN_TRACE_ID

    @property
    def span_id(self):
        return None

    @property
    def name(self):
        return None

    @property
    def start_time_ns(self):
        return None

    @property
    def end_time_ns(self):
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
    def _trace_id(self):
        return None

    def set_inputs(self, inputs: dict[str, Any]):
        pass

    def set_outputs(self, outputs: dict[str, Any]):
        pass

    def set_attributes(self, attributes: dict[str, Any]):
        pass

    def set_attribute(self, key: str, value: Any):
        pass

    def set_status(self, status: SpanStatus):
        pass

    def add_event(self, event: SpanEvent):
        pass

    def record_exception(self, exception: str | Exception):
        pass

    def end(
        self,
        outputs: Any | None = None,
        attributes: dict[str, Any] | None = None,
        status: SpanStatus | str | None = None,
        end_time_ns: int | None = None,
    ):
        pass


class _SpanAttributesRegistry:
    """
    A utility class to manage the span attributes.

    In MLflow users can add arbitrary key-value pairs to the span attributes, however,
    OpenTelemetry only allows a limited set of types to be stored in the attribute values.
    Therefore, we serialize all values into JSON string before storing them in the span.
    This class provides simple getter and setter methods to interact with the span attributes
    without worrying about the serde process.
    """

    def __init__(self, otel_span: OTelSpan):
        self._span = otel_span

    def get_all(self) -> dict[str, Any]:
        return {key: self.get(key) for key in self._span.attributes.keys()}

    def get(self, key: str):
        serialized_value = self._span.attributes.get(key)
        if serialized_value:
            try:
                return json.loads(serialized_value)
            except Exception:
                # If failed to deserialize (e.g., string value is directly set to OTel span),
                # return the original value as is.
                return serialized_value

    def set(self, key: str, value: Any):
        if not isinstance(key, str):
            _logger.warning(f"Attribute key must be a string, but got {type(key)}. Skipping.")
            return

        # NB: OpenTelemetry attribute can store not only string but also a few primitives like
        #   int, float, bool, and list of them. However, we serialize all into JSON string here
        #   for the simplicity in deserialization process.
        self._span.set_attribute(key, json.dumps(value, cls=TraceJSONEncoder, ensure_ascii=False))


class _CachedSpanAttributesRegistry(_SpanAttributesRegistry):
    """
    A cache-enabled version of the SpanAttributesRegistry.

    The caching helps to avoid the redundant deserialization of the attribute, however, it does
    not handle the value change well. Therefore, this class should only be used for the persisted
    spans that are immutable, and thus implemented as a subclass of _SpanAttributesRegistry.
    """

    @lru_cache(maxsize=128)
    def get(self, key: str):
        return super().get(key)

    def set(self, key: str, value: Any):
        raise MlflowException(
            "The attributes of the immutable span must not be updated.", INVALID_PARAMETER_VALUE
        )
