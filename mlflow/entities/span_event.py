import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime

from opentelemetry.util.types import AttributeValue

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.databricks_trace_server_pb2 import Span as ProtoSpan
from mlflow.utils.proto_json_utils import set_pb_value


@dataclass
class SpanEvent(_MlflowObject):
    """
    An event that records a specific occurrences or moments in time
    during a span, such as an exception being thrown. Compatible with OpenTelemetry.

    Args:
        name: Name of the event.
        timestamp:  The exact time the event occurred, measured in microseconds.
            If not provided, the current time will be used.
        attributes: A collection of key-value pairs representing detailed
            attributes of the event, such as the exception stack trace.
            Attributes value must be one of ``[str, int, float, bool, bytes]``
            or a sequence of these types.
    """

    name: str
    # Use current time if not provided. We need to use default factory otherwise
    # the default value will be fixed to the build time of the class.
    timestamp: int = field(default_factory=lambda: int(time.time() * 1e6))
    attributes: dict[str, AttributeValue] = field(default_factory=dict)

    @classmethod
    def from_exception(cls, exception: Exception):
        "Create a span event from an exception."

        stack_trace = cls._get_stacktrace(exception)
        return cls(
            name="exception",
            attributes={
                "exception.message": str(exception),
                "exception.type": exception.__class__.__name__,
                "exception.stacktrace": stack_trace,
            },
        )

    @staticmethod
    def _get_stacktrace(error: BaseException) -> str:
        """Get the stacktrace of the parent error."""
        msg = repr(error)
        try:
            if sys.version_info < (3, 10):
                tb = traceback.format_exception(error.__class__, error, error.__traceback__)
            else:
                tb = traceback.format_exception(error)
            return "".join(tb).strip()
        except Exception:
            return msg

    def json(self):
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": json.dumps(self.attributes, cls=CustomEncoder)
            if self.attributes
            else None,
        }

    def to_proto(self):
        """Convert into OTLP compatible proto object to sent to the Databricks Trace Server."""
        proto = ProtoSpan.Event(
            name=self.name,
            time_unix_nano=self.timestamp,
        )
        # Trace server's proto uses map<string, google.protobuf.Value> for attributes
        for key, value in self.attributes.items():
            set_pb_value(proto.attributes[key], value)
        return proto


class CustomEncoder(json.JSONEncoder):
    """
    Custom encoder to handle json serialization.
    """

    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            # convert datetime to string format by default
            if isinstance(o, datetime):
                return o.isoformat()
            # convert object direct to string to avoid error in serialization
            return str(o)
