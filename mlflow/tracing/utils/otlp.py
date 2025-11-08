import os
from typing import Any

from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.environment_variables import MLFLOW_ENABLE_OTLP_EXPORTER
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

# Constants for OpenTelemetry integration
MLFLOW_EXPERIMENT_ID_HEADER = "x-mlflow-experiment-id"
OTLP_TRACES_PATH = "/v1/traces"


def should_use_otlp_exporter() -> bool:
    """
    Determine if OTLP traces should be exported based on environment configuration.
    """
    return _get_otlp_endpoint() is not None and MLFLOW_ENABLE_OTLP_EXPORTER.get()


def should_export_otlp_metrics() -> bool:
    """
    Determine if OTLP metrics should be exported based on environment configuration.

    Returns True if metrics endpoint is configured.
    """
    return _get_otlp_metrics_endpoint() is not None


def get_otlp_exporter() -> SpanExporter:
    """
    Get the OTLP exporter based on the configured protocol.
    """
    endpoint = _get_otlp_endpoint()
    protocol = _get_otlp_protocol()
    if protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        except ImportError:
            raise MlflowException(
                "gRPC OTLP exporter is not available. Please install the required dependency by "
                "running `pip install opentelemetry-exporter-otlp-proto-grpc`.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return OTLPSpanExporter(endpoint=endpoint)
    elif protocol == "http/protobuf":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as e:
            raise MlflowException(
                "HTTP OTLP exporter is not available. Please install the required dependency by "
                "running `pip install opentelemetry-exporter-otlp-proto-http`.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from e

        return OTLPSpanExporter(endpoint=endpoint)
    else:
        raise MlflowException.invalid_parameter_value(
            f"Unsupported OTLP protocol '{protocol}' is configured. Please set "
            "the protocol to either 'grpc' or 'http/protobuf'."
        )


def _get_otlp_endpoint() -> str | None:
    """
    Get the OTLP endpoint from the environment variables.
    Ref: https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/#endpoint-configuration
    """
    # Use `or` instead of default value to do lazy eval
    return os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )


def _get_otlp_metrics_endpoint() -> str | None:
    """
    Get the OTLP metrics endpoint from the environment variables.
    """
    return os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )


def _get_otlp_protocol(default_value: str = "grpc") -> str:
    """
    Get the OTLP traces protocol from environment variables.

    Returns the value of OTEL_EXPORTER_OTLP_TRACES_PROTOCOL if set,
    otherwise falls back to OTEL_EXPORTER_OTLP_PROTOCOL, then to default_value.

    Args:
        default_value: The default protocol to use if no environment variables are set.
    """
    return os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or os.environ.get(
        "OTEL_EXPORTER_OTLP_PROTOCOL", default_value
    )


def _get_otlp_metrics_protocol(default_value: str = "grpc") -> str:
    """
    Get the OTLP metrics protocol from environment variables.

    Returns the value of OTEL_EXPORTER_OTLP_METRICS_PROTOCOL if set,
    otherwise falls back to OTEL_EXPORTER_OTLP_PROTOCOL, then to default_value.

    Args:
        default_value: The default protocol to use if no environment variables are set.
    """
    return os.environ.get("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL") or os.environ.get(
        "OTEL_EXPORTER_OTLP_PROTOCOL", default_value
    )


def _otel_proto_bytes_to_id(id_bytes: bytes) -> int:
    """Convert OTel protobuf bytes to integer ID."""
    return int.from_bytes(id_bytes, byteorder="big", signed=False)


def _set_otel_proto_anyvalue(pb_any_value: AnyValue, value: Any) -> None:
    """Set a value on an OTel protobuf AnyValue message.

    Args:
        pb_any_value: The OTel protobuf AnyValue message to populate.
        value: The value to set.
    """
    if value is None:
        # Leave the value unset for None
        pass
    elif isinstance(value, bool):
        pb_any_value.bool_value = value
    elif isinstance(value, str):
        pb_any_value.string_value = value
    elif isinstance(value, int):
        pb_any_value.int_value = value
    elif isinstance(value, float):
        pb_any_value.double_value = value
    elif isinstance(value, bytes):
        pb_any_value.bytes_value = value
    elif isinstance(value, (list, tuple)):
        # Handle arrays
        for item in value:
            _set_otel_proto_anyvalue(pb_any_value.array_value.values.add(), item)
    elif isinstance(value, dict):
        # Handle key-value lists
        for k, v in value.items():
            kv = pb_any_value.kvlist_value.values.add()
            kv.key = str(k)
            _set_otel_proto_anyvalue(kv.value, v)
    else:
        # For unknown types, convert to string
        pb_any_value.string_value = str(value)


def _decode_otel_proto_anyvalue(pb_any_value: AnyValue) -> Any:
    """Decode an OTel protobuf AnyValue.

    Args:
        pb_any_value: The OTel protobuf AnyValue message to decode.

    Returns:
        The decoded value.
    """
    value_type = pb_any_value.WhichOneof("value")
    if not value_type:
        return None

    # Handle complex types that need recursion
    if value_type == "array_value":
        return [_decode_otel_proto_anyvalue(v) for v in pb_any_value.array_value.values]
    elif value_type == "kvlist_value":
        return {
            kv.key: _decode_otel_proto_anyvalue(kv.value) for kv in pb_any_value.kvlist_value.values
        }
    else:
        # For simple types, just get the attribute directly
        return getattr(pb_any_value, value_type)
