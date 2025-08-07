import os

from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST


def should_use_otlp_exporter() -> bool:
    return _get_otlp_endpoint() is not None


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


def _get_otlp_protocol() -> str:
    return os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or os.environ.get(
        "OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"
    )
