from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import trace as trace_api


@dataclass
class SpanStatus:
    """
    Status of the span or the trace.

    Args:
        status_code: The status code of the span or the trace.
        description: Description of the status. Optional.
    """

    status_code: trace_api.StatusCode
    description: str = ""

    # NB: Using the OpenTelemetry native StatusCode values here, because span's set_status
    #     method only accepts a StatusCode enum in their definition.
    #     https://github.com/open-telemetry/opentelemetry-python/blob/8ed71b15fb8fc9534529da8ce4a21e686248a8f3/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L949
    #     Working around this is possible, but requires some hack to handle automatic status
    #     propagation mechanism, so here we just use the native object that meets our
    #     current requirements at least. Nevertheless, declaring the new class extending
    #     the OpenTelemetry Status class so users code doesn't have to import the OTel's
    #     StatusCode object, which makes future migration easier.
    class StatusCode:
        UNSET = trace_api.StatusCode.UNSET
        OK = trace_api.StatusCode.OK
        ERROR = trace_api.StatusCode.ERROR
