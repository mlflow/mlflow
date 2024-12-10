import io
import json
from typing import Any, Callable, Union

from botocore.client import BaseClient
from botocore.eventstream import EventStream

import mlflow
from mlflow.bedrock import FLAVOR_NAME
from mlflow.entities import SpanType
from mlflow.utils.autologging_utils import safe_patch
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_BEDROCK_RUNTIME_SERVICE_NAME = "bedrock-runtime"
_BEDROCK_SPAN_PREFIX = "BedrockRuntime."


def skip_if_trace_disabled(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to apply the function only if trace autologging is enabled.
    This decorator is used to skip the test if the trace autologging is disabled.
    """

    def wrapper(original, self, *args, **kwargs):
        config = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
        if not config.log_traces:
            return original(self, *args, **kwargs)

        return func(original, self, *args, **kwargs)

    return wrapper


def patched_create_client(original, self, *args, **kwargs):
    """
    Patched version of the boto3 ClientCreator.create_client method that returns
    a patched client class.
    """
    if kwargs.get("service_name") != _BEDROCK_RUNTIME_SERVICE_NAME:
        return original(self, *args, **kwargs)

    client = original(self, *args, **kwargs)
    patch_bedrock_runtime_client(client.__class__)

    return client


def patch_bedrock_runtime_client(client_class: type[BaseClient]):
    """
    Patch the BedrockRuntime client to log traces and models.
    """
    # The most basic model invocation API
    safe_patch(FLAVOR_NAME, client_class, "invoke_model", _patched_invoke_model)

    if hasattr(client_class, "converse"):
        # The new "converse" API was introduced in boto3 1.35 to access all models
        # with the consistent chat format.
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
        safe_patch(FLAVOR_NAME, client_class, "converse", _patched_converse)


@skip_if_trace_disabled
def _patched_invoke_model(original, self, *args, **kwargs):
    if not AutoLoggingConfig.init(flavor_name=FLAVOR_NAME).log_traces:
        return original(self, *args, **kwargs)

    with mlflow.start_span(name=f"{_BEDROCK_SPAN_PREFIX}{original.__name__}") as span:
        # NB: Bedrock client doesn't accept any positional arguments
        span.set_inputs(kwargs)

        result = original(self, *args, **kwargs)

        result["body"] = _buffer_stream(result["body"])
        parsed_response_body = _parse_invoke_model_response_body(result["body"])

        # Determine the span type based on the key in the response body.
        # As of 2024 Dec 9th, all supported embedding models in Bedrock returns the response body
        # with the key "embedding". This might change in the future.
        span_type = SpanType.EMBEDDING if "embedding" in parsed_response_body else SpanType.LLM
        span.set_span_type(span_type)

        span.set_outputs({**result, "body": parsed_response_body})

        return result


def _buffer_stream(raw_stream: EventStream) -> io.BytesIO:
    """
    Create a buffered stream from the raw byte stream.

    The boto3's invoke_model() API returns the LLM response as a byte stream.
    We need to read the stream data to set the span outputs, however, the stream
    can only be read once and not seekable (https://github.com/boto/boto3/issues/564).
    To work around this, we create a buffered stream that can be read multiple times.
    """
    buffered_response = io.BytesIO(raw_stream.read())
    buffered_response.seek(0)
    return buffered_response


def _parse_invoke_model_response_body(response_body: dict[str, Any]) -> Union[dict[str, Any], str]:
    content = response_body.read()
    try:
        return json.loads(content)
    except Exception:
        # When failed to parse the response body as JSON, return the raw response
        return content
    finally:
        response_body.seek(0)


@skip_if_trace_disabled
def _patched_converse(original, self, *args, **kwargs):
    if not AutoLoggingConfig.init(flavor_name=FLAVOR_NAME).log_traces:
        return original(self, *args, **kwargs)

    with mlflow.start_span(
        name=f"{_BEDROCK_SPAN_PREFIX}{original.__name__}",
        span_type=SpanType.CHAT_MODEL,
    ) as span:
        # NB: Bedrock client doesn't accept any positional arguments
        span.set_inputs(kwargs)
        result = original(self, *args, **kwargs)
        span.set_outputs(result)
        return result
