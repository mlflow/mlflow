"""Shared helpers for pydantic-ai autologging tests.

pydantic-ai 1.95.0 moved model-request instrumentation off ``InstrumentedModel``
and onto the ``Instrumentation`` capability (see
https://github.com/pydantic/pydantic-ai/pull/4967). Tests inject a dummy model
response by patching the *concrete* model's ``request``/``request_stream``, which
sits below instrumentation in every supported version:

* On older versions, ``InstrumentedModel.request`` (patched by MLflow autolog)
  internally calls the wrapped concrete model's ``request``.
* On >= 1.95.0, the capability's ``wrap_model_request`` hook calls the concrete
  model's ``request`` via its handler.

So patching the concrete model works regardless of pydantic-ai version, while
MLflow's autolog still produces the LLM span.
"""

import importlib.metadata
from unittest.mock import patch

from packaging.version import Version

PYDANTIC_AI_VERSION = Version(importlib.metadata.version("pydantic_ai"))

# Instrumentation moved to the Instrumentation capability in 1.95.0. From this
# version the single ``wrap_model_request`` hook handles both streaming and
# non-streaming model calls, so the LLM span is always named
# ``InstrumentedModel.request`` (older versions used a separate
# ``InstrumentedModel.request_stream`` span for streaming).
USES_CAPABILITY_INSTRUMENTATION = PYDANTIC_AI_VERSION >= Version("1.95.0")
LLM_STREAM_SPAN_NAME = (
    "InstrumentedModel.request"
    if USES_CAPABILITY_INSTRUMENTATION
    else "InstrumentedModel.request_stream"
)


def _concrete_openai_model_cls():
    from pydantic_ai.models import openai

    # OpenAIModel was renamed/split into OpenAIChatModel in recent versions.
    return getattr(openai, "OpenAIChatModel", None) or openai.OpenAIModel


def patch_model_request(**kwargs):
    """Patch the concrete OpenAI model's ``request`` (accepts ``new``/``side_effect``)."""
    return patch.object(_concrete_openai_model_cls(), "request", **kwargs)


def patch_model_request_stream(**kwargs):
    """Patch the concrete OpenAI model's ``request_stream`` (accepts ``new``/``side_effect``)."""
    return patch.object(_concrete_openai_model_cls(), "request_stream", **kwargs)
