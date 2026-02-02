"""
Sampling logic for MLflow tracing.

This module provides sampling functionality to control which traces are recorded
and exported, helping manage costs and performance for high-volume applications.
"""

import contextvars

from opentelemetry.sdk.trace.sampling import (
    Decision,
    Sampler,
    SamplingResult,
    TraceIdRatioBased,
)

# Context variable to force sampling, bypassing the global sampling ratio.
# When set to True, the custom sampler will always sample the trace.
_FORCE_SAMPLE = contextvars.ContextVar("force_sample", default=False)


class _OverridableSampler(Sampler):
    """
    A custom sampler that allows per-trace override of the global sampling ratio.

    When _FORCE_SAMPLE context variable is True, this sampler always samples.
    Otherwise, it delegates to the underlying ratio-based sampler.
    """

    def __init__(self, ratio: float):
        self._ratio_sampler = TraceIdRatioBased(ratio)

    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        links=None,
        trace_state=None,
    ) -> SamplingResult:
        if _FORCE_SAMPLE.get():
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes, trace_state)
        return self._ratio_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return f"OverridableSampler(ratio={self._ratio_sampler.rate})"
