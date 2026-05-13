import contextvars

from opentelemetry.sdk.trace.sampling import (
    Sampler,
    SamplingResult,
    TraceIdRatioBased,
)

# Context variable to override the sampling ratio for a specific trace.
# When set, the sampler uses this ratio instead of the default.
_SAMPLING_RATIO_OVERRIDE = contextvars.ContextVar("sampling_ratio_override", default=None)


class _MlflowSampler(Sampler):
    """
    A custom OTel sampler that delegates to TraceIdRatioBased but allows
    per-trace overrides via a ContextVar.

    When _SAMPLING_RATIO_OVERRIDE is set, uses that ratio instead of the default.
    Otherwise, falls back to the default ratio (from MLFLOW_TRACE_SAMPLING_RATIO).
    """

    def __init__(self, default_ratio: float = 1.0):
        self._default_ratio = default_ratio
        self._default_sampler = TraceIdRatioBased(default_ratio)

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
        override = _SAMPLING_RATIO_OVERRIDE.get()
        if override is not None:
            sampler = TraceIdRatioBased(override)
            return sampler.should_sample(
                parent_context, trace_id, name, kind, attributes, links, trace_state
            )
        return self._default_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return f"MlflowSampler(default_ratio={self._default_ratio})"
