from __future__ import annotations

import logging
import random

import mlflow
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import (
    SAMPLE_POOL_MULTIPLIER,
    SAMPLE_RANDOM_SEED,
)
from mlflow.genai.discovery.utils import group_traces_by_session

_logger = logging.getLogger(__name__)


def sample_traces(
    sample_size: int,
    search_kwargs: dict[str, object] | None = None,
    *,
    traces: list[Trace] | None = None,
) -> list[Trace]:
    """
    Randomly sample traces, grouping by session when session IDs exist.

    Either fetches a pool via ``search_kwargs`` or samples from a
    pre-fetched ``traces`` list. Groups by session (or treats each trace
    as its own group when no sessions exist), then randomly selects
    ``sample_size`` groups and returns all traces from those groups.

    Args:
        sample_size: Number of groups (sessions or individual traces) to sample.
        search_kwargs: Keyword arguments passed to ``mlflow.search_traces``.
            Ignored when ``traces`` is provided.
        traces: Pre-fetched traces to sample from. When provided, skips
            the ``search_traces`` call.

    Returns:
        List of sampled Trace objects.
    """
    if traces is not None:
        pool = traces
    else:
        pool_size = sample_size * SAMPLE_POOL_MULTIPLIER
        pool = mlflow.search_traces(
            max_results=pool_size, return_type="list", **(search_kwargs or {})
        )
    if not pool:
        return []

    groups = group_traces_by_session(pool)

    rng = random.Random(SAMPLE_RANDOM_SEED)
    group_keys = sorted(groups.keys())
    num_samples = min(sample_size, len(group_keys))
    selected = rng.sample(group_keys, num_samples)
    result = [trace for key in selected for trace in groups[key]]
    _logger.debug(
        "Sampled %d groups (%d traces) from pool of %d groups",
        num_samples,
        len(result),
        len(group_keys),
    )
    return result
