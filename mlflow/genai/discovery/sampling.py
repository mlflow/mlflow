from __future__ import annotations

import logging
import random
from collections import defaultdict

import mlflow
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import (
    SAMPLE_POOL_MULTIPLIER,
    SAMPLE_RANDOM_SEED,
)
from mlflow.genai.discovery.utils import get_session_id

_logger = logging.getLogger(__name__)


def sample_traces(
    sample_size: int,
    search_kwargs: dict[str, object],
) -> list[Trace]:
    """
    Randomly sample traces, grouping by session when session IDs exist.

    Fetches a pool of traces, groups them by session (or treats each trace
    as its own group when no sessions exist), then randomly selects
    ``sample_size`` groups and returns all traces from those groups.

    Args:
        sample_size: Number of groups (sessions or individual traces) to sample.
        search_kwargs: Keyword arguments passed to ``mlflow.search_traces``.

    Returns:
        List of sampled Trace objects.
    """
    pool_size = sample_size * SAMPLE_POOL_MULTIPLIER
    pool = mlflow.search_traces(max_results=pool_size, **search_kwargs)
    if not pool:
        return []

    # Group traces by session; traces without a session become their own group
    groups: dict[str, list[Trace]] = defaultdict(list)
    for trace in pool:
        key = get_session_id(trace) or trace.info.trace_id
        groups[key].append(trace)

    rng = random.Random(SAMPLE_RANDOM_SEED)
    group_keys = sorted(groups.keys())
    num_samples = min(sample_size, len(group_keys))
    selected = rng.sample(group_keys, num_samples)
    result = [trace for key in selected for trace in groups[key]]
    _logger.info(
        "Sampled %d groups (%d traces) from pool of %d groups",
        num_samples,
        len(result),
        len(group_keys),
    )
    return result
