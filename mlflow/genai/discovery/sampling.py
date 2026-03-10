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
    pool = mlflow.search_traces(max_results=pool_size, return_type="list", **search_kwargs)
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
