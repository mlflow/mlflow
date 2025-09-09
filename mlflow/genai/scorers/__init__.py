from typing import TYPE_CHECKING

from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig, scorer
from mlflow.genai.scorers.registry import delete_scorer, get_scorer, list_scorers

# For static analysis only - helps linters understand the module structure
if TYPE_CHECKING:
    from mlflow.genai.scorers.builtin_scorers import (
        Correctness,
        ExpectationsGuidelines,
        Guidelines,
        RelevanceToQuery,
        RetrievalGroundedness,
        RetrievalRelevance,
        RetrievalSufficiency,
        Safety,
        get_all_scorers,
    )

# Defer import of builtin_scorers to avoid circular imports:
# - builtin_scorers imports Judge from mlflow.genai.judges.base
# - mlflow.genai.judges.base imports Scorer from mlflow.genai.scorers.base
# Using __getattr__ for lazy loading to address

_builtin_scorers_cache = {}


def __getattr__(name):
    """Lazily import builtin scorers to avoid circular dependency."""
    builtin_names = {
        "Correctness",
        "ExpectationsGuidelines",
        "Guidelines",
        "RelevanceToQuery",
        "RetrievalGroundedness",
        "RetrievalRelevance",
        "RetrievalSufficiency",
        "Safety",
        "get_all_scorers",
    }

    if name in builtin_names:
        # Check if already cached
        if name in _builtin_scorers_cache:
            return _builtin_scorers_cache[name]

        # Import builtin_scorers module and get the specific attribute
        from mlflow.genai.scorers import builtin_scorers

        attr = getattr(builtin_scorers, name)
        _builtin_scorers_cache[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Correctness",
    "ExpectationsGuidelines",
    "Guidelines",
    "RelevanceToQuery",
    "RetrievalGroundedness",
    "RetrievalRelevance",
    "RetrievalSufficiency",
    "Safety",
    "Scorer",
    "scorer",
    "ScorerSamplingConfig",
    "get_all_scorers",
    "get_scorer",
    "list_scorers",
    "delete_scorer",
]
