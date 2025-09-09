from typing import TYPE_CHECKING

from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig, scorer
from mlflow.genai.scorers.registry import delete_scorer, get_scorer, list_scorers

# Import builtin scorers only for type checking to avoid circular imports
# At runtime, these are accessed via __getattr__
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
else:
    # Lazy imports to avoid circular dependencies at runtime
    def __getattr__(name):
        if name in [
            "Correctness",
            "ExpectationsGuidelines",
            "Guidelines",
            "RelevanceToQuery",
            "RetrievalGroundedness",
            "RetrievalRelevance",
            "RetrievalSufficiency",
            "Safety",
            "get_all_scorers",
        ]:
            from mlflow.genai.scorers import builtin_scorers

            return getattr(builtin_scorers, name)
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
