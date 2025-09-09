from typing import TYPE_CHECKING

from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig, scorer
from mlflow.genai.scorers.registry import delete_scorer, get_scorer, list_scorers

# NB: We use lazy imports for builtin_scorers to avoid a circular dependency issue.
#
# The circular dependency chain:
# 1. scorers/__init__.py imports from builtin_scorers (if done eagerly)
# 2. builtin_scorers.py has `class BuiltInScorer(Judge)` which imports Judge from judges.base
# 3. judges.base.py has `class Judge(Scorer)` which imports Scorer from scorers.base
# 4. When importing scorers.base, Python first initializes the scorers package (__init__.py)
# 5. This would try to import builtin_scorers, creating a circular import
#
# Solution:
# - We use Python's __getattr__ mechanism to defer the import of builtin_scorers until
#   the specific scorer classes are actually accessed at runtime
# - The TYPE_CHECKING block provides type hints for IDEs and static analysis tools
#   without actually importing the modules at runtime (TYPE_CHECKING is only True
#   during static analysis, never during actual execution)
# - This allows Sphinx autodoc and IDEs to understand the module structure while
#   avoiding the circular import at runtime


# Define the attributes that should be lazily loaded
_LAZY_IMPORTS = {
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


def __getattr__(name):
    """Lazily import builtin scorers to avoid circular dependency."""
    if name in _LAZY_IMPORTS:
        # Import the module when first accessed
        from mlflow.genai.scorers import builtin_scorers

        return getattr(builtin_scorers, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return the list of attributes available in this module.

    This is necessary for Sphinx autodoc and other introspection tools
    to discover the lazily-loaded scorer classes.
    """
    # Get the default module attributes
    module_attrs = list(globals().keys())
    # Add the lazy imports
    return sorted(set(module_attrs) | _LAZY_IMPORTS)


# The TYPE_CHECKING block below is for static analysis tools only.
# TYPE_CHECKING is False at runtime, so these imports never execute during normal operation.
# This gives us the best of both worlds: type hints without circular imports.
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

# Explicitly declare module-level attributes for Sphinx autodoc.
# This __annotations__ dict helps documentation generators (like Sphinx) understand
# what attributes are available in this module without having to execute __getattr__.
# It provides type information for each lazily-loaded scorer class.
__annotations__ = {
    "Correctness": "type[mlflow.genai.scorers.builtin_scorers.Correctness]",
    "ExpectationsGuidelines": "type[mlflow.genai.scorers.builtin_scorers.ExpectationsGuidelines]",
    "Guidelines": "type[mlflow.genai.scorers.builtin_scorers.Guidelines]",
    "RelevanceToQuery": "type[mlflow.genai.scorers.builtin_scorers.RelevanceToQuery]",
    "RetrievalGroundedness": "type[mlflow.genai.scorers.builtin_scorers.RetrievalGroundedness]",
    "RetrievalRelevance": "type[mlflow.genai.scorers.builtin_scorers.RetrievalRelevance]",
    "RetrievalSufficiency": "type[mlflow.genai.scorers.builtin_scorers.RetrievalSufficiency]",
    "Safety": "type[mlflow.genai.scorers.builtin_scorers.Safety]",
    "get_all_scorers": (
        "Callable[[], dict[str, type[mlflow.genai.scorers.builtin_scorers.BuiltInScorer]]]"
    ),
}


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

# For Sphinx autodoc: explicitly set the lazy-loaded classes as module attributes
# This allows Sphinx to document them even though they're not imported at module initialization
# These will be replaced by the actual classes when accessed via __getattr__
import sys

if "sphinx" in sys.modules:
    # Only do this when building docs to avoid the circular import
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
