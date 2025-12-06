from typing import TYPE_CHECKING

from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig, scorer
from mlflow.genai.scorers.registry import delete_scorer, get_scorer, list_scorers

# Metadata keys for scorer feedback
FRAMEWORK_METADATA_KEY = "mlflow.scorer.framework"

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
    "Completeness",
    "ConversationalRoleAdherence",
    "ConversationalSafety",
    "ConversationCompleteness",
    "ConversationalToolCallEfficiency",
    "Correctness",
    "ExpectationsGuidelines",
    "Fluency",
    "Guidelines",
    "Equivalence",
    "RelevanceToQuery",
    "RetrievalGroundedness",
    "RetrievalRelevance",
    "RetrievalSufficiency",
    "Safety",
    "Summarization",
    "ToolCallCorrectness",
    "ToolCallEfficiency",
    "UserFrustration",
    "get_all_scorers",
}

# Phoenix scorer integrations
_PHOENIX_IMPORTS = {
    "PhoenixHallucinationScorer",
    "PhoenixRelevanceScorer",
    "PhoenixToxicityScorer",
    "PhoenixQAScorer",
    "PhoenixSummarizationScorer",
}

# TruLens scorer integrations
_TRULENS_IMPORTS = {
    "TruLensGroundednessScorer",
    "TruLensContextRelevanceScorer",
    "TruLensAnswerRelevanceScorer",
    "TruLensCoherenceScorer",
}


def __getattr__(name):
    """Lazily import builtin scorers and third-party integrations to avoid circular dependency."""
    if name in _LAZY_IMPORTS:
        # Import the module when first accessed
        from mlflow.genai.scorers import builtin_scorers

        return getattr(builtin_scorers, name)

    if name in _PHOENIX_IMPORTS:
        # Import Phoenix integrations when first accessed
        from mlflow.genai.scorers import phoenix

        return getattr(phoenix, name)

    if name in _TRULENS_IMPORTS:
        # Import TruLens integrations when first accessed
        from mlflow.genai.scorers import trulens

        return getattr(trulens, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return the list of attributes available in this module.

    This is necessary for Sphinx autodoc and other introspection tools
    to discover the lazily-loaded scorer classes.
    """
    # Get the default module attributes
    module_attrs = list(globals().keys())
    # Add the lazy imports and Phoenix/TruLens imports
    return sorted(set(module_attrs) | _LAZY_IMPORTS | _PHOENIX_IMPORTS | _TRULENS_IMPORTS)


# The TYPE_CHECKING block below is for static analysis tools only.
# TYPE_CHECKING is False at runtime, so these imports never execute during normal operation.
# This gives us the best of both worlds: type hints without circular imports.
if TYPE_CHECKING:
    from mlflow.genai.scorers.builtin_scorers import (
        Completeness,
        ConversationalRoleAdherence,
        ConversationalSafety,
        ConversationalToolCallEfficiency,
        ConversationCompleteness,
        Correctness,
        Equivalence,
        ExpectationsGuidelines,
        Fluency,
        Guidelines,
        RelevanceToQuery,
        RetrievalGroundedness,
        RetrievalRelevance,
        RetrievalSufficiency,
        Safety,
        Summarization,
        ToolCallCorrectness,
        ToolCallEfficiency,
        UserFrustration,
        get_all_scorers,
    )
    from mlflow.genai.scorers.phoenix import (
        PhoenixHallucinationScorer,
        PhoenixQAScorer,
        PhoenixRelevanceScorer,
        PhoenixSummarizationScorer,
        PhoenixToxicityScorer,
    )
    from mlflow.genai.scorers.trulens import (
        TruLensAnswerRelevanceScorer,
        TruLensCoherenceScorer,
        TruLensContextRelevanceScorer,
        TruLensGroundednessScorer,
    )

__all__ = [
    # Built-in scorers
    "Completeness",
    "ConversationalRoleAdherence",
    "ConversationalSafety",
    "ConversationalToolCallEfficiency",
    "ConversationCompleteness",
    "Correctness",
    "ExpectationsGuidelines",
    "Fluency",
    "Guidelines",
    "Equivalence",
    "RelevanceToQuery",
    "RetrievalGroundedness",
    "RetrievalRelevance",
    "RetrievalSufficiency",
    "Safety",
    "Summarization",
    "ToolCallCorrectness",
    "ToolCallEfficiency",
    "UserFrustration",
    # Third-party scorers - Phoenix
    "PhoenixHallucinationScorer",
    "PhoenixRelevanceScorer",
    "PhoenixToxicityScorer",
    "PhoenixQAScorer",
    "PhoenixSummarizationScorer",
    # Third-party scorers - TruLens
    "TruLensGroundednessScorer",
    "TruLensContextRelevanceScorer",
    "TruLensAnswerRelevanceScorer",
    "TruLensCoherenceScorer",
    # Base classes and utilities
    "Scorer",
    "scorer",
    "ScorerSamplingConfig",
    "get_all_scorers",
    "get_scorer",
    "list_scorers",
    "delete_scorer",
]
