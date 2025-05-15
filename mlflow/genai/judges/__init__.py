"""
Databricks Agent Eval Built-in Judges. For more details see Databricks Agent Evaluation:
<https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>
"""

from mlflow.utils.annotations import experimental

try:
    from databricks.agents.evals.judges import (
        chunk_relevance as _chunk_relevance,
        context_sufficiency as _context_sufficiency,
        correctness as _correctness,
        groundedness as _groundedness,
        guideline_adherence as _guideline_adherence,
        relevance_to_query as _relevance_to_query,
        safety as _safety,
    )
except ImportError:
    raise ImportError(
        "The `databricks-agents` package is required to use mlflow.genai.judges. "
        "Please install it with `pip install databricks-agents`."
    )


@experimental
def chunk_relevance(*args, **kwargs):
    return _chunk_relevance(*args, **kwargs)


@experimental
def context_sufficiency(*args, **kwargs):
    return _context_sufficiency(*args, **kwargs)


@experimental
def correctness(*args, **kwargs):
    return _correctness(*args, **kwargs)


@experimental
def groundedness(*args, **kwargs):
    return _groundedness(*args, **kwargs)


@experimental
def guideline_adherence(*args, **kwargs):
    return _guideline_adherence(*args, **kwargs)


@experimental
def relevance_to_query(*args, **kwargs):
    return _relevance_to_query(*args, **kwargs)


@experimental
def safety(*args, **kwargs):
    return _safety(*args, **kwargs)


__all__ = [
    "chunk_relevance",
    "context_sufficiency",
    "correctness",
    "groundedness",
    "guideline_adherence",
    "relevance_to_query",
    "safety",
]
