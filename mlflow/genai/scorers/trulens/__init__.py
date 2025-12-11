"""
TruLens evaluation framework integration for MLflow GenAI scorers.

This module wraps TruLens feedback functions as MLflow scorers, enabling use of
TruLens' evaluation metrics within the MLflow evaluation framework.

**Available Scorers:**

Basic Scorers (input/output evaluation):
- ``TruLensGroundednessScorer``: Evaluates if outputs are grounded in context
- ``TruLensContextRelevanceScorer``: Evaluates context relevance to query
- ``TruLensAnswerRelevanceScorer``: Evaluates answer relevance to query
- ``TruLensCoherenceScorer``: Evaluates logical flow of outputs

Agent Trace Scorers (goal-plan-action alignment evaluation):
- ``TruLensLogicalConsistencyScorer``: Evaluates logical consistency of agent reasoning
- ``TruLensExecutionEfficiencyScorer``: Evaluates agent execution efficiency
- ``TruLensPlanAdherenceScorer``: Evaluates if agent follows its plan
- ``TruLensPlanQualityScorer``: Evaluates quality of agent's plan
- ``TruLensToolSelectionScorer``: Evaluates correctness of tool selection
- ``TruLensToolCallingScorer``: Evaluates tool calling accuracy

**Installation:**
    pip install trulens trulens-providers-openai

For LiteLLM provider support:
    pip install trulens trulens-providers-litellm

**Example (Basic Scorer):**

.. code-block:: python

    from mlflow.genai.scorers import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer()
    result = scorer(
        outputs="Paris is the capital.",
        context="France's capital is Paris.",
    )
    print(result.value)  # Score between 0.0 and 1.0

**Example (Agent Trace Scorer):**

.. code-block:: python

    import mlflow
    from mlflow.genai.scorers import TruLensLogicalConsistencyScorer

    # Get traces from your agent
    traces = mlflow.search_traces(experiment_ids=["1"])

    # Evaluate traces
    results = mlflow.genai.evaluate(
        data=traces,
        scorers=[TruLensLogicalConsistencyScorer()],
    )

For more information on TruLens, see:
https://www.trulens.org/
"""

from mlflow.genai.scorers.trulens.agent_trace import (
    TruLensExecutionEfficiencyScorer,
    TruLensLogicalConsistencyScorer,
    TruLensPlanAdherenceScorer,
    TruLensPlanQualityScorer,
    TruLensToolCallingScorer,
    TruLensToolSelectionScorer,
)
from mlflow.genai.scorers.trulens.basic import (
    TruLensAnswerRelevanceScorer,
    TruLensCoherenceScorer,
    TruLensContextRelevanceScorer,
    TruLensGroundednessScorer,
)

__all__ = [
    # Basic scorers
    "TruLensGroundednessScorer",
    "TruLensContextRelevanceScorer",
    "TruLensAnswerRelevanceScorer",
    "TruLensCoherenceScorer",
    # Agent trace scorers
    "TruLensLogicalConsistencyScorer",
    "TruLensExecutionEfficiencyScorer",
    "TruLensPlanAdherenceScorer",
    "TruLensPlanQualityScorer",
    "TruLensToolSelectionScorer",
    "TruLensToolCallingScorer",
]
