"""
TruLens agent trace scorers for goal-plan-action alignment evaluation.

This module provides TruLens agent trace evaluators as MLflow scorers, enabling
trace-aware evaluation of LLM agents. These evaluators analyze agent execution
traces to detect internal errors, with benchmarked 95% error coverage against
TRAIL (compared to 55% for standard LLM judges).

See: https://arxiv.org/abs/2510.08847

**Available Agent Trace Scorers:**

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

**Usage Examples:**

All agent trace scorers require a trace parameter (MLflow Trace object or JSON string).

.. code-block:: python

    import mlflow
    from mlflow.entities.span import SpanType
    from mlflow.genai.scorers import (
        TruLensLogicalConsistencyScorer,
        TruLensExecutionEfficiencyScorer,
        TruLensPlanAdherenceScorer,
        TruLensPlanQualityScorer,
        TruLensToolSelectionScorer,
        TruLensToolCallingScorer,
    )

    # First, create an agent trace to evaluate
    @mlflow.trace(name="research_agent", span_type=SpanType.AGENT)
    def research_agent(query):
        # Planning phase
        with mlflow.start_span(name="plan", span_type=SpanType.CHAIN) as plan_span:
            plan_span.set_inputs({"query": query})
            plan = "1. Search for information 2. Analyze results 3. Summarize findings"
            plan_span.set_outputs({"plan": plan})

        # Tool execution phase
        with mlflow.start_span(name="web_search", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs({"search_query": query})
            results = ["Result 1: MLflow is an ML platform", "Result 2: MLflow tracks experiments"]
            tool_span.set_outputs({"results": results})

        return "MLflow is an open-source platform for managing ML lifecycle."

    # Run the agent to generate a trace
    research_agent("What is MLflow?")
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    # Evaluate the trace with all scorers
    results = mlflow.genai.evaluate(
        data=[trace],
        scorers=[
            TruLensLogicalConsistencyScorer(),
            TruLensExecutionEfficiencyScorer(),
            TruLensPlanAdherenceScorer(),
            TruLensPlanQualityScorer(),
            TruLensToolSelectionScorer(),
            TruLensToolCallingScorer(),
        ],
    )

**Individual Scorer Examples:**

TruLensLogicalConsistencyScorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mlflow.genai.scorers import TruLensLogicalConsistencyScorer

    scorer = TruLensLogicalConsistencyScorer()
    result = scorer(trace=trace)
    print(f"Name: {result.name}")           # trulens_logical_consistency
    print(f"Value: {result.value}")         # 0.87 (score from 0-1)
    print(f"Rationale: {result.rationale}") # Detailed reasoning

TruLensExecutionEfficiencyScorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mlflow.genai.scorers import TruLensExecutionEfficiencyScorer

    scorer = TruLensExecutionEfficiencyScorer()
    result = scorer(trace=trace)
    print(f"Name: {result.name}")           # trulens_execution_efficiency
    print(f"Value: {result.value}")         # 0.92 (1.0 = highly efficient)
    print(f"Rationale: {result.rationale}")

TruLensPlanAdherenceScorer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mlflow.genai.scorers import TruLensPlanAdherenceScorer

    scorer = TruLensPlanAdherenceScorer()
    result = scorer(trace=trace)
    print(f"Name: {result.name}")           # trulens_plan_adherence
    print(f"Value: {result.value}")         # 0.95 (1.0 = followed plan exactly)
    print(f"Rationale: {result.rationale}")

TruLensPlanQualityScorer
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mlflow.genai.scorers import TruLensPlanQualityScorer

    scorer = TruLensPlanQualityScorer()
    result = scorer(trace=trace)
    print(f"Name: {result.name}")           # trulens_plan_quality
    print(f"Value: {result.value}")         # 0.88 (1.0 = excellent plan)
    print(f"Rationale: {result.rationale}")

TruLensToolSelectionScorer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mlflow.genai.scorers import TruLensToolSelectionScorer

    scorer = TruLensToolSelectionScorer()
    result = scorer(trace=trace)
    print(f"Name: {result.name}")           # trulens_tool_selection
    print(f"Value: {result.value}")         # 0.91 (1.0 = perfect tool choice)
    print(f"Rationale: {result.rationale}")

TruLensToolCallingScorer
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mlflow.genai.scorers import TruLensToolCallingScorer

    scorer = TruLensToolCallingScorer()
    result = scorer(trace=trace)
    print(f"Name: {result.name}")           # trulens_tool_calling
    print(f"Value: {result.value}")         # 0.89 (1.0 = correct tool invocation)
    print(f"Rationale: {result.rationale}")

**Custom Configuration:**

.. code-block:: python

    # Use custom model and evaluation criteria
    scorer = TruLensLogicalConsistencyScorer(
        name="custom_logical_check",
        model_name="gpt-4",
        model_provider="openai",  # or "litellm"
        criteria="Focus on whether the agent's reasoning follows from its observations",
        custom_instructions="Be strict about logical fallacies",
        temperature=0.0,
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

__all__ = [
    "TruLensLogicalConsistencyScorer",
    "TruLensExecutionEfficiencyScorer",
    "TruLensPlanAdherenceScorer",
    "TruLensPlanQualityScorer",
    "TruLensToolSelectionScorer",
    "TruLensToolCallingScorer",
]
