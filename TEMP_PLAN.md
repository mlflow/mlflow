# DeepEval Integration Plan for MLflow.genai

## Executive Summary

This document outlines a comprehensive plan to integrate 30 DeepEval LLM evaluation metrics into MLflow's `mlflow.genai` module. The integration will provide two levels of access: a generic `get_judge()` wrapper for any DeepEval metric (P0) and direct namespaced scorer classes for each metric (P1).

**Note**: This integration excludes multimodal metrics, custom framework metrics (G-Eval, DAG, Arena variants), and MCP metrics to focus on standard LLM evaluation use cases.

## Background

### MLflow Judge Interface
MLflow judges/scorers use the following interface (from `mlflow/genai/scorers/base.py:432-526`):

```python
def __call__(
    self,
    *,  # keyword-only
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
) -> Feedback
```

Key characteristics:
- All parameters are keyword-only and optional
- MLflow filters parameters based on function signature inspection
- Returns `Feedback` object with `value`, `rationale`, and `source`
- Model specified via URI (e.g., `"databricks"`, `"openai:/gpt-4"`, `"anthropic:/claude-3-5-sonnet"`)

### DeepEval Metric Interface
DeepEval metrics use the following interface (from `deepeval/metrics/base_metric.py`):

```python
class BaseMetric:
    def __init__(
        self,
        threshold: float = 0.5,
        model: str | DeepEvalBaseLLM | None = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        ...

    def measure(self, test_case: LLMTestCase) -> float:
        ...
```

DeepEval's `LLMTestCase` structure (from `deepeval/test_case/llm_test_case.py`):

```python
class LLMTestCase(BaseModel):
    # Required
    input: str

    # Optional
    actual_output: str | None
    expected_output: str | None
    context: List[str] | None
    retrieval_context: List[str] | None
    additional_metadata: Dict | None
    tools_called: List[ToolCall] | None
    comments: str | None
    expected_tools: List[ToolCall] | None
    # ... and more
```

DeepEval's `ConversationalTestCase` structure for multi-turn metrics:

```python
class ConversationalTestCase(BaseModel):
    # Required
    turns: List[LLMTestCase]  # List of conversation turns

    # Optional
    additional_metadata: Dict | None
    comments: str | None
```

**Multi-Turn Metrics Behavior**:
- Accept a list of MLflow traces representing conversation turns
- Each trace is converted to an `LLMTestCase`
- All test cases are wrapped in a `ConversationalTestCase`
- Metrics evaluate the entire conversation context

## Available DeepEval Metrics

DeepEval provides 30 metrics across the following categories:

**Note**: Some metrics appear in both agentic and conversational categories as they evaluate multi-turn agent interactions.

### 1. RAG (Retrieval-Augmented Generation) Metrics
- **AnswerRelevancy** - Evaluates output relevance to input
- **Faithfulness** - Checks factual alignment with retrieval context
- **ContextualRecall** - Measures retrieval coverage
- **ContextualPrecision** - Evaluates retriever ranking quality
- **ContextualRelevancy** - Assesses overall context relevance

### 2. Agentic Metrics
- **TaskCompletion** - Evaluates agent's ability to complete assigned tasks
- **ToolCorrectness** - Assesses correct tool selection and usage
- **ArgumentCorrectness** - Validates correctness of tool arguments
- **StepEfficiency** - Evaluates efficiency of agent steps
- **PlanAdherence** - Assesses adherence to planned approach
- **PlanQuality** - Evaluates quality of generated plans

### 3. Conversational Metrics (Multi-Turn)

**Note**: These metrics use `ConversationalTestCase` instead of `LLMTestCase` and accept a list of conversation turns.

- **TurnRelevancy** - Evaluates relevance of each individual conversation turn
- **RoleAdherence** - Assesses staying in character throughout conversation
- **KnowledgeRetention** - Evaluates chatbot's ability to retain information
- **ConversationCompleteness** - Evaluates whether conversation satisfies user needs
- **GoalAccuracy** - Evaluates accuracy in achieving goals in multi-turn context
- **ToolUse** - Measures effectiveness of tool usage in conversations
- **TopicAdherence** - Measures adherence to specified topics in conversation

### 4. Safety & Responsible AI Metrics
- **Bias** - Detects gender, racial, or political bias
- **Toxicity** - Evaluates harmful or offensive content
- **NonAdvice** - Checks for inappropriate advice
- **Misuse** - Detects potential misuse scenarios
- **PIILeakage** - Identifies personal information leakage
- **RoleViolation** - Detects violations of assigned role

### 5. General Metrics
- **Hallucination** - Detects fabricated information
- **Summarization** - Evaluates summary quality and accuracy
- **JsonCorrectness** - Validates JSON schema compliance
- **PromptAlignment** - Measures alignment with prompt instructions

### 6. Non-LLM Deterministic Metrics
- **ExactMatch** - Exact string matching evaluation
- **PatternMatch** - Regex pattern matching evaluation

## Architecture Design

### Directory Structure

```
mlflow/genai/scorers/
├── __init__.py                  # Update to export deepeval module
├── base.py                      # Existing base classes
├── builtin_scorers.py           # Existing MLflow scorers
├── deepeval/                    # New DeepEval integration
│   ├── __init__.py              # Exports get_judge and all metric classes
│   ├── adapter.py               # Core adapter logic
│   ├── models.py                # Custom model implementations
│   ├── rag_metrics.py           # RAG metric wrappers (5 metrics)
│   ├── agentic_metrics.py       # Agentic metric wrappers (6 metrics)
│   ├── conversational_metrics.py # Conversational metric wrappers (7 metrics)
│   ├── safety_metrics.py        # Safety metric wrappers (6 metrics)
│   ├── general_metrics.py       # General metric wrappers (4 metrics)
│   ├── deterministic_metrics.py # Non-LLM metric wrappers (2 metrics)
│   └── registry.py              # Metric name -> class mapping
```

### P0: Generic get_judge() Function

**Goal**: Support any DeepEval scorer with a single function.

```python
from mlflow.genai.scorers.deepeval import get_judge

# Single-turn metrics (standard)
deep_eval_answer_relevance = get_judge("AnswerRelevancy", threshold=0.7, model="openai:/gpt-4")
deep_eval_answer_relevance(trace=trace)

# Multi-turn metrics (conversational)
deep_eval_knowledge_retention = get_judge("KnowledgeRetention", include_reason=True)
deep_eval_knowledge_retention(traces=[trace1, trace2, trace3])  # List of conversation turns
```

**Implementation** (`mlflow/genai/scorers/deepeval/adapter.py`):

```python
from typing import Any, Literal
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.genai.scorers.base import Scorer

class DeepEvalScorer(Scorer):
    """Adapter that wraps any DeepEval metric to work with MLflow's judge interface."""

    def __init__(
        self,
        metric_name: str,
        threshold: float = 0.5,
        model: str | None = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        **metric_kwargs,
    ):
        """
        Initialize a DeepEval metric adapter.

        Args:
            metric_name: Name of the DeepEval metric (e.g., "answer_relevancy")
            threshold: Score threshold for success (default: 0.5)
            model: Model URI in MLflow format (e.g., "openai:/gpt-4", "databricks")
            include_reason: Whether to include reasoning in feedback
            strict_mode: Whether to use strict mode (forces threshold to 1.0)
            **metric_kwargs: Additional metric-specific parameters
        """
        self.metric_name = metric_name
        self.threshold = threshold
        self.model_uri = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.metric_kwargs = metric_kwargs

        # Lazy import to avoid dependency issues
        from .registry import get_metric_class

        metric_class = get_metric_class(metric_name)

        # Convert MLflow model URI to DeepEval model
        deepeval_model = self._create_deepeval_model(model)

        # Initialize the DeepEval metric
        self.metric = metric_class(
            threshold=threshold,
            model=deepeval_model,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=False,
            async_mode=False,  # Use sync mode for simplicity
            **metric_kwargs,
        )

    def _create_deepeval_model(self, model_uri: str | None):
        """Convert MLflow model URI to DeepEval model instance."""
        if model_uri is None:
            # Use DeepEval's default (usually GPT-4)
            return None

        if model_uri == "databricks":
            # Use custom Databricks model
            from .models import DatabricksDeepEvalModel
            return DatabricksDeepEvalModel()

        # Parse MLflow model URI format: "provider:/model_name"
        if ":" in model_uri:
            provider, model_name = model_uri.split(":", 1)
            provider = provider.lstrip("/")
            model_name = model_name.lstrip("/")

            # Use DeepEval's LiteLLMModel for standard providers
            from deepeval.models import LiteLLMModel

            # LiteLLM uses format "provider/model_name"
            litellm_model_uri = f"{provider}/{model_name}"
            return LiteLLMModel(model=litellm_model_uri)

        # Fallback: assume it's a direct model name
        from deepeval.models import LiteLLMModel
        return LiteLLMModel(model=model_uri)

    def _map_mlflow_to_test_case(
        self,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ):
        """Map MLflow __call__ parameters to DeepEval LLMTestCase."""
        from deepeval.test_case import LLMTestCase

        # Convert inputs to string
        input_str = self._convert_to_string(inputs) if inputs is not None else ""

        # Convert outputs to string
        actual_output = self._convert_to_string(outputs) if outputs is not None else None

        # Extract optional fields from expectations
        expected_output = None
        context = None
        retrieval_context = None

        if expectations:
            expected_output = expectations.get("expected_output")
            if expected_output:
                expected_output = self._convert_to_string(expected_output)

            # Handle context fields
            if "context" in expectations:
                ctx = expectations["context"]
                if isinstance(ctx, list):
                    context = [self._convert_to_string(c) for c in ctx]
                else:
                    context = [self._convert_to_string(ctx)]

            if "retrieval_context" in expectations:
                ret_ctx = expectations["retrieval_context"]
                if isinstance(ret_ctx, list):
                    retrieval_context = [self._convert_to_string(c) for c in ret_ctx]
                else:
                    retrieval_context = [self._convert_to_string(ret_ctx)]

        # Extract tool calls from trace if available
        tools_called = None
        if trace:
            tools_called = self._extract_tools_from_trace(trace)

        return LLMTestCase(
            input=input_str,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
            tools_called=tools_called,
        )

    def _convert_to_string(self, value: Any) -> str:
        """Convert any value to string representation."""
        if isinstance(value, str):
            return value
        elif isinstance(value, dict):
            import json
            return json.dumps(value, indent=2)
        elif isinstance(value, list):
            import json
            return json.dumps(value, indent=2)
        else:
            return str(value)

    def _extract_tools_from_trace(self, trace: Trace):
        """Extract tool calls from MLflow trace."""
        # Implementation depends on trace structure
        # For now, return None - can be enhanced later
        return None

    def _is_multi_turn_metric(self) -> bool:
        """Check if this is a multi-turn conversational metric."""
        multi_turn_metrics = {
            "TurnRelevancy",
            "RoleAdherence",
            "KnowledgeRetention",
            "ConversationCompleteness",
            "GoalAccuracy",
            "ToolUse",
            "TopicAdherence",
        }
        return self.metric_name in multi_turn_metrics

    def _map_mlflow_to_conversational_test_case(self, traces: list[Trace]):
        """Map list of MLflow traces to DeepEval ConversationalTestCase."""
        from deepeval.test_case import ConversationalTestCase, LLMTestCase

        # Convert each trace to an LLMTestCase
        turns = []
        for trace in traces:
            test_case = self._map_mlflow_to_test_case(trace=trace)
            turns.append(test_case)

        return ConversationalTestCase(turns=turns)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
        traces: list[Trace] | None = None,
    ) -> Feedback:
        """
        Evaluate using the wrapped DeepEval metric.

        Args:
            inputs: The input to evaluate (single-turn)
            outputs: The output to evaluate (single-turn)
            expectations: Expected values and context for evaluation
            trace: MLflow trace for single-turn evaluation
            traces: List of MLflow traces for multi-turn evaluation

        Returns:
            Feedback object with score, rationale, and source
        """
        # Check if this is a multi-turn metric
        if self._is_multi_turn_metric():
            if traces is None:
                raise ValueError(
                    f"Multi-turn metric '{self.metric_name}' requires 'traces' parameter"
                )
            # Map list of traces to ConversationalTestCase
            test_case = self._map_mlflow_to_conversational_test_case(traces)
        else:
            # Map MLflow parameters to DeepEval LLMTestCase
            test_case = self._map_mlflow_to_test_case(
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
            )

        # Run the DeepEval metric
        self.metric.measure(test_case)

        # Extract results
        score = self.metric.score
        reason = self.metric.reason if self.include_reason else None
        success = self.metric.is_successful()

        # Create Feedback object
        return Feedback(
            name=self.metric_name,
            value=score,
            rationale=reason,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=f"deepeval/{self.metric_name}",
            ),
            metadata={
                "success": success,
                "threshold": self.threshold,
            },
        )


def get_judge(
    metric_name: str,
    threshold: float = 0.5,
    model: str | None = None,
    include_reason: bool = True,
    strict_mode: bool = False,
    **metric_kwargs,
) -> DeepEvalScorer:
    """
    Get a DeepEval metric as an MLflow judge.

    Args:
        metric_name: Name of the DeepEval metric (e.g., "answer_relevancy", "faithfulness")
        threshold: Score threshold for success (default: 0.5)
        model: Model URI in MLflow format (e.g., "openai:/gpt-4", "databricks")
        include_reason: Whether to include reasoning in feedback
        strict_mode: Whether to use strict mode (forces threshold to 1.0)
        **metric_kwargs: Additional metric-specific parameters

    Returns:
        DeepEvalScorer instance that can be called with MLflow's judge interface

    Examples:
        >>> judge = get_judge("answer_relevancy", threshold=0.7, model="openai:/gpt-4")
        >>> feedback = judge(inputs="What is MLflow?", outputs="MLflow is a platform...")

        >>> judge = get_judge("faithfulness", model="databricks")
        >>> feedback = judge(
        ...     outputs="Paris is the capital of France",
        ...     expectations={"retrieval_context": ["Paris is France's capital city"]}
        ... )
    """
    return DeepEvalScorer(
        metric_name=metric_name,
        threshold=threshold,
        model=model,
        include_reason=include_reason,
        strict_mode=strict_mode,
        **metric_kwargs,
    )
```

### P1: Namespaced Scorer Classes

**Goal**: Provide direct access to each DeepEval metric as a named class.

```python
from mlflow.genai.scorers.deepeval import AnswerRelevancy, ContextualRecall, Faithfulness

# Use specific metric classes
answer_relevancy = AnswerRelevancy(threshold=0.7, model="openai:/gpt-4")
answer_relevancy(inputs="question", outputs="answer")

contextual_recall = ContextualRecall(model="databricks")
contextual_recall(outputs="answer", expectations={"retrieval_context": ["context"]})
```

**Implementation** (example for RAG metrics in `mlflow/genai/scorers/deepeval/rag_metrics.py`):

```python
from typing import Any
from mlflow.genai.scorers.deepeval.adapter import DeepEvalScorer

class AnswerRelevancy(DeepEvalScorer):
    """
    Evaluates how relevant the LLM's output is to the input query.

    This metric uses an LLM to extract statements from the output and determine
    which statements are relevant to addressing the input.

    Args:
        threshold: Score threshold for success (default: 0.5)
        model: Model URI in MLflow format (e.g., "openai:/gpt-4", "databricks")
        include_reason: Whether to include reasoning in feedback (default: True)
        strict_mode: Whether to use strict mode (default: False)

    Examples:
        >>> scorer = AnswerRelevancy(threshold=0.7, model="openai:/gpt-4")
        >>> feedback = scorer(
        ...     inputs="What is the capital of France?",
        ...     outputs="Paris is the capital of France, known for the Eiffel Tower."
        ... )
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model: str | None = None,
        include_reason: bool = True,
        strict_mode: bool = False,
    ):
        super().__init__(
            metric_name="answer_relevancy",
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            strict_mode=strict_mode,
        )


class Faithfulness(DeepEvalScorer):
    """
    Evaluates whether the LLM's output is factually consistent with the retrieval context.

    This metric checks if claims in the output are supported by the provided context,
    helping detect hallucinations in RAG systems.

    Args:
        threshold: Score threshold for success (default: 0.5)
        model: Model URI in MLflow format (e.g., "openai:/gpt-4", "databricks")
        include_reason: Whether to include reasoning in feedback (default: True)
        strict_mode: Whether to use strict mode (default: False)

    Examples:
        >>> scorer = Faithfulness(threshold=0.8)
        >>> feedback = scorer(
        ...     outputs="Paris is the capital of France",
        ...     expectations={"retrieval_context": ["Paris is France's capital city"]}
        ... )
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model: str | None = None,
        include_reason: bool = True,
        strict_mode: bool = False,
    ):
        super().__init__(
            metric_name="faithfulness",
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            strict_mode=strict_mode,
        )


class ContextualRecall(DeepEvalScorer):
    """
    Evaluates whether the retrieval context contains all information needed for the expected output.

    This metric measures the quality of the retriever by checking if retrieved documents
    contain the necessary information to produce the ideal response.

    Args:
        threshold: Score threshold for success (default: 0.5)
        model: Model URI in MLflow format (e.g., "openai:/gpt-4", "databricks")
        include_reason: Whether to include reasoning in feedback (default: True)
        strict_mode: Whether to use strict mode (default: False)

    Examples:
        >>> scorer = ContextualRecall(model="databricks")
        >>> feedback = scorer(
        ...     expectations={
        ...         "expected_output": "Paris is the capital",
        ...         "retrieval_context": ["Paris is France's capital city"]
        ...     }
        ... )
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model: str | None = None,
        include_reason: bool = True,
        strict_mode: bool = False,
    ):
        super().__init__(
            metric_name="contextual_recall",
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            strict_mode=strict_mode,
        )


# Additional RAG metrics...
class ContextualPrecision(DeepEvalScorer):
    """Evaluates whether relevant context is ranked higher than irrelevant context."""
    def __init__(self, threshold=0.5, model=None, include_reason=True, strict_mode=False):
        super().__init__("contextual_precision", threshold, model, include_reason, strict_mode)


class ContextualRelevancy(DeepEvalScorer):
    """Evaluates the overall relevance of the retrieval context."""
    def __init__(self, threshold=0.5, model=None, include_reason=True, strict_mode=False):
        super().__init__("contextual_relevancy", threshold, model, include_reason, strict_mode)
```

Similar implementations would be created for:
- `agentic_metrics.py`: `TaskCompletion`, `ToolCorrectness`, `ArgumentCorrectness`, `StepEfficiency`, `PlanAdherence`, `PlanQuality`
- `conversational_metrics.py`: `TurnRelevancy`, `RoleAdherence`, `KnowledgeRetention`, `ConversationCompleteness`, `GoalAccuracy`, `ToolUse`, `TopicAdherence`
- `safety_metrics.py`: `Bias`, `Toxicity`, `NonAdvice`, `Misuse`, `PIILeakage`, `RoleViolation`
- `general_metrics.py`: `Hallucination`, `Summarization`, `JsonCorrectness`, `PromptAlignment`
- `deterministic_metrics.py`: `ExactMatch`, `PatternMatch`

### Model Integration

**Strategy**: Use DeepEval's `LiteLLMModel` for standard providers (OpenAI, Anthropic, etc.) and create custom `DatabricksDeepEvalModel` implementing `DeepEvalBaseLLM` interface for Databricks endpoints.

**Implementation** (`mlflow/genai/scorers/deepeval/models.py`):

```python
class DatabricksDeepEvalModel(DeepEvalBaseLLM):
    """Wraps MLflow's Databricks adapter for DeepEval."""
    def generate(self, prompt: str) -> str:
        # Use _invoke_databricks_serving_endpoint from MLflow's adapter
        ...
```

## Implementation Plan

### Phase 1: Core Infrastructure (P0)

**Files to create**:
1. `mlflow/genai/scorers/deepeval/__init__.py` - Module exports
2. `mlflow/genai/scorers/deepeval/adapter.py` - Core `DeepEvalScorer` and `get_judge()`
3. `mlflow/genai/scorers/deepeval/registry.py` - Metric name → class mapping
4. `mlflow/genai/scorers/deepeval/models.py` - Custom model implementations
5. `mlflow/genai/scorers/deepeval/utils.py` - Helper functions for multi-turn support

**Tasks**:
- [ ] Implement `DeepEvalScorer` base class
- [ ] Implement `get_judge()` factory function
- [ ] Create metric registry with all 30 metric mappings (23 single-turn + 7 multi-turn)
- [ ] Implement `DatabricksDeepEvalLLM` for default Databricks endpoint
- [ ] Implement `DatabricksServingEndpointDeepEvalLLM` for custom serving endpoints
- [ ] Add multi-turn metric support with `ConversationalTestCase`
- [ ] Add comprehensive docstrings with examples
- [ ] Update `mlflow/genai/scorers/__init__.py` to export new module

**Testing**:
```python
# tests/genai/scorers/deepeval/test_adapter.py
def test_get_judge_answer_relevancy():
    judge = get_judge("AnswerRelevancy", threshold=0.7, model="openai:/gpt-4")
    feedback = judge(inputs="What is MLflow?", outputs="MLflow is a platform...")
    assert feedback.value >= 0.0
    assert feedback.value <= 1.0
    assert feedback.rationale is not None

def test_get_judge_with_databricks_model():
    judge = get_judge("Faithfulness", model="databricks")
    feedback = judge(
        outputs="Paris is the capital",
        expectations={"retrieval_context": ["Paris is France's capital"]}
    )
    assert feedback.value >= 0.0

def test_get_judge_databricks_serving_endpoint():
    judge = get_judge("AnswerRelevancy", model="databricks:/my-endpoint")
    feedback = judge(inputs="question", outputs="answer")
    assert feedback.value >= 0.0

def test_get_judge_multi_turn_metric():
    judge = get_judge("KnowledgeRetention", model="openai:/gpt-4")
    feedback = judge(traces=[trace1, trace2, trace3])
    assert feedback.value >= 0.0
    assert feedback.rationale is not None
```

### Phase 2: Namespaced Classes (P1)

**Files to create**:
1. `mlflow/genai/scorers/deepeval/rag_metrics.py` - 5 RAG metrics
2. `mlflow/genai/scorers/deepeval/agentic_metrics.py` - 6 agentic metrics
3. `mlflow/genai/scorers/deepeval/conversational_metrics.py` - 7 conversational metrics
4. `mlflow/genai/scorers/deepeval/safety_metrics.py` - 6 safety metrics
5. `mlflow/genai/scorers/deepeval/general_metrics.py` - 4 general metrics
6. `mlflow/genai/scorers/deepeval/deterministic_metrics.py` - 2 non-LLM metrics

**Tasks**:
- [ ] Implement wrapper class for each metric category
- [ ] Add detailed docstrings with parameter descriptions
- [ ] Include usage examples for each metric
- [ ] Export all classes from `__init__.py`

**Testing**:
```python
# tests/genai/scorers/deepeval/test_rag_metrics.py
def test_answer_relevancy_class():
    scorer = AnswerRelevancy(threshold=0.7, model="openai:/gpt-4")
    feedback = scorer(inputs="question", outputs="answer")
    assert feedback.name == "answer_relevancy"

def test_faithfulness_class():
    scorer = Faithfulness(model="databricks")
    feedback = scorer(
        outputs="fact",
        expectations={"retrieval_context": ["supporting context"]}
    )
    assert feedback.name == "faithfulness"
```

### Phase 3: Documentation & Testing
- Add user guide with examples
- Integration tests with real LLM APIs
- Comprehensive docstrings for all classes

## Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| P0 Core | Adapter + get_judge() | 3-5 days |
| P1 Classes | Namespaced wrappers | 3-4 days |
| Testing | Unit + integration tests | 2-3 days |
| Documentation | Guide + API docs | 2-3 days |
| Review | Code review + iterations | 2-3 days |
| **Total** | | **12-18 days** |

## Summary

Integrate 30 DeepEval metrics into MLflow via:
1. **P0**: `get_judge()` wrapper function
   - 23 single-turn metrics using `LLMTestCase`
   - 7 multi-turn conversational metrics using `ConversationalTestCase`
2. **P1**: Namespaced classes for each metric

Use adapter pattern to wrap DeepEval metrics, supporting:
- Standard LLM providers (OpenAI, Anthropic, etc.) via LiteLLM
- Databricks managed judge endpoint (`databricks`)
- Databricks custom serving endpoints (`databricks:/<endpoint_name>`)
- Multi-turn conversational evaluation with list of traces
