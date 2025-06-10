# overview

Predefined scorers provide default implementations of `Scorer` for the predefined judges for common GenAI evaluation use cases.

They can be passed directly to the evaluate(..., scorers=[]) parameter. Predefined scorers require your trace and evaluate() input data to meet certain requirements. If your trace or input data does not meet these requirements or you wish to customize the behavior of the scorer, you can create a custom scorer.

:::important
Predefined scorers only work if:

1. You use MLflow Tracing with `mlflow.genai.evaluate()`, either by
   - passing a (`predict_fn`)[/genai/eval-monitor/concepts/eval-harness#1-have-evaluate-call-your-app-to-produce-traces-for-a-set-of-inputs] that is traced
   - passing an [answer sheet](/genai/eval-monitor/concepts/eval-harness#2-pass-pre-computed-outputs-from-your-app-to-evaluate) that includes pre-computed traces.
2. Your Trace has inputs and outputs set on the root span - by default, these fields are set when instrumenting your application with tracing
   :::

Predefined scorers work as follows:

1. For each row, `mlflow.genai.evaluate()` passes the Trace to the scorer
   - For some predefined scorers, the `expectations` key from the input data provided to the `data` arguement
2. Scorer parses the `trace` (and, if required, `expectations`) for the required data to score.
3. Scorer calls a predefined LLM judge to generate a [`Feedback`](/mlflow3/genai/tracing/data-model#feedbacks)
4. `mlflow.genai.evaluate()` attaches the resulting `Feedback` to the evaluation results

- What it evaluates?
- Trace requirements
- Input data requirements
- Predefined judges used
- How it works
- Example usage (leave as a blank python snippet for now)

|                       | What it does?                                                                                                           | Additional trace requirements                                                                                                                                      | Input data requirements                                                                                                                                                  | Predefined judges used |
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------- |
| Correctness           | Checks that the agent's responses are correct and accurate.                                                             | n/a                                                                                                                                                                | `expectations` must contain 1 of the following keys (but not both) `expected_facts` `Expected_response` Can be passed as expectations on the trace OR in the input data. | is_correct             |
| RelevanceToQuery      | Ensures that the agent's response directly addresses the user's input without deviating into unrelated topics.          | n/a                                                                                                                                                                | n/a                                                                                                                                                                      | is_context_relevant    |
| Safety                | Ensures that the agent's responses do not contain harmful, offensive, or toxic content.                                 | n/a                                                                                                                                                                | n/a                                                                                                                                                                      | is_safe                |
| PerRowGuidelines      | Evaluates whether the agent's response follows specific constraints or instructions provided in per-example guidelines. | n/a                                                                                                                                                                | `expectations` must contain the key `guidelines` Can be passed as expectations on the trace OR in the input data.                                                        | meets_guidelines       |
| GlobalGuidelines      | Evaluates whether the agent's response follows specific constraints or instructions provided in global guidelines.      | n/a                                                                                                                                                                | `guidelines` are passed directly to the scorer when provided to `evaluate()`                                                                                             | meets_guidelines       |
| RetrievalGroundedness | Assesses whether the agent's response is aligned with the information provided in the retrieved context.                | Trace must contain 1+ spans of type RETRIEVER                                                                                                                      | n/a                                                                                                                                                                      | is_grounded            |
| ToolGroundedness      | Assesses whether the agent's response is aligned with the information provided by tool calls.                           | Trace must contain 1+ CHAT_MODEL spans with OpenAI format tool calls. If you use LangChain/LangGraph or OpenAI autologging, your trace will meet this requirement. | n/a                                                                                                                                                                      | is_grounded            |
| RetrievalSufficiency  | Evaluates whether the retrieved documents provide all necessary information to generate the expected response.          | Trace must contain 1+ spans of type RETRIEVER                                                                                                                      | `expectations` must contain 1 of the following keys (but not both) `expected_facts` `Expected_response` Can be passed as expectations on the trace OR in the input data. | is_context_sufficient  |
| ToolSufficiency       | Evaluates whether the tools provide all necessary information to generate the expected response.                        | Trace must contain 1+ CHAT_MODEL spans with OpenAI format tool calls. If you use LangChain/LangGraph or OpenAI autologging, your trace will meet this requirement. | `expectations` must contain 1 of the following keys (but not both) `expected_facts` `Expected_response` Can be passed as expectations on the trace OR in the input data. | is_context_sufficient  |
| RetrievalRelevance    | Measures whether each retrieved chunk is relevant to the input request.                                                 | Trace must contain 1+ spans of type RETRIEVER                                                                                                                      |                                                                                                                                                                          | is_context_relevant    |

from mlflow.genai import scorers, judges

### throughout all scorers, inputs/outputs must be json-serializable

### throughout all judges take name= as optional parameter

### --- RENAMED JUDGES ---

### Renamed from judges.correctness

✅ judges.is_correct(request: str, response: str, \*, expected_facts: List[str], expected_response: str, name: Optional[str] = None) -> Feedback
scorers.response_correctness(name: Optional[str])(inputs: Any, outputs:Any, expectations: Dict[str, Any]) -> Feedback

### Renamed from judges.relevance_to_query

judges.is_relevant_to_query(request: str, response: str, name: Optional[str] = None) -> Feedback
✅ scorers.relevance_to_query(name: Optional[str])(inputs: Any, outputs:Any) -> Feedback
-> calls is_context_relevant() judge under the hood

### Renamed from judges.safety

### Note: Change from today's state, no need for request

✅ judges.is_safe(content: str, name: Optional[str] = None) -> Feedback
scorers.safety(name: Optional[str])(outputs: Any) -> Feedback

### New guidelines judge

### Renamed from judges.guidelines

✅ judges.meets_guidelines(guidelines: Union[str, List[str]], context: Dict[str, Any], , name: Optional[str] = None) -> Feedback

### Per row guidelines, works with guidelines in `expectations` that are Union[str, List[str], Dict[str, Union[str, List[str]]]

scorers.guidelines()(inputs: Any, outputs:Any, expectations: Dict[str, Any])

### Global guidlines, if 1 guideline, returns a single Feedback, otherwise List[Feedback]

scorers.guidelines(guidelines=Union[str, List[str]], name: str)(inputs: Any, outputs:Any) -> Feedback

### context must be json-serializable

### Renamed from judges.groundedness

✅ judges.is_grounded(request: str, response: str, context:Any, name: Optional[str] = None) -> Feedback ### Added request based on typical groundedness checks, adjust if not needed.

### this will produce a Feedback for EACH span of type=retriever in the Trace.

### Assuming this scorer might use judges.is_grounded internally or similar logic.

scorers.retrieval_groundedness()(inputs: Any, outputs:Any, trace:mlflow.Trace) -> List[Feedback] ### Corrected return type

### this will work over all tool outputs

scorers.tool_groundedness()(inputs: Any, outputs:Any, trace:mlflow.Trace) -> List[Feedback] ### Corrected return type

### context must be json-serializable

### Renamed from judges.context_sufficiency

✅ judges.is_context_sufficient(request: str, context: Any, \*, expected_facts: List[str], expected_response: str, name: Optional[str] = None) -> Feedback

### this will produce a Feedback for EACH span of type=retriever in the Trace.

scorers.retrieval_sufficiency()(inputs: Any, trace:mlflow.Trace, expectations: Dict[str, Any]) -> List[Feedback] ### Corrected return type

### this will work over all tool outputs

scorers.tool_sufficiency()(inputs: Any, trace:mlflow.Trace, expectations: Dict[str, Any]) -> List[Feedback] ### Corrected return type

### context must be json-serializable

### Renamed from judges.context_relevance

✅ judges.is_context_relevant(request: str, context: Any, name: Optional[str] = None) -> Feedback

### this will produce a set of Feedback outputs (as specified in the signature) for EACH span of type=retriever in the Trace. In each retriver, all chunks are considered by default, unless K is passed.

scorers.retrieval_relevance(k=Optional[int])(inputs: Any, trace:mlflow.Trace) -> List[Feedback] ### Simplified return type for clarity, detailed structure still applies.

### For all scorers.retrieval\_\*, we will do the following:

### - use the set of inputs to the retriever span as the query.

### For all scorers.tool\_\*, we will use the trace's inputs as the query.

## How a scorer is called in mlflow.genai.evaluate()

from mlfow.genai.scorers import NameOfScorer

mlflow.genai.evaluate(..., scorers=[NameOfScorer()])

# Scorers doc strings - any conflicts with `overview` - use that

from abc import abstractmethod
from typing import Any, Optional, Union

from mlflow.entities import Assessment
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai import judges
from mlflow.genai.judges.databricks import requires_databricks_agents
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.utils.trace_utils import (
extract_retrieval_context_from_trace,
parse_inputs_to_str,
parse_output_to_str,
)
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.utils.annotations import experimental

GENAI_CONFIG_NAME = "databricks-agent"

class BuiltInScorer(Scorer):
"""
Base class for built-in scorers that share a common implementation. All built-in scorers should
inherit from this class.
"""

    required_columns: set[str] = set()

    ### Avoid direct mutation of built-in scorer fields like name. Overriding the default setting must
    ### be done through the `with_config` method. This is because the short-hand syntax like
    ### `mlflow.genai.scorers.chunk_relevance` returns a single instance rather than a new instance.
    def __setattr__(self, name: str, value: Any) -> None:
        raise MlflowException(
            "Built-in scorer fields are immutable. Please use the `with_config` method to "
            "get a new instance with the custom field values.",
            error_code=BAD_REQUEST,
        )

    @abstractmethod
    def with_config(self, **kwargs) -> "BuiltInScorer":
        """
        Get a new scorer instance with the given configuration, such as name, global guidelines.

        Override this method with the appropriate config keys. This method must return the scorer
        instance itself with the updated configuration.
        """

    def validate_columns(self, columns: set[str]) -> None:
        missing_columns = self.required_columns - columns
        if missing_columns:
            raise MissingColumnsException(self.name, missing_columns)

### === Builtin Scorers ===

@experimental
class RetrievalRelevance(BuiltInScorer):
"""
Retrieval relevance measures whether each chunk is relevant to the input request.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.retrieval_relevance` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import retrieval_relevance

        trace = mlflow.get_trace("<your-trace-id>")
        feedbacks = retrieval_relevance(trace=trace)
        print(feedbacks)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[retrieval_relevance])
    """

    name: str = "retrieval_relevance"
    required_columns: set[str] = {"inputs", "trace"}

    def __call__(self, *, trace: Trace) -> Feedback:
        """
        Evaluate chunk relevance for each context chunk.

        Args:
            trace: The trace of the model's execution. Must contains at least one span with
                type `RETRIEVER`. MLflow will extract the retrieved context from that span.
                If multiple spans are found, MLflow will use the **last** one.

        Returns:
            A list of assessments evaluating the relevance of each context chunk.
            If the number of retrievers is N and each retriever has M chunks, the list will
            contain N * (M + 1) assessments. Each retriever span will emit M assessments
            for the relevance of its chunks and 1 assessment for the average relevance of all
            chunks.
        """
        request = parse_inputs_to_str(trace.data.spans[0].inputs)
        span_id_to_context = extract_retrieval_context_from_trace(trace)

        feedbacks = []
        for span_id, context in span_id_to_context.items():
            feedbacks.extend(self._compute_span_relevance(span_id, request, context))
        return feedbacks

    @requires_databricks_agents
    def _compute_span_relevance(
        self, span_id: str, request: str, chunks: dict[str, str]
    ) -> list[Feedback]:
        """Compute the relevance of retrieved context for one retriever span."""
        from databricks.agents.evals.judges import chunk_relevance

        ### Compute relevance for each chunk. Call `chunk_relevance` judge directly
        ### to get a list of feedbacks with ids.
        chunk_feedbacks = chunk_relevance(
            request=request, retrieved_context=chunks, assessment_name=self.name
        )
        for feedback in chunk_feedbacks:
            feedback.span_id = span_id

        if len(chunk_feedbacks) == 0:
            return []

        ### Compute average relevance across all chunks.
        ### NB: Handling error feedback as 0.0 relevance for simplicity.
        average = sum(f.value == "yes" for f in chunk_feedbacks) / len(chunk_feedbacks)

        span_level_feedback = Feedback(
            name=self.name,
            value=average,
            source=chunk_feedbacks[0].source,
            span_id=span_id,
        )
        return [span_level_feedback] + chunk_feedbacks

    def with_config(self, *, name: str = "retrieval_relevance") -> "RetrievalRelevance":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "retrieval_relevance".
        """
        return RetrievalRelevance(name=name)

@experimental
class RetrievalSufficiency(BuiltInScorer):
"""
Retrieval sufficiency evaluates whether the retrieved documents provide all necessary
information to generate the expected response.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.retrieval_sufficiency` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import retrieval_sufficiency

        trace = mlflow.get_trace("<your-trace-id>")
        feedback = retrieval_sufficiency(trace=trace)
        print(feedback)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[retrieval_sufficiency])
    """

    name: str = "retrieval_sufficiency"
    required_columns: set[str] = {"inputs", "trace"}

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        if (
            "expectations/expected_response" not in columns
            and "expectations/expected_facts" not in columns
        ):
            raise MissingColumnsException(
                self.name, ["expectations/expected_response or expectations/expected_facts"]
            )

    def __call__(
        self, *, trace: Trace, expectations: Optional[dict[str, Any]] = None
    ) -> list[Feedback]:
        """
        Evaluate context sufficiency based on retrieved documents.

        Args:
            trace: The trace of the model's execution. Must contains at least one span with
                type `RETRIEVER`. MLflow will extract the retrieved context from that span.
                If multiple spans are found, MLflow will use the **last** one.
            expectations: A dictionary of expectations for the response. Either `expected_facts` or
                `expected_response` key is required. Alternatively, you can pass a trace annotated
                with `expected_facts` or `expected_response` label(s) and omit this argument.
        """
        request = parse_inputs_to_str(trace.data.spans[0].inputs)
        span_id_to_context = extract_retrieval_context_from_trace(trace)

        ### If expectations are explicitly provided, use them.
        expectations = expectations or {}
        expected_facts = expectations.get("expected_facts")
        expected_response = expectations.get("expected_response")

        ### As a fallback, use the trace annotations as expectations.
        if expected_facts is None or expected_response is None:
            for assessment in trace.info.assessments:
                if assessment.name == "expected_facts" and expected_facts is None:
                    expected_facts = assessment.value
                if assessment.name == "expected_response" and expected_response is None:
                    expected_response = assessment.value

        ### This scorer returns a list of feedbacks, one for retriever span in the trace.
        feedbacks = []
        for span_id, context in span_id_to_context.items():
            feedback = judges.is_context_sufficient(
                request=request,
                context=context,
                expected_response=expected_response,
                expected_facts=expected_facts,
                name=self.name,
            )
            feedback.span_id = span_id
            feedbacks.append(feedback)

        return feedbacks

    def with_config(self, *, name: str = "retrieval_sufficiency") -> "RetrievalSufficiency":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "retrieval_sufficiency".
        """
        return RetrievalSufficiency(name=name)

@experimental
class RetrievalGroundedness(BuiltInScorer):
"""
RetrievalGroundedness assesses whether the agent's response is aligned with the information
provided in the retrieved context.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.retrieval_groundedness` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import retrieval_groundedness

        trace = mlflow.get_trace("<your-trace-id>")
        feedback = retrieval_groundedness(trace=trace)
        print(feedback)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[retrieval_groundedness])
    """

    name: str = "retrieval_groundedness"
    required_columns: set[str] = {"inputs", "trace"}

    def __call__(self, *, trace: Trace) -> list[Feedback]:
        """
        Evaluate groundedness of response against retrieved context.

        Args:
            trace: The trace of the model's execution. Must contains at least one span with
                type `RETRIEVER`. MLflow will extract the retrieved context from that span.
                If multiple spans are found, MLflow will use the **last** one.

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the groundedness of the response.
        """
        request = parse_inputs_to_str(trace.data.spans[0].inputs)
        response = parse_output_to_str(trace.data.spans[0].outputs)
        span_id_to_context = extract_retrieval_context_from_trace(trace)
        feedbacks = []
        for span_id, context in span_id_to_context.items():
            feedback = judges.is_grounded(
                request=request, response=response, context=context, name=self.name
            )
            feedback.span_id = span_id
            feedbacks.append(feedback)
        return feedbacks

    def with_config(self, *, name: str = "retrieval_groundedness") -> "RetrievalGroundedness":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "retrieval_groundedness".

        Returns:
            The updated RetrievalGroundedness scorer instance.
        """
        return RetrievalGroundedness(name=name)

@experimental
class GuidelineAdherence(BuiltInScorer):
"""
Guideline adherence evaluates whether the agent's response follows specific constraints
or instructions provided in the guidelines.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.guideline_adherence` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    There are two different ways to specify judges, depending on the use case:

    **1. Global Guidelines**

    If you want to evaluate all the response with a single set of guidelines, you can specify
    the guidelines in the `guidelines` parameter of this scorer.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import guideline_adherence

        ### Create a global judge
        english = guideline_adherence.with_config(
            name="english_guidelines",
            global_guidelines=["The response must be in English"],
        )
        feedback = english(outputs="The capital of France is Paris.")
        print(feedback)

    Example (with evaluate):

    In the following example, the guidelines specified in the `english` and `clarify` scorers
    will be uniformly applied to all the examples in the dataset. The evaluation result will
    contains two scores "english" and "clarify".

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import guideline_adherence

        english = guideline_adherence.with_config(
            name="english",
            global_guidelines=["The response must be in English"],
        )
        clarify = guideline_adherence.with_config(
            name="clarify",
            global_guidelines=["The response must be clear, coherent, and concise"],
        )

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
            },
            {
                "inputs": {"question": "What is the capital of Germany?"},
                "outputs": "The capital of Germany is Berlin.",
            },
        ]
        mlflow.genai.evaluate(data=data, scorers=[english, clarify])

    **2. Per-Example Guidelines**

    When you have a different set of guidelines for each example, you can specify the guidelines
    in the `guidelines` field of the `expectations` column of the input dataset. Alternatively,
    you can annotate a trace with "guidelines" expectation and use the trace as an input data.

    Example:

    In this example, the guidelines specified in the `guidelines` field of the `expectations`
    column will be applied to each example individually. The evaluation result will contain a
    single "guideline_adherence" score.

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import guideline_adherence

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
                "expectations": {
                    "guidelines": ["The response must be factual and concise"],
                },
            },
            {
                "inputs": {"question": "How to learn Python?"},
                "outputs": "You can read a book or take a course.",
                "expectations": {
                    "guidelines": ["The response must be helpful and encouraging"],
                },
            },
        ]
        mlflow.genai.evaluate(data=data, scorers=[guideline_adherence])
    """

    name: str = "guideline_adherence"
    global_guidelines: Optional[Union[str, list[str]]] = None
    required_columns: set[str] = {"inputs", "outputs"}

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        ### If no global guidelines are specified, the guidelines must exist in the input dataset
        if not self.global_guidelines and "expectations/guidelines" not in columns:
            raise MissingColumnsException(self.name, ["expectations/guidelines"])

    def __call__(
        self,
        *,
        outputs: Any,
        expectations: Optional[dict[str, Any]] = None,
    ) -> Assessment:
        """
        Evaluate adherence to specified guidelines.

        Args:
            outputs: The response from the model, e.g. "The capital of France is Paris."
            expectations: A dictionary of expectations for the response. This must contain either
                `guidelines` key, which is used to evaluate the response against the guidelines
                specified in the `guidelines` field of the `expectations` column of the dataset.
                E.g., {"guidelines": ["The response must be factual and concise"]}

        Returns:
            An :py:class:`mlflow.entities.assessment.Assessment~` object with a boolean value
            indicating the adherence to the specified guidelines.
        """
        guidelines = (expectations or {}).get("guidelines", self.global_guidelines)
        if not guidelines:
            raise MlflowException(
                "Guidelines must be specified either in the `expectations` parameter or "
                "by the `with_config` method of the scorer."
            )

        return judges.meets_guidelines(
            guidelines=guidelines,
            context={"response": outputs},
            name=self.name,
        )

    def with_config(
        self,
        *,
        name: str = "guideline_adherence",
        global_guidelines: Optional[list[str]] = None,
    ) -> "GuidelineAdherence":
        """
        Get a new scorer instance with the given name and global guidelines.

        Args:
            name: The name of the scorer. Default is "guideline_adherence".
            global_guidelines: A list of global guidelines to be used for evaluation.
                If not provided, the scorer will use the per-row guidelines in the input dataset.

        Returns:
            The updated GuidelineAdherence scorer instance.

        Example:

        .. code-block:: python

            from mlflow.genai.scorers import guideline_adherence

            is_english = guideline_adherence.with_config(
                name="is_english", global_guidelines=["The response must be in English"]
            )

            mlflow.genai.evaluate(data=data, scorers=[is_english])
        """
        return GuidelineAdherence(name=name, global_guidelines=global_guidelines)

@experimental
class RelevanceToQuery(BuiltInScorer):
"""
Relevance ensures that the agent's response directly addresses the user's input without
deviating into unrelated topics.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.relevance_to_query` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import relevance_to_query

        assessment = relevance_to_query(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import relevance_to_query

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[relevance_to_query])
    """

    name: str = "relevance_to_query"
    required_columns: set[str] = {"inputs", "outputs"}

    def __call__(self, *, inputs: dict[str, Any], outputs: Any) -> Feedback:
        """
        Evaluate relevance to the user's query.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
            outputs: The response from the model, e.g. "The capital of France is Paris."

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the relevance of the response to the query.
        """
        request = parse_inputs_to_str(inputs)
        ### NB: Reuse is_context_relevant judge to evaluate response
        feedback = judges.is_context_relevant(request=request, context=outputs, name=self.name)
        ### drop chunk id metadata
        feedback.metadata.pop("chunk_index", None)
        return feedback

    def with_config(self, *, name: str = "relevance_to_query") -> "RelevanceToQuery":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "relevance_to_query".
        """
        return RelevanceToQuery(name=name)

@experimental
class Safety(BuiltInScorer):
"""
Safety ensures that the agent's responses do not contain harmful, offensive, or toxic content.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.safety` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import safety

        assessment = safety(outputs="The capital of France is Paris.")
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import safety

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[safety])
    """

    name: str = "safety"
    required_columns: set[str] = {"inputs", "outputs"}

    def __call__(self, *, outputs: Any) -> Feedback:
        """
        Evaluate safety of the response.

        Args:
            outputs: The response from the model, e.g. "The capital of France is Paris."

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the safety of the response.
        """
        return judges.is_safe(content=parse_output_to_str(outputs), name=self.name)

    def with_config(self, *, name: str = "safety") -> "Safety":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The name of the scorer. Default is "safety".
        """
        return Safety(name=name)

@experimental
class Correctness(BuiltInScorer):
"""
Correctness ensures that the agent's responses are correct and accurate.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Use `mlflow.genai.scorers.correctness` to get an instance of this scorer with
    default setting. You can override the setting by the :py:meth:`with_config` method.

    Example (direct usage):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import correctness

        assessment = correctness(
            inputs={
                "question": "What is the difference between reduceByKey and groupByKey in Spark?"
            },
            outputs=(
                "reduceByKey aggregates data before shuffling, whereas groupByKey "
                "shuffles all data, making reduceByKey more efficient."
            ),
            expectations=[
                {"expected_response": "reduceByKey aggregates data before shuffling"},
                {"expected_response": "groupByKey shuffles all data"},
            ],
        )
        print(assessment)

    Example (with evaluate):

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import correctness

        data = [
            {
                "inputs": {
                    "question": (
                        "What is the difference between reduceByKey and groupByKey in Spark?"
                    )
                },
                "outputs": (
                    "reduceByKey aggregates data before shuffling, whereas groupByKey "
                    "shuffles all data, making reduceByKey more efficient."
                ),
                "expectations": [
                    {"expected_response": "reduceByKey aggregates data before shuffling"},
                    {"expected_response": "groupByKey shuffles all data"},
                ],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[correctness])
    """

    name: str = "correctness"
    required_columns: set[str] = {"inputs", "outputs"}

    def validate_columns(self, columns: set[str]) -> None:
        super().validate_columns(columns)
        if (
            "expectations/expected_response" not in columns
            and "expectations/expected_facts" not in columns
        ):
            raise MissingColumnsException(
                self.name, ["expectations/expected_response or expectations/expected_facts"]
            )

    def __call__(
        self, *, inputs: dict[str, Any], outputs: Any, expectations: dict[str, Any]
    ) -> Feedback:
        """
        Evaluate correctness of the response against expectations.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
            outputs: The response from the model, e.g. "The capital of France is Paris."
            expectations: A dictionary of expectations for the response. This must contain either
                `expected_response` or `expected_facts` key, which is used to evaluate the response
                against the expected response or facts respectively.
                E.g., {"expected_facts": ["Paris", "France", "Capital"]}

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the correctness of the response.
        """
        request = parse_inputs_to_str(inputs)
        response = parse_output_to_str(outputs)
        expected_facts = expectations.get("expected_facts")
        expected_response = expectations.get("expected_response")

        if expected_response is None and expected_facts is None:
            raise MlflowException(
                "Correctness scorer requires either `expected_response` or `expected_facts` "
                "in the `expectations` dictionary."
            )

        return judges.is_correct(
            request=request,
            response=response,
            expected_response=expected_response,
            expected_facts=expected_facts,
            name=self.name,
        )

    def with_config(self, *, name: str = "correctness") -> "Correctness":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The new name of the scorer. Default is "correctness".
        """
        return Correctness(name=name)

### === Shorthand for getting builtin scorer instances ===

retrieval_groundedness = RetrievalGroundedness()
retrieval_relevance = RetrievalRelevance()
retrieval_sufficiency = RetrievalSufficiency()
guideline_adherence = GuidelineAdherence()
relevance_to_query = RelevanceToQuery()
safety = Safety()
correctness = Correctness()

### === Shorthand for all builtin RAG scorers ===

@experimental
def get_rag_scorers() -> list[BuiltInScorer]:
"""
Returns a list of built-in scorers for evaluating RAG models. Contains scorers
chunk_relevance, context_sufficiency, groundedness, and relevance_to_query.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import get_rag_scorers

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=get_rag_scorers())
    """
    return [
        retrieval_relevance,
        retrieval_sufficiency,
        retrieval_groundedness,
        relevance_to_query,
    ]

@experimental
def get_all_scorers() -> list[BuiltInScorer]:
"""
Returns a list of all built-in scorers.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import get_all_scorers

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
                "expectations": [
                    {"expected_response": "Paris is the capital city of France."},
                ],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=get_all_scorers())
    """
    return get_rag_scorers() + [
        guideline_adherence,
        safety,
        correctness,
    ]

class MissingColumnsException(MlflowException):
def **init**(self, scorer: str, missing_columns: set[str]):
self.scorer = scorer
self.missing_columns = list(missing_columns)
super().**init**(
f"The following columns are required for the scorer {scorer}: {missing_columns}"
)
