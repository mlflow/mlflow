from abc import abstractmethod
from typing import Any, Optional, Union

from mlflow.entities import Assessment
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.utils.trace_utils import extract_retrieval_context_from_trace, parse_inputs_to_str
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.utils.annotations import experimental

GENAI_CONFIG_NAME = "databricks-agent"


class BuiltInScorer(Scorer):
    """
    Base class for built-in scorers that share a common implementation. All built-in scorers should
    inherit from this class.
    """

    required_columns: set[str] = set()

    # Avoid direct mutation of built-in scorer fields like name. Overriding the default setting must
    # be done through the `with_config` method. This is because the short-hand syntax like
    # `mlflow.genai.scorers.chunk_relevance` returns a single instance rather than a new instance.
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


# === Builtin Scorers ===
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
        feedback = retrieval_relevance(trace)
        print(feedback)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[retrieval_relevance])
    """

    name: str = "retrieval_relevance"
    required_columns: set[str] = {"inputs", "trace"}

    def __call__(self, *, trace: Trace) -> list[Feedback]:
        """
        Evaluate chunk relevance for each context chunk.

        Args:
            trace: The trace of the model's execution. Must contains at least one span with
                type `RETRIEVER`. MLflow will extract the retrieved context from that span.
                If multiple spans are found, MLflow will use the **last** one.

        Returns:
            A list of assessments evaluating the relevance of each context chunk.
        """
        from databricks.agents.evals import judges

        request = parse_inputs_to_str(trace.data.spans[0].inputs)
        retrieved_context = extract_retrieval_context_from_trace(trace)
        return judges.chunk_relevance(
            request=request, retrieved_context=retrieved_context, assessment_name=self.name
        )

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
        feedback = retrieval_sufficiency(trace)
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

    def __call__(self, *, trace: Trace) -> Feedback:
        """
        Evaluate context sufficiency based on retrieved documents.

        Args:
            trace: The trace of the model's execution. Must contains at least one span with
                type `RETRIEVER`. MLflow will extract the retrieved context from that span.
                If multiple spans are found, MLflow will use the **last** one.
                Optionally, you can annotate the trace with `expected_facts` or `expected_response`
                label(s) by calling :py:func:`mlflow.log_expectation` API. The annotated label(s)
                will be considered when computing the sufficiency of retrieved context.
        """
        from databricks.agents.evals import judges

        request = parse_inputs_to_str(trace.data.spans[0].inputs)
        retrieved_context = extract_retrieval_context_from_trace(trace)

        expected_facts = None
        expected_response = None
        for assessment in trace.info.assessments:
            if assessment.name == "expected_facts":
                expected_facts = assessment.value
            if assessment.name == "expected_response":
                expected_response = assessment.value

        return judges.context_sufficiency(
            request=request,
            retrieved_context=retrieved_context,
            expected_response=expected_response,
            expected_facts=expected_facts,
            assessment_name=self.name,
        )

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
        feedback = retrieval_groundedness(trace)
        print(feedback)

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[retrieval_groundedness])
    """

    name: str = "retrieval_groundedness"
    required_columns: set[str] = {"inputs", "trace"}

    def __call__(self, *, trace: Trace) -> Feedback:
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
        from databricks.agents.evals import judges

        request = parse_inputs_to_str(trace.data.spans[0].inputs)
        response = trace.data.spans[0].outputs
        retrieved_context = extract_retrieval_context_from_trace(trace)
        return judges.groundedness(
            request=request,
            response=response,
            retrieved_context=retrieved_context,
            assessment_name=self.name,
        )

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

        # Create a global judge
        english = guideline_adherence.with_config(
            name="english_guidelines",
            global_guidelines=["The response must be in English"],
        )
        feedback = english(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )
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
        # If no global guidelines are specified, the guidelines must exist in the input dataset
        if not self.global_guidelines and "expectations/guidelines" not in columns:
            raise MissingColumnsException(self.name, ["expectations/guidelines"])

    def __call__(
        self,
        *,
        inputs: dict[str, Any],
        outputs: Any,
        expectations: Optional[dict[str, Any]] = None,
    ) -> Assessment:
        """
        Evaluate adherence to specified guidelines.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
            outputs: The response from the model, e.g. "The capital of France is Paris."
            expectations: A dictionary of expectations for the response. This must contain either
                `guidelines` key, which is used to evaluate the response against the guidelines
                specified in the `guidelines` field of the `expectations` column of the dataset.
                E.g., {"guidelines": ["The response must be factual and concise"]}

        Returns:
            An :py:class:`mlflow.entities.assessment.Assessment~` object with a boolean value
            indicating the adherence to the specified guidelines.
        """
        from databricks.agents.evals import judges

        request = parse_inputs_to_str(inputs)
        response = outputs
        guidelines = (expectations or {}).get("guidelines", self.global_guidelines)
        if not guidelines:
            raise MlflowException(
                "Guidelines must be specified either in the `expectations` parameter or "
                "by the :py:meth:`with_config` method of the scorer."
            )

        return judges.guideline_adherence(
            request=request, response=response, guidelines=guidelines, assessment_name=self.name
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
        from databricks.agents.evals import judges

        request = parse_inputs_to_str(inputs)
        response = outputs
        return judges.relevance_to_query(
            request=request, response=response, assessment_name=self.name
        )

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

        assessment = safety(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )
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

    def __call__(self, *, inputs: dict[str, Any], outputs: Any) -> Feedback:
        """
        Evaluate safety of the response.

        Args:
            inputs: A dictionary of input data, e.g. {"question": "What is the capital of France?"}.
            outputs: The response from the model, e.g. "The capital of France is Paris."

        Returns:
            An :py:class:`mlflow.entities.assessment.Feedback~` object with a boolean value
            indicating the safety of the response.
        """
        from databricks.agents.evals import judges

        request = parse_inputs_to_str(inputs)
        response = outputs
        return judges.safety(request=request, response=response, assessment_name=self.name)

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
        from databricks.agents.evals import judges

        request = parse_inputs_to_str(inputs)
        response = outputs
        expected_facts = expectations.get("expected_facts")
        expected_response = expectations.get("expected_response")

        if expected_response is None and expected_facts is None:
            raise MlflowException(
                "Correctness scorer requires either `expected_response` or `expected_facts` "
                "in the `expectations` dictionary."
            )

        return judges.correctness(
            request=request,
            response=response,
            expected_response=expected_response,
            expected_facts=expected_facts,
            assessment_name=self.name,
        )

    def with_config(self, *, name: str = "correctness") -> "Correctness":
        """
        Get a new scorer instance with a specified name.

        Args:
            name: The new name of the scorer. Default is "correctness".
        """
        return Correctness(name=name)


# === Shorthand for getting builtin scorer instances ===
retrieval_groundedness = RetrievalGroundedness()
retrieval_relevance = RetrievalRelevance()
retrieval_sufficiency = RetrievalSufficiency()
guideline_adherence = GuidelineAdherence()
relevance_to_query = RelevanceToQuery()
safety = Safety()
correctness = Correctness()


# === Shorthand for all builtin RAG scorers ===
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
    def __init__(self, scorer: str, missing_columns: set[str]):
        self.scorer = scorer
        self.missing_columns = list(missing_columns)
        super().__init__(
            f"The following columns are required for the scorer {scorer}: {missing_columns}"
        )
