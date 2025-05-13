from copy import deepcopy

from mlflow.genai.scorers import BuiltInScorer

GENAI_CONFIG_NAME = "databricks-agent"


class _BaseBuiltInScorer(BuiltInScorer):
    """
    Base class for built-in scorers that share a common implementation. All built-in scorers should
    inherit from this class.
    """

    def update_evaluation_config(self, evaluation_config) -> dict:
        config = deepcopy(evaluation_config)
        metrics = config.setdefault(GENAI_CONFIG_NAME, {}).setdefault("metrics", [])
        if self.name not in metrics:
            metrics.append(self.name)
        return config


# === Builtin Scorers ===
class _ChunkRelevance(_BaseBuiltInScorer):
    name: str = "chunk_relevance"


def chunk_relevance():
    """
    Chunk relevance measures whether each chunk is relevant to the input request.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate`for running full evaluation on a dataset.

    Example (direct usage):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import chunk_relevance

        assessment = chunk_relevance()(
            inputs={"question": "What is the capital of France?"},
            retrieved_context=[
                {"content": "Paris is the capital city of France."},
                {"content": "The chicken crossed the road."},
            ],
        )
        print(assessment)

    Example (with evaluate):
    .. code-block:: python

        import mlflow

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "retrieved_context": [
                    {"content": "Paris is the capital city of France."},
                    {"content": "The chicken crossed the road."},
                ],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[chunk_relevance()])
        print(result)
    """
    return _ChunkRelevance()


class _ContextSufficiency(_BaseBuiltInScorer):
    name: str = "context_sufficiency"


def context_sufficiency():
    """
    Context sufficiency evaluates whether the retrieved documents provide all necessary
    information to generate the expected response.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Example (direct usage):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import context_sufficiency

        assessment = context_sufficiency()(
            inputs={"question": "What is the capital of France?"},
            retrieved_context=[{"content": "Paris is the capital city of France."}],
        )
        print(assessment)

    Example (with evaluate):
    .. code-block:: python

        import mlflow

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "retrieved_context": [{"content": "Paris is the capital city of France."}],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[context_sufficiency()])
        print(result)
    """
    return _ContextSufficiency()


class _Groundedness(_BaseBuiltInScorer):
    name: str = "groundedness"


def groundedness():
    """
    Groundedness assesses whether the agent's response is aligned with the information provided
    in the retrieved context.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Example (direct usage):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import groundedness

        assessment = groundedness()(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
            retrieved_context=[{"content": "Paris is the capital city of France."}],
        )
        print(assessment)

    Example (with evaluate):
    .. code-block:: python

        import mlflow

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
                "retrieved_context": [{"content": "Paris is the capital city of France."}],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[groundedness()])
        print(result)
    """
    return _Groundedness()


class _GuidelineAdherence(_BaseBuiltInScorer):
    name: str = "guideline_adherence"


def guideline_adherence():
    """
    Guideline adherence evaluates whether the agent's response follows specific constraints
    or instructions provided in the guidelines.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    This judge should be used when each example has a different set of guidelines. The guidelines
    must be specified in the `guidelines` column of the input dataset.

    You can also specify contextual information for guidelines using the `guidelines_context`
    column in your dataset (requires `databricks-agents>=0.20.0`).

    Example (direct usage):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import guideline_adherence

        assessment = guideline_adherence()(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
            guidelines={
                "english": ["The response must be in English"],
            },
            guidelines_context={
                "tool_call_result": "{'country': 'France', 'capital': 'Paris'}",
            },
        )
        print(assessment)

    Example (with evaluate):
    .. code-block:: python

        import mlflow

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
                "guidelines": {
                    "english": ["The response must be in English"],
                    "clarity": ["The response must be clear, coherent, and concise"],
                    "grounded": ["The response must be grounded in the tool call result"],
                },
                "guidelines_context": {
                    "tool_call_result": "{'country': 'France', 'capital': 'Paris'}",
                },
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[guideline_adherence()])
        print(result)
    """
    return _GuidelineAdherence()


class _GlobalGuidelineAdherence(_GuidelineAdherence):
    guidelines: list[str]

    def update_evaluation_config(self, evaluation_config) -> dict:
        config = deepcopy(evaluation_config)
        metrics = config.setdefault(GENAI_CONFIG_NAME, {}).setdefault("metrics", [])
        if "guideline_adherence" not in metrics:
            metrics.append("guideline_adherence")

        # NB: The agent eval harness will take multiple global guidelines in a dictionary format
        #   where the key is the name of the global guideline judge. Therefore, when multiple
        #   judges are specified, we merge them into a single dictionary.
        #   https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/llm-judge-reference#examples
        global_guidelines = config[GENAI_CONFIG_NAME].get("global_guidelines", {})
        global_guidelines[self.name] = self.guidelines
        config[GENAI_CONFIG_NAME]["global_guidelines"] = global_guidelines
        return config


def global_guideline_adherence(
    guidelines: list[str],
    name: str = "guideline_adherence",
):
    """
    Guideline adherence evaluates whether the agent's response follows specific global
    constraints or instructions provided in the guidelines.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        guidelines: A list of global guidelines to evaluate the agent's response against.
        name: The name of the judge. Defaults to "guideline_adherence".

    Example (direct usage):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import global_guideline_adherence

        # Create a global judge
        english = global_guideline_adherence(
            guidelines=["The response must be in English"],
            name="english_guidelines",
        )
        assessment = english()(
            inputs={"question": "What is the capital of France?"},
            outputs="The capital of France is Paris.",
        )
        print(assessment)

    Example (with evaluate):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import global_guideline_adherence

        guideline = global_guideline_adherence(["Be polite", "Be kind"])
        english = global_guideline_adherence(
            guidelines=["The response must be in English"],
            name="english_guidelines",
        )
        clarify = global_guideline_adherence(
            guidelines=["The response must be clear, coherent, and concise"],
            name="clarify_guidelines",
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
        result = mlflow.genai.evaluate(
            data=data,
            scorers=[guideline, english, clarify],
        )
        print(result)
    """
    return _GlobalGuidelineAdherence(guidelines=guidelines, name=name)


class _RelevanceToQuery(_BaseBuiltInScorer):
    name: str = "relevance_to_query"


def relevance_to_query():
    """
    Relevance ensures that the agent's response directly addresses the user's input without
    deviating into unrelated topics.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Example (direct usage):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import relevance_to_query

        assessment = relevance_to_query()(
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
        result = mlflow.genai.evaluate(data=data, scorers=[relevance_to_query()])
        print(result)
    """
    return _RelevanceToQuery()


class _Safety(_BaseBuiltInScorer):
    name: str = "safety"


def safety():
    """
    Safety ensures that the agent's responses do not contain harmful, offensive, or toxic content.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Example (direct usage):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import safety

        assessment = safety()(
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
        result = mlflow.genai.evaluate(data=data, scorers=[safety()])
        print(result)
    """
    return _Safety()


class _Correctness(_BaseBuiltInScorer):
    name: str = "correctness"


def correctness():
    """
    Correctness ensures that the agent's responses are correct and accurate.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Example (direct usage):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import correctness

        assessment = correctness()(
            inputs={
                "question": "What is the difference between reduceByKey and groupByKey in Spark?"
            },
            outputs=(
                "reduceByKey aggregates data before shuffling, whereas groupByKey "
                "shuffles all data, making reduceByKey more efficient."
            ),
            expectations=[
                "reduceByKey aggregates data before shuffling",
                "groupByKey shuffles all data",
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
                    "reduceByKey aggregates data before shuffling",
                    "groupByKey shuffles all data",
                ],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=[correctness()])
        print(result)
    """
    return _Correctness()


# === Shorthand for all builtin RAG scorers ===
def rag_scorers() -> list[BuiltInScorer]:
    """
    Returns a list of built-in scorers for evaluating RAG models. Contains scorers
    chunk_relevance, context_sufficiency, groundedness, and relevance_to_query.

    Example (with evaluate):
    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import rag_scorers

        data = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": "The capital of France is Paris.",
                "retrieved_context": [
                    {"content": "Paris is the capital city of France."},
                ],
            }
        ]
        result = mlflow.genai.evaluate(data=data, scorers=rag_scorers())
        print(result)
    """
    return [
        chunk_relevance(),
        context_sufficiency(),
        groundedness(),
        relevance_to_query(),
    ]
