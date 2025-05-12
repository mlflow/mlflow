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
    """
    return _ChunkRelevance()


class _ContextSufficiency(_BaseBuiltInScorer):
    name: str = "context_sufficiency"


def context_sufficiency():
    """
    Context sufficiency evaluates whether the retrieved documents provide all necessary
    information to generate the expected response.
    """
    return _ContextSufficiency()


class _Groundedness(_BaseBuiltInScorer):
    name: str = "groundedness"


def groundedness():
    """
    Groundedness assesses whether the agent’s response is aligned with the information provided
    in the retrieved context.
    """
    return _Groundedness()


class _GuidelineAdherence(_BaseBuiltInScorer):
    name: str = "guideline_adherence"


def guideline_adherence():
    """
    Guideline adherence evaluates whether the agent's response follows specific constraints
    or instructions provided in the guidelines.

    This judge should be used when each example has a different set of guidelines. The guidelines
    must be specified in the `guidelines` column of the input dataset.

    If you want to apply the same set of guidelines to all examples, use the
    :py:func:`global_guideline_adherence` scorer instead.

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import guideline_adherence

        eval_set = [
            {
                "inputs": "Translate the following text to English: 'Hello, world!'",
                "guidelines": ["The response must be in English"],
            },
            {
                "inputs": "Translate the following text to German: 'Hello, world!'",
                "guidelines": ["The response must be in German"],
            },
        ]

        # Run evaluation
        mlflow.genai.evaluate(
            data=data,
            scorers=[guideline_adherence()],
        )
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
    Guideline adherence evaluates whether the agent's response follows specific constraints or
    instructions provided in the guidelines.

    Args:
        guidelines: A list of global guidelines to evaluate the agent's response against.
        name: The name of the judge. Defaults to "guideline_adherence".

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import global_guideline_adherence

        # A single judge with multiple guidelines
        guideline = global_guideline_adherence(["Be polite", "Be kind"])

        # Create a judge with different names
        english = global_guideline_adherence(
            name="english_guidelines",
            guidelines=["The response must be in English"],
        )

        clarify = global_guideline_adherence(
            name="clarify_guidelines",
            guidelines=["The response must be clear, coherent, and concise"],
        )

        # Dataset
        eval_set = [
            {
                "inputs": "What is the capital of France?",
                "outputs": "The capital of France is Paris.",
            },
            {
                "inputs": "What is the capital of Germany?",
                "outputs": "The capital of Germany is Berlin.",
            },
        ]

        # Run evaluation
        mlflow.genai.evaluate(
            data=data,
            scorers=[guideline, english, clarify],
        )
    """
    return _GlobalGuidelineAdherence(guidelines=guidelines, name=name)


class _RelevanceToQuery(_BaseBuiltInScorer):
    name: str = "relevance_to_query"


def relevance_to_query():
    """
    Relevance ensures that the agent’s response directly addresses the user’s input without
    deviating into unrelated topics.
    """
    return _RelevanceToQuery()


class _Safety(_BaseBuiltInScorer):
    name: str = "safety"


def safety():
    """
    Safety ensures that the agent’s responses do not contain harmful, offensive, or toxic content.
    """
    return _Safety()


class _Correctness(_BaseBuiltInScorer):
    name: str = "correctness"


def correctness():
    """
    Correctness ensures that the agent’s responses are correct and accurate.
    """
    return _Correctness()


# === Shorthand for all builtin RAG scorers ===
def rag_scorers() -> list[BuiltInScorer]:
    """
    Returns a list of built-in scorers for evaluating RAG models. Contains scorers
    chunk_relevance, context_sufficiency, global_guideline_adherence,
    groundedness, and relevance_to_query.
    """
    return [
        chunk_relevance(),
        context_sufficiency(),
        groundedness(),
        relevance_to_query(),
    ]
