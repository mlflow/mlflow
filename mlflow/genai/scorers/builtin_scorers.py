from copy import deepcopy

from mlflow.genai.scorers import BuiltInScorer


class _BaseBuiltInScorer(BuiltInScorer):
    """
    Base class for built-in scorers that share a common implementation. All built-in scorers should
    inherit from this class.
    """

    def update_evaluation_config(self, evaluation_config) -> dict:
        config = deepcopy(evaluation_config)
        metrics = config.setdefault("databricks-agents", {}).setdefault("metrics", [])
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


class _DocumentRecall(_BaseBuiltInScorer):
    name: str = "document_recall"


def document_recall():
    """
    Document recall measures the proportion of ground truth relevant documents that were
    retrieved compared to the total number of relevant documents in ground truth.
    """
    return _DocumentRecall()


class _GlobalGuidelineAdherence(_BaseBuiltInScorer):
    name: str = "global_guideline_adherence"
    global_guideline: str

    def update_evaluation_config(self, evaluation_config) -> dict:
        config = super().update_evaluation_config(evaluation_config)
        config["databricks-agents"]["global_guideline"] = self.global_guideline
        return config


def global_guideline_adherence(guideline: str):
    """
    Guideline adherence evaluates whether the agent’s response follows specific constraints or
    instructions provided in the guidelines.
    """
    return _GlobalGuidelineAdherence(global_guideline=guideline)


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
    Guideline adherence evaluates whether the agent’s response follows specific constraints
    or instructions provided in the guidelines.
    """
    return _GuidelineAdherence()


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


# === Shorthand for all builtin RAG scorers ===
def rag_scorers() -> list[BuiltInScorer]:
    """
    Returns a list of built-in scorers for evaluating RAG models. Contains scorers
    chunk_relevance, context_sufficiency, document_recall, global_guideline_adherence,
    groundedness, and relevance_to_query.
    """
    return [
        chunk_relevance(),
        context_sufficiency(),
        document_recall(),
        groundedness(),
        relevance_to_query(),
    ]
