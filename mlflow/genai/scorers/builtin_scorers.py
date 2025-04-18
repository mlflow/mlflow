from copy import deepcopy
from typing import List

from mlflow.genai.scorers import BuiltInScorer


class _BaseBuiltInScorer(BuiltInScorer):
    """
    Base class for built-in scorers that share a common implementation. All built-in scorers should
    inherit from this class.
    """

    def update_evaluation_config(self, evaluation_config) -> dict:
        config = deepcopy(evaluation_config)
        config.setdefault("databricks-agents", {}).setdefault("metrics", []).append(self.name)
        return config


# === Builtin Scorers ===
class _ChunkRelevance(_BaseBuiltInScorer):
    name: str = "chunk_relevance"


chunk_relevance = _ChunkRelevance()


class _ContextSufficiency(_BaseBuiltInScorer):
    name: str = "context_sufficiency"


context_sufficiency = _ContextSufficiency()


class _DocumentRecall(_BaseBuiltInScorer):
    name: str = "document_recall"


document_recall = _DocumentRecall()


class _GlobalGuidelineAdherence(_BaseBuiltInScorer):
    name: str = "global_guideline_adherence"
    global_guideline: str

    def update_evaluation_config(self, evaluation_config) -> dict:
        config = super().update_evaluation_config(evaluation_config)
        config["databricks-agents"]["global_guideline"] = self.global_guideline
        return config


def global_guideline_adherence(guideline: str):
    return _GlobalGuidelineAdherence(global_guideline=guideline)


class _Groundedness(_BaseBuiltInScorer):
    name: str = "groundedness"


groundedness = _Groundedness()


class _GuidelineAdherence(_BaseBuiltInScorer):
    name: str = "guideline_adherence"


guideline_adherence = _GuidelineAdherence()


class _RelevanceToQuery(_BaseBuiltInScorer):
    name: str = "relevance_to_query"


relevance_to_query = _RelevanceToQuery()


class _Safety(_BaseBuiltInScorer):
    name: str = "safety"


safety = _Safety()


# === Shorthand for all builtin scorers ===
def rag_evaluators(global_guideline: str) -> List[BuiltInScorer]:
    return [
        chunk_relevance,
        context_sufficiency,
        document_recall,
        global_guideline_adherence(global_guideline),
        groundedness,
        guideline_adherence,
        relevance_to_query,
        safety,
    ]
