from mlflow.genai.judges.base import Judge
from mlflow.genai.scorers.builtin_scorers import BuiltInScorer
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class BuiltinJudge(BuiltInScorer, Judge):
    """
    Base class for built-in AI judge scorers that use LLMs for evaluation.
    """
