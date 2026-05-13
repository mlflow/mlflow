from mlflow.genai.judges.base import Judge
from mlflow.genai.scorers.builtin_scorers import BuiltInScorer


class BuiltinJudge(BuiltInScorer, Judge):
    """
    Base class for built-in AI judge scorers that use LLMs for evaluation.
    """
