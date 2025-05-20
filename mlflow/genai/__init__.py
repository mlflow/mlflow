from mlflow.genai.evaluation import evaluate, to_predict_fn
from mlflow.genai.scorers import Scorer, scorer

try:
    from databricks.agents.review_app import (
        ReviewApp,
        get_review_app,
    )
except ImportError:
    raise ImportError(
        "The `databricks-agents` package is required to use `mlflow.genai.ReviewApp` and `mlflow.genai.get_review_app`. "
        "Please install it with `pip install databricks-agents`."
    )

__all__ = ["evaluate", "to_predict_fn", "Scorer", "scorer", "ReviewApp", "get_review_app"]
