from mlflow.genai.evaluation import evaluate, to_predict_fn
from mlflow.genai.scorers import Scorer, scorer

try:
    from mlflow.genai.labeling import (
        Agent,
        LabelingSession,
        ReviewApp,
        get_review_app,
        create_labeling_session,
        get_labeling_sessions,
        get_labeling_session,
        delete_labeling_session,
    )
    from mlflow.genai.datasets import EvaluationDataset
except ImportError:
    # Silently fail if the databricks-agents package is not installed
    pass

# Stick this at the end to avoid unnecessary warnings (thrown by EvaluationDataset)
from mlflow.genai.datasets import (
    create_dataset,
    delete_dataset,
    get_dataset,
)


__all__ = [
    "evaluate",
    "to_predict_fn",
    "Scorer",
    "scorer",
    # Labeling
    "Agent",
    "LabelingSession",
    "ReviewApp",
    "get_review_app",
    "create_labeling_session",
    "get_labeling_sessions",
    "get_labeling_session",
    "delete_labeling_session",
    # Datasets
    "EvaluationDataset",
    "create_dataset",
    "delete_dataset",
    "get_dataset",
]
