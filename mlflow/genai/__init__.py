from mlflow.genai import (
    judges,
    scorers,
)
from mlflow.genai.evaluation import evaluate, to_predict_fn
from mlflow.genai.scorers import Scorer, scorer

try:
    from mlflow.genai.labeling import (
        Agent,  # noqa: F401
        LabelingSession,  # noqa: F401
        ReviewApp,  # noqa: F401
        create_labeling_session,  # noqa: F401
        delete_labeling_session,  # noqa: F401
        get_labeling_session,  # noqa: F401
        get_labeling_sessions,  # noqa: F401
        get_review_app,  # noqa: F401
    )
except ImportError:
    # Silently fail if the databricks-agents package is not installed
    pass

from mlflow.genai.datasets import (
    create_dataset,
    delete_dataset,
    get_dataset,
)
from mlflow.genai.optimize import optimize_prompt

__all__ = [
    "evaluate",
    "to_predict_fn",
    "Scorer",
    "scorer",
    "judges",
    "scorers",
    "create_dataset",
    "delete_dataset",
    "get_dataset",
    "optimize_prompt",
    # Labeling exports (only included when databricks-agents is installed)
    *(
        [
            "Agent",
            "LabelingSession",
            "ReviewApp",
            "get_review_app",
            "create_labeling_session",
            "get_labeling_sessions",
            "get_labeling_session",
            "delete_labeling_session",
        ]
        if "Agent" in locals()
        else []
    ),
]
