import warnings

try:
    with warnings.catch_warnings():
        # Ignore warnings from the mlflow.genai.datasets module
        warnings.filterwarnings("ignore", message="The `databricks-agents` package is required")
        from mlflow.genai.datasets import EvaluationDataset  # noqa: F401

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

__all__ = [
    # TODO (B-Step62): Add these back once we release the new evaluate API
    # "to_predict_fn",
    # "Scorer",
    # "scorer",
    "create_dataset",
    "delete_dataset",
    "get_dataset",
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
    # Dataset exports (only included when databricks-agents is installed)
    *(
        [
            "EvaluationDataset",
        ]
        if "EvaluationDataset" in locals()
        else []
    ),
]
