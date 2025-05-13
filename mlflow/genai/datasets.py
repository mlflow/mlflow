from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For type checking and IDE completion, import EvaluationDataset
    from databricks.rag_eval.datasets.entities import Dataset as EvaluationDataset

__all__ = ["EvaluationDataset"]


def __getattr__(name: str):
    if name == "EvaluationDataset":
        try:
            from databricks.rag_eval.datasets.entities import Dataset as EvaluationDataset
        except ImportError:
            raise ImportError(
                "The `databricks-agents` package is required to use `EvaluationDataset`. "
                "Please install it with `pip install databricks-agents`."
            )
        globals()[name] = EvaluationDataset
        return EvaluationDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
