import logging
from collections import defaultdict
from typing import Any, Callable, Optional

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import BuiltInScorer, Scorer
from mlflow.genai.scorers.builtin_scorers import MissingColumnsException

try:
    # `pandas` is not required for `mlflow-skinny`.
    import pandas as pd
except ImportError:
    pass

_logger = logging.getLogger(__name__)


def validate_scorers(scorers: list[Any]) -> tuple[list[BuiltInScorer], list[Scorer]]:
    """
    Validate a list of specified scorers and split them into
    a tuple of builtin scorers and custom scorers.

    Args:
        scorers: A list of scorers to validate.

    Returns:
        A tuple of builtin scorers and custom scorers.
    """
    from databricks.rag_eval.evaluation.metrics import Metric

    if not isinstance(scorers, list) or len(scorers) == 0:
        raise MlflowException.invalid_parameter_value(
            "The `scorers` argument must be a list of scorers with at least one scorer. "
            "If you are unsure about which scorer to use, you can specify "
            "`scorers=mlflow.genai.scorers.all_scorers()` to jump start with all "
            "available built-in scorers."
        )

    builtin_scorers = []
    custom_scorers = []
    legacy_metrics = []

    for scorer in scorers:
        if isinstance(scorer, BuiltInScorer):
            builtin_scorers.append(scorer)
        elif isinstance(scorer, Scorer):
            custom_scorers.append(scorer)
        elif isinstance(scorer, Metric):
            legacy_metrics.append(scorer)
            custom_scorers.append(scorer)
        elif isinstance(scorer, Callable) and getattr(scorer, "__is_mlflow_builtin_scorer", False):
            raise MlflowException.invalid_parameter_value(
                f"A built-in scorer {scorer.__name__} is specified, but the constructor function "
                "is specified, not the scorer object itself. Please pass the returned scorer "
                "object from the constructor function instead.\n"
                "Example:\n"
                "  - Correct:   `mlflow.genai.evaluate(scorers=[correctness()])`\n"
                "  - Incorrect: `mlflow.genai.evaluate(scorers=[correctness])`"
            )
        else:
            raise MlflowException.invalid_parameter_value(
                f"Scorer {scorer} is not a valid scorer. Please use the @scorer decorator "
                "to convert a function into a scorer or inherit from the Scorer class"
            )

    if legacy_metrics:
        legacy_metric_names = [metric.name for metric in legacy_metrics]
        _logger.warning(
            f"Scorers {legacy_metric_names} are legacy metrics and will soon be deprecated "
            "in future releases. Please use the builtin scorers defined in `mlflow.genai.scorers` "
            "or custom scorers defined with the @scorer decorator instead."
        )

    return builtin_scorers, custom_scorers


def valid_data_for_builtin_scorers(
    data: "pd.DataFrame",
    builtin_scorers: list[BuiltInScorer],
    predict_fn: Optional[Callable] = None,
) -> None:
    """
    Validate that the required columns are present in the data for running the builtin scorers.

    Args:
        data: The data to validate. This must be a pandas DataFrame converted to
            the legacy evaluation set schema via `_convert_to_legacy_eval_set`.
        builtin_scorers: The list of builtin scorers to validate the data for.
        predict_fn: The predict function to validate the data for.
    """
    input_columns = set(data.columns.tolist())

    # Revert the replacement of "inputs"->"request" and "outputs"->"response"
    # in the upstream processing.
    if "request" in input_columns:
        input_columns.remove("request")
        input_columns.add("inputs")
    if "response" in input_columns:
        input_columns.remove("response")
        input_columns.add("outputs")

    if predict_fn is not None:
        # If the predict function is provided, the data doesn't need to
        # contain the "outputs" column.
        input_columns.add("outputs")

    if "trace" in input_columns:
        # Inputs and outputs are inferred from the trace.
        input_columns |= {"inputs", "outputs"}

    if predict_fn is not None or "trace" in input_columns:
        # NB: The retrieved_context is only inferred when a trace contains a retriever span,
        #     however, it is not impractical to check all traces and see if any of them
        #     contains a retriever span (it is valid case that some trace misses a retriever
        #     span). Therefore, we don't rigorously check the retrieved_context presence for
        #     traces and let scorers handle the missing retrieved_context gracefully.
        input_columns |= {"retrieved_context"}

    # Explode keys in the "expectations" column for easier processing.
    if "expectations" in input_columns:
        for value in data["expectations"].values:
            if pd.isna(value):
                continue
            if not isinstance(value, dict):
                raise MlflowException.invalid_parameter_value(
                    "The 'expectations' column must be a dictionary of each expectation name "
                    "to its value. For example, `{'expected_response': 'answer to the question'}`."
                )
            for k in value:
                input_columns.add(f"expectations/{k}")

    # Missing column -> list of scorers that require the column.
    missing_col_to_scorers = defaultdict(list)
    for scorer in builtin_scorers:
        try:
            scorer.validate_columns(input_columns)
        except MissingColumnsException as e:
            for col in e.missing_columns:
                missing_col_to_scorers[col].append(scorer.name)

    if missing_col_to_scorers:
        msg = (
            "The input data is missing following columns that are required by the specified "
            "scorers. The results will be null for those scorers."
        )
        for col, scorers in missing_col_to_scorers.items():
            if col.startswith("expectations/"):
                col = col.replace("expectations/", "")
                msg += (
                    f"\n - `{col}` field in `expectations` column "
                    f"is required by [{', '.join(scorers)}]."
                )
            else:
                msg += f"\n - `{col}` column is required by [{', '.join(scorers)}]."
        _logger.info(msg)
