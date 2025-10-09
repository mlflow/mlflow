import functools
from typing import Any, Callable

from pydantic import BaseModel, create_model

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer


def infer_type_from_value(value: Any, model_name: str = "Output") -> type:
    """
    Infer the type from the value.
    Only supports primitive types, lists, and dict and Pydantic models.
    """
    if value is None:
        return type(None)
    elif isinstance(value, (bool, int, float, str)):
        return type(value)
    elif isinstance(value, list):
        if not value:
            return list[Any]
        element_types = set()
        for item in value:
            element_types.add(infer_type_from_value(item))
        return list[functools.reduce(lambda x, y: x | y, element_types)]
    elif isinstance(value, dict):
        fields = {k: (infer_type_from_value(v, model_name=k), ...) for k, v in value.items()}
        return create_model(model_name, **fields)
    elif isinstance(value, BaseModel):
        return type(value)
    return Any


def create_metric_from_scorers(
    scorers: list[Scorer],
    objective: Callable[[dict[str, Any]], float] | None = None,
) -> Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], float]:
    """
    Create a metric function from scorers and an optional objective function.

    Args:
        scorers: List of scorers to evaluate inputs, outputs, and expectations.
        objective: Optional function that aggregates scorer outputs into a single score.
                  Takes a dict mapping scorer names to scores and returns a float.
                  If None and all scorers return numerical values, sums them by default.

    Returns:
        A callable that takes (inputs, outputs, expectations) and returns a float score.

    Raises:
        MlflowException: If scorers return non-numerical values and no objective is provided.
    """

    def metric(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        expectations: dict[str, Any],
    ) -> float:
        scores = {}

        for scorer in scorers:
            scores[scorer.name] = scorer.run(
                inputs=inputs, outputs=outputs, expectations=expectations
            )

        if objective is not None:
            return objective(scores)
        elif all(isinstance(score, (int, float, bool)) for score in scores.values()):
            # Use total score by default if no objective is provided
            return sum(scores.values())
        else:
            non_numerical_scorers = [
                k for k, v in scores.items() if not isinstance(v, (int, float, bool))
            ]
            raise MlflowException(
                f"Scorer [{','.join(non_numerical_scorers)}] return a string, Assessment or "
                "a list of Assessment. Please provide `objective` function to aggregate "
                "non-numerical values into a single value for optimization."
            )

    return metric
