import functools
import inspect
from typing import Callable, Literal, Optional, Union

from pydantic import BaseModel

from mlflow.evaluation import Assessment


class Scorer(BaseModel):
    name: str
    aggregations: Optional[list] = None

    def __call__(
        self,
        *,
        inputs,
        outputs=None,
        expectations=None,
        trace=None,
    ) -> Union[float, bool, str, Assessment, list[Assessment]]:
        """

        Args:
            inputs (required): A single input to the target model/app.
            outputs (optional): A single output from the target model/app.
            expectations (optional): Ground truth, or a dictionary of ground
                truths for individual output fields.
            trace (optional): A single trace object corresponding to the prediction
                for the row. Only required when any of scorers requires a trace in
                order to compute assessments/metrics.
        """
        raise NotImplementedError("Implementation of __call__ is required for Scorer class")


class BuiltInScorer(Scorer):
    def update_evaluation_config(evaluation_config) -> dict:
        """
        The builtin scorer will take in an evaluation_config and return an updated version
        of it as necessary to comply with the expected format for mlflow.evaluate().
        More details about built-in judges can be found at
        https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/llm-judge-reference
        """
        raise NotImplementedError("Please use an instance of BuiltInScorer")


def scorer(
    func=None,
    *,
    name: Optional[str] = None,
    aggregations: Optional[
        list[Union[Literal["min", "max", "mean", "median", "variance", "p90", "p99"], Callable]]
    ] = None,
):
    """
    Syntactic sugar that ensures the decorated function has the correct parameters (with inputs
    as required, outputs, expectations, trace as optional) and returns float | bool |
    str | Assessment.
    """

    if func is None:
        return functools.partial(scorer, name=name, aggregations=aggregations)

    sig = inspect.signature(func)
    params = sig.parameters

    for optional_arg in ("inputs", "outputs", "expectations", "trace"):
        if optional_arg not in params:
            raise TypeError(
                f"'{optional_arg}' must be present as an optional argument in {func.__name__}."
            )

    class CustomScorer(Scorer):
        def __call__(self, *, inputs, outputs=None, expectations=None, trace=None):
            result = func(inputs=inputs, outputs=outputs, expectations=expectations, trace=trace)
            if not isinstance(result, (float, bool, str, Assessment)):
                raise ValueError(
                    (
                        f"{func.__name__} must return one of float, bool, str, ",
                        f"or Assessment. Got {type(result).__name__}",
                    )
                )
            return result

    return CustomScorer(
        name=name or func.__name__,
        aggregations=aggregations,
    )
