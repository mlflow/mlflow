import functools
import inspect
import types
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel

from mlflow.entities import Trace
from mlflow.evaluation import Assessment


class Scorer(BaseModel):
    name: str

    def __call__(
        self,
        *,
        inputs: dict[str, Any],
        outputs: Optional[dict[str, Any]] = None,
        expectations: Optional[dict[str, Any]] = None,
        trace: Optional[Trace] = None,
    ) -> Union[float, bool, str, Assessment]:
        """
        Implement your custom scoring logic here.
        """
        raise NotImplementedError("Please use an instance of BuiltInScorer")


# TODO: ML-52304: Inherit the following class for every builtin scorer made available
class BuiltInScorer(Scorer):
    def __call__(evaluation_config) -> dict:
        """
        The builtin scorer will take in an evaluation_config and update it as necessary
        to comply with the expected format for mlflow.evaluate().
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

    # Verify required param presence
    if "inputs" not in params:
        raise TypeError(
            f"{func.__name__} must accept 'inputs' as a required keyword-only argument."
        )

    # Check param details
    required_param = params["inputs"]
    # Inputs must be required (no default)
    if required_param.default is not inspect.Parameter.empty:
        raise TypeError(f"'inputs' must not have a default value in {func.__name__}.")

    for optional_arg in ("outputs", "expectations", "trace"):
        if optional_arg not in params:
            raise TypeError(
                f"'{optional_arg}' must be present as an optional argument in {func.__name__}."
            )

    def wrapped_call(self, *, inputs, outputs=None, expectations=None, trace=None):
        result = func(inputs=inputs, outputs=outputs, expectations=expectations, trace=trace)
        if not isinstance(result, (float, bool, str, Assessment)):
            raise ValueError(
                (
                    f"{func.__name__} must return one of float, bool, str, ",
                    "or Assessment. Got {type(result).__name__}",
                )
            )
        return result

    instance = Scorer(name=name or func.__name__)
    instance.__call__ = types.MethodType(wrapped_call, instance)

    return instance
