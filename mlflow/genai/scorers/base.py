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
        inputs=None,
        outputs=None,
        expectations=None,
        trace=None,
        **kwargs,
    ) -> Union[int, float, bool, str, Assessment, list[Assessment]]:
        # TODO: make sure scorer's signature is simply equal to whatever keys are
        # in the eval dataset once we migrate from the agent eval harness
        """

        Args:
            inputs (required): A single input to the target model/app.
            outputs (optional): A single output from the target model/app.
            expectations (optional): Ground truth, or a dictionary of ground
                truths for individual output fields.
            trace (optional): Json representation of a trace object corresponding to
                the prediction for the row. Required when any of scorers requires a
                trace in order to compute assessments/scores.
            retrieved_context (optional): Retrieved context, can be from your input eval dataset
                or from trace
            custom_expected (optional): Custom expected results from input eval dataset
            custom_inputs (optional): Custom inputs from your input eval dataset
            custom_outputs (optional): Custom outputs from the agent's response
            tool_calls (optional): Tool calls from the agent's response.
            **kwargs (optional): Additional keyword arguments passed to the scorer.
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

    class CustomScorer(Scorer):
        def __call__(self, *, inputs=None, outputs=None, expectations=None, trace=None, **kwargs):
            # Normalize singular/plural keys
            merged = {
                "inputs": inputs,
                "outputs": outputs,
                "expectations": expectations,
                "trace": trace,
                **kwargs,
            }
            # Filter to only the parameters the function actually expects
            sig = inspect.signature(func)
            filtered = {k: v for k, v in merged.items() if k in sig.parameters}
            result = func(**filtered)
            if not (
                isinstance(result, (int, float, bool, str, Assessment))
                or (
                    isinstance(result, list)
                    and all(isinstance(item, Assessment) for item in result)
                )
            ):
                raise ValueError(
                    f"{func.__name__} must return one of int, float, bool, str, "
                    f"Assessment, or list[Assessment]. Got {type(result).__name__}"
                )
            return result

    return CustomScorer(
        name=name or func.__name__,
        aggregations=aggregations,
    )
