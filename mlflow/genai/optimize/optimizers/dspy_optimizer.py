import importlib.metadata
import importlib.util
import inspect
from typing import TYPE_CHECKING, Callable, Optional

from packaging.version import Version

from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers import _BaseOptimizer
from mlflow.genai.optimize.types import OBJECTIVE_FN, OptimizerConfig
from mlflow.genai.optimize.util import infer_type_from_value
from mlflow.genai.scorers import Scorer

if TYPE_CHECKING:
    import dspy
    import pandas as pd


class _DSPyOptimizer(_BaseOptimizer):
    def __init__(self, optimizer_config: OptimizerConfig):
        super().__init__(optimizer_config)

        if not importlib.util.find_spec("dspy"):
            raise ImportError("dspy is not installed. Please install it with `pip install dspy`.")

        dspy_version = importlib.metadata.version("dspy")
        if Version(dspy_version) < Version("2.6.0"):
            raise MlflowException(
                f"Current dspy version {dspy_version} is unsupported. "
                "Please upgrade to version >= 2.6.0"
            )

    def _get_input_fields(self, train_data: "pd.DataFrame") -> dict[str, type]:
        if "inputs" in train_data.columns:
            sample_input = train_data["inputs"].values[0]
            return {k: infer_type_from_value(v) for k, v in sample_input.items()}
        return {}

    def _get_output_fields(self, train_data: "pd.DataFrame") -> dict[str, type]:
        if "expectations" in train_data.columns:
            sample_output = train_data["expectations"].values[0]
            return {k: infer_type_from_value(v) for k, v in sample_output.items()}
        return {}

    def _convert_to_dspy_dataset(self, data: "pd.DataFrame") -> list["dspy.Example"]:
        import dspy

        examples = []
        for _, row in data.iterrows():
            expectations = row["expectations"] if "expectations" in row else {}
            examples.append(
                dspy.Example(**row["inputs"], **expectations).with_inputs(*row["inputs"].keys())
            )
        return examples

    def _convert_to_dspy_metric(
        self,
        input_fields: dict[str, type],
        output_fields: dict[str, type],
        scorers: list[Scorer],
        objective: Optional[OBJECTIVE_FN] = None,
    ) -> Callable[["dspy.Example"], float]:
        def metric(example: "dspy.Example", pred: "dspy.Example", trace=None) -> float:
            scores = {}
            inputs = {key: example.get(key) for key in input_fields.keys()}
            expectations = {key: example.get(key) for key in output_fields.keys()}
            outputs = {key: pred.get(key) for key in output_fields.keys()}

            for scorer in scorers:
                kwargs = {"inputs": inputs, "outputs": outputs, "expectations": expectations}
                signature = inspect.signature(scorer)
                kwargs = {
                    key: value for key, value in kwargs.items() if key in signature.parameters
                }
                scores[scorer.name] = scorer(**kwargs)
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
                    f"Scorer [{','.join(non_numerical_scorers)}] return a string, Assessment or a "
                    "list of Assessment. Please provide `objective` function to aggregate "
                    "non-numerical values into a single value for optimization."
                )

        return metric
