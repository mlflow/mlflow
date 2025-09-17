import contextlib
import importlib.metadata
import importlib.util
import inspect
import io
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

from packaging.version import Version

from mlflow.entities.model_registry import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers import BasePromptOptimizer
from mlflow.genai.optimize.types import LLMParams, ObjectiveFn, OptimizerConfig, OptimizerOutput
from mlflow.genai.optimize.util import infer_type_from_value
from mlflow.genai.scorers import Scorer
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import dspy
    import pandas as pd

_logger = logging.getLogger(__name__)


@experimental(version="3.3.0")
class DSPyPromptOptimizer(BasePromptOptimizer):
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

    def _parse_model_name(self, model_name: str) -> str:
        """
        Parse model name from URI format to DSPy format.

        Accepts two formats:
        - URI format: 'openai:/gpt-4o' -> converted to 'openai/gpt-4o'
        - DSPy format: 'openai/gpt-4o' -> returned unchanged

        Raises MlflowException for invalid formats.
        """
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        if not model_name:
            raise MlflowException.invalid_parameter_value(
                "Model name cannot be empty. Please provide a model name in the format "
                "'<provider>:/<model>' or '<provider>/<model>'."
            )

        try:
            scheme, path = _parse_model_uri(model_name)
            return f"{scheme}/{path}"
        except MlflowException:
            if "/" in model_name and ":" not in model_name:
                parts = model_name.split("/")
                if len(parts) == 2 and parts[0] and parts[1]:
                    return model_name

            raise MlflowException.invalid_parameter_value(
                f"Invalid model name format: '{model_name}'. "
                "Model name must be in one of the following formats:\n"
                "  - '<provider>/<model>' (e.g., 'openai/gpt-4')\n"
                "  - '<provider>:/<model>' (e.g., 'openai:/gpt-4')"
            )

    def optimize(
        self,
        prompt: PromptVersion,
        target_llm_params: LLMParams,
        train_data: "pd.DataFrame",
        scorers: list[Scorer],
        objective: ObjectiveFn | None = None,
        eval_data: Optional["pd.DataFrame"] = None,
    ) -> OptimizerOutput:
        import dspy

        _logger.info(
            f"🎯 Starting prompt optimization for: {prompt.uri}\n"
            f"⏱️ This may take several minutes or longer depending on dataset size...\n"
            f"📊 Training with {len(train_data)} examples."
        )

        input_fields = self._get_input_fields(train_data)
        self._validate_input_fields(input_fields, prompt)
        output_fields = self._get_output_fields(train_data)

        lm = dspy.LM(
            model=self._parse_model_name(target_llm_params.model_name),
            temperature=target_llm_params.temperature,
            api_base=target_llm_params.base_uri,
        )

        if self.optimizer_config.optimizer_llm:
            teacher_lm = dspy.LM(
                model=self._parse_model_name(self.optimizer_config.optimizer_llm.model_name),
                temperature=self.optimizer_config.optimizer_llm.temperature,
                api_base=self.optimizer_config.optimizer_llm.base_uri,
            )
        else:
            teacher_lm = lm

        if self.optimizer_config.extract_instructions:
            instructions = self._extract_instructions(prompt.template, teacher_lm)
        else:
            instructions = prompt.template

        signature = dspy.make_signature(
            {key: (type_, dspy.InputField()) for key, type_ in input_fields.items()}
            | {key: (type_, dspy.OutputField()) for key, type_ in output_fields.items()},
            instructions,
        )

        # Define main student program
        program = dspy.Predict(signature)
        adapter = dspy.JSONAdapter()

        train_data = self._convert_to_dspy_dataset(train_data)
        eval_data = self._convert_to_dspy_dataset(eval_data) if eval_data is not None else None

        with dspy.context(lm=lm, adapter=adapter):
            return self.run_optimization(
                prompt=prompt,
                program=program,
                metric=self._convert_to_dspy_metric(
                    input_fields, output_fields, scorers, objective
                ),
                train_data=train_data,
                eval_data=eval_data,
            )

    def run_optimization(
        self,
        prompt: PromptVersion,
        program: "dspy.Module",
        metric: Callable[["dspy.Example"], float],
        train_data: list["dspy.Example"],
        eval_data: list["dspy.Example"],
    ) -> OptimizerOutput:
        """
        Run the optimization process for the given prompt and program.

        Parameters
        ----------
        prompt : PromptVersion
            The prompt version to optimize.
        program : dspy.Module
            The DSPy program/module to optimize.
        metric : Callable[[dspy.Example], float]
            A callable that computes a metric score for a given example.
        train_data : list[dspy.Example]
            List of training examples for optimization.
        eval_data : list[dspy.Example]
            List of evaluation examples for validation.

        Returns
        -------
        OptimizerOutput
            The result of the optimization, including the optimized prompt and metrics.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses of DSPyPromptOptimizer must implement `run_optimization`."
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
        objective: ObjectiveFn | None = None,
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

    def _validate_input_fields(self, input_fields: dict[str, type], prompt: PromptVersion) -> None:
        if missing_fields := set(prompt.variables) - set(input_fields.keys()):
            raise MlflowException(
                f"Validation failed. Missing prompt variables in dataset: {missing_fields}. "
                "Please ensure your dataset contains columns for all prompt variables."
            )

    def _extract_instructions(self, template: str | dict[str, Any], lm: "dspy.LM") -> str:
        import dspy

        extractor = dspy.Predict(
            dspy.make_signature(
                {
                    "prompt": (str, dspy.InputField()),
                    "instruction": (str, dspy.OutputField()),
                },
                "Extract the core instructions from the prompt "
                "to use as the system message for the LLM.",
            )
        )

        with dspy.context(lm=lm):
            return extractor(prompt=template).instruction

    @contextlib.contextmanager
    def _maybe_suppress_stdout_stderr(self):
        """Context manager for redirecting stdout/stderr based on verbose setting.
        If verbose is False, redirects output to devnull or StringIO.
        If verbose is True, doesn't redirect output.
        """
        if not self.optimizer_config.verbose:
            try:
                output_sink = open(os.devnull, "w")  # noqa: SIM115
            except (OSError, IOError):
                output_sink = io.StringIO()

            with output_sink:
                with (
                    contextlib.redirect_stdout(output_sink),
                    contextlib.redirect_stderr(output_sink),
                ):
                    yield
        else:
            yield
