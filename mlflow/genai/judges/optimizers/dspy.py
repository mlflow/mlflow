"""DSPy-based alignment optimizer implementation."""

import logging
from abc import abstractmethod
from typing import Any, Callable, ClassVar, Collection

from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.base import AlignmentOptimizer, Judge
from mlflow.genai.judges.optimizers.dspy_utils import (
    _check_dspy_installed,
    agreement_metric,
    append_input_fields_section,
    construct_dspy_lm,
    create_dspy_signature,
    format_demos_as_examples,
    trace_to_dspy_example,
)
from mlflow.genai.judges.utils import (
    _suppress_litellm_nonfatal_errors,
    get_default_model,
)
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental

_check_dspy_installed()
import dspy

_logger = logging.getLogger(__name__)


@experimental(version="3.4.0")
class DSPyAlignmentOptimizer(AlignmentOptimizer):
    """
    Abstract base class for DSPy-based alignment optimizers.

    Provides common functionality for converting MLflow traces to DSPy examples
    and handling DSPy program compilation.
    """

    _logger: logging.Logger
    _model: str

    _MINIMUM_TRACES_REQUIRED_FOR_OPTIMIZATION: ClassVar[int] = 10

    @classmethod
    def get_min_traces_required(cls) -> int:
        """Get the minimum number of traces required for optimization.

        Returns:
            The minimum number of traces required for optimization.
        """
        return cls._MINIMUM_TRACES_REQUIRED_FOR_OPTIMIZATION

    @property
    def model(self) -> str:
        """Get the model used by this optimizer."""
        return self._model

    def __init__(self, model: str | None = None, **kwargs):
        """
        Initialize DSPy optimizer with common parameters.

        Args:
            model: Model to use for DSPy optimization. If None, uses get_default_model().
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model if model is not None else get_default_model()

    @abstractmethod
    def _dspy_optimize(
        self,
        program: "dspy.Predict",
        examples: Collection["dspy.Example"],
        metric_fn: Callable[["dspy.Example", Any, Any | None], bool],
    ) -> "dspy.Predict":
        """
        Perform DSPy optimization with algorithm-specific parameters.

        Each implementation can decide how to split the data internally if needed.

        Args:
            program: The DSPy Predict program to optimize
            examples: Examples for optimization (implementations decide how to split)
            metric_fn: Metric function for optimization

        Returns:
            Optimized DSPy Predict program
        """

    def _create_judge_from_dspy_program(
        self,
        optimized_program: "dspy.Predict",
        original_judge: Judge,
    ) -> Judge:
        """
        Create a judge from an optimized DSPy program.

        This method combines instruction post-processing (appending input fields section)
        and demo formatting into a single operation that returns a ready-to-use judge.

        Args:
            optimized_program: The optimized DSPy Predict program
            original_judge: The original judge (to get name, model, field names, etc.)

        Returns:
            A new Judge instance with processed instructions and demos included
        """
        optimized_instructions = optimized_program.signature.instructions

        instructions = append_input_fields_section(optimized_instructions, original_judge)

        demos = getattr(optimized_program, "demos", [])
        if demos_text := format_demos_as_examples(demos, original_judge):
            instructions = demos_text + "\n\n" + instructions
            self._logger.info(f"Including {len(demos)} demos from optimization")

        return make_judge(
            name=original_judge.name,
            instructions=instructions,
            model=original_judge.model,
            feedback_value_type=original_judge.feedback_value_type,
        )

    def _get_dspy_program_from_judge(self, judge: Judge) -> Any:
        """Convert a judge into a DSPy Predict module."""
        create_judge_from_dspy_program = self._create_judge_from_dspy_program

        class CustomPredict(dspy.Predict):
            """
            Custom DSPy Predict class that uses the judge's model for evaluations.

            This ensures the optimized DSPy program uses the judge's model,
            while allowing the optimizer itself to use a different model.
            """

            def __init__(self, original_judge: Judge):
                super().__init__(create_dspy_signature(original_judge))
                self._original_judge: Judge = original_judge

            def forward(self, *args, **kwargs):
                # Extract _trace before filtering (DSPy convention for disabling trace)
                should_trace = kwargs.pop("_trace", True)

                # Filter kwargs to only include the judge's input fields
                input_field_names = {f.name for f in self._original_judge.get_input_fields()}
                judge_kwargs = {k: v for k, v in kwargs.items() if k in input_field_names}

                created_judge: Judge = create_judge_from_dspy_program(
                    optimized_program=self,
                    original_judge=self._original_judge,
                )
                feedback: Feedback = created_judge(**judge_kwargs)

                pred = dspy.Prediction(
                    result=feedback.value,
                    rationale=feedback.rationale,
                )

                # Manually record a consistent trace for optimizers that depend on it (e.g., GEPA)
                if should_trace and dspy.settings.trace is not None:
                    trace = dspy.settings.trace
                    max_trace_size = getattr(dspy.settings, "max_trace_size", float("inf"))
                    if max_trace_size > 0:
                        if len(trace) >= max_trace_size:
                            trace.pop(0)
                        trace.append((self, {**kwargs}, pred))

                return pred

        return CustomPredict(judge)

    @_suppress_litellm_nonfatal_errors
    def align(self, judge: Judge, traces: list[Trace]) -> Judge:
        """
        Main alignment method that orchestrates the DSPy optimization process.

        1. Extract judge instructions and create DSPy signature
        2. Convert traces to DSPy examples
        3. Create and compile DSPy optimizer
        4. Generate optimized judge from results

        Args:
            judge: The judge to be optimized
            traces: List of traces containing alignment data.
                   The implementation will split these traces internally for train/validation.

        Returns:
            A new optimized Judge instance
        """
        try:
            if not traces:
                raise MlflowException(
                    "No traces provided for alignment",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            self._logger.debug(f"Setting up DSPy context with model: {self._model}")

            # Configure DSPy to use the optimizer's model
            # This ensures the optimizer uses its own model, separate from the judge's model
            optimizer_lm = construct_dspy_lm(self._model)

            with dspy.context(lm=optimizer_lm):
                # Create DSPy program that will simulate the judge
                program = self._get_dspy_program_from_judge(judge)
                self._logger.debug("Created DSPy program with signature using judge's model")

                # Convert traces to DSPy format
                dspy_examples = []
                for trace in traces:
                    example = trace_to_dspy_example(trace, judge)
                    if example is not None:
                        dspy_examples.append(example)

                self._logger.info(
                    f"Preparing optimization with {len(dspy_examples)} examples "
                    f"from {len(traces)} traces"
                )

                if not dspy_examples:
                    raise MlflowException(
                        f"No valid examples could be created from traces. "
                        f"Ensure that the provided traces contain Feedback entries "
                        f"with name {judge.name}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

                min_traces = self.get_min_traces_required()
                if len(dspy_examples) < min_traces:
                    raise MlflowException(
                        f"At least {min_traces} valid traces are required for optimization. "
                        f"Label more traces with Feedback entries with name {judge.name}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

                self._logger.debug("Starting DSPy optimization...")

                # Use the algorithm-specific optimization method
                # Each implementation decides how to handle data splitting
                optimized_program = self._dspy_optimize(program, dspy_examples, agreement_metric)

                self._logger.debug("DSPy optimization completed")

                if not isinstance(optimized_program, dspy.Predict):
                    raise MlflowException(
                        f"Optimizer returned {type(optimized_program).__name__}, "
                        "expected dspy.Predict. Custom optimizers must return a "
                        "Predict instance from _dspy_optimize().",
                        error_code=INTERNAL_ERROR,
                    )

                return self._create_judge_from_dspy_program(
                    optimized_program=optimized_program,
                    original_judge=judge,
                )

        except Exception as e:
            raise MlflowException(
                f"Alignment optimization failed: {e!s}", error_code=INTERNAL_ERROR
            ) from e
