"""DSPy-based alignment optimizer implementation."""

import logging
from abc import abstractmethod
from typing import Any, Callable, ClassVar, Collection

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.base import AlignmentOptimizer, Judge
from mlflow.genai.judges.optimizers.dspy_utils import (
    agreement_metric,
    convert_mlflow_uri_to_litellm,
    create_dspy_signature,
    trace_to_dspy_example,
)
from mlflow.genai.judges.utils import _suppress_litellm_nonfatal_errors, get_default_model
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental

# Import dspy - raise exception if not installed
try:
    import dspy
except ImportError:
    raise MlflowException(
        "DSPy library is required but not installed. Please install it with: pip install dspy",
        error_code=INTERNAL_ERROR,
    )

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
        program: "dspy.Module",
        examples: Collection["dspy.Example"],
        metric_fn: Callable[["dspy.Example", Any, Any | None], bool],
    ) -> "dspy.Module":
        """
        Perform DSPy optimization with algorithm-specific parameters.

        Each implementation can decide how to split the data internally if needed.

        Args:
            program: The DSPy program to optimize
            examples: Examples for optimization (implementations decide how to split)
            metric_fn: Metric function for optimization

        Returns:
            Optimized DSPy program
        """

    def _get_dspy_program_from_judge(self, judge: Judge) -> Any:
        """Convert a judge into a DSPy Predict module."""

        class CustomPredict(dspy.Predict):
            """
            Custom DSPy Predict class that allows passing an LM to the forward method.
            This is necessary to ensure that the optimized dspy program uses the judge's model,
            while we allow for the optimizer itself to use a different model.
            """

            def __init__(self, judge):
                super().__init__(create_dspy_signature(judge))
                # Convert MLflow model URI to LiteLLM format for DSPy
                judge_model_litellm = convert_mlflow_uri_to_litellm(judge.model)
                self._lm = dspy.LM(model=judge_model_litellm)

            def forward(self, *args, **kwargs):
                # If an LM is supplied via kwargs, use that, else use self.lm
                lm = kwargs.pop("lm", self._lm)
                return super().forward(*args, lm=lm, **kwargs)

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

            self._logger.info(f"Setting up DSPy context with model: {self._model}")

            # Configure DSPy to use the optimizer's model
            # This ensures the optimizer uses its own model, separate from the judge's model
            optimizer_model_litellm = convert_mlflow_uri_to_litellm(self._model)
            with dspy.context(lm=dspy.LM(model=optimizer_model_litellm)):
                # Create DSPy program that will simulate the judge
                program = self._get_dspy_program_from_judge(judge)
                self._logger.info("Created DSPy program with signature using judge's model")

                # Convert traces to DSPy format
                dspy_examples = []
                for trace in traces:
                    example = trace_to_dspy_example(trace, judge.name)
                    if example is not None:
                        dspy_examples.append(example)

                self._logger.info(
                    f"Created {len(dspy_examples)} valid examples from {len(traces)} traces"
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

                self._logger.info("Starting DSPy optimization...")

                # Use the algorithm-specific optimization method
                # Each implementation decides how to handle data splitting
                optimized_program = self._dspy_optimize(program, dspy_examples, agreement_metric)

                self._logger.info("DSPy optimization completed")

                # Create optimized judge with DSPy-optimized instructions

                optimized_instructions = optimized_program.signature.instructions
                return make_judge(
                    name=judge.name, instructions=optimized_instructions, model=judge.model
                )

        except Exception as e:
            raise MlflowException(
                f"Alignment optimization failed: {e!s}", error_code=INTERNAL_ERROR
            ) from e
