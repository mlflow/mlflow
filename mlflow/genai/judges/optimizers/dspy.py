"""DSPy-based alignment optimizer implementation."""

import logging
from abc import abstractmethod
from typing import Any

from pydantic import PrivateAttr

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.base import AlignmentOptimizer, Judge
from mlflow.genai.judges.optimizers.dspy_utils import (
    agreement_metric,
    create_dspy_signature,
    trace_to_dspy_example,
)
from mlflow.genai.judges.utils import get_default_model
from mlflow.utils.annotations import experimental

logger = logging.getLogger(__name__)


@experimental(version="3.4.0")
class DSPyAlignmentOptimizer(AlignmentOptimizer):
    """
    Abstract base class for DSPy-based alignment optimizers.

    Provides common functionality for converting MLflow traces to DSPy examples
    and handling DSPy program compilation.
    """

    _logger: logging.Logger = PrivateAttr()
    _model: str = PrivateAttr()

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
        # Initialize private variables using PrivateAttr
        self._logger = logging.getLogger(self.__class__.__name__)
        # Private member variable for the model, defaulted to get_default_model()
        self._model = model if model is not None else get_default_model()

    @abstractmethod
    def _dspy_optimize(self, program, examples, metric_fn) -> Any:
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

    def _lower_to_dspy(self, judge: Judge) -> Any:
        """Lowers the judge into a Predict dspy module"""
        try:
            import dspy

            class CustomPredict(dspy.Predict):
                """Custom DSPy Predict class that allows passing an LM to the forward method."""

                def __init__(self, judge):
                    super().__init__(create_dspy_signature(judge))
                    self._lm = dspy.LM(model=judge.model)

                def forward(self, *args, **kwargs):
                    # If an LM is supplied via kwargs, use that, else use self.lm
                    lm = kwargs.pop("lm", self._lm)
                    return super().forward(*args, lm=lm, **kwargs)

            return CustomPredict(judge)
        except ImportError:
            raise MlflowException("DSPy library is required but not installed")

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

        Raises:
            MlflowException: If optimization fails or insufficient data is provided
        """
        try:
            import dspy

            if not traces:
                raise MlflowException("No traces provided for alignment")

            # Set up DSPy context to use the optimizer's model
            self._logger.info(f"Setting up DSPy context with model: {self._model}")

            # Configure DSPy to use the optimizer's model
            # This ensures the optimizer uses its own model, separate from the judge's model
            dspy_model = dspy.LM(model=self._model)

            # Use DSPy context manager to ensure proper model usage
            with dspy.context(lm=dspy_model):
                # Create DSPy program that will simulate the judge
                program = self._lower_to_dspy(judge)
                self._logger.info(
                    "Created DSPy program with signature using judge's model"
                )

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
                        "No valid examples could be created from traces"
                    )

                if len(dspy_examples) < 2:
                    raise MlflowException(
                        "At least 2 valid examples are required for optimization"
                    )

                self._logger.info("Starting DSPy optimization...")

                # Use the algorithm-specific optimization method
                # Each implementation decides how to handle data splitting
                optimized_program = self._dspy_optimize(
                    program, dspy_examples, agreement_metric
                )

                self._logger.info("DSPy optimization completed")

                # Create optimized judge with DSPy-optimized instructions
                # Use the same name as the original judge
                optimized_name = judge.name

                # Get the model from the original judge using the model property
                judge_model = judge.model

                # Extract optimized instructions from the DSPy program
                optimized_instructions = optimized_program.signature.instructions

                self._logger.info(
                    f"Creating optimized judge '{optimized_name}' with DSPy-optimized instructions"
                )

                return make_judge(
                    name=optimized_name,
                    instructions=optimized_instructions,
                    model=judge_model,
                )

        except ImportError:
            raise MlflowException("DSPy library is required but not installed")
        except Exception as e:
            raise MlflowException(f"Alignment optimization failed: {e!s}")
