"""SIMBA alignment optimizer implementation."""

from typing import Any, ClassVar

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.utils.annotations import experimental

# Import dspy - raise exception if not installed
try:
    import dspy
except ImportError:
    raise MlflowException(
        "DSPy library is required but not installed. "
        "Please install it with: pip install 'mlflow[genai-dspy]'",
        error_code=INTERNAL_ERROR,
    )


@experimental(version="3.4.0")
class SIMBAAlignmentOptimizer(DSPyAlignmentOptimizer):
    """
    SIMBA (Simplified Multi-Bootstrap Aggregation) alignment optimizer.

    Uses DSPy's SIMBA algorithm to optimize judge prompts through
    bootstrap aggregation with simplified parametrization.
    """

    # Class constants for default SIMBA parameters
    DEFAULT_BSIZE: ClassVar[int] = 4
    DEFAULT_SEED: ClassVar[int] = 42

    def __init__(self, model: str | None = None, **kwargs):
        """
        Initialize SIMBA optimizer with default parameters.

        Args:
            model: Model to use for DSPy optimization. If None, uses get_default_model().
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(model=model, **kwargs)
        self._bsize = self.DEFAULT_BSIZE
        self._seed = self.DEFAULT_SEED

    def _dspy_optimize(self, program, examples, metric_fn) -> Any:
        """
        Perform SIMBA optimization with algorithm-specific parameters.

        SIMBA uses all examples as training data (no separate validation set).

        Args:
            program: The DSPy program to optimize
            examples: Examples for optimization
            metric_fn: Metric function for optimization

        Returns:
            Optimized DSPy program
        """
        # Create SIMBA optimizer
        optimizer = dspy.SIMBA(metric=metric_fn, bsize=self._bsize)

        # Compile with SIMBA-specific parameters
        # SIMBA uses all examples as training data
        return optimizer.compile(
            student=program,
            trainset=examples,
            seed=self._seed,
        )
