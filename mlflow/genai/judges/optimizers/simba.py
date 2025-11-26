"""SIMBA alignment optimizer implementation."""

import logging
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Collection, Iterator

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
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


@contextmanager
def _suppress_verbose_logging(
    logger_name: str, threshold_level: int = logging.DEBUG
) -> Iterator[None]:
    """
    Context manager to suppress verbose logging from a specific logger.

    Args:
        logger_name: Name of the logger to control
        threshold_level: Only suppress if MLflow logger is above this level
    """
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    try:
        if _logger.getEffectiveLevel() > threshold_level:
            logger.setLevel(logging.WARNING)
        yield
    finally:
        logger.setLevel(original_level)


@experimental(version="3.4.0")
class SIMBAAlignmentOptimizer(DSPyAlignmentOptimizer):
    """
    SIMBA (Simplified Multi-Bootstrap Aggregation) alignment optimizer.

    Uses DSPy's SIMBA algorithm to optimize judge prompts through
    bootstrap aggregation with simplified parametrization.

    Note on Logging:
        By default, SIMBA optimization suppresses DSPy's verbose output.
        To see detailed optimization progress from DSPy, set the MLflow logger to DEBUG::

            import logging

            logging.getLogger("mlflow.genai.judges.optimizers.simba").setLevel(logging.DEBUG)
    """

    # Class constants for default SIMBA parameters
    DEFAULT_SEED: ClassVar[int] = 42

    def __init__(
        self,
        model: str | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        simba_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize SIMBA optimizer with customizable parameters.

        Args:
            model: Model to use for DSPy optimization. If None, uses get_default_model().
            batch_size: Batch size for SIMBA evaluation. If None, uses get_min_traces_required().
            seed: Random seed for reproducibility. If None, uses DEFAULT_SEED (42).
            simba_kwargs: Additional keyword arguments to pass directly to dspy.SIMBA().
                         Supported parameters include:
                         - metric: Custom metric function (overrides default agreement_metric)
                         - max_demos: Maximum number of demonstrations to use
                         - num_threads: Number of threads for parallel optimization
                         - max_steps: Maximum number of optimization steps
                         See https://dspy.ai/api/optimizers/SIMBA/ for full list.
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(model=model, **kwargs)
        self._batch_size = batch_size
        self._seed = seed or self.DEFAULT_SEED
        self._simba_kwargs = simba_kwargs or {}

    def _get_batch_size(self) -> int:
        """
        Get the batch size for SIMBA optimization.

        Returns:
            The batch size to use for SIMBA optimization.
        """
        return self._batch_size if self._batch_size is not None else self.get_min_traces_required()

    def _dspy_optimize(
        self,
        program: "dspy.Module",
        examples: Collection["dspy.Example"],
        metric_fn: Callable[["dspy.Example", Any, Any | None], bool],
    ) -> "dspy.Module":
        """
        Perform SIMBA optimization with algorithm-specific parameters.

        SIMBA uses all examples as training data (no separate validation set).

        Args:
            program: The DSPy program to optimize
            examples: Examples for optimization
            metric_fn: Default metric function for optimization

        Returns:
            Optimized DSPy program
        """
        with _suppress_verbose_logging("dspy.teleprompt.simba"):
            # Build SIMBA optimizer kwargs starting with required parameters
            # If metric is in simba_kwargs, it will override the default metric_fn
            optimizer_kwargs = {
                "metric": metric_fn,
                "bsize": self._get_batch_size(),
                **self._simba_kwargs,  # Pass through any additional SIMBA parameters
            }

            optimizer = dspy.SIMBA(**optimizer_kwargs)

            _logger.info(
                f"Starting SIMBA optimization with {len(examples)} examples "
                f"(set logging to DEBUG for detailed output)"
            )

            # Compile with SIMBA-specific parameters
            result = optimizer.compile(
                student=program,
                trainset=examples,
                seed=self._seed,
            )

            _logger.info("SIMBA optimization completed")
            return result
