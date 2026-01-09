"""GEPA alignment optimizer implementation."""

import logging
from typing import Any, Callable, Collection

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


@experimental(version="3.8.0")
class GePaAlignmentOptimizer(DSPyAlignmentOptimizer):
    """
    GEPA (Genetic-Pareto) alignment optimizer for judges.

    Uses DSPy's GEPA algorithm to optimize judge instructions through
    genetic-pareto optimization, learning from human feedback in traces.

    GEPA uses iterative refinement to improve text components like judge instructions
    by reflecting on system behavior and proposing improvements based on human feedback.

    Args:
        model: Model to use for DSPy/GEPA optimization. If None, uses get_default_model().
        max_metric_calls: Maximum number of evaluation calls during optimization.
            Higher values may lead to better results but increase optimization time.
            Default: 100
        gepa_kwargs: Additional keyword arguments to pass directly to dspy.GEPA().
            Useful for accessing advanced GEPA features not directly exposed
            through MLflow's GEPA interface.

            Note: Parameters already handled by MLflow's GEPA class will be overridden by the direct
            parameters and should not be passed through gepa_kwargs. List of predefined params:

            - max_metric_calls
            - metric

    Example:

        .. code-block:: python

            import mlflow
            from mlflow.genai.judges import make_judge
            from mlflow.genai.judges.optimizers import GePaAlignmentOptimizer

            # Create a judge
            judge = make_judge(
                name="relevance",
                instructions="Evaluate if the {{ outputs }} is relevant to {{ inputs }}.",
                model="openai:/gpt-4o-mini",
            )

            # Get traces with human feedback for this judge
            traces = mlflow.search_traces()

            # Optimize the judge instructions
            optimizer = GePaAlignmentOptimizer(
                model="openai:/gpt-4o",
                max_metric_calls=50,
            )

            optimized_judge = optimizer.align(judge, traces)
            print(optimized_judge.instructions)
    """

    def __init__(
        self,
        model: str | None = None,
        max_metric_calls: int = 100,
        gepa_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize GEPA optimizer with customizable parameters.

        Args:
            model: Model to use for DSPy/GEPA optimization. If None, uses get_default_model().
            max_metric_calls: Maximum number of evaluation calls during optimization.
                Higher values may lead to better results but increase optimization time.
                Default: 100
            gepa_kwargs: Additional keyword arguments to pass directly to dspy.GEPA().
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(model=model, **kwargs)
        self._max_metric_calls = max_metric_calls
        self._gepa_kwargs = gepa_kwargs or {}

    def _dspy_optimize(
        self,
        program: "dspy.Module",
        examples: Collection["dspy.Example"],
        metric_fn: Callable[["dspy.Example", Any, Any | None], bool],
    ) -> "dspy.Module":
        """
        Perform GEPA optimization with algorithm-specific parameters.

        GEPA uses all examples as training data (no separate validation set).

        Args:
            program: The DSPy program to optimize
            examples: Examples for optimization
            metric_fn: Default metric function for optimization (3-arg signature)

        Returns:
            Optimized DSPy program
        """

        # GEPA requires a metric with signature: (gold, pred, trace, pred_name, pred_trace)
        # But our metric_fn has signature: (example, pred, trace)
        # Create an adapter that bridges the two signatures
        def gepa_metric_adapter(gold, pred, trace, pred_name, pred_trace):
            """Adapt DSPy's 3-argument metric to GEPA's 5-argument format."""
            # gold is the dspy.Example
            # pred is the prediction output
            # We ignore pred_name and pred_trace for now, use the standard metric
            return metric_fn(gold, pred, trace)

        # Get the current LM from dspy context (set by parent class's align() method)
        # This LM will be used for reflection in GEPA
        reflection_lm = dspy.settings.lm

        # Build GEPA optimizer kwargs starting with required parameters
        # If metric or reflection_lm is in gepa_kwargs, they will override defaults
        optimizer_kwargs = {
            "metric": gepa_metric_adapter,
            "max_metric_calls": self._max_metric_calls,
            "reflection_lm": reflection_lm,
            **self._gepa_kwargs,  # Pass through any additional GEPA parameters
        }

        optimizer = dspy.GEPA(**optimizer_kwargs)

        self._logger.info(
            f"Starting GEPA optimization with {len(examples)} examples "
            f"and max {self._max_metric_calls} metric calls"
        )

        # Compile with GEPA-specific parameters
        result = optimizer.compile(
            student=program,
            trainset=examples,
        )

        self._logger.info("GEPA optimization completed")
        return result
