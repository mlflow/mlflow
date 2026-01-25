"""GEPA alignment optimizer implementation."""

import logging
from typing import Any, Callable, Collection

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer
from mlflow.genai.judges.optimizers.dspy_utils import create_gepa_metric_adapter
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.utils.annotations import experimental

# Import dspy - raise exception if not installed
try:
    import dspy
except ImportError:
    raise MlflowException(
        "DSPy library is required but not installed. Please install it with: `pip install dspy`",
        error_code=INTERNAL_ERROR,
    )

_logger = logging.getLogger(__name__)


@experimental(version="3.8.0")
class GEPAAlignmentOptimizer(DSPyAlignmentOptimizer):
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
            If None (default), automatically set to 4x the number of training examples.
            This ensures sufficient budget for initial evaluation plus reflection iterations.
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
            from mlflow.genai.judges.optimizers import GEPAAlignmentOptimizer

            # Create a judge
            judge = make_judge(
                name="relevance",
                instructions="Evaluate if the {{ outputs }} is relevant to {{ inputs }}.",
                model="openai:/gpt-4o-mini",
            )

            # Get traces with human feedback for this judge
            traces = mlflow.search_traces()

            # Optimize the judge instructions
            optimizer = GEPAAlignmentOptimizer(
                model="openai:/gpt-4o",
                max_metric_calls=50,
            )

            optimized_judge = optimizer.align(judge, traces)
            print(optimized_judge.instructions)
    """

    _DEFAULT_BUDGET_MULTIPLIER: int = 4

    def __init__(
        self,
        model: str | None = None,
        max_metric_calls: int | None = None,
        gepa_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Args:
            model: Model to use for DSPy/GEPA optimization. If None, uses get_default_model().
            max_metric_calls: Maximum number of evaluation calls during optimization.
                Higher values may lead to better results but increase optimization time.
                If None (default), automatically set to 4x the number of training examples.
            gepa_kwargs: Additional keyword arguments to pass directly to dspy.GEPA().
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(model=model, **kwargs)
        self._max_metric_calls = max_metric_calls
        self._gepa_kwargs = gepa_kwargs or {}

    def _dspy_optimize(
        self,
        program: "dspy.Predict",
        examples: Collection["dspy.Example"],
        metric_fn: Callable[["dspy.Example", Any, Any | None], bool],
    ) -> "dspy.Predict":
        gepa_metric_adapter = create_gepa_metric_adapter(metric_fn)

        reflection_lm = dspy.settings.lm

        # Calculate max_metric_calls if not explicitly set
        # GEPA needs at least num_examples calls for initial evaluation,
        # plus additional calls for reflection iterations
        max_metric_calls = self._max_metric_calls
        if max_metric_calls is None:
            max_metric_calls = len(examples) * self._DEFAULT_BUDGET_MULTIPLIER
            self._logger.info(
                f"max_metric_calls not specified, using {self._DEFAULT_BUDGET_MULTIPLIER}x "
                f"number of examples: {max_metric_calls}"
            )

        optimizer_kwargs = self._gepa_kwargs | {
            "metric": gepa_metric_adapter,
            "max_metric_calls": max_metric_calls,
            "reflection_lm": reflection_lm,
        }

        optimizer = dspy.GEPA(**optimizer_kwargs)

        self._logger.info(
            f"Starting GEPA optimization with {len(examples)} examples "
            f"and max {max_metric_calls} metric calls"
        )

        result = optimizer.compile(
            student=program,
            trainset=examples,
        )

        self._logger.info("GEPA optimization completed")
        return result
