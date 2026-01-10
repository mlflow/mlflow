"""GEPA alignment optimizer implementation."""

import logging
from typing import Any, Callable, Collection

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer
from mlflow.genai.judges.optimizers.dspy_utils import suppress_verbose_logging
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

    @staticmethod
    def _create_gepa_metric_adapter(
        metric_fn: Callable[["dspy.Example", Any, Any | None], bool]
    ) -> Callable[[Any, Any, Any | None, Any | None, Any | None], bool]:
        """
        Create a metric adapter that bridges DSPy's standard metric to GEPA's format.

        GEPA requires a metric with signature: (gold, pred, trace, pred_name, pred_trace)
        but our standard metric_fn has signature: (example, pred, trace).
        This method creates an adapter that bridges the two signatures.

        Args:
            metric_fn: Standard metric function with signature (example, pred, trace)

        Returns:
            Adapter function with GEPA's expected signature
        """

        def gepa_metric_adapter(gold, pred, trace=None, pred_name=None, pred_trace=None):
            """Adapt DSPy's 3-argument metric to GEPA's 5-argument format."""
            # gold is the dspy.Example
            # pred is the prediction output
            # trace/pred_name/pred_trace are optional GEPA-specific args
            # We pass None for our metric's trace parameter since GEPA's trace is different
            return metric_fn(gold, pred, trace=None)

        return gepa_metric_adapter

    def _dspy_optimize(
        self,
        program: "dspy.Module",
        examples: Collection["dspy.Example"],
        metric_fn: Callable[["dspy.Example", Any, Any | None], bool],
    ) -> "dspy.Module":
        # Create metric adapter for GEPA's expected signature
        gepa_metric_adapter = self._create_gepa_metric_adapter(metric_fn)

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

        # Compile with GEPA-specific parameters, suppressing verbose DSPy output
        with suppress_verbose_logging("dspy.teleprompt.gepa.gepa"):
            result = optimizer.compile(
                student=program,
                trainset=examples,
            )

        self._logger.info("GEPA optimization completed")
        return result
