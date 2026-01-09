"""GEPA alignment optimizer implementation."""

import importlib.metadata
import logging
from typing import TYPE_CHECKING, Any

from packaging.version import Version

from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.base import AlignmentOptimizer, Judge
from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.utils.trace_utils import (
    extract_expectations_from_trace,
    extract_request_from_trace,
    extract_response_from_trace,
)
from mlflow.metrics.genai.model_utils import _parse_model_uri
from mlflow.prompt.constants import PROMPT_TEMPLATE_VARIABLE_PATTERN
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import gepa

_logger = logging.getLogger(__name__)


def _sanitize_assessment_name(name: str) -> str:
    return name.lower().strip()


@experimental(version="3.8.0")
class GePaAlignmentOptimizer(AlignmentOptimizer):
    """
    GEPA (Genetic-Pareto) alignment optimizer for judges.

    Uses GEPA's iterative mutation, reflection, and Pareto-aware candidate
    selection to optimize judge instructions by learning from human feedback
    in traces.

    GEPA uses iterative refinement to improve text components like judge instructions
    by reflecting on system behavior and proposing improvements based on human feedback.

    Args:
        model: Model to use for GEPA's reflection and optimization.
            Format: "<provider>:/<model>"
            (e.g., "openai:/gpt-4o", "anthropic:/claude-3-5-sonnet-20241022").
            If None, uses get_default_model(). Default: None
        max_metric_calls: Maximum number of evaluation calls during optimization.
            Higher values may lead to better results but increase optimization time.
            Default: 100
        gepa_kwargs: Additional keyword arguments to pass directly to
            gepa.optimize <https://github.com/gepa-ai/gepa/blob/main/src/gepa/api.py>.
            Useful for accessing advanced GEPA features not directly exposed
            through MLflow's GEPA interface.

            Note: Parameters already handled by MLflow's GEPA class will be overridden by the direct
            parameters and should not be passed through gepa_kwargs. List of predefined params:

            - max_metric_calls
            - seed_candidate
            - trainset
            - adapter
            - reflection_lm
            - use_mlflow

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

    _MINIMUM_TRACES_REQUIRED = 10

    def __init__(
        self,
        model: str | None = None,
        max_metric_calls: int = 100,
        gepa_kwargs: dict[str, Any] | None = None,
    ):
        self._model = model if model is not None else get_default_model()
        self._max_metric_calls = max_metric_calls
        self._gepa_kwargs = gepa_kwargs or {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def align(self, judge: Judge, traces: list[Trace]) -> Judge:
        """
        Align judge using GEPA optimization.

        Args:
            judge: The judge to be optimized
            traces: List of traces containing human feedback for alignment

        Returns:
            A new Judge instance with optimized instructions

        Raises:
            MlflowException: If no traces provided, no valid traces with human feedback,
                           or insufficient traces for optimization
            ImportError: If GEPA library is not installed
        """
        try:
            import gepa
        except ImportError as e:
            raise ImportError(
                "GEPA is not installed. Please install it with: `pip install gepa`"
            ) from e

        if not traces:
            raise MlflowException(
                "No traces provided for alignment",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Filter traces to only those with human feedback for this judge
        valid_traces = [
            trace for trace in traces if self._has_human_feedback_for_judge(trace, judge.name)
        ]

        self._logger.info(
            f"Found {len(valid_traces)} valid traces with human feedback "
            f"for judge '{judge.name}' out of {len(traces)} total traces"
        )

        if len(valid_traces) < self._MINIMUM_TRACES_REQUIRED:
            raise MlflowException(
                f"At least {self._MINIMUM_TRACES_REQUIRED} traces with human feedback "
                f"are required for optimization. Found {len(valid_traces)} valid traces. "
                f"Label more traces with human assessments for judge '{judge.name}'",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Parse model URI for GEPA
        provider, model = _parse_model_uri(self._model)

        # Create GEPA adapter
        adapter = self._MlflowGEPAAdapter(
            base_judge=judge,
            valid_traces=valid_traces,
        )

        # Set up GEPA optimization
        kwargs = self._gepa_kwargs | {
            "seed_candidate": {"instructions": judge.instructions},
            "trainset": valid_traces,
            "adapter": adapter,
            "reflection_lm": f"{provider}/{model}",
            "max_metric_calls": self._max_metric_calls,
            "use_mlflow": True,
        }

        # Handle version compatibility
        if Version(importlib.metadata.version("gepa")) < Version("0.0.18"):
            kwargs.pop("use_mlflow")

        self._logger.info(
            f"Starting GEPA optimization with {len(valid_traces)} traces "
            f"and max {self._max_metric_calls} metric calls"
        )

        # Run GEPA optimization
        gepa_result = gepa.optimize(**kwargs)

        self._logger.info("GEPA optimization completed")

        # Extract optimized instructions
        optimized_instructions = gepa_result.best_candidate["instructions"]

        # Validate that optimized instructions have exactly the same template variables
        original_template_vars = set(
            PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(judge.instructions)
        )
        optimized_template_vars = set(
            PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(optimized_instructions)
        )

        if optimized_template_vars != original_template_vars:
            raise MlflowException(
                f"Optimized instructions have different template variables. "
                f"Original: {sorted(original_template_vars)}, "
                f"Optimized: {sorted(optimized_template_vars)}. "
                f"The template variables must match exactly.",
                error_code=INTERNAL_ERROR,
            )

        # Create and return new judge with optimized instructions
        return make_judge(
            name=judge.name,
            instructions=optimized_instructions,
            model=judge.model,
            feedback_value_type=getattr(judge, "_feedback_value_type", str),
        )

    def _has_human_feedback_for_judge(self, trace: Trace, judge_name: str) -> bool:
        """Check if trace has human feedback for the specified judge."""
        if not trace.info.assessments:
            return False

        sanitized_judge_name = _sanitize_assessment_name(judge_name)
        return any(
            _sanitize_assessment_name(assessment.name) == sanitized_judge_name
            and assessment.source.source_type == AssessmentSourceType.HUMAN
            and assessment.feedback is not None
            for assessment in trace.info.assessments
        )

    class _MlflowGEPAAdapter:
        """Adapter that bridges MLflow traces to GEPA's evaluation interface."""

        def __init__(self, base_judge: Judge, valid_traces: list[Trace]):
            self.base_judge = base_judge
            self.valid_traces = valid_traces

            # Determine what input fields the judge requires
            judge_input_fields = base_judge.get_input_fields()
            self.judge_requires_inputs = any(f.name == "inputs" for f in judge_input_fields)
            self.judge_requires_outputs = any(f.name == "outputs" for f in judge_input_fields)
            self.judge_requires_expectations = any(
                f.name == "expectations" for f in judge_input_fields
            )
            self.judge_requires_trace = any(f.name == "trace" for f in judge_input_fields)

            # Extract template variables from original instructions for validation
            self.original_template_vars = set(
                PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(base_judge.instructions)
            )

        def evaluate(
            self,
            batch: list[Trace],
            candidate: dict[str, str],
            capture_traces: bool = False,
        ) -> "gepa.EvaluationBatch":
            """
            Evaluate candidate judge instructions against traces.

            Args:
                batch: List of traces to evaluate
                candidate: Proposed instructions {"instructions": "..."}
                capture_traces: Whether to capture execution traces

            Returns:
                EvaluationBatch with outputs, scores, and optional trajectories
            """
            import gepa

            candidate_instructions = candidate["instructions"]

            # Validate that candidate instructions contain the same template variables
            candidate_template_vars = set(
                PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(candidate_instructions)
            )
            if candidate_template_vars != self.original_template_vars:
                _logger.warning(
                    f"Candidate instructions have different template variables. "
                    f"Original: {self.original_template_vars}, "
                    f"Candidate: {candidate_template_vars}"
                )

            # Create temporary judge with candidate instructions
            temp_judge = make_judge(
                name=self.base_judge.name,
                instructions=candidate_instructions,
                model=self.base_judge.model,
                feedback_value_type=getattr(self.base_judge, "_feedback_value_type", str),
            )

            outputs = []
            scores = []
            trajectories = [] if capture_traces else None

            for trace in batch:
                try:
                    # Build judge kwargs based on requirements
                    judge_kwargs = self._build_judge_kwargs(trace)

                    # Call temporary judge
                    feedback = temp_judge(**judge_kwargs)
                    outputs.append(feedback.value)

                    # Get human feedback
                    human_value, _ = self._extract_human_feedback(trace)

                    # Calculate agreement score
                    score = self._agreement_score(feedback.value, human_value)
                    scores.append(score)

                    if capture_traces:
                        # Store trace information for reflective dataset
                        trajectories.append(
                            {
                                "trace": trace,
                                "inputs": judge_kwargs.get("inputs"),
                                "outputs": judge_kwargs.get("outputs"),
                                "expectations": judge_kwargs.get("expectations"),
                                "human_value": human_value,
                                "predicted_value": feedback.value,
                                "rationale": feedback.rationale,
                            }
                        )

                except Exception as e:
                    _logger.warning(f"Failed to evaluate trace {trace.info.trace_id}: {e}")
                    outputs.append(None)
                    scores.append(0.0)
                    if capture_traces:
                        trajectories.append(None)

            return gepa.EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories,
            )

        def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_batch: "gepa.EvaluationBatch",
            components_to_update: list[str],
        ) -> dict[str, list[dict[str, Any]]]:
            """
            Build reflective dataset for GEPA's reflection phase.

            Args:
                candidate: The evaluated candidate {"instructions": "..."}
                eval_batch: Result of evaluate with capture_traces=True
                components_to_update: Component names to update (should be ["instructions"])

            Returns:
                Dict mapping component name -> list of reflection records
            """
            reflective_datasets = {}

            for component_name in components_to_update:
                component_data = []

                trajectories = eval_batch.trajectories
                if trajectories is None:
                    trajectories = []

                for i, (trajectory, score) in enumerate(zip(trajectories, eval_batch.scores)):
                    if trajectory is None:
                        continue

                    trace = trajectory.get("trace")
                    spans = []
                    if trace and trace.data:
                        spans = [
                            {
                                "name": span.name,
                                "inputs": span.inputs,
                                "outputs": span.outputs,
                            }
                            for span in trace.data.spans
                        ]

                    component_data.append(
                        {
                            "component_name": component_name,
                            "current_text": candidate.get(component_name, ""),
                            "trace": spans,
                            "score": score,
                            "inputs": trajectory.get("inputs"),
                            "outputs": trajectory.get("outputs"),
                            "expectations": trajectory.get("expectations"),
                            "rationales": trajectory.get("rationale"),
                            "human_value": trajectory.get("human_value"),
                            "predicted_value": trajectory.get("predicted_value"),
                            "index": i,
                        }
                    )

                reflective_datasets[component_name] = component_data

            return reflective_datasets

        def _build_judge_kwargs(self, trace: Trace) -> dict[str, Any]:
            """Build kwargs for judge call based on required input fields."""
            judge_kwargs = {}

            if self.judge_requires_inputs:
                inputs = extract_request_from_trace(trace)
                if inputs is not None:
                    judge_kwargs["inputs"] = inputs

            if self.judge_requires_outputs:
                outputs = extract_response_from_trace(trace)
                if outputs is not None:
                    judge_kwargs["outputs"] = outputs

            if self.judge_requires_expectations:
                expectations = extract_expectations_from_trace(trace)
                if expectations is not None:
                    judge_kwargs["expectations"] = expectations

            if self.judge_requires_trace:
                judge_kwargs["trace"] = trace

            return judge_kwargs

        def _extract_human_feedback(self, trace: Trace) -> tuple[Any | None, str | None]:
            """
            Extract human feedback (value and rationale) for the judge from trace.

            Returns:
                (feedback_value, rationale) or (None, None) if not found
            """
            if not trace.info.assessments:
                return None, None

            # Sort by creation time (most recent first)
            sorted_assessments = sorted(
                trace.info.assessments,
                key=lambda a: (
                    a.create_time_ms if hasattr(a, "create_time_ms") and a.create_time_ms else 0
                ),
                reverse=True,
            )

            sanitized_judge_name = _sanitize_assessment_name(self.base_judge.name)
            for assessment in sorted_assessments:
                sanitized_assessment_name = _sanitize_assessment_name(assessment.name)
                if (
                    sanitized_assessment_name == sanitized_judge_name
                    and assessment.source.source_type == AssessmentSourceType.HUMAN
                ):
                    if assessment.feedback:
                        return assessment.feedback.value, assessment.rationale

            return None, None

        def _agreement_score(self, predicted: Any, expected: Any) -> float:
            """
            Calculate agreement between predicted and expected values.

            Returns:
                1.0 for match, 0.0 for mismatch
            """
            if predicted is None or expected is None:
                return 0.0

            # Normalize to lowercase strings for comparison
            pred_str = str(predicted).lower().strip()
            exp_str = str(expected).lower().strip()

            return 1.0 if pred_str == exp_str else 0.0
