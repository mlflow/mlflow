import logging
from typing import TYPE_CHECKING, Any, Callable

from mlflow.entities.model_registry import PromptVersion
from mlflow.genai.optimize.optimizers.dspy_optimizer import DSPyPromptOptimizer
from mlflow.genai.optimize.types import OptimizerOutput

if TYPE_CHECKING:
    import dspy


_logger = logging.getLogger(__name__)


class _DSPyGEPAOptimizer(DSPyPromptOptimizer):
    def run_optimization(
        self,
        prompt: PromptVersion,
        program: "dspy.Module",
        metric: Callable[["dspy.Example", "dspy.Example", Any], float],
        train_data: list["dspy.Example"],
        eval_data: list["dspy.Example"] | None,
    ) -> OptimizerOutput:
        import dspy

        from mlflow.genai.optimize.optimizers.utils.dspy_optimizer_utils import (
            format_dspy_prompt,
        )

        # Wrap the metric to match GEPA's expected signature
        def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            # Call the original metric with DSPy's standard signature
            return metric(gold, pred, trace)

        if self.optimizer_config.optimizer_llm:
            reflection_lm = dspy.LM(
                model=self._parse_model_name(self.optimizer_config.optimizer_llm.model_name),
                temperature=self.optimizer_config.optimizer_llm.temperature,
                api_base=self.optimizer_config.optimizer_llm.base_uri,
            )
        else:
            reflection_lm = dspy.settings.lm

        optimizer = dspy.GEPA(
            metric=gepa_metric,
            auto="light",
            track_stats=True,
            reflection_lm=reflection_lm,
            num_threads=self.optimizer_config.num_threads,
        )

        with self._maybe_suppress_stdout_stderr():
            optimized_program = optimizer.compile(
                program,
                trainset=train_data,
                valset=eval_data,
            )

            template = format_dspy_prompt(
                program=optimized_program,
                convert_to_single_text=self.optimizer_config.convert_to_single_text,
            )

        initial_score, final_score = self._extract_eval_scores(optimized_program)
        self._display_optimization_result(initial_score, final_score)

        return OptimizerOutput(
            final_eval_score=final_score,
            initial_eval_score=initial_score,
            optimized_prompt=template,
            optimizer_name="DSPy/GEPA",
        )

    def _extract_eval_scores(self, program: "dspy.Predict") -> tuple[float | None, float | None]:
        """Extract initial and final scores from GEPA optimized program.

        GEPA stores detailed results in program.detailed_results when track_stats=True.
        The detailed_results.val_aggregate_scores contains scores for each candidate,
        with the highest score being the final score.
        """
        final_score = None
        initial_score = None

        # Extract scores from detailed_results if available (when track_stats=True)
        if hasattr(program, "detailed_results") and program.detailed_results is not None:
            detailed_results = program.detailed_results

            if hasattr(detailed_results, "val_aggregate_scores"):
                scores = detailed_results.val_aggregate_scores
                if scores and len(scores) > 0:
                    # The first score is the initial baseline score
                    initial_score = scores[0]
                    # The highest score is the final optimized score
                    final_score = max(scores)

        return initial_score, final_score

    def _display_optimization_result(self, initial_score: float | None, final_score: float | None):
        """Display optimization results."""
        if final_score is None:
            return

        if initial_score is not None:
            if initial_score == final_score:
                _logger.info(f"Optimization complete! Score remained stable at: {final_score}.")
            else:
                improvement = final_score - initial_score
                _logger.info(
                    f"🎉 Optimization complete! "
                    f"Initial score: {initial_score}. "
                    f"Final score: {final_score} "
                    f"(+{improvement:.4f})."
                )
        else:
            _logger.info(f"Optimization complete! Final score: {final_score}.")
