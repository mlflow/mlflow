import logging
import math
from typing import TYPE_CHECKING, Callable, Optional

from mlflow.entities.model_registry import PromptVersion
from mlflow.genai.optimize.optimizers.dspy_optimizer import DSPyPromptOptimizer
from mlflow.genai.optimize.types import OptimizerOutput

if TYPE_CHECKING:
    import dspy
    import pandas as pd


_logger = logging.getLogger(__name__)


class _DSPyMIPROv2Optimizer(DSPyPromptOptimizer):
    def run_optimization(
        self,
        prompt: PromptVersion,
        program: "dspy.Module",
        metric: Callable[["dspy.Example"], float],
        train_data: list["dspy.Example"],
        eval_data: list["dspy.Example"] | None,
    ) -> OptimizerOutput:
        import dspy

        from mlflow.genai.optimize.optimizers.utils.dspy_mipro_callback import _DSPyMIPROv2Callback
        from mlflow.genai.optimize.optimizers.utils.dspy_optimizer_utils import (
            format_dspy_prompt,
        )

        teacher_settings = {}

        if self.optimizer_config.optimizer_llm:
            teacher_lm = dspy.LM(
                model=self._parse_model_name(self.optimizer_config.optimizer_llm.model_name),
                temperature=self.optimizer_config.optimizer_llm.temperature,
                api_base=self.optimizer_config.optimizer_llm.base_uri,
            )
            teacher_settings["lm"] = teacher_lm

        num_candidates = self.optimizer_config.num_instruction_candidates
        optimizer = dspy.MIPROv2(
            metric=metric,
            max_bootstrapped_demos=self.optimizer_config.max_few_show_examples,
            num_candidates=num_candidates,
            num_threads=self.optimizer_config.num_threads,
            teacher_settings=teacher_settings,
            auto=None,
        )

        callbacks = (
            [
                _DSPyMIPROv2Callback(
                    prompt_name=prompt.name,
                    input_fields=program.signature.input_fields,
                    convert_to_single_text=self.optimizer_config.convert_to_single_text,
                ),
            ]
            if self.optimizer_config.autolog
            else []
        )

        with dspy.context(callbacks=callbacks):
            with self._maybe_suppress_stdout_stderr():
                optimized_program = optimizer.compile(
                    program,
                    trainset=train_data,
                    valset=eval_data,
                    num_trials=self._get_num_trials(num_candidates),
                    minibatch_size=self._get_minibatch_size(train_data, eval_data),
                    requires_permission_to_run=False,
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
            optimizer_name="DSPy/MIPROv2",
        )

    def _get_num_trials(self, num_candidates: int) -> int:
        # MAX(2*log(num_candidates), 3/2*num_candidates)
        return int(max(2 * math.log2(num_candidates), 1.5 * num_candidates))

    def _get_minibatch_size(
        self,
        train_data: "pd.DataFrame",
        eval_data: Optional["pd.DataFrame"] = None,
    ) -> int:
        # The default minibatch size is 35 in MIPROv2.
        if eval_data is not None:
            return min(35, len(eval_data) // 2)
        return min(35, len(train_data) // 2)

    def _extract_eval_scores(self, program: "dspy.Predict") -> tuple[float | None, float | None]:
        final_score = getattr(program, "score", None)
        initial_score = None
        # In DSPy < 2.6.17, trial_logs contains initial score in key=-1.
        trial_logs = getattr(program, "trial_logs", {})
        if 1 in trial_logs:
            initial_score = trial_logs[1].get("full_eval_score")
        elif -1 in trial_logs:
            initial_score = trial_logs[-1].get("full_eval_score")
        return initial_score, final_score

    def _display_optimization_result(self, initial_score: float | None, final_score: float | None):
        _logger.info(f"Optimization complete! Score remained stable at: {final_score}.")
        if final_score is None:
            return

        if initial_score is not None:
            if initial_score == final_score:
                _logger.info(f"Optimization complete! Score remained stable at: {final_score}.")
                return
            else:
                _logger.info(
                    f"🎉 Optimization complete! "
                    f"Initial score: {initial_score}. "
                    f"Final score: {final_score}."
                )
