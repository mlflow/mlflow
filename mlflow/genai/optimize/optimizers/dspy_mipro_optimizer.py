import contextlib
import io
import logging
import math
import os
from typing import TYPE_CHECKING, Optional

from mlflow.entities.model_registry import Prompt
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers.dspy_optimizer import _DSPyOptimizer
from mlflow.genai.optimize.types import OBJECTIVE_FN, LLMParams
from mlflow.genai.prompts import register_prompt
from mlflow.genai.scorers import Scorer
from mlflow.tracking._model_registry.fluent import active_run
from mlflow.tracking.fluent import log_metric, log_param

if TYPE_CHECKING:
    import dspy
    import pandas as pd


_logger = logging.getLogger(__name__)


class _DSPyMIPROv2Optimizer(_DSPyOptimizer):
    def optimize(
        self,
        prompt: Prompt,
        target_llm_params: LLMParams,
        train_data: "pd.DataFrame",
        scorers: list[Scorer],
        objective: Optional[OBJECTIVE_FN] = None,
        eval_data: Optional["pd.DataFrame"] = None,
    ) -> Prompt:
        import dspy

        from mlflow.genai.optimize.optimizers.utils.dspy_mipro_callback import _DSPyMIPROv2Callback
        from mlflow.genai.optimize.optimizers.utils.dspy_mipro_utils import format_optimized_prompt

        _logger.info(
            f"🎯 Starting prompt optimization for: {prompt.uri}\n"
            f"⏱️ This may take several minutes or longer depending on dataset size...\n"
            f"📊 Training with {len(train_data)} examples."
        )

        input_fields = self._get_input_fields(train_data)
        self._validate_input_fields(input_fields, prompt)
        output_fields = self._get_output_fields(train_data)

        teacher_settings = {}
        if self.optimizer_config.optimizer_llm:
            teacher_lm = dspy.LM(
                model=self.optimizer_config.optimizer_llm.model_name,
                temperature=self.optimizer_config.optimizer_llm.temperature,
                api_base=self.optimizer_config.optimizer_llm.base_uri,
            )
            teacher_settings["lm"] = teacher_lm

        lm = dspy.LM(
            model=target_llm_params.model_name,
            temperature=target_llm_params.temperature,
            api_base=target_llm_params.base_uri,
        )

        instructions = self._extract_instructions(prompt.template, teacher_settings.get("lm", lm))

        signature = dspy.make_signature(
            {
                **{key: (_type, dspy.InputField()) for key, _type in input_fields.items()},
                **{key: (_type, dspy.OutputField()) for key, _type in output_fields.items()},
            },
            instructions,
        )

        # Define main student program
        program = dspy.Predict(signature)

        train_data = self._convert_to_dspy_dataset(train_data)
        eval_data = self._convert_to_dspy_dataset(eval_data) if eval_data is not None else None

        num_candidates = self.optimizer_config.num_instruction_candidates
        optimizer = dspy.MIPROv2(
            metric=self._convert_to_dspy_metric(input_fields, output_fields, scorers, objective),
            max_bootstrapped_demos=self.optimizer_config.max_few_show_examples,
            num_candidates=num_candidates,
            num_threads=self.optimizer_config.num_threads,
            teacher_settings=teacher_settings,
            auto=None,
        )

        adapter = dspy.JSONAdapter()
        callbacks = (
            [
                _DSPyMIPROv2Callback(prompt.name, input_fields),
            ]
            if self.optimizer_config.autolog
            else []
        )
        with dspy.context(lm=lm, adapter=adapter, callbacks=callbacks):
            with self._maybe_suppress_stdout_stderr():
                optimized_program = optimizer.compile(
                    program,
                    trainset=train_data,
                    valset=eval_data,
                    num_trials=self._get_num_trials(num_candidates),
                    minibatch_size=self._get_minibatch_size(train_data, eval_data),
                    requires_permission_to_run=False,
                )

            template = format_optimized_prompt(
                program=optimized_program,
                input_fields=input_fields,
            )

        self._display_optimization_result(optimized_program)
        final_score = getattr(optimized_program, "score", None)
        optimized_prompt = register_prompt(
            name=prompt.name,
            template=template,
            tags={
                "overall_eval_score": str(final_score),
            },
        )

        if self.optimizer_config.autolog:
            self._log_optimization_result(final_score, optimized_prompt)

        return optimized_prompt

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

    def _validate_input_fields(self, input_fields: dict[str, type], prompt: Prompt) -> None:
        if missing_fields := set(prompt.variables) - set(input_fields.keys()):
            raise MlflowException(
                f"Validation failed. Missing prompt variables in dataset: {missing_fields}. "
                "Please ensure your dataset contains columns for all prompt variables."
            )

    def _extract_instructions(self, template: str, lm: "dspy.LM") -> str:
        import dspy

        extractor = dspy.Predict(
            dspy.make_signature(
                {
                    "prompt": (str, dspy.InputField()),
                    "instruction": (str, dspy.OutputField()),
                },
                "Extract the core instructions from the prompt "
                "to use as the system message for the LLM.",
            )
        )

        with dspy.context(lm=lm):
            return extractor(prompt=template).instruction

    @contextlib.contextmanager
    def _maybe_suppress_stdout_stderr(self):
        """Context manager for redirecting stdout/stderr based on verbose setting.
        If verbose is False, redirects output to devnull or StringIO.
        If verbose is True, doesn't redirect output.
        """
        if not self.optimizer_config.verbose:
            try:
                output_sink = open(os.devnull, "w")  # noqa: SIM115
            except (OSError, IOError):
                output_sink = io.StringIO()

            with output_sink:
                with (
                    contextlib.redirect_stdout(output_sink),
                    contextlib.redirect_stderr(output_sink),
                ):
                    yield
        else:
            yield

    def _display_optimization_result(self, program: "dspy.Predict"):
        score = getattr(program, "score", None)
        if score is None:
            return
        # In DSPy < 2.6.17, trial_logs contains initial score in key=-1.
        trial_logs = getattr(program, "trial_logs", {})
        if 1 in trial_logs:
            initial_score = trial_logs[1].get("full_eval_score")
        elif -1 in trial_logs:
            initial_score = trial_logs[-1].get("full_eval_score")
        else:
            initial_score = None
        if initial_score is not None:
            if initial_score == score:
                _logger.info(f"Optimization complete! Score remained stable at: {score}.")
                return
            else:
                _logger.info(
                    f"🎉 Optimization complete! "
                    f"Initial score: {initial_score}. "
                    f"Final score: {score}."
                )

    def _log_optimization_result(self, final_score: Optional[float], optimized_prompt: Prompt):
        if not active_run():
            return

        if final_score:
            log_metric(
                "final_eval_score",
                final_score,
            )
        log_param("optimized_prompt_uri", optimized_prompt.uri)
