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
from mlflow.genai.scorers import Scorer
from mlflow.tracking._model_registry.fluent import register_prompt

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

        _logger.info(
            f"Started optimizing prompt {prompt.uri}. "
            "Please wait as this process typically takes several minutes, "
            "but can take longer with large datasets..."
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
        with dspy.context(lm=lm, adapter=adapter):
            with self._maybe_suppress_stdout_stderr():
                optimized_program = optimizer.compile(
                    program,
                    trainset=train_data,
                    valset=eval_data,
                    num_trials=self._get_num_trials(num_candidates),
                    minibatch_size=self._get_minibatch_size(train_data, eval_data),
                    requires_permission_to_run=False,
                )

        template = self._format_optimized_prompt(
            adapter=adapter,
            program=optimized_program,
            input_fields=input_fields,
        )

        self._display_optimization_result(optimized_program)

        return register_prompt(
            name=prompt.name,
            template=template,
            version_metadata={
                "overall_eval_score": getattr(optimized_program, "score", None),
            },
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

    def _format_optimized_prompt(
        self, adapter: "dspy.Adapter", program: "dspy.Predict", input_fields: dict[str, type]
    ) -> str:
        messages = adapter.format(
            signature=program.signature,
            demos=program.demos,
            inputs={key: "{{" + key + "}}" for key in input_fields.keys()},
        )

        return "\n\n".join(
            [
                f"<{message['role']}>\n{message['content']}\n</{message['role']}>"
                for message in messages
            ]
        )

    def _validate_input_fields(self, input_fields: dict[str, type], prompt: Prompt) -> None:
        if missing_fields := set(prompt.variables) - set(input_fields.keys()):
            raise MlflowException(
                "The following variables of the prompt are missing from "
                f"the dataset: {missing_fields}"
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
        if hasattr(program, "score") and hasattr(program, "trial_logs"):
            initial_score = program.trial_logs[1]["full_eval_score"]
            _logger.info(
                "Prompt optimization completed. Evaluation score changed "
                f"from {initial_score} to {program.score}."
            )
