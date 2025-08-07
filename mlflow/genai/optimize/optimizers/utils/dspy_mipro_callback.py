from collections import defaultdict
from typing import TYPE_CHECKING, Any

from dspy import Prediction
from dspy.utils.callback import BaseCallback

import mlflow
from mlflow.genai.optimize.optimizers.utils.dspy_optimizer_utils import format_dspy_prompt

_FULL_EVAL_NAME = "eval_full"

if TYPE_CHECKING:
    import dspy


class _DSPyMIPROv2Callback(BaseCallback):
    def __init__(
        self,
        prompt_name: str,
        input_fields: dict[str, type],
        convert_to_single_text: bool,
    ):
        self.prompt_name = prompt_name
        self.input_fields = input_fields
        # call_id: (key, step, program)
        self._call_id_to_values: dict[str, tuple[str, int, "dspy.Predict"]] = {}
        self._evaluation_counter = defaultdict(int)
        self._best_score = None
        self.convert_to_single_text = convert_to_single_text

    def on_evaluate_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        key = inputs.get("callback_metadata", {}).get("metric_key", "eval")
        step = self._evaluation_counter[key]
        self._evaluation_counter[key] += 1
        self._call_id_to_values[call_id] = (key, step, inputs.get("program"))

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any,
        exception: Exception | None = None,
    ):
        if exception:
            return

        if call_id not in self._call_id_to_values:
            return
        key, step, program = self._call_id_to_values.pop(call_id)

        # Only log the full evaluation result
        if key != _FULL_EVAL_NAME:
            return

        if isinstance(outputs, float):
            score = outputs
        elif isinstance(outputs, tuple):
            score = outputs[0]
        elif isinstance(outputs, Prediction):
            score = float(outputs)
        else:
            return

        if self._best_score is None:
            # This is the first evaluation with initial prompt,
            # we don't register this prompt
            self._best_score = score
        elif score > self._best_score:
            # When best score is updated, register the new prompt
            self._best_score = score
            template = format_dspy_prompt(program, self.convert_to_single_text)
            mlflow.genai.register_prompt(
                name=self.prompt_name,
                template=template,
                tags={"overall_eval_score": score, key: step},
            )

        if mlflow.active_run() is not None:
            mlflow.log_metric(
                key,
                score,
                step=step,
            )
