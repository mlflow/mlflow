import importlib
import importlib.metadata
import inspect
import logging
import math
from typing import TYPE_CHECKING, Callable, Optional, Union, Any

from packaging.version import Version

from mlflow.entities.model_registry import Prompt
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.types import OBJECTIVE_FN, LLMParam, OptimizerParam
from mlflow.genai.optimize.util import infer_type_from_value
from mlflow.genai.scorers import Scorer

if TYPE_CHECKING:
    import dspy
    import pandas as pd

_logger = logging.getLogger(__name__)


class _BaseOptimizer:
    def __init__(self, optimizer_params: OptimizerParam):
        self.optimizer_params = optimizer_params

    def optimize(
        self,
        prompt: Prompt,
        agent_lm: LLMParam,
        train_data: "pd.DataFrame",
        scorers: list[Scorer],
        objective: Optional[OBJECTIVE_FN] = None,
        eval_data: Optional["pd.DataFrame"] = None,
    ) -> Union[str, list[dict[str, Any]]]:
        raise NotImplementedError("Method not implemented")


class _DSPyOptimizer(_BaseOptimizer):
    def __init__(self, optimizer_params: OptimizerParam):
        super().__init__(optimizer_params)

        if not importlib.util.find_spec("dspy"):
            raise ImportError("dspy is not installed. Please install it with `pip install dspy`.")

        dspy_version = importlib.metadata.version("dspy")
        if Version(dspy_version) < Version("2.6.0"):
            raise MlflowException("dspy version is too old. Please upgrade to version >= 2.6.0")

    def _get_input_fields(self, train_data: "pd.DataFrame") -> list[dict[str, type]]:
        if "request" in train_data.columns:
            sample_input = train_data["request"].values[0]
            return {k: infer_type_from_value(v) for k, v in sample_input.items()}
        return {}

    def _get_output_fields(self, train_data: "pd.DataFrame") -> list[dict[str, type]]:
        if "expectations" in train_data.columns:
            sample_output = train_data["expectations"].values[0]
            return {k: infer_type_from_value(v) for k, v in sample_output.items()}
        return {}

    def _convert_to_dspy_dataset(self, data: "pd.DataFrame") -> list["dspy.Example"]:
        import dspy

        examples = []
        for _, row in data.iterrows():
            expectations = row["expectations"] if "expectations" in row else {}
            examples.append(
                dspy.Example(**row["request"], **expectations).with_inputs(*row["request"].keys())
            )
        return examples

    def _convert_to_dspy_metric(
        self,
        input_fields: dict[str, type],
        output_fields: dict[str, type],
        scorers: list[Scorer],
        objective: Optional[OBJECTIVE_FN] = None,
    ) -> Callable[["dspy.Example"], float]:
        def metric(example: "dspy.Example", pred: "dspy.Example", trace=None) -> float:
            scores = {}
            inputs = {key: example.get(key) for key in input_fields.keys()}
            expectations = {key: example.get(key) for key in output_fields.keys()}
            outputs = {key: pred.get(key) for key in output_fields.keys()}

            for scorer in scorers:
                kwargs = {"inputs": inputs, "outputs": outputs, "expectations": expectations}
                signature = inspect.signature(scorer)
                kwargs = {
                    key: value for key, value in kwargs.items() if key in signature.parameters
                }
                scores[scorer.name] = scorer(**kwargs)
            if objective is not None:
                return objective(scores)
            elif all(isinstance(score, (int, float, bool)) for score in scores.values()):
                # Use total score by default if no objective is provided
                return sum(scores.values())
            else:
                raise ValueError(
                    "Non numerical score value found. "
                    "Please provide `objective` to use non-numerical scores."
                )

        return metric


class _DSPyMIPROv2Optimizer(_DSPyOptimizer):
    def __init__(self, optimizer_params: OptimizerParam):
        super().__init__(optimizer_params)

    def optimize(
        self,
        prompt: Prompt,
        agent_lm: LLMParam,
        train_data: "pd.DataFrame",
        scorers: list[Scorer],
        objective: Optional[OBJECTIVE_FN] = None,
        eval_data: Optional["pd.DataFrame"] = None,
    ) -> Union[str, list[dict[str, Any]]]:
        import dspy

        _logger.info(f"Start optimizing prompt {prompt.uri}...")

        input_fields = self._get_input_fields(train_data)
        output_fields = self._get_output_fields(train_data)
        signature = dspy.make_signature(
            {
                **{key: (_type, dspy.InputField()) for key, _type in input_fields.items()},
                **{key: (_type, dspy.OutputField()) for key, _type in output_fields.items()},
            },
            prompt.template,  # TODO: Extract only instructions from the existing prompt template
        )

        # Define main student program
        program = dspy.Predict(signature)

        train_data = self._convert_to_dspy_dataset(train_data)
        eval_data = self._convert_to_dspy_dataset(eval_data) if eval_data is not None else None

        teacher_settings = {}
        if self.optimizer_params.optimizer_llm:
            teacher_lm = dspy.LM(
                model=self.optimizer_params.optimizer_llm.model_name,
                temperature=self.optimizer_params.optimizer_llm.temperature,
                api_base=self.optimizer_params.optimizer_llm.base_uri,
            )
            teacher_settings["lm"] = teacher_lm

        num_candidates = self.optimizer_params.num_instruction_candidates
        optimizer = dspy.MIPROv2(
            metric=self._convert_to_dspy_metric(input_fields, output_fields, scorers, objective),
            max_bootstrapped_demos=self.optimizer_params.max_few_show_examples,
            num_candidates=num_candidates,
            num_threads=self.optimizer_params.num_threads,
            teacher_settings=teacher_settings,
            auto=None,
        )

        adapter = dspy.JSONAdapter()
        lm = dspy.LM(
            model=agent_lm.model_name,
            temperature=agent_lm.temperature,
            api_base=agent_lm.base_uri,
        )
        with dspy.context(lm=lm, adapter=adapter):
            dspy_logger = logging.getLogger("dspy")
            original_level = dspy_logger.level
            dspy_logger.setLevel(logging.ERROR)
            try:
                optimized_program = optimizer.compile(
                    program,
                    trainset=train_data,
                    valset=eval_data,
                    num_trials=self._get_num_trials(num_candidates),
                    minibatch_size=self._get_minibatch_size(train_data, eval_data),
                    requires_permission_to_run=False,
                )
            finally:
                # Restore original logging level
                dspy_logger.setLevel(original_level)

            return self._format_optimized_prompt(
                adapter=adapter,
                program=optimized_program,
                input_fields=input_fields,
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
