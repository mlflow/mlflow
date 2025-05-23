from typing import TYPE_CHECKING, Callable, Optional, Union

from mlflow.entities.model_registry import Prompt
from mlflow.genai.optimize.types import OBJECTIVE_FN, LLMParam, OptimizerParam
from mlflow.genai.scorers import Scorer
from mlflow.types.chat import ChatMessage

if TYPE_CHECKING:
    import dspy
    import pandas as pd


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
    ) -> Union[str, list[ChatMessage]]:
        raise NotImplementedError("Method not implemented")


class _DSPyOptimizer(_BaseOptimizer):
    def __init__(self, optimizer_params: OptimizerParam):
        super().__init__(optimizer_params)

        try:
            import dspy  # noqa: F401
        except ImportError:
            raise ImportError("dspy is not installed. Please install it with `pip install dspy`.")

    def _get_input_fields(self, train_data: "pd.DataFrame") -> list[dict[str, type]]:
        input_fields = {}
        for item in train_data["request"].values:
            for key, value in item.items():
                # TODO: handle nested and union types
                input_fields[key] = type(value)
        return input_fields

    def _get_output_fields(self, train_data: "pd.DataFrame") -> list[str]:
        input_fields = {}
        for item in train_data["predictions"].values:
            for key, value in item.items():
                # TODO: handle nested and union types
                input_fields[key] = type(value)
        return input_fields

    def _convert_to_dspy_dataset(self, data: "pd.DataFrame") -> list["dspy.Example"]:
        examples = []
        for _, row in data.iterrows():
            examples.append(dspy.Example(**row["request"], **row["predictions"]))
        return examples

    def _convert_to_dspy_metric(
        self,
        input_fields: dict[str, type],
        output_fields: dict[str, type],
        scorers: list[Scorer],
        objective: Optional[OBJECTIVE_FN] = None,
    ) -> Callable[["dspy.Example"], float]:
        def metric(example: "dspy.Example") -> float:
            scores = {}
            inputs = {key: example[key] for key in input_fields.keys()}
            outputs = {key: example[key] for key in output_fields.keys()}
            for scorer in scorers:
                scores[scorer.name] = scorer(inputs=inputs, outputs=outputs)
            if objective:
                return objective(scores)
            elif all(isinstance(score, (int, float, bool)) for score in scores.values()):
                # Use total score by default if no objective is provided
                return sum(scores.values())
            else:
                raise ValueError(
                    "Non Numerical Score value found. "
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
    ) -> Union[str, list[ChatMessage]]:
        import dspy

        input_fields = self._get_input_fields(train_data)
        output_fields = self._get_output_fields(train_data)
        signature = dspy.make_signature(
            {
                **{key: (_type, dspy.InputField()) for key, _type in input_fields},
                **{key: (_type, dspy.OutputField()) for key, _type in output_fields},
            },
            prompt.template,
        )

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

        optimizer = dspy.MIPROv2(
            metric=self._convert_to_dspy_metric(input_fields, output_fields, scorers, objective),
            max_bootstrapped_demos=self.optimizer_params.max_few_show_examples,
            num_candidates=self.optimizer_params.num_instruction_candidates,
            num_threads=self.optimizer_params.num_threads,
            teacher_settings=teacher_settings,
        )

        adapter = dspy.JsonAdapter()
        lm = dspy.LM(
            model=agent_lm.model_name,
            temperature=agent_lm.temperature,
            api_base=agent_lm.base_uri,
        )
        with dspy.context(lm=lm, adapter=adapter):
            optimized_program = optimizer.compile(
                program,
                trainset=train_data,
                valset=eval_data,
                teacher_settings=teacher_settings,
            )

        return adapter.format(
            signature=adapter.signature,
            demos=optimized_program.demos,
            inputs={key: "{{" + key + "}}" for key in input_fields.keys()},
        )
