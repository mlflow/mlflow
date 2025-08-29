import json
from typing import Any, ClassVar

from pydantic import PrivateAttr

from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.judges.utils import (
    RESPONSE_KEY_RATIONALE,
    RESPONSE_KEY_RESULT,
    format_prompt,
    get_default_model,
    invoke_judge_model,
)
from mlflow.genai.scorers.base import ScorerKind
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class InstructionsJudge(Judge):
    """
    A judge that evaluates traces based on user-provided instructions.

    This judge uses natural language instructions to guide evaluation,
    making it flexible for various assessment criteria.
    """

    # Private template variable names (read-only access via class methods)
    _TEMPLATE_VARIABLE_INPUTS: ClassVar[str] = "inputs"
    _TEMPLATE_VARIABLE_OUTPUTS: ClassVar[str] = "outputs"
    _TEMPLATE_VARIABLE_RESULT: ClassVar[str] = RESPONSE_KEY_RESULT
    _TEMPLATE_VARIABLE_RATIONALE: ClassVar[str] = RESPONSE_KEY_RATIONALE
    _TEMPLATE_VARIABLE_TRACE: ClassVar[str] = "trace"
    _TEMPLATE_VARIABLE_EXPECTATIONS: ClassVar[str] = "expectations"

    # Public class methods for read-only access to template variables
    @classmethod
    def get_template_variable_inputs(cls) -> str:
        """Get the template variable name for inputs."""
        return cls._TEMPLATE_VARIABLE_INPUTS

    @classmethod
    def get_template_variable_outputs(cls) -> str:
        """Get the template variable name for outputs."""
        return cls._TEMPLATE_VARIABLE_OUTPUTS

    @classmethod
    def get_template_variable_result(cls) -> str:
        """Get the template variable name for result."""
        return cls._TEMPLATE_VARIABLE_RESULT

    @classmethod
    def get_template_variable_rationale(cls) -> str:
        """Get the template variable name for rationale."""
        return cls._TEMPLATE_VARIABLE_RATIONALE

    @classmethod
    def get_template_variable_trace(cls) -> str:
        """Get the template variable name for trace."""
        return cls._TEMPLATE_VARIABLE_TRACE

    @classmethod
    def get_template_variable_expectations(cls) -> str:
        """Get the template variable name for expectations."""
        return cls._TEMPLATE_VARIABLE_EXPECTATIONS

    # Reserved template variables that cannot be used in instructions
    _RESERVED_INSTRUCTION_TEMPLATE_VARIABLES: ClassVar[list] = [
        _TEMPLATE_VARIABLE_INPUTS,
        _TEMPLATE_VARIABLE_OUTPUTS,
        _TEMPLATE_VARIABLE_TRACE,
        _TEMPLATE_VARIABLE_EXPECTATIONS,
    ]

    _instructions: str = PrivateAttr()
    _model: str = PrivateAttr()
    _instructions_prompt: PromptVersion = PrivateAttr()
    _custom_template_variables: set[str] = PrivateAttr()

    def __init__(self, name: str, instructions: str, model: str | None = None, **kwargs):
        """
        Initialize the InstructionsJudge.

        Args:
            name: The name of the judge
            instructions: Natural language instructions for evaluation
            model: The model identifier to use for evaluation (e.g., "openai:/gpt-4")
            kwargs: Additional configuration parameters
        """
        super().__init__(name=name, **kwargs)

        if not name or not isinstance(name, str):
            raise MlflowException(
                "name must be a non-empty string", error_code=INVALID_PARAMETER_VALUE
            )
        if not instructions or not isinstance(instructions, str):
            raise MlflowException(
                "instructions must be a non-empty string", error_code=INVALID_PARAMETER_VALUE
            )

        self._instructions = instructions
        self._model = model or get_default_model()

        # NB: We create a dummy PromptVersion here to leverage its existing template variable
        # extraction logic. This allows us to reuse the well-tested regex patterns and variable
        # parsing functionality without reimplementing it.
        self._instructions_prompt = PromptVersion(
            name=name,
            version=1,
            template=instructions,
        )

        self._custom_template_variables = self._instructions_prompt.variables - set(
            self._RESERVED_INSTRUCTION_TEMPLATE_VARIABLES
        )

        self._validate_model_format()
        self._validate_instructions_template()

    @property
    def instructions(self) -> str:
        """Get the instructions for this judge."""
        return self._instructions

    @property
    def model(self) -> str:
        """Get the model for this judge."""
        return self._model

    @property
    def template_variables(self) -> set[str]:
        """Get the template variables from the instructions."""
        return self._instructions_prompt.variables

    @property
    def description(self) -> str:
        """Get the description of this judge."""
        header = f"Instructions-based judge: {self.name}"
        return f"{header}\n\nInstructions:\n-------------\n\n{self._instructions}"

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
    ) -> Any:
        """
        Evaluate the provided data using the judge's instructions.

        Args:
            inputs: Input dictionary to evaluate. Cannot be used with 'trace'.
            outputs: Output dictionary to evaluate. Cannot be used with 'trace'.
            expectations: Expected outcomes or ground truth.

        Returns:
            Evaluation results

        """

        if inputs is not None or outputs is not None:
            self._validate_call_args_contain_template_fields(inputs, outputs, expectations)

            template_values = {}
            if inputs is not None:
                template_values.update(inputs)
            if outputs is not None:
                template_values.update(outputs)
            if expectations is not None:
                if self._TEMPLATE_VARIABLE_EXPECTATIONS in self.template_variables:
                    template_values[self._TEMPLATE_VARIABLE_EXPECTATIONS] = json.dumps(
                        expectations, default=str, indent=2
                    )
                template_values.update(expectations)

            formatted_prompt = format_prompt(self._instructions, **template_values)

            return invoke_judge_model(
                model_uri=self._model,
                prompt=formatted_prompt,
                assessment_name=self.name,
            )
        raise MlflowException(
            "Must specify 'inputs' or 'outputs' for evaluation.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    @property
    def kind(self) -> ScorerKind:
        """Return the kind of scorer this judge represents."""
        return ScorerKind.CLASS

    def _validate_model_format(self) -> None:
        """
        Validate that the model is in a valid format.

        Valid formats:
        - "databricks" for Databricks-native integration
        - "provider:/<model-name>" URI format (e.g., "openai:/gpt-4")
        - "endpoints:/<endpoint-name>" for Databricks model serving endpoints
        """
        if self._model == _DATABRICKS_DEFAULT_JUDGE_MODEL:
            return

        # Import here to avoid circular dependency and pandas requirement
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        _parse_model_uri(self._model)

    def _validate_instructions_template(self) -> None:
        """
        Validate that instructions contain at least one variable and don't contain
        a mix of trace and inputs/outputs/expectations variables.

        """
        template_vars = self.template_variables

        if not template_vars:
            raise MlflowException(
                "Instructions template must contain at least one variable (e.g., {{inputs}}, "
                "{{outputs}}, {{trace}}, or custom variables).",
                error_code=INVALID_PARAMETER_VALUE,
            )

        has_trace = self._TEMPLATE_VARIABLE_TRACE in template_vars
        has_inputs = self._TEMPLATE_VARIABLE_INPUTS in template_vars
        has_outputs = self._TEMPLATE_VARIABLE_OUTPUTS in template_vars

        if has_trace:
            if self._custom_template_variables:
                raise MlflowException(
                    "When submitting a 'trace' variable, no other variables are permitted. "
                    f"found: {self._custom_template_variables}. A submitted trace contains "
                    "the complete context for evaluation and should not be mixed with "
                    "other variables.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if has_inputs or has_outputs:
                raise MlflowException(
                    "Instructions template cannot contain both 'trace' and 'inputs'/'outputs' "
                    "variables. Use either 'trace' for trace-based evaluation or "
                    "'inputs'/'outputs' for field-based evaluation.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if self._model == _DATABRICKS_DEFAULT_JUDGE_MODEL:
                raise MlflowException(
                    "Model cannot be 'databricks' when using 'trace' variable in "
                    "the instructions template. Specify a different model "
                    "(e.g., model='openai:/gpt-4o').",
                    error_code=INVALID_PARAMETER_VALUE,
                )

    def _validate_call_args_contain_template_fields(
        self,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
    ) -> None:
        """
        Validate that required template variables are present in inputs, outputs, or expectations.

        Args:
            inputs: Input dictionary to validate
            outputs: Output dictionary to validate
            expectations: Expectations dictionary to validate

        Raises:
            MlflowException: If any required template variable is missing
        """
        if not self._custom_template_variables:
            return

        input_keys = set(inputs.keys()) if inputs is not None else set()
        output_keys = set(outputs.keys()) if outputs is not None else set()
        expectation_keys = set(expectations.keys()) if expectations is not None else set()
        available_vars = input_keys | output_keys | expectation_keys

        missing_vars = self._custom_template_variables - available_vars

        if missing_vars:
            raise MlflowException(
                f"Required template variables {missing_vars} are missing from inputs, outputs, "
                "and expectations. Each variable must be present in at least one of them.",
                error_code=INVALID_PARAMETER_VALUE,
            )


__all__ = ["InstructionsJudge"]
