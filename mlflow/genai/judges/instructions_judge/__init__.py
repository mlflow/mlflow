import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr

import mlflow
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge, JudgeField
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.judges.instructions_judge.constants import (
    INSTRUCTIONS_JUDGE_SYSTEM_PROMPT,
    INSTRUCTIONS_JUDGE_TRACE_PROMPT_TEMPLATE,
)
from mlflow.genai.judges.instructions_judge.utils import extract_evaluation_fields_from_trace
from mlflow.genai.judges.utils import (
    add_output_format_instructions,
    format_prompt,
    get_default_model,
    invoke_judge_model,
)
from mlflow.genai.scorers.base import _SERIALIZATION_VERSION, ScorerKind, SerializedScorer
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from mlflow.types.llm import ChatMessage  # noqa: F401


@experimental(version="3.4.0")
class InstructionsJudge(Judge):
    """
    A judge that evaluates traces based on user-provided instructions.

    This judge uses natural language instructions to guide evaluation,
    making it flexible for various assessment criteria.
    """

    _TEMPLATE_VARIABLE_INPUTS = "inputs"
    _TEMPLATE_VARIABLE_OUTPUTS = "outputs"
    _TEMPLATE_VARIABLE_TRACE = "trace"
    _TEMPLATE_VARIABLE_EXPECTATIONS = "expectations"
    _RESERVED_INSTRUCTION_TEMPLATE_VARIABLES = [
        _TEMPLATE_VARIABLE_INPUTS,
        _TEMPLATE_VARIABLE_OUTPUTS,
        _TEMPLATE_VARIABLE_TRACE,
        _TEMPLATE_VARIABLE_EXPECTATIONS,
    ]

    _instructions: str = PrivateAttr()
    _model: str = PrivateAttr()
    _instructions_prompt: PromptVersion = PrivateAttr()

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

        # Reject any custom template variables
        custom_template_variables = self._instructions_prompt.variables - set(
            self._RESERVED_INSTRUCTION_TEMPLATE_VARIABLES
        )
        if custom_template_variables:
            allowed_vars = ", ".join(self._RESERVED_INSTRUCTION_TEMPLATE_VARIABLES)
            raise MlflowException(
                f"Instructions template contains unsupported variables: "
                f"{custom_template_variables}. "
                f"Only the following variables are allowed: {allowed_vars}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        self._validate_model_format()
        self._validate_instructions_template()

    @property
    def model(self) -> str:
        """Get the model for this judge."""
        return self._model

    @property
    def template_variables(self) -> set[str]:
        """Get the template variables from the instructions."""
        return self._instructions_prompt.variables

    @property
    def instructions(self) -> str:
        """Get the instructions of this judge."""
        return self._instructions

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for this judge based on the template variables.

        Returns:
            List of JudgeField objects defining the input fields.
        """
        fields = []

        if self._TEMPLATE_VARIABLE_INPUTS in self.template_variables:
            fields.append(JudgeField(name="inputs", description="Input dictionary to evaluate"))

        if self._TEMPLATE_VARIABLE_OUTPUTS in self.template_variables:
            fields.append(JudgeField(name="outputs", description="Output dictionary to evaluate"))

        if self._TEMPLATE_VARIABLE_EXPECTATIONS in self.template_variables:
            fields.append(
                JudgeField(name="expectations", description="Expected outcomes or ground truth")
            )

        if self._TEMPLATE_VARIABLE_TRACE in self.template_variables:
            fields.append(JudgeField(name="trace", description="Trace to evaluate"))

        return fields

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Any:
        """
        Evaluate the provided data using the judge's instructions.

        Args:
            inputs: Input dictionary to evaluate. If not provided and a trace is given,
                will be extracted from the trace's root span inputs.
            outputs: Output dictionary to evaluate. If not provided and a trace is given,
                will be extracted from the trace's root span outputs.
            expectations: Expected outcomes or ground truth. If not provided and a trace is given,
                will be extracted from the trace's expectation assessments.
            trace: Trace object for evaluation. When the template uses {{ inputs }}, {{ outputs }},
                or {{ expectations }} (not {{ trace }}), the values will be extracted from the
                trace.

        Returns:
            Evaluation results

        **Note on Trace Behavior**:
        - If template uses {{ trace }}: The trace is passed to an agent for evaluation.
        - If template uses {{ inputs }}/{{ outputs }}/{{ expectations }}: Values are extracted from:
          - inputs/outputs: From the trace's root span
          - expectations: From the trace's human-set expectation assessments (ground truth only)

        """
        if inputs is not None and not isinstance(inputs, dict):
            raise MlflowException(
                f"'inputs' must be a dictionary, got {type(inputs).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if outputs is not None and not isinstance(outputs, dict):
            raise MlflowException(
                f"'outputs' must be a dictionary, got {type(outputs).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if expectations is not None and not isinstance(expectations, dict):
            raise MlflowException(
                f"'expectations' must be a dictionary, got {type(expectations).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if trace is not None and not isinstance(trace, Trace):
            raise MlflowException(
                f"'trace' must be a Trace instance, got {type(trace).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Extract fields from trace if needed (when not using {{ trace }} template)
        if trace is not None and self._TEMPLATE_VARIABLE_TRACE not in self.template_variables:
            extracted = extract_evaluation_fields_from_trace(trace)

            # Use extracted values if not already provided
            if inputs is None and self._TEMPLATE_VARIABLE_INPUTS in self.template_variables:
                inputs = extracted.inputs
            if outputs is None and self._TEMPLATE_VARIABLE_OUTPUTS in self.template_variables:
                outputs = extracted.outputs
            if (
                expectations is None
                and self._TEMPLATE_VARIABLE_EXPECTATIONS in self.template_variables
            ):
                expectations = extracted.expectations

        # Check if the input arguments match the template variables
        missing_params = []

        if self._TEMPLATE_VARIABLE_INPUTS in self.template_variables and inputs is None:
            missing_params.append("inputs")
        if self._TEMPLATE_VARIABLE_OUTPUTS in self.template_variables and outputs is None:
            missing_params.append("outputs")
        if self._TEMPLATE_VARIABLE_EXPECTATIONS in self.template_variables and expectations is None:
            missing_params.append("expectations")
        if self._TEMPLATE_VARIABLE_TRACE in self.template_variables and trace is None:
            missing_params.append("trace")

        if missing_params:
            missing_str = "', '".join(missing_params)
            raise MlflowException(
                f"Must specify '{missing_str}' - required by template variables in instructions.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Determine evaluation mode based on template variables
        is_trace_based = self._TEMPLATE_VARIABLE_TRACE in self.template_variables

        # Handle field-based evaluation (inputs/outputs)
        if not is_trace_based:
            # Build the system message with instructions template and output format
            system_content = format_prompt(
                INSTRUCTIONS_JUDGE_SYSTEM_PROMPT, instructions=self._instructions
            )
            system_content = add_output_format_instructions(
                system_content, output_fields=self.get_output_fields()
            )

            # Build the user message with variable substitutions
            template_values = {}
            if inputs is not None and self._TEMPLATE_VARIABLE_INPUTS in self.template_variables:
                template_values[self._TEMPLATE_VARIABLE_INPUTS] = json.dumps(
                    inputs, default=str, indent=2
                )
            if outputs is not None and self._TEMPLATE_VARIABLE_OUTPUTS in self.template_variables:
                template_values[self._TEMPLATE_VARIABLE_OUTPUTS] = json.dumps(
                    outputs, default=str, indent=2
                )
            if (
                expectations is not None
                and self._TEMPLATE_VARIABLE_EXPECTATIONS in self.template_variables
            ):
                template_values[self._TEMPLATE_VARIABLE_EXPECTATIONS] = json.dumps(
                    expectations, default=str, indent=2
                )

            # Create user content with the actual values for each variable
            user_message_parts = []
            for var_name in sorted(self.template_variables):
                if var_name in template_values:
                    value = template_values[var_name]
                    formatted_value = (
                        value if isinstance(value, str) else json.dumps(value, default=str)
                    )
                    user_message_parts.append(f"{var_name}: {formatted_value}")

            user_content = "\n".join(user_message_parts)

            # Create messages list using ChatMessage objects
            from mlflow.types.llm import ChatMessage

            messages = [
                ChatMessage(role="system", content=system_content),
                ChatMessage(role="user", content=user_content),
            ]

            return invoke_judge_model(
                model_uri=self._model,
                prompt=messages,
                assessment_name=self.name,
            )

        # Handle trace-based evaluation
        else:  # is_trace_based
            output_fields = self.get_output_fields()
            evaluation_rating_fields = "\n".join(
                [f"- {field.name}: {field.description}" for field in output_fields]
            )

            base_prompt = INSTRUCTIONS_JUDGE_TRACE_PROMPT_TEMPLATE.format(
                evaluation_rating_fields=evaluation_rating_fields, instructions=self._instructions
            )
            # Add structured output format instructions
            augmented_prompt = add_output_format_instructions(
                base_prompt, output_fields=output_fields
            )
            return invoke_judge_model(
                model_uri=self._model,
                prompt=augmented_prompt,
                assessment_name=self.name,
                trace=trace,
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
                "Instructions template must contain at least one variable (e.g., {{ inputs }}, "
                "{{ outputs }}, {{ trace }}, or {{ expectations }}).",
                error_code=INVALID_PARAMETER_VALUE,
            )

        has_trace = self._TEMPLATE_VARIABLE_TRACE in template_vars
        has_inputs = self._TEMPLATE_VARIABLE_INPUTS in template_vars
        has_outputs = self._TEMPLATE_VARIABLE_OUTPUTS in template_vars
        has_expectations = self._TEMPLATE_VARIABLE_EXPECTATIONS in template_vars

        if has_trace:
            # TODO: Allow expectations variable with trace in followup implementation
            if has_expectations:
                raise MlflowException(
                    "When submitting a 'trace' variable, expectations are not yet supported. "
                    "This will be implemented in a future release.",
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

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to serialize as a SerializedScorer."""
        serialized_scorer = SerializedScorer(
            name=self.name,
            aggregations=self.aggregations,
            mlflow_version=mlflow.__version__,
            serialization_version=_SERIALIZATION_VERSION,
            instructions_judge_pydantic_data={
                "instructions": self._instructions,
                "model": self._model,
            },
            builtin_scorer_class=None,
            builtin_scorer_pydantic_data=None,
            call_source=None,
            call_signature=None,
            original_func_name=None,
        )
        return asdict(serialized_scorer)


__all__ = ["InstructionsJudge"]
