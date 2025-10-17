import json
import logging
from dataclasses import asdict
from typing import Any

from pydantic import PrivateAttr

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge, JudgeField
from mlflow.genai.judges.instructions_judge.constants import (
    INSTRUCTIONS_JUDGE_SYSTEM_PROMPT,
    INSTRUCTIONS_JUDGE_TRACE_PROMPT_TEMPLATE,
)
from mlflow.genai.judges.utils import (
    add_output_format_instructions,
    format_prompt,
    get_default_model,
    invoke_judge_model,
    validate_judge_model,
)
from mlflow.genai.scorers.base import (
    _SERIALIZATION_VERSION,
    ScorerKind,
    SerializedScorer,
)
from mlflow.genai.utils.trace_utils import (
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)
from mlflow.prompt.constants import PROMPT_TEMPLATE_VARIABLE_PATTERN, PROMPT_TEXT_DISPLAY_LIMIT
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


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
    _ordered_template_variables: list[str] = PrivateAttr()

    def __init__(self, name: str, instructions: str, model: str | None = None, **kwargs):
        """
        Initialize the InstructionsJudge.

        Args:
            name: The name of the judge
            instructions: Natural language instructions for evaluation
            model: The model identifier to use for evaluation (e.g., "openai:/gpt-4")
            kwargs: Additional configuration parameters
        """
        # TODO: Allow aggregations once we support boolean/numeric judge outputs
        super().__init__(name=name, aggregations=[], **kwargs)

        if not name or not isinstance(name, str):
            raise MlflowException(
                "name must be a non-empty string", error_code=INVALID_PARAMETER_VALUE
            )
        if not instructions or not isinstance(instructions, str):
            raise MlflowException(
                "instructions must be a non-empty string",
                error_code=INVALID_PARAMETER_VALUE,
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

        # Extract variables in the order they appear in the template
        all_vars = PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(instructions)
        seen = set()
        self._ordered_template_variables = []
        for var in all_vars:
            if var not in seen and var in self._RESERVED_INSTRUCTION_TEMPLATE_VARIABLES:
                seen.add(var)
                self._ordered_template_variables.append(var)

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
            fields.append(JudgeField(name="inputs", description="Input data to evaluate"))

        if self._TEMPLATE_VARIABLE_OUTPUTS in self.template_variables:
            fields.append(JudgeField(name="outputs", description="Output data to evaluate"))

        if self._TEMPLATE_VARIABLE_EXPECTATIONS in self.template_variables:
            fields.append(
                JudgeField(name="expectations", description="Expected outcomes or ground truth")
            )

        if self._TEMPLATE_VARIABLE_TRACE in self.template_variables:
            fields.append(JudgeField(name="trace", description="Trace to evaluate"))

        return fields

    def _validate_parameter_types(
        self, expectations: dict[str, Any] | None, trace: Trace | None
    ) -> None:
        """Validate that parameters have correct types."""
        if expectations is not None and not isinstance(expectations, dict):
            raise MlflowException(
                f"'expectations' must be a dictionary, got {type(expectations).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if trace is not None and not isinstance(trace, Trace):
            raise MlflowException(
                f"'trace' must be a Trace object, got {type(trace).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _check_required_parameters(
        self,
        inputs: Any | None,
        outputs: Any | None,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
    ) -> None:
        """Check that all required parameters are provided."""
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

    def _warn_unused_parameters(
        self,
        inputs: Any | None,
        outputs: Any | None,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
    ) -> None:
        """Warn about parameters that were provided but aren't used."""
        # Don't warn about unused parameters when using trace-based evaluation
        # since these parameters may be extracted from the trace for context
        if self._TEMPLATE_VARIABLE_TRACE in self.template_variables:
            return

        unused_params = []
        if inputs is not None and self._TEMPLATE_VARIABLE_INPUTS not in self.template_variables:
            unused_params.append("inputs")
        if outputs is not None and self._TEMPLATE_VARIABLE_OUTPUTS not in self.template_variables:
            unused_params.append("outputs")
        if (
            expectations is not None
            and self._TEMPLATE_VARIABLE_EXPECTATIONS not in self.template_variables
        ):
            unused_params.append("expectations")

        if unused_params:
            unused_str = "', '".join(unused_params)
            _logger.warning(
                f"The following parameters were provided but are not used by this judge's "
                f"instructions: '{unused_str}'. The judge only uses template variables that "
                f"appear in the instructions: {self.template_variables}"
            )

    def _build_system_message(self, is_trace_based: bool) -> str:
        """Build the system message based on whether this is trace-based or field-based."""
        output_fields = self.get_output_fields()

        if is_trace_based:
            evaluation_rating_fields = "\n".join(
                [f"- {field.name}: {field.description}" for field in output_fields]
            )
            return INSTRUCTIONS_JUDGE_TRACE_PROMPT_TEMPLATE.format(
                evaluation_rating_fields=evaluation_rating_fields,
                instructions=self._instructions,
            )
        else:
            base_prompt = format_prompt(
                INSTRUCTIONS_JUDGE_SYSTEM_PROMPT, instructions=self._instructions
            )
            return add_output_format_instructions(base_prompt, output_fields=output_fields)

    def _build_user_message(
        self,
        inputs: Any | None,
        outputs: Any | None,
        expectations: dict[str, Any] | None,
        is_trace_based: bool,
    ) -> str:
        """Build the user message with field values."""
        template_values = self._build_template_values(inputs, outputs, expectations)

        field_vars = [
            var for var in self._ordered_template_variables if var != self._TEMPLATE_VARIABLE_TRACE
        ]

        # Build user message parts in order
        user_message_parts = []
        for var_name in field_vars:
            if var_name in template_values:
                user_message_parts.append(f"{var_name}: {template_values[var_name]}")

        # Some model providers (like Anthropic) require a user message
        # (i.e. a single-message chat history with role 'system' is not supported),
        # *and* they require the message to have non-empty content (empty string is not allowed)
        return (
            "\n".join(user_message_parts)
            if user_message_parts
            else "Follow the instructions from the first message"
        )

    def _build_template_values(
        self, inputs: Any | None, outputs: Any | None, expectations: dict[str, Any] | None
    ) -> dict[str, str]:
        """Build dictionary of template variable values."""
        template_values = {}

        if inputs is not None and self._TEMPLATE_VARIABLE_INPUTS in self.template_variables:
            template_values[self._TEMPLATE_VARIABLE_INPUTS] = self._safe_json_dumps(inputs)

        if outputs is not None and self._TEMPLATE_VARIABLE_OUTPUTS in self.template_variables:
            template_values[self._TEMPLATE_VARIABLE_OUTPUTS] = self._safe_json_dumps(outputs)

        if (
            expectations is not None
            and self._TEMPLATE_VARIABLE_EXPECTATIONS in self.template_variables
        ):
            template_values[self._TEMPLATE_VARIABLE_EXPECTATIONS] = self._safe_json_dumps(
                expectations
            )

        return template_values

    def _safe_json_dumps(self, value: Any) -> str:
        """Safely serialize a value to JSON, falling back to str() if JSON serialization fails."""
        try:
            return json.dumps(value, default=str, indent=2)
        except Exception:
            return str(value)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate the provided data using the judge's instructions.

        Args:
            inputs: Input data to evaluate. If not provided and a trace is given,
                will be extracted from the trace's root span inputs.
            outputs: Output data to evaluate. If not provided and a trace is given,
                will be extracted from the trace's root span outputs.
            expectations: Expected outcomes or ground truth. If not provided and a trace is given,
                will be extracted from the trace's expectation assessments.
            trace: Trace object for evaluation. When the template uses {{ inputs }}, {{ outputs }},
                or {{ expectations }}, the values will be extracted from the trace.

        Returns:
            Evaluation results

        **Note on Trace Behavior**:
        - If template uses {{ trace }}: The trace metadata is used by an agent-based judge that uses
          tools to fetch aspects of the trace's span data. If inputs/outputs/expectations are also
          provided, they can augment the agent's context if the template has corresponding
          placeholders ({{ inputs }}/{{ outputs }}/{{ expectations }}). The agent will still use
          tools to fetch span data but will have this additional context in the user prompt.
        - If template uses {{ inputs }}/{{ outputs }}/{{ expectations }} without {{ trace }}:
          Values are extracted from the trace, if specified, as follows:
          - inputs/outputs: From the trace's root span
          - expectations: From the trace's human-set expectation assessments (ground truth only)

        """
        self._validate_parameter_types(expectations, trace)

        original_inputs = inputs
        original_outputs = outputs
        original_expectations = expectations

        if trace is not None:
            inputs = resolve_inputs_from_trace(
                inputs,
                trace,
                extract_if_none=self._TEMPLATE_VARIABLE_INPUTS in self.template_variables,
            )
            outputs = resolve_outputs_from_trace(
                outputs,
                trace,
                extract_if_none=self._TEMPLATE_VARIABLE_OUTPUTS in self.template_variables,
            )
            expectations = resolve_expectations_from_trace(
                expectations,
                trace,
                extract_if_none=self._TEMPLATE_VARIABLE_EXPECTATIONS in self.template_variables,
            )

        self._check_required_parameters(inputs, outputs, expectations, trace)
        self._warn_unused_parameters(
            original_inputs, original_outputs, original_expectations, trace
        )

        is_trace_based = self._TEMPLATE_VARIABLE_TRACE in self.template_variables

        system_content = self._build_system_message(is_trace_based)
        user_content = self._build_user_message(inputs, outputs, expectations, is_trace_based)

        from mlflow.types.llm import ChatMessage

        messages = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]

        return invoke_judge_model(
            model_uri=self._model,
            prompt=messages,
            assessment_name=self.name,
            trace=trace if is_trace_based else None,
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
        validate_judge_model(self._model)

    def _validate_instructions_template(self) -> None:
        """
        Validate that instructions contain at least one variable.

        """
        template_vars = self.template_variables

        if not template_vars:
            raise MlflowException(
                "Instructions template must contain at least one variable (e.g., {{ inputs }}, "
                "{{ outputs }}, {{ trace }}, or {{ expectations }}).",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def __repr__(self) -> str:
        """Return string representation of the InstructionsJudge."""
        instructions_preview = (
            self._instructions[:PROMPT_TEXT_DISPLAY_LIMIT] + "..."
            if len(self._instructions) > PROMPT_TEXT_DISPLAY_LIMIT
            else self._instructions
        )
        return (
            f"InstructionsJudge(name='{self.name}', model='{self._model}', "
            f"instructions='{instructions_preview}', "
            f"template_variables={sorted(self.template_variables)})"
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
