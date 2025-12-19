import json
import logging
from dataclasses import asdict
from typing import Any, Literal, get_origin

import pydantic
from pydantic import PrivateAttr

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge, JudgeField
from mlflow.genai.judges.constants import (
    _RATIONALE_FIELD_DESCRIPTION,
    _RESULT_FIELD_DESCRIPTION,
    USE_CASE_AGENTIC_JUDGE,
)
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
    resolve_conversation_from_session,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)
from mlflow.prompt.constants import PROMPT_TEMPLATE_VARIABLE_PATTERN, PROMPT_TEXT_DISPLAY_LIMIT
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracing.constant import TraceMetadataKey

_logger = logging.getLogger(__name__)


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
    _TEMPLATE_VARIABLE_CONVERSATION = "conversation"
    _RESERVED_INSTRUCTION_TEMPLATE_VARIABLES = [
        _TEMPLATE_VARIABLE_INPUTS,
        _TEMPLATE_VARIABLE_OUTPUTS,
        _TEMPLATE_VARIABLE_TRACE,
        _TEMPLATE_VARIABLE_EXPECTATIONS,
        _TEMPLATE_VARIABLE_CONVERSATION,
    ]

    _instructions: str = PrivateAttr()
    _model: str = PrivateAttr()
    _instructions_prompt: PromptVersion = PrivateAttr()
    _ordered_template_variables: list[str] = PrivateAttr()
    _feedback_value_type: Any = PrivateAttr()
    _generate_rationale_first: bool = PrivateAttr(default=False)
    _include_tool_calls_in_conversation: bool = PrivateAttr(default=False)
    _inference_params: dict[str, Any] | None = PrivateAttr(default=None)

    def __init__(
        self,
        name: str,
        instructions: str,
        model: str | None = None,
        description: str | None = None,
        feedback_value_type: Any = str,
        generate_rationale_first: bool = False,
        include_tool_calls_in_conversation: bool = False,
        inference_params: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the InstructionsJudge.

        Args:
            name: The name of the judge
            instructions: Natural language instructions for evaluation
            model: The model identifier to use for evaluation (e.g., "openai:/gpt-4")
            description: A description of what the judge evaluates
            feedback_value_type: Optional type for the 'value' field in the Feedback response.
                           Default is str. Supported types (FeedbackValueType): int, float,
                           str, bool, Literal types, as well as a dict and list of these types.
            generate_rationale_first: Whether to generate rationale before the final value
            include_tool_calls_in_conversation: If True, include tool call information from
                           TOOL type spans when extracting conversation from session traces.
                           Default is False for backward compatibility.
            inference_params: Optional dictionary of inference parameters to pass to the
                           model (e.g., temperature, top_p, max_tokens). These parameters
                           allow fine-grained control over the model's behavior.
            kwargs: Additional configuration parameters
        """
        # TODO: Allow aggregations once we support boolean/numeric judge outputs
        super().__init__(name=name, description=description, aggregations=[], **kwargs)

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
        self._feedback_value_type = feedback_value_type
        self._generate_rationale_first = generate_rationale_first
        self._include_tool_calls_in_conversation = include_tool_calls_in_conversation
        self._inference_params = inference_params

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
    def kind(self) -> ScorerKind:
        return ScorerKind.INSTRUCTIONS

    @property
    def model(self) -> str:
        """Get the model for this judge."""
        return self._model

    @property
    def template_variables(self) -> set[str]:
        """Get the template variables from the instructions."""
        return self._instructions_prompt.variables

    @property
    def is_session_level_scorer(self) -> bool:
        """Get whether this judge is a session-level judge based on template variables."""
        return self._TEMPLATE_VARIABLE_CONVERSATION in self.template_variables

    @property
    def instructions(self) -> str:
        """Get the instructions of this judge."""
        return self._instructions

    @property
    def inference_params(self) -> dict[str, Any] | None:
        """Get the inference parameters for this judge."""
        return self._inference_params

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
        self,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
        session: list[Trace] | None,
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
        if session is not None and not isinstance(session, list):
            raise MlflowException(
                f"'session' must be a list of Trace objects, got {type(session).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if session is not None:
            for i, trace in enumerate(session):
                if not isinstance(trace, Trace):
                    raise MlflowException(
                        f"All elements in 'session' must be Trace objects, "
                        f"got {type(trace).__name__} at index {i}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

    def _check_required_parameters(
        self,
        inputs: Any | None,
        outputs: Any | None,
        expectations: dict[str, Any] | None,
        trace: Trace | None,
        session: list[Trace] | None,
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
        if self._TEMPLATE_VARIABLE_CONVERSATION in self.template_variables and session is None:
            missing_params.append("session")

        if missing_params:
            missing_str = "', '".join(missing_params)
            raise MlflowException(
                f"Must specify '{missing_str}' - required by template variables in instructions.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _validate_session(self, session: list[Trace]) -> None:
        """Validate that all traces in session belong to the same session."""
        session_id_to_trace_ids: dict[str, list[str]] = {}
        for trace in session:
            session_id = trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
            if session_id is None:
                raise MlflowException(
                    f"All traces in 'session' must have a session_id. "
                    f"Trace {trace.info.trace_id} is missing session_id. "
                    f"See https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/ "
                    f"for information on how to set session_id on traces.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if session_id not in session_id_to_trace_ids:
                session_id_to_trace_ids[session_id] = []
            session_id_to_trace_ids[session_id].append(trace.info.trace_id)

        if len(session_id_to_trace_ids) != 1:
            session_details = "\n".join(
                f"session_id '{sid}': trace_ids {trace_ids[:3]}"
                + (
                    f" and {len(trace_ids) - 3} more trace{'s' if len(trace_ids) - 3 != 1 else ''}"
                    if len(trace_ids) > 3
                    else ""
                )
                for sid, trace_ids in session_id_to_trace_ids.items()
            )
            raise MlflowException.invalid_parameter_value(
                f"All traces in 'session' must belong to the same session. "
                f"Found {len(session_id_to_trace_ids)} different session(s):\n{session_details}"
            )

    def _warn_unused_parameters(
        self,
        inputs: Any | None,
        outputs: Any | None,
        expectations: dict[str, Any] | None,
        conversation: list[dict[str, str]] | None,
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
        if (
            conversation is not None
            and self._TEMPLATE_VARIABLE_CONVERSATION not in self.template_variables
        ):
            unused_params.append("conversation")

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
            # include the value type in the description too so that models that don't support
            # structured outputs with tool calls can still understand the type.
            evaluation_rating_fields = "\n".join(
                [
                    f"- {field.name} ({self._format_type(field.value_type)}): {field.description}"
                    for field in output_fields
                ]
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

    def get_output_fields(self) -> list[JudgeField]:
        """Get the output fields for this judge."""
        # Use generic field description, not self.description, to avoid the LLM
        # echoing the scorer's description as the assessment value
        result_field = JudgeField(
            name="result",
            description=_RESULT_FIELD_DESCRIPTION,
            value_type=self._feedback_value_type,
        )
        rationale_field = JudgeField(
            name="rationale",
            description=_RATIONALE_FIELD_DESCRIPTION,
            value_type=str,
        )
        return (
            [rationale_field, result_field]
            if self._generate_rationale_first
            else [result_field, rationale_field]
        )

    def _format_type(self, value_type: Any) -> str:
        if value_type in (str, int, float, bool):
            return value_type.__name__
        elif get_origin(value_type) is Literal:
            return str(value_type).replace("typing.", "")
        # dict and list
        return str(value_type)

    def _build_user_message(
        self,
        inputs: Any | None,
        outputs: Any | None,
        expectations: dict[str, Any] | None,
        conversation: list[dict[str, str]] | None,
    ) -> str:
        """Build the user message with field values."""
        template_values = self._build_template_values(inputs, outputs, expectations, conversation)

        field_vars = [
            var for var in self._ordered_template_variables if var != self._TEMPLATE_VARIABLE_TRACE
        ]

        # Build user message parts in order
        user_message_parts = [
            f"{var_name}: {template_values[var_name]}"
            for var_name in field_vars
            if var_name in template_values
        ]

        # Some model providers (like Anthropic) require a user message
        # (i.e. a single-message chat history with role 'system' is not supported),
        # *and* they require the message to have non-empty content (empty string is not allowed)
        return (
            "\n".join(user_message_parts)
            if user_message_parts
            else "Follow the instructions from the first message"
        )

    def _build_template_values(
        self,
        inputs: Any | None,
        outputs: Any | None,
        expectations: dict[str, Any] | None,
        conversation: list[dict[str, str]] | None,
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

        if (
            conversation is not None
            and self._TEMPLATE_VARIABLE_CONVERSATION in self.template_variables
        ):
            template_values[self._TEMPLATE_VARIABLE_CONVERSATION] = self._safe_json_dumps(
                conversation
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
        session: list[Trace] | None = None,
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
            session: List of traces from the same session. When the template uses
                {{ conversation }}, the conversation history will be extracted from these traces.

        Returns:
            Evaluation results

        **Note on Trace Behavior**:
        - If template uses {{ trace }}: The trace object is passed to an agent-based judge that
          uses tools to fetch aspects of the trace's span data. The {{ trace }} placeholder itself
          is not replaced in the prompt - instead, the trace enables tool calling.
        - If template uses {{ inputs }}/{{ outputs }}/{{ expectations }} alongside {{ trace }}:
          These placeholders ARE replaced in the prompt with their values (either from the provided
          parameters or extracted from the trace), providing additional context to the agent.
        - If template uses {{ inputs }}/{{ outputs }}/{{ expectations }} without {{ trace }}:
          Values are extracted from the trace parameter (if provided) as follows:
          - inputs/outputs: From the trace's root span
          - expectations: From the trace's expectation assessments

        **Note on Session Behavior**:
        - Traces are expected to be in the same session and exception will be raised
          if they are not.
        - The conversation history will be extracted from the traces in chronological order.
        """
        self._validate_parameter_types(expectations, trace, session)

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

        conversation = None
        if session is not None and session:
            self._validate_session(session)
            conversation = resolve_conversation_from_session(
                session, include_tool_calls=self._include_tool_calls_in_conversation
            )

        self._check_required_parameters(inputs, outputs, expectations, trace, conversation)
        self._warn_unused_parameters(
            original_inputs, original_outputs, original_expectations, conversation
        )

        is_trace_based = self._TEMPLATE_VARIABLE_TRACE in self.template_variables

        system_content = self._build_system_message(is_trace_based)
        user_content = self._build_user_message(inputs, outputs, expectations, conversation)

        from mlflow.types.llm import ChatMessage

        messages = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]

        response_format = self._create_response_format_model()

        return invoke_judge_model(
            model_uri=self._model,
            prompt=messages,
            assessment_name=self.name,
            trace=trace if is_trace_based else None,
            response_format=response_format,
            use_case=USE_CASE_AGENTIC_JUDGE,
            inference_params=self._inference_params,
        )

    def _create_response_format_model(self) -> type[pydantic.BaseModel]:
        output_fields = self.get_output_fields()

        fields = {}
        for field in output_fields:
            fields[field.name] = (
                field.value_type,
                pydantic.Field(description=field.description),
            )

        return pydantic.create_model("ResponseFormat", **fields)

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
        Validate that instruction:
        1. contains at least one variable.
        2. does not contain any template variables other than expectation
           if conversation is provided.
        """
        template_vars = self.template_variables

        if not template_vars:
            raise MlflowException(
                "Instructions template must contain at least one variable (e.g., {{ inputs }}, "
                "{{ outputs }}, {{ trace }}, {{ expectations }}, or {{ conversation }}).",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if self._TEMPLATE_VARIABLE_CONVERSATION in template_vars and template_vars - {
            self._TEMPLATE_VARIABLE_EXPECTATIONS,
            self._TEMPLATE_VARIABLE_CONVERSATION,
        }:
            raise MlflowException(
                "Instructions template must not contain any template variables "
                "other than {{ expectations }} if {{ conversation }} is provided.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def __repr__(self) -> str:
        """Return string representation of the InstructionsJudge."""
        instructions_preview = (
            self._instructions[:PROMPT_TEXT_DISPLAY_LIMIT] + "..."
            if len(self._instructions) > PROMPT_TEXT_DISPLAY_LIMIT
            else self._instructions
        )
        inference_params_str = (
            f", inference_params={self._inference_params}" if self._inference_params else ""
        )
        return (
            f"InstructionsJudge(name='{self.name}', model='{self._model}', "
            f"instructions='{instructions_preview}', "
            f"template_variables={sorted(self.template_variables)}{inference_params_str})"
        )

    @staticmethod
    def _serialize_feedback_value_type(feedback_value_type: Any) -> dict[str, Any]:
        """
        Serialize a feedback_value_type to JSON Schema format.

        Supports all FeedbackValueType types:
        - PbValueType: float, int, str, bool
        - Literal types (as enum)
        - dict[str, PbValueType] (as object with additionalProperties)
        - list[PbValueType] (as array with items)

        Returns a JSON Schema representation of the type.
        """
        model = pydantic.create_model(
            "FeedbackValueSchema",
            result=(feedback_value_type, ...),
        )
        return model.model_json_schema()["properties"]["result"]

    @staticmethod
    def _deserialize_feedback_value_type(serialized: dict[str, Any]) -> type:
        """
        Deserialize a feedback_value_type from JSON Schema format.

        Supports all FeedbackValueType types:
        - PbValueType: str, int, float, bool
        - Literal types (from enum)
        - dict[str, PbValueType] (from object with additionalProperties)
        - list[PbValueType] (from array with items)
        """
        if not isinstance(serialized, dict) or "type" not in serialized:
            raise MlflowException.invalid_parameter_value(
                f"Invalid feedback_value_type serialization: {serialized}"
            )

        schema_type = serialized["type"]

        # Map JSON Schema types back to Python types
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }

        # Handle enum (Literal types)
        if "enum" in serialized:
            enum_values = serialized["enum"]
            if not enum_values:
                raise MlflowException.invalid_parameter_value(
                    f"Enum must have at least one value: {serialized}"
                )
            return Literal[tuple(enum_values)]

        # Handle basic types
        if schema_type in type_map:
            return type_map[schema_type]

        # Handle object (dict) type
        if schema_type == "object":
            if "additionalProperties" not in serialized:
                raise MlflowException.invalid_parameter_value(
                    f"Object type missing 'additionalProperties' field: {serialized}"
                )
            value_schema = serialized["additionalProperties"]
            if "type" not in value_schema:
                raise MlflowException.invalid_parameter_value(
                    f"additionalProperties missing 'type' field: {serialized}"
                )
            value_type = type_map.get(value_schema["type"])
            if value_type is None:
                raise MlflowException.invalid_parameter_value(
                    f"Unsupported value type in object: {value_schema['type']}"
                )
            return dict[str, value_type]

        # Handle array (list) type
        if schema_type == "array":
            if "items" not in serialized:
                raise MlflowException.invalid_parameter_value(
                    f"Array type missing 'items' field: {serialized}"
                )
            items_schema = serialized["items"]
            if "type" not in items_schema:
                raise MlflowException.invalid_parameter_value(
                    f"items missing 'type' field: {serialized}"
                )
            element_type = type_map.get(items_schema["type"])
            if element_type is None:
                raise MlflowException.invalid_parameter_value(
                    f"Unsupported element type in array: {items_schema['type']}"
                )
            return list[element_type]

        # Unsupported type
        raise MlflowException.invalid_parameter_value(
            f"Unsupported JSON Schema type: {schema_type}. "
            f"Only string, integer, number, boolean, object, and array are supported."
        )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to serialize as a SerializedScorer."""
        pydantic_data = {
            "instructions": self._instructions,
            "model": self._model,
        }
        if self._feedback_value_type is not None:
            pydantic_data["feedback_value_type"] = self._serialize_feedback_value_type(
                self._feedback_value_type
            )
        if self._inference_params is not None:
            pydantic_data["inference_params"] = self._inference_params

        serialized_scorer = SerializedScorer(
            name=self.name,
            description=self.description,
            aggregations=self.aggregations,
            is_session_level_scorer=self.is_session_level_scorer,
            mlflow_version=mlflow.__version__,
            serialization_version=_SERIALIZATION_VERSION,
            instructions_judge_pydantic_data=pydantic_data,
            builtin_scorer_class=None,
            builtin_scorer_pydantic_data=None,
            call_source=None,
            call_signature=None,
            original_func_name=None,
        )
        return asdict(serialized_scorer)


__all__ = ["InstructionsJudge"]
