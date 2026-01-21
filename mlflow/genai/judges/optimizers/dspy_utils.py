"""Utility functions for DSPy-based alignment optimizers."""

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

from mlflow import __version__ as VERSION
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.constants import (
    _DATABRICKS_DEFAULT_JUDGE_MODEL,
    USE_CASE_JUDGE_ALIGNMENT,
)
from mlflow.genai.utils.trace_utils import (
    extract_expectations_from_trace,
    extract_request_from_trace,
    extract_response_from_trace,
)
from mlflow.metrics.genai.model_utils import _parse_model_uri
from mlflow.utils import AttrDict

# Import dspy - raise exception if not installed
try:
    import dspy
except ImportError:
    raise MlflowException("DSPy library is required but not installed")

if TYPE_CHECKING:
    from mlflow.genai.judges.base import Judge

_logger = logging.getLogger(__name__)


@contextmanager
def suppress_verbose_logging(
    logger_name: str, threshold_level: int = logging.DEBUG
) -> Iterator[None]:
    """
    Context manager to suppress verbose logging from a specific logger.

    This is useful when running optimization algorithms that produce verbose output
    by default. The suppression only applies if the parent MLflow logger is above
    the threshold level, allowing users to see detailed output when they explicitly
    set logging to DEBUG.

    Args:
        logger_name: Name of the logger to control.
        threshold_level: Only suppress if MLflow logger is above this level.
            Defaults to logging.DEBUG.

    Example:
        >>> with suppress_verbose_logging("some.verbose.library"):
        ...     # Run operations that would normally log verbosely
        ...     result = run_optimization()
    """
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    try:
        if _logger.getEffectiveLevel() > threshold_level:
            logger.setLevel(logging.WARNING)
        yield
    finally:
        logger.setLevel(original_level)


def create_gepa_metric_adapter(
    metric_fn: Callable[["dspy.Example", Any, Any | None], bool],
) -> Any:
    """
    Create a metric adapter that bridges DSPy's standard metric to GEPA's format.

    GEPA requires a metric with signature: (gold, pred, trace, pred_name, pred_trace)
    but our standard metric_fn has signature: (example, pred, trace).
    This function creates an adapter that bridges the two signatures.

    Args:
        metric_fn: Standard metric function with signature (example, pred, trace)

    Returns:
        Adapter function with GEPA's expected signature
    """

    def gepa_metric_adapter(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """Adapt DSPy's 3-argument metric to GEPA's 5-argument format."""
        # gold is the dspy.Example
        # pred is the prediction output
        # trace/pred_name/pred_trace are optional GEPA-specific args
        # We pass None for our metric's trace parameter since GEPA's trace is different
        return metric_fn(gold, pred, trace=None)

    return gepa_metric_adapter


def _check_dspy_installed():
    try:
        import dspy  # noqa: F401
    except ImportError as e:
        raise MlflowException(
            "The DSPy library is required but not installed. "
            "Please install it using: `pip install dspy`"
        ) from e


def construct_dspy_lm(model: str):
    """
    Create a dspy.LM instance from a given model.

    Args:
        model: The model identifier/URI

    Returns:
        A dspy.LM instance configured for the given model
    """
    if model == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        return AgentEvalLM()
    else:
        model_litellm = convert_mlflow_uri_to_litellm(model)
        api_base, api_key = _get_api_base_key(model)
        if api_base:
            return dspy.LM(model=model_litellm, api_base=api_base, api_key=api_key)
        return dspy.LM(model=model_litellm)


def _get_api_base_key(model: str) -> tuple[str | None, str | None]:
    """
    Get the api_base URL and api_key for a model.

    Args:
        model: MLflow model URI (e.g., 'endpoints:/my-endpoint', 'databricks:/my-endpoint')

    Returns:
        Tuple of (api_base, api_key) - both None if not applicable
    """
    try:
        scheme, _ = _parse_model_uri(model)
        if scheme in ("endpoints", "databricks"):
            return _get_databricks_api_base_key()
        return None, None
    except Exception:
        return None, None


def _get_databricks_api_base_key() -> tuple[str | None, str | None]:
    """
    Get the api_base URL and api_key for Databricks serving endpoints.

    For Databricks endpoints with OpenAI-compatible API, the URL format is:
    - api_base: https://host/serving-endpoints
    - model: endpoint-name (passed in request body)

    LiteLLM appends /chat/completions to api_base, making the final URL:
    https://host/serving-endpoints/chat/completions with model=endpoint-name in body.

    Returns:
        Tuple of (api_base, api_key) - both None if credentials cannot be determined
    """
    # Get Databricks host and token from environment or SDK
    host = os.environ.get("DATABRICKS_HOST")
    api_key = os.environ.get("DATABRICKS_TOKEN")

    if not host or not api_key:
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError:
            _logger.warning(
                "Could not determine Databricks credentials. "
                "Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables."
            )
            return None, None

        client = WorkspaceClient()
        if not host:
            host = client.config.host
        if not api_key:
            # Get token from SDK authentication (supports OAuth, PAT, etc.)
            headers = client.config.authenticate()
            if "Authorization" in headers:
                api_key = headers["Authorization"].replace("Bearer ", "")

    host = host.rstrip("/")
    # Return api_base with just /serving-endpoints
    # LiteLLM will append /chat/completions and pass the endpoint name as model
    return f"{host}/serving-endpoints", api_key


def _to_attrdict(obj):
    """Recursively convert nested dicts/lists to AttrDicts."""
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_attrdict(item) for item in obj]
    else:
        return obj


def _process_chat_completions(
    user_prompt: str, system_prompt: str | None = None
) -> AttrDict[str, Any]:
    """Call managed RAG client and return formatted response."""
    response = call_chat_completions(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        session_name=f"mlflow-judge-optimizer-v{VERSION}",
        use_case=USE_CASE_JUDGE_ALIGNMENT,
    )

    if response.output is not None:
        result_dict = {
            "object": "chat.completion",
            "model": "databricks",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": response.output},
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "response_format": "json_object",
        }
    else:
        result_dict = {
            "object": "response",
            "error": response.error_message,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "response_format": "json_object",
        }

    return _to_attrdict(result_dict)


class AgentEvalLM(dspy.BaseLM):
    """Special DSPy LM for Databricks environment using managed RAG client."""

    def __init__(self):
        super().__init__("databricks")

    def dump_state(self):
        return {}

    def load_state(self, state):
        pass

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AttrDict[str, Any]:
        """Forward pass for the language model."""
        user_prompt = None
        system_prompt = None

        if messages:
            for message in messages:
                if message.get("role") == "user":
                    user_prompt = message.get("content", "")
                elif message.get("role") == "system":
                    system_prompt = message.get("content", "")

        if not user_prompt and prompt:
            user_prompt = prompt

        return _process_chat_completions(user_prompt, system_prompt)


def _sanitize_assessment_name(name: str) -> str:
    """
    Sanitize a name by converting it to lowercase and stripping whitespace.
    """
    return name.lower().strip()


def convert_mlflow_uri_to_litellm(model_uri: str) -> str:
    """
    Convert MLflow model URI format to LiteLLM format.

    MLflow uses URIs like 'openai:/gpt-4' while LiteLLM expects 'openai/gpt-4'.
    For Databricks endpoints, MLflow uses 'endpoints:/endpoint-name' which needs
    to be converted to 'databricks/endpoints/endpoint-name' for LiteLLM.

    Args:
        model_uri: MLflow model URI (e.g., 'openai:/gpt-4', 'endpoints:/my-endpoint')

    Returns:
        LiteLLM-compatible model string (e.g., 'openai/gpt-4', 'databricks/endpoints/my-endpoint')
    """
    try:
        scheme, path = _parse_model_uri(model_uri)
        # MLflow's "endpoints:/my-endpoint" is a Databricks serving endpoint URI.
        # LiteLLM expects "databricks/{endpoint-name}" for Databricks endpoints,
        # so we normalize both "endpoints" and "databricks" schemes to "databricks".
        if scheme in ("endpoints", "databricks"):
            return f"databricks/{path}"
        return f"{scheme}/{path}"
    except Exception as e:
        raise MlflowException(f"Failed to convert MLflow URI to LiteLLM format: {e}")


def convert_litellm_to_mlflow_uri(litellm_model: str) -> str:
    """
    Convert LiteLLM model format to MLflow URI format.

    LiteLLM uses formats like 'openai/gpt-4' while MLflow expects 'openai:/gpt-4'.

    Args:
        litellm_model: LiteLLM model string (e.g., 'openai/gpt-4')

    Returns:
        MLflow-compatible model URI (e.g., 'openai:/gpt-4')

    Raises:
        MlflowException: If the model string is not in the expected format

    Examples:
        >>> convert_litellm_to_mlflow_uri("openai/gpt-4")
        'openai:/gpt-4'
        >>> convert_litellm_to_mlflow_uri("anthropic/claude-3")
        'anthropic:/claude-3'
    """
    if not litellm_model:
        raise MlflowException(
            "Model string cannot be empty or None",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if "/" not in litellm_model:
        raise MlflowException(
            f"Invalid LiteLLM model format: '{litellm_model}'. "
            "Expected format: 'provider/model' (e.g., 'openai/gpt-4')",
            error_code=INVALID_PARAMETER_VALUE,
        )

    try:
        provider, model = litellm_model.split("/", 1)
        if not provider or not model:
            raise MlflowException(
                f"Invalid LiteLLM model format: '{litellm_model}'. "
                "Both provider and model name must be non-empty",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return f"{provider}:/{model}"
    except ValueError as e:
        raise MlflowException(f"Failed to convert LiteLLM format to MLflow URI: {e}")


def trace_to_dspy_example(trace: Trace, judge: Judge) -> Optional["dspy.Example"]:
    """
    Convert MLflow trace to DSPy example format.

    Extracts:
    - inputs/outputs from trace spans
    - expected result from human assessments
    - rationale from assessment feedback

    Args:
        trace: MLflow trace object
        judge: Judge instance to find assessments for

    Returns:
        DSPy example object or None if conversion fails
    """
    try:
        judge_input_fields = judge.get_input_fields()

        judge_requires_trace = any(field.name == "trace" for field in judge_input_fields)
        judge_requires_inputs = any(field.name == "inputs" for field in judge_input_fields)
        judge_requires_outputs = any(field.name == "outputs" for field in judge_input_fields)
        judge_requires_expectations = any(
            field.name == "expectations" for field in judge_input_fields
        )

        request = extract_request_from_trace(trace)
        response = extract_response_from_trace(trace)
        expectations = extract_expectations_from_trace(trace)

        # Check for missing required fields
        if not request and judge_requires_inputs:
            _logger.warning(f"Missing required request in trace {trace.info.trace_id}")
            return None
        elif not response and judge_requires_outputs:
            _logger.warning(f"Missing required response in trace {trace.info.trace_id}")
            return None
        elif not expectations and judge_requires_expectations:
            _logger.warning(f"Missing required expectations in trace {trace.info.trace_id}")
            return None

        # Find human assessment for this judge
        expected_result = None

        if trace.info.assessments:
            # Sort assessments by creation time (most recent first) then process
            sorted_assessments = sorted(
                trace.info.assessments,
                key=lambda a: (
                    a.create_time_ms if hasattr(a, "create_time_ms") and a.create_time_ms else 0
                ),
                reverse=True,
            )
            for assessment in sorted_assessments:
                sanitized_assessment_name = _sanitize_assessment_name(assessment.name)
                sanitized_judge_name = _sanitize_assessment_name(judge.name)
                if (
                    sanitized_assessment_name == sanitized_judge_name
                    and assessment.source.source_type == AssessmentSourceType.HUMAN
                ):
                    expected_result = assessment
                    break

        if not expected_result:
            _logger.warning(
                f"No human assessment found for judge '{judge.name}' in trace {trace.info.trace_id}"
            )
            return None

        if not expected_result.feedback:
            _logger.warning(f"No feedback found in assessment for trace {trace.info.trace_id}")
            return None

        # Create DSPy example
        example_kwargs = {}
        example_inputs = []
        if judge_requires_trace:
            example_kwargs["trace"] = trace
            example_inputs.append("trace")
        if judge_requires_inputs:
            example_kwargs["inputs"] = request
            example_inputs.append("inputs")
        if judge_requires_outputs:
            example_kwargs["outputs"] = response
            example_inputs.append("outputs")
        if judge_requires_expectations:
            example_kwargs["expectations"] = expectations
            example_inputs.append("expectations")
        example = dspy.Example(
            result=str(expected_result.feedback.value).lower(),
            rationale=expected_result.rationale or "",
            **example_kwargs,
        )

        # Set inputs (what the model should use as input)
        return example.with_inputs(*example_inputs)

    except Exception as e:
        _logger.error(f"Failed to create DSPy example from trace: {e}")
        return None


def create_dspy_signature(judge: "Judge") -> "dspy.Signature":
    """
    Create DSPy signature for judge evaluation.

    Args:
        judge: The judge to create signature for

    Returns:
        DSPy signature object
    """
    try:
        # Build signature fields dictionary using the judge's field definitions
        signature_fields = {}

        # Get input fields from the judge
        input_fields = judge.get_input_fields()
        for field in input_fields:
            signature_fields[field.name] = (
                field.value_type,
                dspy.InputField(desc=field.description),
            )

        # Get output fields from the judge
        output_fields = judge.get_output_fields()
        for field in output_fields:
            signature_fields[field.name] = (
                field.value_type,
                dspy.OutputField(desc=field.description),
            )

        return dspy.make_signature(signature_fields, judge.instructions)

    except Exception as e:
        raise MlflowException(f"Failed to create DSPy signature: {e}")


def agreement_metric(example: "dspy.Example", pred: Any, trace: Any | None = None):
    """Simple agreement metric for judge optimization."""
    try:
        # Extract result from example and prediction
        expected = getattr(example, "result", None)
        predicted = getattr(pred, "result", None)

        if expected is None or predicted is None:
            return False

        # Normalize both to consistent format
        expected_norm = str(expected).lower().strip()
        predicted_norm = str(predicted).lower().strip()

        _logger.debug(f"expected_norm: {expected_norm}, predicted_norm: {predicted_norm}")

        return expected_norm == predicted_norm
    except Exception as e:
        _logger.warning(f"Error in agreement_metric: {e}")
        return False


def append_input_fields_section(instructions: str, judge: "Judge") -> str:
    """
    Append a section listing the input fields for assessment to the instructions.

    DSPy optimizers may modify instructions during optimization. This function
    ensures that the input field names are clearly documented at the end of
    the instructions, but only if they're not already present.

    Args:
        instructions: The optimized instructions
        judge: The original judge (to get field names)

    Returns:
        Instructions with input fields section appended, or original instructions
        if all input fields are already present
    """
    input_fields = judge.get_input_fields()
    field_names = [f.name for f in input_fields]

    if not field_names:
        return instructions

    # Check if all input fields are already present in mustached format
    # Support both {{field}} and {{ field }} formats
    def has_mustached_field(name: str) -> bool:
        return f"{{{{{name}}}}}" in instructions or f"{{{{ {name} }}}}" in instructions

    if all(has_mustached_field(name) for name in field_names):
        return instructions

    fields_list = ", ".join(f"{{{{ {name} }}}}" for name in field_names)
    return f"{instructions}\n\nInputs for assessment: {fields_list}"


def format_demos_as_examples(demos: list[Any], judge: "Judge") -> str:
    """Format demos as few-shot examples to include in judge instructions.

    SIMBA optimization adds successful examples to program.demos. This function
    converts them to a text format suitable for inclusion in the judge prompt.

    Args:
        demos: List of dspy.Example objects containing successful judge evaluations
        judge: The original judge (to get field names)

    Returns:
        Formatted text with examples, or empty string if no valid demos
    """
    if not demos:
        return ""

    # Get field names from the judge
    input_fields = [f.name for f in judge.get_input_fields()]
    output_fields = [f.name for f in judge.get_output_fields()]

    examples_text = []
    for i, demo in enumerate(demos):
        example_parts = []

        # Access demo fields using dict-style access
        # (getattr conflicts with dspy.Example's built-in methods like inputs())
        if not hasattr(demo, "items"):
            raise MlflowException(
                f"Demo at index {i} cannot be converted to dict. "
                f"Expected dspy.Example, got {type(demo).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        demo_dict = dict(demo)

        # Format input fields
        for field in input_fields:
            if field in demo_dict:
                value = demo_dict[field]
                example_parts.append(f"{field}: {value}")

        # Format output fields
        for field in output_fields:
            if field in demo_dict:
                value = demo_dict[field]
                example_parts.append(f"{field}: {value}")

        if example_parts:
            examples_text.append(f"Example {i + 1}:\n" + "\n".join(example_parts))

    if examples_text:
        return "Here are some examples of good assessments:\n\n" + "\n\n".join(examples_text)
    return ""
