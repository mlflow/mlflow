import json

import pytest

from mlflow.entities import Assessment, Feedback, Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.scorer_utils import (
    BUILTIN_SCORER_PYDANTIC_DATA,
    INSTRUCTIONS_JUDGE_PYDANTIC_DATA,
    build_gateway_model,
    extract_endpoint_ref,
    extract_model_from_serialized_scorer,
    get_tool_call_signature,
    is_gateway_model,
    normalize_tool_call_arguments,
    parse_tool_call_expectations,
    recreate_function,
    update_model_in_serialized_scorer,
)
from mlflow.genai.utils.type import FunctionCall

# ============================================================================
# HAPPY PATH TESTS
# ============================================================================


def test_simple_function_recreation():
    source = "return x + y"
    signature = "(x, y)"
    func_name = "add_func"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated.__name__ == func_name
    assert recreated(3, 4) == 7
    assert recreated(10, -5) == 5


def test_function_with_control_flow():
    source = """if x > 0:
    return "positive"
else:
    return "non-positive" """
    signature = "(x)"
    func_name = "classify_number"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated(5) == "positive"
    assert recreated(-3) == "non-positive"
    assert recreated(0) == "non-positive"


def test_function_with_loop():
    source = """total = 0
for i in range(n):
    total += i
return total"""
    signature = "(n)"
    func_name = "sum_range"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated(5) == 10  # 0+1+2+3+4
    assert recreated(3) == 3  # 0+1+2
    assert recreated(0) == 0


def test_function_with_multiple_parameters():
    source = """if threshold is None:
    threshold = 5
return len(text) > threshold"""
    signature = "(text, threshold=None)"
    func_name = "length_check"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated("hello") is False  # len=5, not > 5
    assert recreated("hello world") is True  # len=11, > 5
    assert recreated("hi", 1) is True  # len=2, > 1


def test_function_creating_feedback_object():
    source = """import re
words = re.findall(r'\\b\\w+\\b', text)
return Feedback(value=len(words), rationale=f"Found {len(words)} words")"""
    signature = "(text)"
    func_name = "word_counter"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    result = recreated("hello world test")
    assert isinstance(result, Feedback)
    assert result.value == 3
    assert "Found 3 words" in result.rationale


def test_function_creating_assessment_object():
    # Note: Assessment constructor doesn't take 'value' directly - it's an abstract base
    # Use Feedback instead, which is a concrete subclass of Assessment
    source = """score = 1 if "good" in response else 0
return Feedback(name=name, value=score, rationale="Assessment result")"""
    signature = "(response, name='test_assessment')"
    func_name = "assess_response"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    result = recreated("This is good")
    assert isinstance(result, Feedback)
    assert isinstance(result, Assessment)  # Feedback is a subclass of Assessment
    assert result.value == 1
    assert result.name == "test_assessment"


def test_complex_logic_function():
    source = """result = {}
for item in items:
    if isinstance(item, str):
        result[item] = len(item)
    elif isinstance(item, (int, float)):
        result[str(item)] = item * 2
return result"""
    signature = "(items)"
    func_name = "process_items"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    result = recreated(["hello", 5, "world", 3.5])
    expected = {"hello": 5, "5": 10, "world": 5, "3.5": 7.0}
    assert result == expected


# ============================================================================
# SIGNATURE PARSING TESTS
# ============================================================================


def test_empty_signature():
    source = "return 42"
    signature = "()"
    func_name = "get_answer"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated() == 42


def test_single_parameter_signature():
    source = "return x * 2"
    signature = "(x)"
    func_name = "double"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated(5) == 10


def test_signature_with_whitespace():
    source = "return a + b"
    signature = "( a , b )"
    func_name = "add_with_spaces"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated(3, 7) == 10


def test_signature_with_defaults():
    source = "return base ** exponent"
    signature = "(base, exponent=2)"
    func_name = "power"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated(3) == 9  # 3^2
    assert recreated(2, 3) == 8  # 2^3


def test_complex_signature():
    source = """if data is None:
    data = []
return f"{prefix}: {len(data)} items" + (suffix or "")"""
    signature = "(data=None, prefix='Result', suffix=None)"
    func_name = "format_result"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated() == "Result: 0 items"
    assert recreated([1, 2, 3]) == "Result: 3 items"
    assert recreated([1, 2], "Count", "!") == "Count: 2 items!"


def test_empty_signature_string():
    from mlflow.exceptions import MlflowException

    source = "return 1"
    signature = ""
    func_name = "empty_sig"

    with pytest.raises(MlflowException, match="Invalid signature format"):
        recreate_function(source, signature, func_name)


# ============================================================================
# IMPORT NAMESPACE TESTS
# ============================================================================


def test_function_with_unavailable_import():
    # Import errors occur at execution time, not definition time
    source = """from some_nonexistent_module import NonExistentClass
return NonExistentClass()"""
    signature = "()"
    func_name = "use_bad_import"

    recreated = recreate_function(source, signature, func_name)

    # Function should be created successfully
    assert recreated is not None

    # But should fail when called due to import error
    with pytest.raises(ModuleNotFoundError, match="some_nonexistent_module"):
        recreated()


def test_function_with_undefined_variable():
    source = "return undefined_variable * 2"
    signature = "()"
    func_name = "use_undefined"

    recreated = recreate_function(source, signature, func_name)

    # Function is created but will fail when called
    assert recreated is not None

    # Should raise NameError when called
    with pytest.raises(NameError, match="undefined_variable"):
        recreated()


def test_function_with_syntax_error():
    source = "if x > 0\n    return True"  # Missing colon
    signature = "(x)"
    func_name = "syntax_error_func"

    with pytest.raises(SyntaxError, match="expected ':'"):
        recreate_function(source, signature, func_name)


def test_function_using_builtin_modules():
    source = """import json
import re
data = {"count": len(re.findall(r'\\w+', text))}
return json.dumps(data)"""
    signature = "(text)"
    func_name = "json_word_count"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    result = recreated("hello world test")

    parsed = json.loads(result)
    assert parsed["count"] == 3


def test_mlflow_imports_available():
    source = """# Test all available MLflow imports
feedback = Feedback(value=True, rationale="test")
# AssessmentSource should be available too
from mlflow.entities.assessment_source import AssessmentSourceType
source_obj = AssessmentSourceType.CODE  # Use the default source type
# Test that Trace is available
from mlflow.entities import TraceInfo, TraceState, TraceData
from mlflow.entities.trace_location import (
    TraceLocation,
    TraceLocationType,
    MlflowExperimentLocation,
)
from mlflow.entities.trace import Trace
mlflow_exp_location = MlflowExperimentLocation(experiment_id="0")
trace_location = TraceLocation(
    type=TraceLocationType.MLFLOW_EXPERIMENT,
    mlflow_experiment=mlflow_exp_location
)
trace_info = TraceInfo(
    trace_id="test_trace_id",
    trace_location=trace_location,
    request_time=1000,
    state=TraceState.OK
)
trace = Trace(info=trace_info, data=TraceData())
return {"feedback": feedback, "source": source_obj, "trace": trace}"""
    signature = "()"
    func_name = "test_mlflow_imports"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    result = recreated()

    assert isinstance(result["feedback"], Feedback)
    # AssessmentSourceType should be available (it's an enum/class)
    assert result["source"] is not None
    assert result["source"] == "CODE"
    # Check that Trace is available and can be instantiated
    assert isinstance(result["trace"], Trace)


def test_function_name_in_namespace():
    source = "return 'success'"
    signature = "()"
    func_name = "test_name_func"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated.__name__ == func_name


def test_indentation_handling():
    # Source without indentation - should be indented by the function
    source = """x = 1
y = 2
return x + y"""
    signature = "()"
    func_name = "indentation_test"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated() == 3


def test_empty_source_code():
    source = ""
    signature = "()"
    func_name = "empty_func"

    # Empty source code should cause syntax error during function definition
    with pytest.raises(SyntaxError, match="expected an indented block"):
        recreate_function(source, signature, func_name)


def test_function_with_import_error_at_runtime():
    # Import that doesn't exist is referenced but not imported in the function
    source = """try:
    return NonExistentClass()
except NameError:
    return "import_failed" """
    signature = "()"
    func_name = "runtime_import_error"

    recreated = recreate_function(source, signature, func_name)

    # Function should be created successfully
    assert recreated is not None
    # But calling it should handle the missing import gracefully
    result = recreated()
    assert result == "import_failed"


def test_function_with_mlflow_trace_type_hint():
    """
    Test that a function with mlflow.entities.Trace type hints can be recreated.

    This reproduces the issue where scorers with type hints like mlflow.entities.Trace
    would fail to register because the mlflow module wasn't available in the namespace
    during function recreation.
    """
    source = """return Feedback(
    value=trace.info.trace_id is not None,
    rationale=f"Trace ID: {trace.info.trace_id}"
)"""
    signature = "(trace: mlflow.entities.Trace) -> mlflow.entities.Feedback"
    func_name = "scorer_with_trace_type_hint"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated.__name__ == func_name

    # Test that it can be called with a Trace object
    from mlflow.entities import TraceData, TraceInfo, TraceState
    from mlflow.entities.trace_location import (
        MlflowExperimentLocation,
        TraceLocation,
        TraceLocationType,
    )

    mlflow_exp_location = MlflowExperimentLocation(experiment_id="0")
    trace_location = TraceLocation(
        type=TraceLocationType.MLFLOW_EXPERIMENT, mlflow_experiment=mlflow_exp_location
    )
    trace_info = TraceInfo(
        trace_id="test_trace_id",
        trace_location=trace_location,
        request_time=1000,
        state=TraceState.OK,
    )
    trace = Trace(info=trace_info, data=TraceData())

    result = recreated(trace)
    assert isinstance(result, Feedback)
    assert result.value is True
    assert "test_trace_id" in result.rationale


# ============================================================================
# GATEWAY MODEL UTILITY TESTS
# ============================================================================


def test_is_gateway_model():
    assert is_gateway_model("gateway:/my-endpoint") is True
    assert is_gateway_model("openai:/gpt-4") is False
    assert is_gateway_model(None) is False


def test_extract_and_build_gateway_model():
    assert extract_endpoint_ref("gateway:/my-endpoint") == "my-endpoint"
    assert build_gateway_model("my-endpoint") == "gateway:/my-endpoint"
    assert extract_endpoint_ref(build_gateway_model("test")) == "test"


def test_extract_model_from_serialized_scorer():
    instructions_judge_scorer = {
        "mlflow_version": "3.3.2",
        "serialization_version": 1,
        "name": "quality_scorer",
        "description": "Evaluates response quality",
        "aggregations": [],
        "is_session_level_scorer": False,
        "builtin_scorer_class": None,
        "builtin_scorer_pydantic_data": None,
        "call_source": None,
        "call_signature": None,
        "original_func_name": None,
        INSTRUCTIONS_JUDGE_PYDANTIC_DATA: {
            "instructions": "Evaluate the response quality",
            "model": "gateway:/my-endpoint",
        },
    }
    assert extract_model_from_serialized_scorer(instructions_judge_scorer) == "gateway:/my-endpoint"

    builtin_scorer = {
        "mlflow_version": "3.3.2",
        "serialization_version": 1,
        "name": "guidelines_scorer",
        "description": None,
        "aggregations": [],
        "is_session_level_scorer": False,
        "builtin_scorer_class": "Guidelines",
        BUILTIN_SCORER_PYDANTIC_DATA: {
            "name": "guidelines_scorer",
            "required_columns": ["outputs", "inputs"],
            "guidelines": ["Be helpful", "Be accurate"],
            "model": "openai:/gpt-4",
        },
        "call_source": None,
        "call_signature": None,
        "original_func_name": None,
        "instructions_judge_pydantic_data": None,
    }
    assert extract_model_from_serialized_scorer(builtin_scorer) == "openai:/gpt-4"
    assert extract_model_from_serialized_scorer({}) is None


def test_update_model_in_serialized_scorer():
    data = {
        "mlflow_version": "3.3.2",
        "serialization_version": 1,
        "name": "quality_scorer",
        INSTRUCTIONS_JUDGE_PYDANTIC_DATA: {
            "instructions": "Evaluate quality",
            "model": "gateway:/old-endpoint",
        },
    }
    result = update_model_in_serialized_scorer(data, "gateway:/new-endpoint")
    assert result[INSTRUCTIONS_JUDGE_PYDANTIC_DATA]["model"] == "gateway:/new-endpoint"
    assert result[INSTRUCTIONS_JUDGE_PYDANTIC_DATA]["instructions"] == "Evaluate quality"
    assert data[INSTRUCTIONS_JUDGE_PYDANTIC_DATA]["model"] == "gateway:/old-endpoint"


# ============================================================================
# TOOL CALL HELPER FUNCTION TESTS
# ============================================================================


@pytest.mark.parametrize(
    "expectations",
    [None, {}, {"expected_tool_calls": []}],
)
def test_parse_tool_call_expectations_returns_none_for_empty(expectations):
    assert parse_tool_call_expectations(expectations) is None


def test_parse_tool_call_expectations_parses_dict():
    expectations = {
        "expected_tool_calls": [
            {"name": "search", "arguments": {"query": "test"}},
            {"name": "summarize"},
        ]
    }
    result = parse_tool_call_expectations(expectations)

    assert len(result) == 2
    assert result[0].name == "search"
    assert result[0].arguments == {"query": "test"}
    assert result[1].name == "summarize"
    assert result[1].arguments is None


def test_parse_tool_call_expectations_parses_function_call_objects():
    expectations = {
        "expected_tool_calls": [
            FunctionCall(name="search", arguments={"query": "test"}),
            FunctionCall(name="summarize"),
        ]
    }
    result = parse_tool_call_expectations(expectations)

    assert len(result) == 2
    assert result[0].name == "search"
    assert result[1].name == "summarize"


@pytest.mark.parametrize(
    ("expectations", "expected_error"),
    [
        (
            {"expected_tool_calls": [{"name": "search", "arguments": "invalid"}]},
            "Arguments must be a dict",
        ),
        ({"expected_tool_calls": ["invalid_string"]}, "Invalid expected tool call format"),
    ],
)
def test_parse_tool_call_expectations_raises_for_invalid_input(expectations, expected_error):
    with pytest.raises(MlflowException, match=expected_error):
        parse_tool_call_expectations(expectations)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (None, {}),
        ({"query": "test", "limit": 10}, {"query": "test", "limit": 10}),
    ],
)
def test_normalize_tool_call_arguments(args, expected):
    assert normalize_tool_call_arguments(args) == expected


def test_normalize_tool_call_arguments_raises_for_invalid_type():
    with pytest.raises(MlflowException, match="Arguments must be a dict"):
        normalize_tool_call_arguments("invalid")


@pytest.mark.parametrize(
    ("call", "include_arguments", "expected"),
    [
        (FunctionCall(name="search", arguments={"query": "test"}), False, "search"),
        (
            FunctionCall(name="search", arguments={"query": "test"}),
            True,
            'search({"query": "test"})',
        ),
        (FunctionCall(name="search"), True, "search({})"),
        (FunctionCall(name=None), False, None),
    ],
)
def test_get_tool_call_signature(call, include_arguments, expected):
    assert get_tool_call_signature(call, include_arguments) == expected


def test_get_tool_call_signature_sorts_arguments():
    call1 = FunctionCall(name="search", arguments={"b": 2, "a": 1})
    call2 = FunctionCall(name="search", arguments={"a": 1, "b": 2})

    sig1 = get_tool_call_signature(call1, include_arguments=True)
    sig2 = get_tool_call_signature(call2, include_arguments=True)
    assert sig1 == sig2


# ============================================================================
# THIRD-PARTY MODEL URI PARSING TESTS
# ============================================================================


def test_parse_third_party_model_uri_databricks_default():
    from mlflow.genai.scorers.scorer_utils import parse_third_party_model_uri

    provider, model_name = parse_third_party_model_uri("databricks")
    assert provider == "databricks"
    assert model_name is None


def test_parse_third_party_model_uri_databricks_endpoint():
    from mlflow.genai.scorers.scorer_utils import parse_third_party_model_uri

    provider, model_name = parse_third_party_model_uri("databricks:/my-endpoint")
    assert provider == "databricks"
    assert model_name == "my-endpoint"


def test_parse_third_party_model_uri_openai():
    from mlflow.genai.scorers.scorer_utils import parse_third_party_model_uri

    provider, model_name = parse_third_party_model_uri("openai:/gpt-4")
    assert provider == "openai"
    assert model_name == "gpt-4"


def test_parse_third_party_model_uri_litellm():
    from mlflow.genai.scorers.scorer_utils import parse_third_party_model_uri

    provider, model_name = parse_third_party_model_uri("litellm:/claude-3-opus")
    assert provider == "litellm"
    assert model_name == "claude-3-opus"


def test_parse_third_party_model_uri_invalid_format():
    from mlflow.genai.scorers.scorer_utils import parse_third_party_model_uri

    with pytest.raises(MlflowException, match="Invalid model_uri format"):
        parse_third_party_model_uri("gpt-4")


# ============================================================================
# CHAT MESSAGE SERIALIZATION TESTS
# ============================================================================


def test_serialize_chat_messages_to_prompts_basic():
    from mlflow.genai.scorers.scorer_utils import serialize_chat_messages_to_prompts

    messages = [
        {"role": "user", "content": "Hello"},
    ]
    user_prompt, system_prompt = serialize_chat_messages_to_prompts(messages)
    assert user_prompt == "Hello"
    assert system_prompt is None


def test_serialize_chat_messages_to_prompts_with_system():
    from mlflow.genai.scorers.scorer_utils import serialize_chat_messages_to_prompts

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    user_prompt, system_prompt = serialize_chat_messages_to_prompts(messages)
    assert user_prompt == "Hello"
    assert system_prompt == "You are a helpful assistant."


def test_serialize_chat_messages_to_prompts_with_assistant():
    from mlflow.genai.scorers.scorer_utils import serialize_chat_messages_to_prompts

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    user_prompt, system_prompt = serialize_chat_messages_to_prompts(messages)
    assert user_prompt == "Hello\n\nAssistant: Hi there!\n\nHow are you?"
    assert system_prompt is None


def test_serialize_chat_messages_to_prompts_multiple_users():
    from mlflow.genai.scorers.scorer_utils import serialize_chat_messages_to_prompts

    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "First message"},
        {"role": "user", "content": "Second message"},
    ]
    user_prompt, system_prompt = serialize_chat_messages_to_prompts(messages)
    assert user_prompt == "First message\n\nSecond message"
    assert system_prompt == "Be helpful"


def test_serialize_chat_messages_to_prompts_custom_role():
    from mlflow.genai.scorers.scorer_utils import serialize_chat_messages_to_prompts

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "Tool result"},
    ]
    user_prompt, system_prompt = serialize_chat_messages_to_prompts(messages)
    assert user_prompt == "Hello\n\ntool: Tool result"
    assert system_prompt is None
