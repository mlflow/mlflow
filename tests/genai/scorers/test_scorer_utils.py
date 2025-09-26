import json
from unittest.mock import patch

import pytest

from mlflow.entities import Assessment, Feedback, Trace
from mlflow.genai.scorers.scorer_utils import recreate_function

# ============================================================================
# HAPPY PATH TESTS
# ============================================================================


def test_simple_function_recreation():
    """Test basic function recreation with simple logic."""
    source = "return x + y"
    signature = "(x, y)"
    func_name = "add_func"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated.__name__ == func_name
    assert recreated(3, 4) == 7
    assert recreated(10, -5) == 5


def test_function_with_control_flow():
    """Test recreation of function with if/else logic."""
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
    """Test recreation of function with loop logic."""
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
    """Test function with multiple parameters and default values."""
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
    """Test recreation of function that creates MLflow Feedback objects."""
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
    """Test recreation of function that creates MLflow Assessment objects."""
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
    """Test recreation of function with complex nested logic."""
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
    """Test function with empty signature."""
    source = "return 42"
    signature = "()"
    func_name = "get_answer"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated() == 42


def test_single_parameter_signature():
    """Test function with single parameter."""
    source = "return x * 2"
    signature = "(x)"
    func_name = "double"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated(5) == 10


def test_signature_with_whitespace():
    """Test signature parsing with extra whitespace."""
    source = "return a + b"
    signature = "( a , b )"
    func_name = "add_with_spaces"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated(3, 7) == 10


def test_signature_with_defaults():
    """Test signature with default parameter values."""
    source = "return base ** exponent"
    signature = "(base, exponent=2)"
    func_name = "power"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated(3) == 9  # 3^2
    assert recreated(2, 3) == 8  # 2^3


def test_complex_signature():
    """Test complex signature with multiple parameters and defaults."""
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
    """Test that empty signature string returns None."""
    source = "return 1"
    signature = ""
    func_name = "empty_sig"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is None


# ============================================================================
# IMPORT NAMESPACE TESTS
# ============================================================================


def test_function_with_unavailable_import():
    """Test that function using unavailable imports fails gracefully."""
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
    """Test that function using undefined variables fails gracefully."""
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
    """Test that function with syntax errors fails gracefully."""
    source = "if x > 0\n    return True"  # Missing colon
    signature = "(x)"
    func_name = "syntax_error_func"

    recreated = recreate_function(source, signature, func_name)

    # Should return None due to syntax error
    assert recreated is None


def test_function_using_builtin_modules():
    """Test that function can use standard library modules."""
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


@patch("mlflow.genai.scorers.scorer_utils._logger.warning")
def test_logging_on_failure(mock_logger):
    """Test that failures are logged appropriately."""
    source = "invalid python syntax !!!"
    signature = "()"
    func_name = "bad_function"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is None
    mock_logger.assert_called_once_with(
        "Failed to recreate function 'bad_function' from serialized source code"
    )


def test_mlflow_imports_available():
    """Test that MLflow imports are available in the namespace."""
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
    """Test that function name is correctly set in the local namespace."""
    source = "return 'success'"
    signature = "()"
    func_name = "test_name_func"

    recreated = recreate_function(source, signature, func_name)

    assert recreated is not None
    assert recreated.__name__ == func_name


def test_indentation_handling():
    """Test that source code indentation is handled correctly."""
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
    """Test handling of empty source code."""
    source = ""
    signature = "()"
    func_name = "empty_func"

    # Empty source code should cause syntax error during function definition
    recreated = recreate_function(source, signature, func_name)

    # Should return None due to invalid function definition
    assert recreated is None


def test_function_with_import_error_at_runtime():
    """Test that functions with import errors at runtime can be recreated but fail when called."""
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
