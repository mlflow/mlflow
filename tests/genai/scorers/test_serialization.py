import json
from unittest.mock import patch

import pytest

from mlflow.entities import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer, scorer
from mlflow.genai.scorers.builtin_scorers import Guidelines

# ============================================================================
# FORMAT VALIDATION TESTS (Minimal - just check serialization structure)
# ============================================================================


def test_decorator_scorer_serialization_format():
    """Test that decorator scorers serialize with correct format."""

    @scorer(name="test_scorer", aggregations=["mean"])
    def test_scorer(outputs):
        return outputs == "correct"

    serialized = test_scorer.model_dump()

    # Check required fields for decorator scorers
    assert serialized["name"] == "test_scorer"
    assert serialized["aggregations"] == ["mean"]
    assert "call_source" in serialized
    assert "original_func_name" in serialized
    assert serialized["original_func_name"] == "test_scorer"
    assert "call_signature" in serialized

    # Check version metadata
    assert "mlflow_version" in serialized
    assert "serialization_version" in serialized
    assert serialized["serialization_version"] == 1

    # Builtin scorer fields should be None (not populated for decorator scorers)
    assert serialized["builtin_scorer_class"] is None
    assert serialized["builtin_scorer_pydantic_data"] is None


def test_builtin_scorer_serialization_format():
    """Test that builtin scorers serialize with correct format."""
    from mlflow.genai.scorers.builtin_scorers import RelevanceToQuery

    serialized = RelevanceToQuery().model_dump()

    # Check required top-level fields for builtin scorers
    assert serialized["name"] == "relevance_to_query"
    assert "builtin_scorer_class" in serialized
    assert serialized["builtin_scorer_class"] == "RelevanceToQuery"
    assert "builtin_scorer_pydantic_data" in serialized

    # Check fields within builtin_scorer_pydantic_data
    pydantic_data = serialized["builtin_scorer_pydantic_data"]
    assert "required_columns" in pydantic_data

    # Check version metadata
    assert "mlflow_version" in serialized
    assert "serialization_version" in serialized
    assert serialized["serialization_version"] == 1

    # Decorator scorer fields should be None (not populated for builtin scorers)
    assert serialized["call_source"] is None
    assert serialized["call_signature"] is None
    assert serialized["original_func_name"] is None


# ============================================================================
# ROUND-TRIP FUNCTIONALITY TESTS (Comprehensive - test complete cycles)
# ============================================================================


def test_simple_scorer_round_trip():
    """Test basic round-trip serialization and execution."""

    @scorer
    def simple_scorer(outputs):
        return outputs == "correct"

    # Test original functionality
    assert simple_scorer(outputs="correct") is True
    assert simple_scorer(outputs="wrong") is False

    # Serialize and deserialize
    serialized = simple_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)

    # Test deserialized functionality matches original
    assert deserialized.name == "simple_scorer"
    assert deserialized(outputs="correct") is True
    assert deserialized(outputs="wrong") is False


def test_custom_name_and_aggregations_round_trip():
    """Test round-trip with custom name and aggregations."""

    @scorer(name="length_check", aggregations=["mean", "max"])
    def my_scorer(inputs, outputs):
        return len(outputs) > len(inputs)

    # Test original
    assert my_scorer(inputs="hi", outputs="hello world") is True
    assert my_scorer(inputs="hello", outputs="hi") is False

    # Round-trip
    serialized = my_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)

    # Test preserved properties and functionality
    assert deserialized.name == "length_check"
    assert deserialized.aggregations == ["mean", "max"]
    assert deserialized(inputs="hi", outputs="hello world") is True
    assert deserialized(inputs="hello", outputs="hi") is False


def test_multiple_parameters_round_trip():
    """Test round-trip with multiple parameters."""

    @scorer
    def multi_param_scorer(inputs, outputs, expectations):
        return outputs.startswith(inputs) and len(outputs) > expectations.get("min_length", 0)

    # Test original
    test_args = {
        "inputs": "Hello",
        "outputs": "Hello world!",
        "expectations": {"min_length": 5},
    }
    assert multi_param_scorer(**test_args) is True

    # Round-trip
    serialized = multi_param_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)

    # Test functionality preserved
    assert deserialized(**test_args) is True
    assert (
        deserialized(inputs="Hi", outputs="Hello world!", expectations={"min_length": 5}) is False
    )


def test_complex_logic_round_trip():
    """Test round-trip with complex control flow and logic."""

    @scorer
    def complex_scorer(outputs):
        if not outputs:
            return 0

        words = outputs.split()
        score = 0
        for word in words:
            if word.isupper():
                score += 2
            elif word.islower():
                score += 1

        return score

    # Test original functionality
    test_cases = [
        ("", 0),
        ("hello world", 2),  # 2 lowercase words
        ("HELLO WORLD", 4),  # 2 uppercase words
        ("Hello WORLD", 2),  # mixed case "Hello" (0) + "WORLD" (2)
    ]

    for test_input, expected in test_cases:
        assert complex_scorer(outputs=test_input) == expected

    # Round-trip
    serialized = complex_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)

    # Test all cases still work
    for test_input, expected in test_cases:
        assert deserialized(outputs=test_input) == expected


def test_imports_and_feedback_round_trip():
    """Test round-trip with imports and Feedback return."""

    @scorer
    def feedback_scorer(outputs):
        import re  # clint: disable=lazy-builtin-import

        pattern = r"\b\w+\b"
        words = re.findall(pattern, outputs)
        return Feedback(value=len(words), rationale=f"Found {len(words)} words")

    # Test original
    result = feedback_scorer(outputs="hello world test")
    assert isinstance(result, Feedback)
    assert result.value == 3
    assert "Found 3 words" in result.rationale

    # Round-trip
    serialized = feedback_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)

    # Test functionality preserved
    result = deserialized(outputs="hello world test")
    assert isinstance(result, Feedback)
    assert result.value == 3
    assert "Found 3 words" in result.rationale


def test_default_parameters_round_trip():
    """Test round-trip with default parameter values."""

    @scorer
    def default_scorer(outputs, threshold=5):
        return len(outputs) > threshold

    # Test original with and without default
    assert default_scorer(outputs="short") is False  # len=5, not > 5
    assert default_scorer(outputs="longer") is True  # len=6, > 5
    assert default_scorer(outputs="hi", threshold=1) is True  # len=2, > 1

    # Round-trip
    serialized = default_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)

    # Test defaults work
    assert deserialized(outputs="short") is False
    assert deserialized(outputs="longer") is True


def test_json_workflow_round_trip():
    """Test complete JSON serialization workflow."""

    @scorer(name="json_test", aggregations=["mean"])
    def json_scorer(outputs):
        return len(outputs.split()) > 3

    # Test original
    assert json_scorer(outputs="one two three four") is True
    assert json_scorer(outputs="one two") is False

    # JSON round-trip
    serialized = json_scorer.model_dump()
    json_str = json.dumps(serialized)
    loaded_dict = json.loads(json_str)
    deserialized = Scorer.model_validate(loaded_dict)

    # Test functionality preserved through JSON
    assert deserialized.name == "json_test"
    assert deserialized.aggregations == ["mean"]
    assert deserialized(outputs="one two three four") is True
    assert deserialized(outputs="one two") is False


def test_end_to_end_complex_round_trip():
    """Test complex end-to-end scenario with all features."""

    @scorer(name="complete_test", aggregations=["mean", "max"])
    def complete_scorer(inputs, outputs, expectations):
        input_words = len(inputs.split())
        output_words = len(outputs.split())
        expected_ratio = expectations.get("word_ratio", 1.0)

        actual_ratio = output_words / input_words if input_words > 0 else 0
        return actual_ratio >= expected_ratio

    test_args = {
        "inputs": "hello world",
        "outputs": "hello beautiful world today",
        "expectations": {"word_ratio": 1.5},
    }

    # Test original
    original_result = complete_scorer(**test_args)
    assert original_result is True

    # Round-trip
    serialized = complete_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)

    # Test everything preserved
    assert deserialized.name == "complete_test"
    assert deserialized.aggregations == ["mean", "max"]
    deserialized_result = deserialized(**test_args)
    assert original_result == deserialized_result is True


def test_deserialized_scorer_runs_without_global_context():
    """Test that deserialized scorer can run without access to original global context."""

    # Create a simple scorer that only uses built-in functions and parameters
    @scorer(name="isolated_test")
    def simple_scorer(outputs):
        # Only use built-in functions and the parameter - no external dependencies
        return len(outputs.split()) > 2

    # Test original works
    assert simple_scorer(outputs="one two three") is True
    assert simple_scorer(outputs="one two") is False

    # Serialize the scorer
    serialized_data = simple_scorer.model_dump()

    # Test deserialized scorer in completely isolated namespace using exec
    test_code = """
# Import required modules in isolated namespace
from mlflow.genai.scorers import Scorer

# Deserialize the scorer (no external context available)
deserialized = Scorer.model_validate(serialized_data)

# Test that it can run successfully in isolation
result1 = deserialized(outputs="one two three")
result2 = deserialized(outputs="one two")
result3 = deserialized(outputs="hello world test case")

# Store results for verification
test_results = {
    "result1": result1,
    "result2": result2,
    "result3": result3,
    "name": deserialized.name,
    "aggregations": deserialized.aggregations
}
"""

    # Execute in isolated namespace with only serialized_data available
    isolated_namespace = {"serialized_data": serialized_data}
    exec(test_code, isolated_namespace)

    # Verify results from isolated execution
    results = isolated_namespace["test_results"]
    assert results["result1"] is True  # "one two three" has 3 words > 2
    assert results["result2"] is False  # "one two" has 2 words, not > 2
    assert results["result3"] is True  # "hello world test case" has 4 words > 2
    assert results["name"] == "isolated_test"
    assert results["aggregations"] is None


def test_builtin_scorer_round_trip():
    """Test builtin scorer serialization and execution with mocking."""
    # from mlflow.genai.scorers import relevance_to_query
    from mlflow.genai.scorers.builtin_scorers import RelevanceToQuery

    # Round-trip serialization
    serialized = RelevanceToQuery().model_dump()
    deserialized = Scorer.model_validate(serialized)

    # Test class type and properties preserved
    assert isinstance(deserialized, RelevanceToQuery)
    assert deserialized.name == "relevance_to_query"
    assert hasattr(deserialized, "required_columns")
    assert deserialized.required_columns == {"inputs", "outputs"}

    # Test execution with mocking
    with patch(
        "mlflow.genai.judges.is_context_relevant",
        return_value=Feedback(name="relevance_to_query", value="yes", metadata={"chunk_index": 0}),
    ) as mock_judge:
        result = deserialized(
            inputs={"question": "What is machine learning?"},
            outputs=(
                "Machine learning is a subset of AI that enables computers to learn without "
                "explicit programming."
            ),
        )

    # Verify execution worked correctly
    mock_judge.assert_called_once_with(
        request="{'question': 'What is machine learning?'}",
        context=(
            "Machine learning is a subset of AI that enables computers to learn without "
            "explicit programming."
        ),
        name="relevance_to_query",
        model=None,
    )

    assert isinstance(result, Feedback)
    assert result.name == "relevance_to_query"
    assert result.value == "yes"
    assert result.metadata == {"chunk_index": 0}  # chunk_index should be preserved


def test_builtin_scorer_with_parameters_round_trip():
    """Test builtin scorer with custom parameters (like Guidelines with guidelines)."""
    from mlflow.genai.scorers.builtin_scorers import Guidelines

    # Create scorer with custom parameters
    tone = (
        "The response must maintain a courteous, respectful tone throughout. "
        "It must show empathy for customer concerns."
    )
    tone_scorer = Guidelines(name="tone", guidelines=[tone])

    # Verify original properties
    assert tone_scorer.name == "tone"
    assert tone_scorer.guidelines == [tone]
    assert isinstance(tone_scorer, Guidelines)

    # Round-trip serialization
    serialized = tone_scorer.model_dump()

    # Verify serialization format includes all fields
    assert "builtin_scorer_class" in serialized
    assert serialized["builtin_scorer_class"] == "Guidelines"
    assert "builtin_scorer_pydantic_data" in serialized
    pydantic_data = serialized["builtin_scorer_pydantic_data"]
    assert "guidelines" in pydantic_data
    assert pydantic_data["guidelines"] == [tone]
    assert pydantic_data["name"] == "tone"

    # Deserialize
    deserialized = Scorer.model_validate(serialized)

    # Test class type and all properties preserved
    assert isinstance(deserialized, Guidelines)
    assert deserialized.name == "tone"
    assert deserialized.guidelines == [tone]
    assert hasattr(deserialized, "required_columns")
    assert deserialized.required_columns == {"inputs", "outputs"}

    # Test that it can be executed with mocking
    with patch(
        "mlflow.genai.judges.meets_guidelines",
        return_value=Feedback(
            name="tone", value=True, rationale="Response is appropriately courteous"
        ),
    ) as mock_judge:
        result = deserialized(
            inputs={"question": "What is the issue?"},
            outputs=(
                "Thank you for bringing this to my attention. I understand your concern and "
                "will help resolve this issue promptly."
            ),
        )

    # Verify execution worked correctly
    mock_judge.assert_called_once_with(
        guidelines=[tone],
        context={
            "request": "{'question': 'What is the issue?'}",
            "response": (
                "Thank you for bringing this to my attention. I understand your concern and "
                "will help resolve this issue promptly."
            ),
        },
        name="tone",
        model=None,
    )

    assert isinstance(result, Feedback)
    assert result.name == "tone"
    assert result.value is True


def test_direct_subclass_scorer_rejected():
    """Test that direct subclassing of Scorer is rejected during serialization."""

    class DirectSubclassScorer(Scorer):
        """An unsupported direct subclass of Scorer."""

        def __init__(self, **data):
            super().__init__(name="direct_subclass", **data)

        def __call__(self, *, outputs):
            return len(outputs) > 5

    # Create instance - this should work
    direct_scorer = DirectSubclassScorer()

    # Calling it should work
    assert direct_scorer(outputs="hello world") is True
    assert direct_scorer(outputs="hi") is False

    # But serialization should raise an error
    with pytest.raises(MlflowException, match="Unsupported scorer type: DirectSubclassScorer"):
        direct_scorer.model_dump()

    # Verify the error message is informative
    try:
        direct_scorer.model_dump()
    except MlflowException as e:
        error_msg = str(e)
        assert "Builtin scorers" in error_msg
        assert "Decorator-created scorers" in error_msg
        assert "@scorer decorator" in error_msg
        assert "Direct subclassing of Scorer is not supported" in error_msg


def test_builtin_scorer_with_aggregations_round_trip():
    """Test builtin scorer with aggregations serialization and execution."""
    from mlflow.genai.scorers.builtin_scorers import RelevanceToQuery

    scorer_with_aggs = RelevanceToQuery(name="relevance_with_aggs", aggregations=["mean", "max"])

    # Test that aggregations were set
    assert scorer_with_aggs.name == "relevance_with_aggs"
    assert scorer_with_aggs.aggregations == ["mean", "max"]

    # Round-trip serialization
    serialized = scorer_with_aggs.model_dump()
    deserialized = Scorer.model_validate(serialized)

    # Test properties preserved
    assert isinstance(deserialized, RelevanceToQuery)
    assert deserialized.name == "relevance_with_aggs"
    assert deserialized.aggregations == ["mean", "max"]
    assert hasattr(deserialized, "required_columns")
    assert deserialized.required_columns == {"inputs", "outputs"}

    # Test that both can be executed with mocking
    test_args = {
        "inputs": {"question": "What is machine learning?"},
        "outputs": "Machine learning is a subset of AI.",
    }

    with patch(
        "mlflow.genai.judges.is_context_relevant",
        return_value=Feedback(name="relevance_with_aggs", value="yes"),
    ) as mock_judge:
        # Test original scorer
        original_result = scorer_with_aggs(**test_args)

        # Test deserialized scorer
        deserialized_result = deserialized(**test_args)

    # Verify both results are equivalent
    assert original_result.name == deserialized_result.name == "relevance_with_aggs"
    assert original_result.value == deserialized_result.value == "yes"

    # Judge should be called twice (once for each scorer)
    assert mock_judge.call_count == 2


# ============================================================================
# COMPATIBILITY TESTS (Fixed serialized strings for backward compatibility)
# ============================================================================


def test_builtin_scorer_with_custom_name_compatibility():
    """Test builtin scorer with custom name from fixed serialized string."""
    # Fixed serialized string for Guidelines scorer with custom name and parameters
    fixed_serialized_data = {
        "name": "custom_guidelines",
        "aggregations": ["mean", "max"],
        "mlflow_version": "3.1.0",
        "serialization_version": 1,
        "builtin_scorer_class": "Guidelines",
        "builtin_scorer_pydantic_data": {
            "name": "custom_guidelines",
            "aggregations": ["mean", "max"],
            "required_columns": ["inputs", "outputs"],
            "guidelines": [
                "Be polite and professional",
                "Provide accurate information",
            ],
        },
        "call_source": None,
        "call_signature": None,
        "original_func_name": None,
    }

    # Test deserialization
    deserialized = Scorer.model_validate(fixed_serialized_data)

    # Verify correct type and properties
    from mlflow.genai.scorers.builtin_scorers import Guidelines

    assert isinstance(deserialized, Guidelines)
    assert deserialized.name == "custom_guidelines"
    assert deserialized.aggregations == ["mean", "max"]
    assert deserialized.guidelines == [
        "Be polite and professional",
        "Provide accurate information",
    ]
    assert deserialized.required_columns == {"inputs", "outputs"}


def test_custom_scorer_compatibility_from_fixed_string():
    """Test that custom scorers can be deserialized from a fixed serialized string."""
    # Fixed serialized string representing a simple custom scorer
    fixed_serialized_data = {
        "name": "word_count_scorer",
        "aggregations": ["mean"],
        "mlflow_version": "3.1.0",
        "serialization_version": 1,
        "builtin_scorer_class": None,
        "builtin_scorer_pydantic_data": None,
        "call_source": "return len(outputs.split())",
        "call_signature": "(outputs)",
        "original_func_name": "word_count_scorer",
    }

    # Test deserialization
    deserialized = Scorer.model_validate(fixed_serialized_data)

    # Verify correct properties
    assert deserialized.name == "word_count_scorer"
    assert deserialized.aggregations == ["mean"]

    # Test functionality
    assert deserialized(outputs="hello world test") == 3
    assert deserialized(outputs="single") == 1
    assert deserialized(outputs="") == 0


def test_complex_custom_scorer_compatibility():
    """Test complex custom scorer with multiple parameters from fixed string."""
    # Fixed serialized string for a more complex custom scorer
    fixed_serialized_data = {
        "name": "length_comparison",
        "aggregations": None,
        "mlflow_version": "2.9.0",
        "serialization_version": 1,
        "builtin_scorer_class": None,
        "builtin_scorer_pydantic_data": None,
        "call_source": (
            "input_len = len(inputs) if inputs else 0\n"
            "output_len = len(outputs) if outputs else 0\n"
            "min_ratio = expectations.get('min_ratio', 1.0) if expectations else 1.0\n"
            "return output_len >= input_len * min_ratio"
        ),
        "call_signature": "(inputs, outputs, expectations)",
        "original_func_name": "length_comparison",
    }

    # Test deserialization
    deserialized = Scorer.model_validate(fixed_serialized_data)

    # Verify properties
    assert deserialized.name == "length_comparison"
    assert deserialized.aggregations is None

    # Test functionality with various inputs
    assert (
        deserialized(inputs="hello", outputs="hello world", expectations={"min_ratio": 1.5}) is True
    )  # 11 >= 5 * 1.5 (7.5)

    assert (
        deserialized(inputs="hello", outputs="hi", expectations={"min_ratio": 1.5}) is False
    )  # 2 < 5 * 1.5 (7.5)

    assert deserialized(inputs="test", outputs="test", expectations={}) is True  # 4 >= 4 * 1.0


def test_decorator_scorer_multiple_serialization_round_trips():
    """Test that decorator scorers can be serialized multiple times after deserialization."""

    @scorer
    def multi_round_scorer(outputs):
        return len(outputs) > 5

    # First serialization
    first_dump = multi_round_scorer.model_dump()

    # Deserialize
    recovered = Scorer.model_validate(first_dump)

    # Second serialization - this should work now with caching
    second_dump = recovered.model_dump()

    # Verify the dumps are identical
    assert first_dump == second_dump

    # Third serialization to ensure it's truly reusable
    third_dump = recovered.model_dump()
    assert first_dump == third_dump

    # Verify functionality is preserved
    assert recovered(outputs="hello world") is True
    assert recovered(outputs="hi") is False


def test_builtin_scorer_instructions_preserved_through_serialization():
    scorer = Guidelines(name="test_guidelines", guidelines=["Be helpful"])

    original_instructions = scorer.instructions

    serialized = scorer.model_dump()
    assert "builtin_scorer_pydantic_data" in serialized
    pydantic_data = serialized["builtin_scorer_pydantic_data"]

    assert "instructions" in pydantic_data
    assert pydantic_data["instructions"] == original_instructions

    deserialized = Scorer.model_validate(serialized)

    assert isinstance(deserialized, Guidelines)
    assert deserialized.instructions == original_instructions
    assert deserialized.name == "test_guidelines"
    assert deserialized.guidelines == ["Be helpful"]
