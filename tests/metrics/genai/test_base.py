import json
import re
from unittest.mock import patch

import pytest

from mlflow.entities import Feedback
from mlflow.genai.scorers import Scorer, scorer
from mlflow.metrics.genai import EvaluationExample


def test_evaluation_example_str():
    example1 = str(
        EvaluationExample(
            input="This is an input",
            output="This is an output",
            score=5,
            justification="This is a justification",
            grading_context={"foo": "bar"},
        )
    )
    example1_expected = """
        Example Input:
        This is an input

        Example Output:
        This is an output

        Additional information used by the model:
        key: foo
        value:
        bar

        Example score: 5
        Example justification: This is a justification
        """
    assert re.sub(r"\s+", "", example1_expected) == re.sub(r"\s+", "", example1)

    example2 = str(
        EvaluationExample(
            input="This is an input", output="This is an output", score=5, justification="It works"
        )
    )
    example2_expected = """
        Example Input:
        This is an input

        Example Output:
        This is an output

        Example score: 5
        Example justification: It works
        """
    assert re.sub(r"\s+", "", example2_expected) == re.sub(r"\s+", "", example2)

    example3 = str(
        EvaluationExample(
            input="This is an input",
            output="This is an output",
            score=5,
            justification="This is a justification",
            grading_context="Baz baz",
        )
    )
    example3_expected = """
        Example Input:
        This is an input

        Example Output:
        This is an output

        Additional information used by the model:
        Baz baz

        Example score: 5
        Example justification: This is a justification
        """
    assert re.sub(r"\s+", "", example3_expected) == re.sub(r"\s+", "", example3)


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

    # Should not have builtin scorer fields
    assert "builtin_scorer_class" not in serialized


def test_builtin_scorer_serialization_format():
    """Test that builtin scorers serialize with correct format."""
    from mlflow.genai.scorers import relevance_to_query

    serialized = relevance_to_query.model_dump()

    # Check required fields for builtin scorers
    assert serialized["name"] == "relevance_to_query"
    assert "builtin_scorer_class" in serialized
    assert serialized["builtin_scorer_class"] == "RelevanceToQuery"
    assert "required_columns" in serialized

    # Check version metadata
    assert "mlflow_version" in serialized
    assert "serialization_version" in serialized
    assert serialized["serialization_version"] == 1

    # Should not have decorator scorer fields
    assert "call_source" not in serialized
    assert "original_func_name" not in serialized


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
    test_args = {"inputs": "Hello", "outputs": "Hello world!", "expectations": {"min_length": 5}}
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
        import re

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
    assert original_result == deserialized_result == True


def test_builtin_scorer_round_trip():
    """Test builtin scorer serialization and execution with mocking."""
    from mlflow.genai.scorers import relevance_to_query
    from mlflow.genai.scorers.builtin_scorers import RelevanceToQuery

    # Round-trip serialization
    serialized = relevance_to_query.model_dump()
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
            outputs="Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        )

    # Verify execution worked correctly
    mock_judge.assert_called_once_with(
        request="What is machine learning?",
        context="Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        name="relevance_to_query",
    )

    assert isinstance(result, Feedback)
    assert result.name == "relevance_to_query"
    assert result.value == "yes"
    assert result.metadata == {}  # chunk_index should be removed


def test_builtin_scorer_with_parameters_round_trip():
    """Test builtin scorer with custom parameters (like GuidelineAdherence with global_guidelines)."""
    from mlflow.genai.scorers import guideline_adherence
    from mlflow.genai.scorers.builtin_scorers import GuidelineAdherence

    # Create scorer with custom parameters
    tone = "The response must maintain a courteous, respectful tone throughout. It must show empathy for customer concerns."
    tone_scorer = guideline_adherence.with_config(name="tone", global_guidelines=[tone])

    # Verify original properties
    assert tone_scorer.name == "tone"
    assert tone_scorer.global_guidelines == [tone]
    assert isinstance(tone_scorer, GuidelineAdherence)

    # Round-trip serialization
    serialized = tone_scorer.model_dump()

    # Verify serialization format includes all fields
    assert "builtin_scorer_class" in serialized
    assert serialized["builtin_scorer_class"] == "GuidelineAdherence"
    assert "global_guidelines" in serialized
    assert serialized["global_guidelines"] == [tone]
    assert serialized["name"] == "tone"

    # Deserialize
    deserialized = Scorer.model_validate(serialized)

    # Test class type and all properties preserved
    assert isinstance(deserialized, GuidelineAdherence)
    assert deserialized.name == "tone"
    assert deserialized.global_guidelines == [tone]
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
            outputs="Thank you for bringing this to my attention. I understand your concern and will help resolve this issue promptly."
        )

    # Verify execution worked correctly
    mock_judge.assert_called_once_with(
        guidelines=[tone],
        context={
            "response": "Thank you for bringing this to my attention. I understand your concern and will help resolve this issue promptly."
        },
        name="tone",
    )

    assert isinstance(result, Feedback)
    assert result.name == "tone"
    assert result.value is True


def test_scorer_serialization():
    """Test that Scorer class can be serialized and deserialized."""

    # Test 1: Simple scorer with decorator
    @scorer
    def simple_scorer(outputs):
        return outputs == "correct"

    serialized = simple_scorer.model_dump()
    assert "name" in serialized
    assert serialized["name"] == "simple_scorer"
    assert "call_source" in serialized
    assert 'outputs == "correct"' in serialized["call_source"]

    # Test 2: Scorer with custom name and aggregations
    @scorer(name="custom", aggregations=["mean", "max"])
    def custom_scorer(inputs, outputs):
        return len(outputs) > len(inputs)

    serialized = custom_scorer.model_dump()
    assert serialized["name"] == "custom"
    assert serialized["aggregations"] == ["mean", "max"]
    assert "len(outputs) > len(inputs)" in serialized["call_source"]

    # Test 3: Deserialization works
    deserialized = Scorer.model_validate(serialized)
    assert deserialized.name == "custom"
    assert deserialized.aggregations == ["mean", "max"]

    # Test functionality preserved
    result = deserialized(inputs="hi", outputs="hello world")
    assert result is True


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
    with pytest.raises(ValueError, match="Unsupported scorer type: DirectSubclassScorer"):
        direct_scorer.model_dump()

    # Verify the error message is informative
    try:
        direct_scorer.model_dump()
    except ValueError as e:
        error_msg = str(e)
        assert "Builtin scorers" in error_msg
        assert "Decorator-created scorers" in error_msg
        assert "@scorer decorator" in error_msg
        assert "Direct subclassing of Scorer is not supported" in error_msg
