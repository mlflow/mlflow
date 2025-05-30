import json
import pytest
import inspect
import warnings

from mlflow.genai.scorers import Scorer, scorer
from mlflow.entities import Feedback


# Test 1: Simple direct Scorer subclass
class SimpleScorer(Scorer):
    def __init__(self, name: str = "simple", aggregations=None):
        super().__init__(name=name, aggregations=aggregations)

    def __call__(self, *, outputs):
        return outputs == "yes"


def test_simple_scorer_serialization():
    """Test that a simple Scorer subclass can be serialized and deserialized."""
    original = SimpleScorer(name="test_simple")
    
    # Serialize to dict
    serialized = original.model_dump()
    
    # Check that serialized data contains expected fields
    assert "name" in serialized
    assert serialized["name"] == "test_simple"
    
    # Check that source code is stored
    assert "__call___source" in serialized
    assert "run_source" in serialized
    
    # Deserialize
    deserialized = SimpleScorer.model_validate(serialized)
    
    # Test that deserialized scorer works
    assert deserialized.name == "test_simple"
    result = deserialized.run(outputs="yes")
    assert result is True


# Test 2: Scorer created with @scorer decorator
def test_scorer_decorator_serialization():
    """Test that a scorer created with @scorer decorator can be serialized."""
    @scorer
    def my_scorer(outputs):
        return outputs == "hello"
    
    # Serialize
    serialized = my_scorer.model_dump()
    
    # Check fields
    assert serialized["name"] == "my_scorer"
    assert "__call___source" in serialized
    
    # Deserialize - for now, just check that serialization works
    # We'll implement full deserialization in the next iteration
    

# Test 3: Scorer with multiple parameters
def test_scorer_with_multiple_params():
    """Test scorer with multiple parameters."""
    @scorer
    def complex_scorer(inputs, outputs, expectations):
        return outputs == expectations.get("expected")
    
    serialized = complex_scorer.model_dump()
    assert "__call___source" in serialized
    assert "outputs == expectations.get" in serialized["__call___source"]


# Test 4: Scorer returning Feedback object
def test_scorer_returning_feedback():
    """Test scorer that returns Feedback object."""
    @scorer
    def feedback_scorer(outputs):
        return Feedback(
            value=len(outputs) > 10,
            rationale="Output is long enough"
        )
    
    serialized = feedback_scorer.model_dump()
    assert "__call___source" in serialized
    assert "Feedback" in serialized["__call___source"]


# Test 5: Scorer with custom name
def test_scorer_with_custom_name():
    """Test scorer with custom name."""
    @scorer(name="custom_name")
    def some_scorer(outputs):
        return True
    
    serialized = some_scorer.model_dump()
    assert serialized["name"] == "custom_name"


# Test 6: Scorer with aggregations
def test_scorer_with_aggregations():
    """Test scorer with aggregations."""
    @scorer(aggregations=["mean", "max"])
    def agg_scorer(outputs):
        return len(outputs)
    
    serialized = agg_scorer.model_dump()
    assert serialized["aggregations"] == ["mean", "max"] 


# Test 7: Scorer with imports in function body
def test_scorer_with_imports():
    """Test scorer that has imports in the function body."""
    @scorer
    def import_scorer(outputs):
        import re
        # Remove all non-alphanumeric characters
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', outputs)
        return len(cleaned) > 5
    
    serialized = import_scorer.model_dump()
    assert "__call___source" in serialized
    assert "import re" in serialized["__call___source"]
    assert "re.sub" in serialized["__call___source"]


# Test 8: Test source code extraction for custom __call__ method
class ComplexScorer(Scorer):
    # Use a private attribute to store threshold
    def __init__(self, threshold: int = 10, **kwargs):
        super().__init__(**kwargs)
        self._threshold = threshold
    
    def __call__(self, *, outputs):
        """Custom call method with docstring."""
        # This is a comment
        word_count = len(outputs.split())
        return word_count > self._threshold


def test_complex_scorer_serialization():
    """Test scorer with custom implementation and docstring."""
    scorer = ComplexScorer(name="word_count_scorer", threshold=20)
    serialized = scorer.model_dump()
    
    assert serialized["name"] == "word_count_scorer"
    assert "__call___source" in serialized
    # Check that the source code (not docstring) is extracted
    assert "word_count = len(outputs.split())" in serialized["__call___source"]
    assert "return word_count > self._threshold" in serialized["__call___source"]


# Test 9: Scorer returning list of Feedback objects
def test_scorer_returning_feedback_list():
    """Test scorer that returns a list of Feedback objects."""
    @scorer
    def multi_feedback_scorer(outputs):
        feedbacks = []
        for i, line in enumerate(outputs.split('\n')):
            feedbacks.append(
                Feedback(
                    name=f"line_{i}",
                    value=len(line) > 0,
                    rationale=f"Line {i} is {'not ' if len(line) == 0 else ''}empty"
                )
            )
        return feedbacks
    
    serialized = multi_feedback_scorer.model_dump()
    assert "__call___source" in serialized
    assert "feedbacks = []" in serialized["__call___source"]
    assert "enumerate(outputs.split" in serialized["__call___source"]


# Test 10: Test JSON serialization/deserialization
def test_scorer_json_serialization():
    """Test that scorer can be serialized to JSON and back."""
    @scorer(name="json_test", aggregations=["mean"])
    def json_scorer(outputs):
        return len(outputs)
    
    # Convert to dict, then to JSON
    serialized_dict = json_scorer.model_dump()
    json_str = json.dumps(serialized_dict)
    
    # Deserialize from JSON
    deserialized_dict = json.loads(json_str)
    
    # Verify the structure
    assert deserialized_dict["name"] == "json_test"
    assert deserialized_dict["aggregations"] == ["mean"]
    assert "__call___source" in deserialized_dict
    assert "return len(outputs)" in deserialized_dict["__call___source"]


# Test 11: Test edge case - scorer with no body
def test_scorer_with_minimal_body():
    """Test scorer with minimal function body."""
    @scorer
    def minimal_scorer(outputs):
        return True
    
    serialized = minimal_scorer.model_dump()
    assert "__call___source" in serialized
    assert "return True" in serialized["__call___source"]


# Test 12: Test that run method source is correctly extracted
def test_run_method_extraction():
    """Test that the run method source code is extracted."""
    scorer_instance = SimpleScorer(name="run_test")
    serialized = scorer_instance.model_dump()
    
    assert "run_source" in serialized
    # The run method filters parameters and calls self.__call__
    assert "sig = inspect.signature(self.__call__)" in serialized["run_source"]
    assert "filtered = {k: v for k, v in merged.items() if k in sig.parameters}" in serialized["run_source"]


# Test 13: Test model_validate preserves source code
def test_model_validate_preserves_source():
    """Test that model_validate preserves source code in attributes."""
    original = SimpleScorer(name="validate_test")
    serialized = original.model_dump()
    
    # Deserialize using model_validate
    deserialized = Scorer.model_validate(serialized)
    
    # Check that private attributes are preserved
    assert hasattr(deserialized, '_run_source')
    assert hasattr(deserialized, '_call_source')
    assert hasattr(deserialized, '_call_signature')
    
    # Check that source code is stored
    assert deserialized._run_source is not None
    assert deserialized._call_source is not None
    assert deserialized._call_signature is not None 


# Test 14: Test scorer with multiline function
def test_scorer_multiline_function():
    """Test scorer with complex multiline logic."""
    @scorer
    def multiline_scorer(inputs, outputs):
        # Count questions in inputs
        question_count = inputs.count('?')
        
        # Count sentences in outputs
        sentences = outputs.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Complex logic
        if question_count > 0:
            ratio = sentence_count / question_count
            return ratio >= 1.0
        else:
            return sentence_count > 0
    
    serialized = multiline_scorer.model_dump()
    assert "__call___source" in serialized
    # Check various parts of the multiline function
    assert "question_count = inputs.count('?')" in serialized["__call___source"]
    assert "sentences = outputs.split('.')" in serialized["__call___source"]
    assert "if question_count > 0:" in serialized["__call___source"]


# Test 15: Test that builtin scorers don't break serialization
def test_builtin_scorer_no_source_extraction():
    """Test that builtin scorers can still be serialized even without source extraction."""
    from mlflow.genai.scorers import relevance_to_query
    
    # This should not raise an error
    serialized = relevance_to_query.model_dump()
    
    # Builtin scorers won't have source code fields
    assert "run_source" not in serialized
    assert "__call___source" not in serialized
    assert "name" in serialized
    assert serialized["name"] == "relevance_to_query"


# Test 16: Test end-to-end serialization workflow
def test_end_to_end_serialization_workflow():
    """Test the complete workflow of creating, serializing, storing, and loading a scorer."""
    # Step 1: Create a scorer
    @scorer(name="sentiment_scorer", aggregations=["mean", "min", "max"])
    def analyze_sentiment(outputs):
        # Simple sentiment analysis based on keywords
        positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "horrible", "poor"]
        
        text_lower = outputs.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)
        
        # Return sentiment score from -1 to 1
        if positive_count + negative_count == 0:
            return 0.0
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    # Step 2: Serialize to dict
    serialized = analyze_sentiment.model_dump()
    
    # Step 3: Convert to JSON (simulating storage)
    json_str = json.dumps(serialized, indent=2)
    
    # Parse JSON to verify contents (more robust than string matching)
    parsed_json = json.loads(json_str)
    assert parsed_json["name"] == "sentiment_scorer"
    assert parsed_json["aggregations"] == ["mean", "min", "max"]
    
    # Verify source code is included
    assert "positive_words =" in json_str
    assert "negative_words =" in json_str
    
    # Step 4: Load from JSON
    loaded_dict = json.loads(json_str)
    
    # Step 5: Reconstruct scorer (partial - just validate structure)
    reconstructed = Scorer.model_validate(loaded_dict)
    
    # Verify reconstructed scorer has correct attributes
    assert reconstructed.name == "sentiment_scorer"
    assert reconstructed.aggregations == ["mean", "min", "max"]
    assert hasattr(reconstructed, '_call_source')
    assert reconstructed._call_source is not None
    assert "positive_count - negative_count" in reconstructed._call_source 


# Test 17: Test that deserialized custom scorers can be invoked
def test_custom_scorer_deserialization_and_invocation():
    """Test that custom scorers can be serialized, deserialized, and invoked."""
    # Test 1: Simple scorer
    @scorer
    def length_scorer(outputs):
        return len(outputs)
    
    # Serialize
    serialized = length_scorer.model_dump()
    
    # Check that source code is included
    assert "__call___source" in serialized
    assert "__call___signature" in serialized
    
    # Deserialize
    deserialized = Scorer.model_validate(serialized)
    
    # Test invocation
    result = deserialized.run(outputs="Hello, World!")
    assert result == 13
    
    # Test 2: Scorer with multiple parameters
    @scorer
    def comparison_scorer(inputs, outputs):
        return len(outputs) > len(str(inputs))
    
    serialized = comparison_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)
    
    # Test invocation
    result = deserialized.run(inputs="Hi", outputs="Hello there!")
    assert result is True
    
    result = deserialized.run(inputs="This is a long input", outputs="Short")
    assert result is False
    
    # Test 3: Scorer returning Feedback
    @scorer
    def feedback_scorer(outputs):
        is_long = len(outputs) > 50
        return Feedback(
            value=is_long,
            rationale=f"Output has {len(outputs)} characters"
        )
    
    serialized = feedback_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)
    
    result = deserialized.run(outputs="This is a short text")
    assert isinstance(result, Feedback)
    assert result.value is False
    assert "20 characters" in result.rationale


# Test 18: Test direct Scorer subclass deserialization and invocation
def test_direct_scorer_subclass_deserialization():
    """Test that direct Scorer subclasses can be deserialized and invoked."""
    class WordCountScorer(Scorer):
        def __call__(self, *, outputs):
            words = outputs.split()
            return len(words)
    
    # Create instance
    scorer = WordCountScorer(name="word_count")
    
    # Serialize
    serialized = scorer.model_dump()
    
    # Deserialize - note we need to use the base Scorer class
    deserialized = Scorer.model_validate(serialized)
    
    # Test invocation
    result = deserialized.run(outputs="Hello world from MLflow")
    assert result == 4


# Test 19: Test scorer with imports inside function
def test_scorer_with_internal_imports_deserialization():
    """Test scorer with imports inside the function can be deserialized."""
    @scorer
    def regex_scorer(outputs):
        import re
        # Count all alphanumeric words
        words = re.findall(r'\b\w+\b', outputs)
        return len(words)
    
    # Serialize
    serialized = regex_scorer.model_dump()
    
    # Deserialize
    deserialized = Scorer.model_validate(serialized)
    
    # Test invocation
    result = deserialized.run(outputs="Hello, World! How are you?")
    assert result == 5  # ["Hello", "World", "How", "are", "you"]


# Test 20: Test complex scorer with multiple operations
def test_complex_scorer_deserialization():
    """Test complex scorer with multiple operations."""
    @scorer(name="sentiment_analysis")
    def analyze_sentiment(outputs):
        positive_words = ["good", "great", "excellent", "love", "wonderful"]
        negative_words = ["bad", "terrible", "hate", "awful", "horrible"]
        
        text_lower = outputs.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            sentiment = "neutral"
            score = 0.5
        else:
            score = positive_count / (positive_count + negative_count)
            if score > 0.6:
                sentiment = "positive"
            elif score < 0.4:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return Feedback(
            value=sentiment,
            rationale=f"Found {positive_count} positive and {negative_count} negative words",
            metadata={"score": score}
        )
    
    # Serialize
    serialized = analyze_sentiment.model_dump()
    
    # Deserialize
    deserialized = Scorer.model_validate(serialized)
    
    # Test various inputs
    result1 = deserialized.run(outputs="This is a great and wonderful day!")
    assert isinstance(result1, Feedback)
    assert result1.value == "positive"
    assert "2 positive and 0 negative" in result1.rationale
    
    result2 = deserialized.run(outputs="This is terrible and awful")
    assert result2.value == "negative"
    assert "0 positive and 2 negative" in result2.rationale
    
    result3 = deserialized.run(outputs="This is a normal day")
    assert result3.value == "neutral"


# Test 21: Test builtin scorer serialization (shouldn't include source)
def test_builtin_scorer_serialization_no_reconstruction():
    """Test that builtin scorers can be serialized but don't include source code."""
    from mlflow.genai.scorers import correctness, safety
    
    # Test correctness scorer
    serialized = correctness.model_dump()
    assert "name" in serialized
    assert serialized["name"] == "correctness"
    # Builtin scorers shouldn't have source code
    assert "__call___source" not in serialized
    assert "run_source" not in serialized
    
    # Test safety scorer
    serialized = safety.model_dump()
    assert serialized["name"] == "safety"
    assert "__call___source" not in serialized
    
    # Built-in scorers maintain their functionality through class inheritance
    # so they don't need source code reconstruction


# Test 22: Test error handling for invalid source code
def test_error_handling_for_invalid_reconstruction():
    """Test that scorers handle reconstruction errors gracefully."""
    import warnings
    
    # Create a scorer and manually corrupt its source code
    @scorer
    def simple_scorer(outputs):
        return True
    
    serialized = simple_scorer.model_dump()
    
    # Corrupt the source code
    serialized["__call___source"] = "invalid python code {"
    
    # Deserialization should succeed but warn about reconstruction failure
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deserialized = Scorer.model_validate(serialized)
        
        # Check that a warning was issued
        assert len(w) == 1
        assert "Failed to set up dynamic call" in str(w[0].message)
    
    # The deserialized scorer should still have basic properties
    assert deserialized.name == "simple_scorer"
    
    # But invocation will fail with NotImplementedError
    with pytest.raises(NotImplementedError):
        deserialized.run(outputs="test")


# Test 23: Test that builtin scorers can be serialized, deserialized and invoked
def test_builtin_scorer_full_workflow():
    """Test that builtin scorers can be serialized, deserialized, and invoked successfully."""
    from mlflow.genai.scorers import correctness, safety, relevance_to_query
    
    # Test 1: Correctness scorer (requires expectations)
    # Serialize
    correctness_serialized = correctness.model_dump()
    assert "name" in correctness_serialized
    assert correctness_serialized["name"] == "correctness"
    # Built-in scorers include their required columns
    assert "required_columns" in correctness_serialized
    
    # Deserialize
    correctness_deserialized = type(correctness).model_validate(correctness_serialized)
    assert correctness_deserialized.name == "correctness"
    assert hasattr(correctness_deserialized, '__call__')
    assert hasattr(correctness_deserialized, 'run')
    # Check that required_columns are preserved
    assert correctness_deserialized.required_columns == correctness.required_columns
    
    # Test 2: Safety scorer (doesn't require expectations)
    # Serialize
    safety_serialized = safety.model_dump()
    assert "name" in safety_serialized
    assert safety_serialized["name"] == "safety"
    
    # Deserialize
    safety_deserialized = type(safety).model_validate(safety_serialized)
    assert safety_deserialized.name == "safety"
    
    # Test 3: Relevance to query scorer
    # Serialize
    relevance_serialized = relevance_to_query.model_dump()
    assert "name" in relevance_serialized
    assert relevance_serialized["name"] == "relevance_to_query"
    
    # Deserialize
    relevance_deserialized = type(relevance_to_query).model_validate(relevance_serialized)
    assert relevance_deserialized.name == "relevance_to_query"
    
    # Test 4: Custom configured builtin scorer
    custom_correctness = correctness.with_config(name="my_correctness")
    
    # Serialize
    custom_serialized = custom_correctness.model_dump()
    assert custom_serialized["name"] == "my_correctness"
    
    # Deserialize - note we need to use the original class type
    custom_deserialized = type(custom_correctness).model_validate(custom_serialized)
    assert custom_deserialized.name == "my_correctness"
    
    # Test 5: Verify that deserialized scorers are functional
    # We can't directly invoke them because they require LLM calls,
    # but we can verify they have all the necessary methods and attributes
    for scorer in [correctness_deserialized, safety_deserialized, relevance_deserialized]:
        assert callable(scorer.__call__)
        assert callable(scorer.run)
        assert hasattr(scorer, 'name')
        assert hasattr(scorer, 'aggregations')
        if hasattr(scorer, 'required_columns'):
            assert isinstance(scorer.required_columns, set)
    
    # Test 6: Verify that deserialized built-in scorers maintain immutability
    from mlflow.exceptions import MlflowException
    with pytest.raises(MlflowException, match="Built-in scorer fields are immutable"):
        correctness_deserialized.name = "new_name"


# Test 24: Test mixed scorer types in the same workflow
def test_mixed_scorer_serialization():
    """Test that custom and builtin scorers can be serialized/deserialized in the same workflow."""
    from mlflow.genai.scorers import scorer, correctness
    
    # Create a custom scorer
    @scorer
    def length_checker(outputs):
        return len(outputs) > 10
    
    # Get a builtin scorer
    builtin = correctness
    
    # Serialize both
    custom_serialized = length_checker.model_dump()
    builtin_serialized = builtin.model_dump()
    
    # Check serialization differences
    assert "__call___source" in custom_serialized  # Custom has source
    assert "__call___source" not in builtin_serialized  # Builtin doesn't
    
    # Deserialize both
    custom_deserialized = Scorer.model_validate(custom_serialized)
    builtin_deserialized = type(builtin).model_validate(builtin_serialized)
    
    # Test custom scorer invocation
    assert custom_deserialized.run(outputs="Hello World!") is True
    assert custom_deserialized.run(outputs="Hi") is False
    
    # Verify builtin scorer properties
    assert builtin_deserialized.name == "correctness"
    assert hasattr(builtin_deserialized, 'required_columns') 