import re

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


def test_scorer_serialization():
    """Test that Scorer class can be serialized and deserialized."""
    from mlflow.genai.scorers import Scorer, scorer
    
    # Test 1: Simple scorer with decorator
    @scorer
    def simple_scorer(outputs):
        return outputs == "correct"
    
    serialized = simple_scorer.model_dump()
    assert "name" in serialized
    assert serialized["name"] == "simple_scorer"
    assert "__call___source" in serialized
    assert "outputs == \"correct\"" in serialized["__call___source"]
    
    # Test 2: Scorer with custom name and aggregations
    @scorer(name="custom", aggregations=["mean", "max"])
    def custom_scorer(inputs, outputs):
        return len(outputs) > len(inputs)
    
    serialized = custom_scorer.model_dump()
    assert serialized["name"] == "custom"
    assert serialized["aggregations"] == ["mean", "max"]
    assert "len(outputs) > len(inputs)" in serialized["__call___source"]
    
    # Test 3: Direct Scorer subclass
    class MyScorer(Scorer):
        def __call__(self, *, outputs):
            return True
    
    my_scorer = MyScorer(name="my_scorer")
    serialized = my_scorer.model_dump()
    assert serialized["name"] == "my_scorer"
    assert "__call___source" in serialized
    assert "return True" in serialized["__call___source"]
