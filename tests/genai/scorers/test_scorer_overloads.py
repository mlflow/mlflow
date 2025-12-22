"""Tests for the @scorer decorator type overloads.

This test verifies that the @scorer decorator has proper @overload annotations
so that type checkers can correctly infer the return type based on usage:

1. Bare usage: `@scorer` should return `Scorer`
2. Parameterized usage: `@scorer(name="...")` should return `Callable[[...], Scorer]`

See: https://github.com/mlflow/mlflow/issues/19567
"""

import inspect

from mlflow.entities import Feedback
from mlflow.genai.scorers import Scorer, scorer


def test_scorer_has_overload_annotations():
    """Test that the scorer function has proper @overload annotations.

    We verify that @overload decorators are defined by checking that
    the module contains the overload function signatures. This ensures
    type checkers can correctly infer:
    - @scorer (bare) -> Scorer
    - @scorer(name="...") (parameterized) -> Callable[[...], Scorer]

    Without overloads, type checkers would infer a union type which causes
    errors like: "Type 'CustomScorer | partial[CustomScorer]' is not
    assignable to type 'list[Scorer]'"
    """
    # Get the source code of the base module
    import mlflow.genai.scorers.base as base_module

    source = inspect.getsource(base_module)

    # Verify that overload decorators are present for the scorer function
    # The pattern should have two @overload decorated functions before the main scorer
    assert source.count("@overload\ndef scorer(") == 2, (
        "Expected 2 @overload annotations for scorer function "
        "(one for bare decorator usage, one for parameterized usage). "
        "See https://github.com/mlflow/mlflow/issues/19567"
    )

    # Verify the return type annotations are correct in the overloads
    assert "-> Scorer: ..." in source, (
        "First overload should return 'Scorer' for bare decorator usage"
    )
    assert "-> Callable[[_F], Scorer]: ..." in source, (
        "Second overload should return 'Callable[[_F], Scorer]' for parameterized usage"
    )


def test_scorer_bare_usage_returns_scorer():
    @scorer
    def my_scorer(outputs: str) -> Feedback:
        return Feedback(value=True)

    assert isinstance(my_scorer, Scorer)


def test_scorer_parameterized_usage_returns_scorer():
    @scorer(name="custom_scorer")
    def my_scorer(outputs: str) -> Feedback:
        return Feedback(value=True)

    assert isinstance(my_scorer, Scorer)
    assert my_scorer.name == "custom_scorer"


def test_scorer_list_type_annotation():
    """Test that decorated scorers can be used in a list[Scorer] without type issues.

    This is the exact use case from the bug report.
    """

    @scorer
    def my_scorer(outputs: str) -> Feedback:
        return Feedback(value=True)

    @scorer(name="another_scorer")
    def another_scorer(outputs: str) -> Feedback:
        return Feedback(value=False)

    # This should work without type errors
    scorers: list[Scorer] = [my_scorer, another_scorer]
    assert len(scorers) == 2
    assert all(isinstance(s, Scorer) for s in scorers)
