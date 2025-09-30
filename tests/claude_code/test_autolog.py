"""Minimal tests for mlflow.claude_code.autolog functionality."""


def test_autolog_function_exists():
    """Test that autolog function is exported."""
    from mlflow.claude_code import autolog

    assert callable(autolog)


def test_autolog_runs_without_sdk():
    """Test that autolog handles missing SDK gracefully."""
    from mlflow.claude_code import autolog

    # Should not raise exception when SDK is not installed
    autolog()
