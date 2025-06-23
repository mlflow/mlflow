import pytest

import mlflow
from mlflow.tracing.config import TracingConfig, get_config


@pytest.fixture(autouse=True)
def reset_tracing_config():
    mlflow.tracing.reset()


def test_tracing_config_default_values():
    """Test that TracingConfig has expected default values."""
    config = TracingConfig()
    assert config.span_processors == []


def test_configure():
    # Default config
    assert get_config().span_processors == []

    def dummy_filter(span):
        pass

    mlflow.tracing.configure(span_processors=[dummy_filter])
    assert get_config().span_processors == [dummy_filter]

    mlflow.tracing.configure(span_processors=[])
    assert get_config().span_processors == []


def test_configure_empty_call():
    def dummy_filter(span):
        pass

    mlflow.tracing.configure(span_processors=[dummy_filter])
    assert get_config().span_processors == [dummy_filter]

    # No-op
    mlflow.tracing.configure()
    assert get_config().span_processors == [dummy_filter]


def test_reset_config():
    def filter1(span):
        pass

    assert get_config().span_processors == []

    mlflow.tracing.configure(span_processors=[filter1])
    assert get_config().span_processors == [filter1]

    mlflow.tracing.reset()
    assert get_config().span_processors == []


def test_configure_context_manager():
    def filter1(span):
        return

    def filter2(span):
        return

    # Set initial config
    mlflow.tracing.configure(span_processors=[filter1])

    assert get_config().span_processors == [filter1]

    with mlflow.tracing.configure(span_processors=[filter2]):
        assert get_config().span_processors == [filter2]

        with mlflow.tracing.configure(span_processors=[filter1, filter2]):
            assert get_config().span_processors == [filter1, filter2]

        # Config should be restored after context exit
        assert get_config().span_processors == [filter2]

    assert get_config().span_processors == [filter1]


def test_context_manager_with_exception():
    """Test context manager restores config even when exception occurs."""

    def filter1(span):
        pass

    def filter2(span):
        pass

    mlflow.tracing.configure(span_processors=[filter1])

    with pytest.raises(ValueError, match="test error"):  # noqa: PT012
        with mlflow.tracing.configure(span_processors=[filter2]):
            assert get_config().span_processors == [filter2]
            raise ValueError("test error")

    # Config should be restored despite exception
    assert get_config().span_processors == [filter1]


def test_context_manager_with_non_copyable_callable():
    """Test context manager handles non-copyable callables gracefully."""

    # Lambda functions are not deepcopyable
    lambda_filter = lambda span: None  # noqa: E731

    # Configure with a lambda function
    mlflow.tracing.configure(span_processors=[lambda_filter])
    assert get_config().span_processors == [lambda_filter]

    def regular_filter(span):
        pass

    # Context manager should still work with non-copyable callables
    with mlflow.tracing.configure(span_processors=[regular_filter]):
        assert get_config().span_processors == [regular_filter]

    # Config should be restored
    assert get_config().span_processors == [lambda_filter]
