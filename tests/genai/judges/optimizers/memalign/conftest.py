"""Shared test fixtures for memalign optimizer tests."""

import pytest

from mlflow.genai.judges.optimizers.memalign.utils import (
    _get_model_max_input_tokens,
    truncate_to_token_limit,
)


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Clear lru_cache before each test to ensure mocks work correctly."""
    _get_model_max_input_tokens.cache_clear()
    truncate_to_token_limit.cache_clear()
    yield
    _get_model_max_input_tokens.cache_clear()
    truncate_to_token_limit.cache_clear()
