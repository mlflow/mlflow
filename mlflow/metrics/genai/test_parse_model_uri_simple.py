from mlflow.metrics.genai.model_utils import _parse_model_uri
import pytest

def test_parse_model_uri_with_underscore_provider():
    provider, model = _parse_model_uri("vertex_ai:/gemini-2.0")
    assert provider == "vertex_ai"
    assert model == "gemini-2.0"

def test_parse_model_uri_standard_provider():
    provider, model = _parse_model_uri("openai:/gpt-4.1-mini")
    assert provider == "openai"
    assert model == "gpt-4.1-mini"

def test_parse_model_uri_invalid_format():
    with pytest.raises(Exception):
        _parse_model_uri("vertex_ai:gemini")  # missing '/'
