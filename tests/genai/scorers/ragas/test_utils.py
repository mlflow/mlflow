from unittest.mock import Mock, patch

import pytest

from mlflow.entities.span import Span, SpanAttributeKey, SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.ragas.models import create_ragas_model
from mlflow.genai.scorers.ragas.utils import map_scorer_inputs_to_ragas_sample


def test_create_ragas_model_databricks():
    model = create_ragas_model("databricks")
    assert model.__class__.__name__ == "DatabricksRagasLLM"


def test_create_ragas_model_databricks_serving_endpoint():
    model = create_ragas_model("databricks:/my-endpoint")
    assert model.__class__.__name__ == "DatabricksServingEndpointRagasLLM"


def test_create_ragas_model_openai():
    model = create_ragas_model("openai:/gpt-4")
    assert model.__class__.__name__ == "LiteLLMStructuredLLM"


def test_create_ragas_model_with_provider_no_slash():
    model = create_ragas_model("openai:gpt-4")
    assert model.__class__.__name__ == "LiteLLMStructuredLLM"


def test_create_ragas_model_rejects_model_name_only():
    with pytest.raises(MlflowException, match="Invalid model_uri format"):
        create_ragas_model("gpt-4")


def test_map_scorer_inputs_to_ragas_sample_basic():
    sample = map_scorer_inputs_to_ragas_sample(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
    )

    assert sample.user_input == "What is MLflow?"
    assert sample.response == "MLflow is a platform"
    assert sample.reference is None
    assert sample.retrieved_contexts is None


def test_map_scorer_inputs_to_ragas_sample_with_expectations():
    expectations = {
        "expected_output": "MLflow is an open source platform",
    }

    sample = map_scorer_inputs_to_ragas_sample(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations=expectations,
    )

    assert sample.reference == "MLflow is an open source platform"


def test_map_scorer_inputs_to_ragas_sample_with_trace():
    span = Mock(spec=Span)
    span.name = "retrieval"
    span.span_type = SpanType.RETRIEVER
    span.attributes = {
        SpanAttributeKey.OUTPUTS: {
            "documents": [
                {"content": "Document 1"},
                {"content": "Document 2"},
            ]
        }
    }
    span.span_id = "span-1"

    trace = Mock()
    trace.search_spans.return_value = [span]
    trace.data.request_metadata = {}

    with patch(
        "mlflow.genai.scorers.ragas.utils.extract_retrieval_context_from_trace"
    ) as mock_extract:
        mock_extract.return_value = {"span-1": ["Document 1", "Document 2"]}

        sample = map_scorer_inputs_to_ragas_sample(
            inputs="What is MLflow?",
            outputs="MLflow is a platform",
            trace=trace,
        )

        assert sample.retrieved_contexts is not None
        assert len(sample.retrieved_contexts) == 2
        assert "Document 1" in sample.retrieved_contexts
        assert "Document 2" in sample.retrieved_contexts
