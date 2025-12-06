from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException


def test_phoenix_hallucination_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixHallucinationScorer

        scorer = PhoenixHallucinationScorer()
        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            scorer(
                inputs={"query": "test"},
                outputs="test output",
                context="test context",
            )


def test_phoenix_relevance_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixRelevanceScorer

        scorer = PhoenixRelevanceScorer()
        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            scorer(
                inputs={"query": "test"},
                context="test context",
            )


def test_phoenix_toxicity_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixToxicityScorer

        scorer = PhoenixToxicityScorer()
        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            scorer(outputs="test output")


def test_phoenix_qa_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixQAScorer

        scorer = PhoenixQAScorer()
        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            scorer(
                inputs={"query": "test"},
                outputs="test output",
                context="test context",
            )


def test_phoenix_summarization_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixSummarizationScorer

        scorer = PhoenixSummarizationScorer()
        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            scorer(
                inputs={"document": "long text"},
                outputs="summary",
            )


def test_phoenix_hallucination_scorer_default_name():
    from mlflow.genai.scorers.phoenix.phoenix import PhoenixHallucinationScorer

    scorer = PhoenixHallucinationScorer()
    assert scorer.name == "phoenix_hallucination"


def test_phoenix_relevance_scorer_default_name():
    from mlflow.genai.scorers.phoenix.phoenix import PhoenixRelevanceScorer

    scorer = PhoenixRelevanceScorer()
    assert scorer.name == "phoenix_relevance"


def test_phoenix_toxicity_scorer_default_name():
    from mlflow.genai.scorers.phoenix.phoenix import PhoenixToxicityScorer

    scorer = PhoenixToxicityScorer()
    assert scorer.name == "phoenix_toxicity"


def test_phoenix_qa_scorer_default_name():
    from mlflow.genai.scorers.phoenix.phoenix import PhoenixQAScorer

    scorer = PhoenixQAScorer()
    assert scorer.name == "phoenix_qa"


def test_phoenix_summarization_scorer_default_name():
    from mlflow.genai.scorers.phoenix.phoenix import PhoenixSummarizationScorer

    scorer = PhoenixSummarizationScorer()
    assert scorer.name == "phoenix_summarization"


def test_phoenix_scorer_custom_name():
    from mlflow.genai.scorers.phoenix.phoenix import PhoenixHallucinationScorer

    scorer = PhoenixHallucinationScorer(name="custom_hallucination_check")
    assert scorer.name == "custom_hallucination_check"


def test_phoenix_scorer_custom_model():
    from mlflow.genai.scorers.phoenix.phoenix import PhoenixHallucinationScorer

    scorer = PhoenixHallucinationScorer(model_name="gpt-4")
    assert scorer.model_name == "gpt-4"


def test_phoenix_hallucination_scorer_with_mock():
    # Phoenix evaluate() returns Tuple[str, Optional[float], Optional[str]]
    # (label, score, explanation)
    # Phoenix hallucination: score=0 means factual (good), score=1 means hallucinated (bad)
    # We test with score=0.1 (mostly factual), which should invert to 0.9 in MLflow
    mock_result = ("factual", 0.1, "The output is grounded in the context.")

    mock_evaluator_class = mock.MagicMock()
    mock_evaluator_instance = mock.MagicMock()
    mock_evaluator_instance.evaluate.return_value = mock_result
    mock_evaluator_class.return_value = mock_evaluator_instance

    mock_model_class = mock.MagicMock()

    mock_phoenix_evals = mock.MagicMock()
    mock_phoenix_evals.HallucinationEvaluator = mock_evaluator_class
    mock_phoenix_evals.OpenAIModel = mock_model_class

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix_evals}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixHallucinationScorer

        scorer = PhoenixHallucinationScorer()
        result = scorer(
            inputs={"query": "What is the capital of France?"},
            outputs="Paris is the capital of France.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "phoenix_hallucination"
        # Phoenix score 0.1 (low hallucination) inverts to 0.9 (high factuality) in MLflow
        assert result.value == 0.9
        assert "grounded" in result.rationale

        # Verify evaluate was called with record dict
        call_args = mock_evaluator_instance.evaluate.call_args
        assert "record" in call_args.kwargs
        record = call_args.kwargs["record"]
        assert record["input"] == "What is the capital of France?"
        assert record["output"] == "Paris is the capital of France."
        assert record["reference"] == "France is a country in Europe. Its capital is Paris."


def test_phoenix_relevance_scorer_with_mock():
    mock_result = ("relevant", 0.9, "The context is relevant to the query.")

    mock_evaluator_class = mock.MagicMock()
    mock_evaluator_instance = mock.MagicMock()
    mock_evaluator_instance.evaluate.return_value = mock_result
    mock_evaluator_class.return_value = mock_evaluator_instance

    mock_model_class = mock.MagicMock()

    mock_phoenix_evals = mock.MagicMock()
    mock_phoenix_evals.RelevanceEvaluator = mock_evaluator_class
    mock_phoenix_evals.OpenAIModel = mock_model_class

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix_evals}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixRelevanceScorer

        scorer = PhoenixRelevanceScorer()
        result = scorer(
            inputs={"query": "What is machine learning?"},
            context="Machine learning is a subset of AI.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "phoenix_relevance"
        assert result.value == 0.9


def test_phoenix_toxicity_scorer_with_mock():
    # Phoenix toxicity: score=0 means non-toxic (good), score=1 means toxic (bad)
    # We test with score=0.05 (mostly non-toxic), which should invert to 0.95 in MLflow
    mock_result = ("non-toxic", 0.05, "The content is safe.")

    mock_evaluator_class = mock.MagicMock()
    mock_evaluator_instance = mock.MagicMock()
    mock_evaluator_instance.evaluate.return_value = mock_result
    mock_evaluator_class.return_value = mock_evaluator_instance

    mock_model_class = mock.MagicMock()

    mock_phoenix_evals = mock.MagicMock()
    mock_phoenix_evals.ToxicityEvaluator = mock_evaluator_class
    mock_phoenix_evals.OpenAIModel = mock_model_class

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix_evals}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixToxicityScorer

        scorer = PhoenixToxicityScorer()
        result = scorer(outputs="This is a friendly response.")

        assert isinstance(result, Feedback)
        assert result.name == "phoenix_toxicity"
        # Phoenix score 0.05 (low toxicity) inverts to 0.95 (high safety) in MLflow
        assert result.value == 0.95


def test_phoenix_qa_scorer_with_mock():
    mock_result = ("correct", 1.0, "The answer is correct.")

    mock_evaluator_class = mock.MagicMock()
    mock_evaluator_instance = mock.MagicMock()
    mock_evaluator_instance.evaluate.return_value = mock_result
    mock_evaluator_class.return_value = mock_evaluator_instance

    mock_model_class = mock.MagicMock()

    mock_phoenix_evals = mock.MagicMock()
    mock_phoenix_evals.QAEvaluator = mock_evaluator_class
    mock_phoenix_evals.OpenAIModel = mock_model_class

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix_evals}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixQAScorer

        scorer = PhoenixQAScorer()
        result = scorer(
            inputs={"query": "What is 2+2?"},
            outputs="4",
            context="Basic arithmetic: 2+2=4",
        )

        assert isinstance(result, Feedback)
        assert result.name == "phoenix_qa"
        assert result.value == 1.0


def test_phoenix_summarization_scorer_with_mock():
    mock_result = ("good", 0.95, "Good summarization quality.")

    mock_evaluator_class = mock.MagicMock()
    mock_evaluator_instance = mock.MagicMock()
    mock_evaluator_instance.evaluate.return_value = mock_result
    mock_evaluator_class.return_value = mock_evaluator_instance

    mock_model_class = mock.MagicMock()

    mock_phoenix_evals = mock.MagicMock()
    mock_phoenix_evals.SummarizationEvaluator = mock_evaluator_class
    mock_phoenix_evals.OpenAIModel = mock_model_class

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix_evals}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixSummarizationScorer

        scorer = PhoenixSummarizationScorer()
        result = scorer(
            inputs={"document": "Long document text..."},
            outputs="Brief summary.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "phoenix_summarization"
        assert result.value == 0.95


def test_phoenix_scorer_label_only_result():
    # Test when Phoenix returns only label (no score)
    mock_result = ("factual", None, None)

    mock_evaluator_class = mock.MagicMock()
    mock_evaluator_instance = mock.MagicMock()
    mock_evaluator_instance.evaluate.return_value = mock_result
    mock_evaluator_class.return_value = mock_evaluator_instance

    mock_model_class = mock.MagicMock()

    mock_phoenix_evals = mock.MagicMock()
    mock_phoenix_evals.HallucinationEvaluator = mock_evaluator_class
    mock_phoenix_evals.OpenAIModel = mock_model_class

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix_evals}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixHallucinationScorer

        scorer = PhoenixHallucinationScorer()
        result = scorer(
            inputs={"query": "test"},
            outputs="test output",
            context="test context",
        )

        assert isinstance(result, Feedback)
        # When label is "factual" and no score, should derive score as 1.0
        assert result.value == 1.0
        assert "factual" in result.rationale.lower()


def test_phoenix_scorer_negative_label_result():
    # Test when Phoenix returns negative label
    mock_result = ("hallucinated", None, "The output contains made-up information.")

    mock_evaluator_class = mock.MagicMock()
    mock_evaluator_instance = mock.MagicMock()
    mock_evaluator_instance.evaluate.return_value = mock_result
    mock_evaluator_class.return_value = mock_evaluator_instance

    mock_model_class = mock.MagicMock()

    mock_phoenix_evals = mock.MagicMock()
    mock_phoenix_evals.HallucinationEvaluator = mock_evaluator_class
    mock_phoenix_evals.OpenAIModel = mock_model_class

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix_evals}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix.phoenix import PhoenixHallucinationScorer

        scorer = PhoenixHallucinationScorer()
        result = scorer(
            inputs={"query": "test"},
            outputs="test output",
            context="test context",
        )

        assert isinstance(result, Feedback)
        # When label is "hallucinated" and no score, should derive score as 0.0
        assert result.value == 0.0


def test_phoenix_scorer_exports():
    from mlflow.genai.scorers.phoenix import (
        PhoenixHallucinationScorer,
        PhoenixQAScorer,
        PhoenixRelevanceScorer,
        PhoenixSummarizationScorer,
        PhoenixToxicityScorer,
    )

    assert PhoenixHallucinationScorer is not None
    assert PhoenixRelevanceScorer is not None
    assert PhoenixToxicityScorer is not None
    assert PhoenixQAScorer is not None
    assert PhoenixSummarizationScorer is not None


def test_phoenix_scorers_available_from_main_module():
    from mlflow.genai.scorers import (
        PhoenixHallucinationScorer,
        PhoenixQAScorer,
        PhoenixRelevanceScorer,
        PhoenixSummarizationScorer,
        PhoenixToxicityScorer,
    )

    assert PhoenixHallucinationScorer is not None
    assert PhoenixRelevanceScorer is not None
    assert PhoenixToxicityScorer is not None
    assert PhoenixQAScorer is not None
    assert PhoenixSummarizationScorer is not None
