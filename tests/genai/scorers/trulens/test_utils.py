from unittest.mock import Mock, patch

from mlflow.genai.scorers.trulens.utils import (
    format_trulens_rationale,
    map_scorer_inputs_to_trulens_args,
)


class TestMapScorerInputsToTrulensArgs:
    def test_groundedness_mapping(self):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="Groundedness",
            outputs="The answer is 42.",
            expectations={"context": "The answer to everything is 42."},
        )

        assert result == {
            "source": "The answer to everything is 42.",
            "statement": "The answer is 42.",
        }

    def test_context_relevance_mapping(self):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="ContextRelevance",
            inputs="What is the answer?",
            expectations={"context": "The answer is 42."},
        )

        assert result == {
            "question": "What is the answer?",
            "context": "The answer is 42.",
        }

    def test_answer_relevance_mapping(self):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="AnswerRelevance",
            inputs="What is MLflow?",
            outputs="MLflow is a platform for ML lifecycle.",
        )

        assert result == {
            "prompt": "What is MLflow?",
            "response": "MLflow is a platform for ML lifecycle.",
        }

    def test_coherence_mapping(self):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="Coherence",
            outputs="This is a well-structured response.",
        )

        assert result == {
            "text": "This is a well-structured response.",
        }

    def test_unknown_metric_fallback(self):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="UnknownMetric",
            inputs="input text",
            outputs="output text",
            expectations={"context": "context text"},
        )

        assert result == {
            "input": "input text",
            "output": "output text",
            "context": "context text",
        }

    def test_context_from_list(self):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="Groundedness",
            outputs="Combined answer.",
            expectations={"context": ["First context.", "Second context."]},
        )

        assert result["source"] == "First context.\nSecond context."

    def test_context_priority_order(self):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="Groundedness",
            outputs="test",
            expectations={
                "context": "primary context",
                "reference": "should be ignored",
            },
        )

        assert result["source"] == "primary context"

    def test_reference_fallback(self):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="Groundedness",
            outputs="test",
            expectations={"reference": "reference context"},
        )

        assert result["source"] == "reference context"

    def test_with_trace(self):
        mock_trace = Mock()

        with (
            patch(
                "mlflow.genai.scorers.trulens.utils.resolve_inputs_from_trace",
                return_value="resolved input",
            ),
            patch(
                "mlflow.genai.scorers.trulens.utils.resolve_outputs_from_trace",
                return_value="resolved output",
            ),
            patch(
                "mlflow.genai.scorers.trulens.utils.resolve_expectations_from_trace",
                return_value={"context": "resolved context"},
            ),
        ):
            result = map_scorer_inputs_to_trulens_args(
                metric_name="Groundedness",
                trace=mock_trace,
            )

            assert result["source"] == "resolved context"
            assert result["statement"] == "resolved output"

    def test_trace_context_fallback(self):
        mock_trace = Mock()

        with (
            patch(
                "mlflow.genai.scorers.trulens.utils.resolve_inputs_from_trace",
                return_value="input",
            ),
            patch(
                "mlflow.genai.scorers.trulens.utils.resolve_outputs_from_trace",
                return_value="output",
            ),
            patch(
                "mlflow.genai.scorers.trulens.utils.resolve_expectations_from_trace",
                return_value=None,
            ),
            patch(
                "mlflow.genai.scorers.trulens.utils.extract_retrieval_context_from_trace",
                return_value={"span1": [{"content": "trace context"}]},
            ),
        ):
            result = map_scorer_inputs_to_trulens_args(
                metric_name="Groundedness",
                trace=mock_trace,
            )

            assert result["source"] == "trace context"


class TestFormatTrulensRationale:
    def test_none_reasons(self):
        assert format_trulens_rationale(None) is None

    def test_empty_reasons(self):
        assert format_trulens_rationale({}) is None

    def test_simple_reasons(self):
        result = format_trulens_rationale({"reason": "Good answer"})
        assert result == "reason: Good answer"

    def test_multiple_reasons(self):
        result = format_trulens_rationale(
            {
                "reason1": "First reason",
                "reason2": "Second reason",
            }
        )
        assert "reason1: First reason" in result
        assert "reason2: Second reason" in result
        assert " | " in result

    def test_list_reason(self):
        result = format_trulens_rationale({"reasons": ["A", "B", "C"]})
        assert result == "reasons: A; B; C"

    def test_dict_reason(self):
        result = format_trulens_rationale({"details": {"key": "value"}})
        assert "details:" in result
        assert "key" in result
