from unittest.mock import MagicMock, patch

import dspy
import pytest

from mlflow.genai.judges.optimizers.memalign.utils import (
    distill_guidelines,
    get_default_embedding_model,
    retrieve_relevant_examples,
    truncate_to_token_limit,
)


def test_get_default_embedding_model():
    assert get_default_embedding_model() == "openai:/text-embedding-3-small"


def test_distill_guidelines_empty_examples():
    with patch(
        "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
    ) as mock_construct_lm:
        result = distill_guidelines(
            examples=[],
            judge_instructions="Test instructions",
            reflection_lm="openai:/gpt-4",
            existing_guidelines=[],
        )
        assert result == []
        mock_construct_lm.assert_not_called()


def test_distill_guidelines_with_examples():
    with (
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
        ) as mock_construct_lm,
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils._find_optimal_batch_size",
            return_value=2,
        ),
    ):
        example1 = MagicMock(spec=dspy.Example)
        example1.__iter__ = lambda self: iter([("input", "test input"), ("output", "good")])
        example1._trace_id = "trace_1"
        example2 = MagicMock(spec=dspy.Example)
        example2.__iter__ = lambda self: iter([("input", "test input 2"), ("output", "bad")])
        example2._trace_id = "trace_2"

        mock_lm = MagicMock()
        mock_lm.return_value = [
            '{"guidelines": [{"guideline_text": "Be concise", "source_trace_ids": [0, 1]}]}'
        ]
        mock_construct_lm.return_value = mock_lm

        result = distill_guidelines(
            examples=[example1, example2],
            judge_instructions="Evaluate quality",
            reflection_lm="openai:/gpt-4",
            existing_guidelines=[],
        )

        assert len(result) == 1
        assert result[0].guideline_text == "Be concise"
        # The LLM returns indices [0, 1] which get mapped to trace IDs
        assert result[0].source_trace_ids == ["trace_1", "trace_2"]
        mock_construct_lm.assert_called_once_with("openai:/gpt-4")


def test_distill_guidelines_filters_existing():
    with (
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
        ) as mock_construct_lm,
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils._find_optimal_batch_size",
            return_value=1,
        ),
    ):
        example1 = MagicMock(spec=dspy.Example)
        example1.__iter__ = lambda self: iter([("input", "test"), ("output", "good")])
        example1._trace_id = "trace_1"

        mock_lm = MagicMock()
        # Guidelines need source_trace_ids to be retained
        mock_lm.return_value = [
            '{"guidelines": [{"guideline_text": "Be concise", "source_trace_ids": [0]}, '
            '{"guideline_text": "Be clear", "source_trace_ids": [0]}]}'
        ]
        mock_construct_lm.return_value = mock_lm

        result = distill_guidelines(
            examples=[example1],
            judge_instructions="Evaluate quality",
            reflection_lm="openai:/gpt-4",
            existing_guidelines=["Be concise"],
        )

        assert len(result) == 1
        assert result[0].guideline_text == "Be clear"


def test_distill_guidelines_handles_lm_error():
    # When LM fails, _find_optimal_batch_size returns 0 and distill_guidelines returns []
    with (
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
        ) as mock_construct_lm,
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils._find_optimal_batch_size",
            return_value=1,
        ),
    ):
        example1 = MagicMock(spec=dspy.Example)
        example1.__iter__ = lambda self: iter([("input", "test"), ("output", "good")])
        example1._trace_id = "trace_1"

        mock_lm = MagicMock()
        mock_lm.side_effect = Exception("API Error")
        mock_construct_lm.return_value = mock_lm

        # The function catches errors per batch and continues, returning empty list
        result = distill_guidelines(
            examples=[example1],
            judge_instructions="Evaluate quality",
            reflection_lm="openai:/gpt-4",
            existing_guidelines=[],
        )
        assert result == []


def test_retrieve_relevant_examples_empty():
    results = retrieve_relevant_examples(
        retriever=None,
        examples=[],
        query_kwargs={"inputs": "test"},
        signature=MagicMock(),
    )
    assert results == []


def test_retrieve_relevant_examples_no_search():
    examples = [MagicMock(), MagicMock()]
    results = retrieve_relevant_examples(
        retriever=None,
        examples=examples,
        query_kwargs={"inputs": "test"},
        signature=MagicMock(),
    )
    assert results == []


def test_retrieve_relevant_examples_success():
    example1 = MagicMock()
    example1._trace_id = "trace_1"
    example2 = MagicMock()
    example2._trace_id = "trace_2"
    example3 = MagicMock()
    example3._trace_id = "trace_3"
    examples = [example1, example2, example3]

    # Mock retriever results
    mock_retriever = MagicMock()
    search_results = MagicMock()
    search_results.indices = [2, 0]  # Return example3 and example1
    mock_retriever.return_value = search_results

    # Mock signature
    signature = MagicMock()
    signature.input_fields = ["inputs", "outputs"]

    results = retrieve_relevant_examples(
        retriever=mock_retriever,
        examples=examples,
        query_kwargs={"inputs": "test query", "outputs": "test output"},
        signature=signature,
    )

    assert len(results) == 2
    assert results[0] == (example3, "trace_3")
    assert results[1] == (example1, "trace_1")
    mock_retriever.assert_called_once_with("test query test output")


def test_retrieve_relevant_examples_filters_none_values():
    examples = [MagicMock()]
    mock_retriever = MagicMock()
    search_results = MagicMock()
    search_results.indices = [0]
    mock_retriever.return_value = search_results

    signature = MagicMock()
    signature.input_fields = ["inputs", "outputs", "context"]

    retrieve_relevant_examples(
        retriever=mock_retriever,
        examples=examples,
        query_kwargs={"inputs": "test", "outputs": None, "context": "ctx"},
        signature=signature,
    )

    # Should only include non-None values
    mock_retriever.assert_called_once_with("test ctx")


def test_retrieve_relevant_examples_out_of_bounds_raises():
    examples = [MagicMock(), MagicMock()]
    mock_retriever = MagicMock()
    search_results = MagicMock()
    search_results.indices = [5]  # Out of bounds
    mock_retriever.return_value = search_results

    signature = MagicMock()
    signature.input_fields = ["inputs"]

    with pytest.raises(IndexError, match="list index out of range"):
        retrieve_relevant_examples(
            retriever=mock_retriever,
            examples=examples,
            query_kwargs={"inputs": "test"},
            signature=signature,
        )


@pytest.mark.parametrize(
    ("token_count", "text"),
    [
        (50, "This is a short text"),
        (100, "This text is exactly at the limit"),
    ],
)
def test_truncate_to_token_limit_no_truncation_needed(token_count, text):
    with (
        patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True),
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.get_model_info",
            return_value={"max_input_tokens": 100},
        ),
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.token_counter",
            return_value=token_count,
        ),
    ):
        result = truncate_to_token_limit(text, "openai:/gpt-4", model_type="chat")
        assert result == text


def test_truncate_to_token_limit_happy_path_with_truncation():
    with (
        patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True),
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.get_model_info",
            return_value={"max_input_tokens": 100},
        ),
        patch("mlflow.genai.judges.optimizers.memalign.utils.token_counter") as mock_counter,
    ):
        mock_counter.side_effect = [150, 90]

        text = "x" * 500
        result = truncate_to_token_limit(text, "openai:/gpt-4", model_type="chat")

        assert len(result) < len(text)
        assert mock_counter.call_count == 2


def test_truncate_to_token_limit_multiple_iterations():
    with (
        patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True),
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.get_model_info",
            return_value={"max_input_tokens": 100},
        ),
        patch("mlflow.genai.judges.optimizers.memalign.utils.token_counter") as mock_counter,
    ):
        mock_counter.side_effect = [200, 120, 95]

        text = "x" * 1000
        result = truncate_to_token_limit(text, "openai:/gpt-4", model_type="chat")

        assert len(result) < len(text)
        assert mock_counter.call_count == 3


def test_truncate_to_token_limit_without_litellm():
    with patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", False):
        text = "a" * 200
        result = truncate_to_token_limit(text, "openai:/gpt-4", model_type="chat")

        # Without litellm, falls back to _MAX_CHAT_MODEL_TOKENS (128000)
        # Since text length (200) < 128000, no truncation occurs
        assert result == text


@pytest.mark.parametrize(
    "get_model_info_side_effect",
    [
        Exception("API Error"),
        {"max_input_tokens": None},
    ],
)
def test_truncate_to_token_limit_get_model_info_fallback(get_model_info_side_effect):
    with patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True):
        if isinstance(get_model_info_side_effect, Exception):
            mock_get_model_info = patch(
                "mlflow.genai.judges.optimizers.memalign.utils.get_model_info",
                side_effect=get_model_info_side_effect,
            )
        else:
            mock_get_model_info = patch(
                "mlflow.genai.judges.optimizers.memalign.utils.get_model_info",
                return_value=get_model_info_side_effect,
            )

        with (
            mock_get_model_info,
            patch("mlflow.genai.judges.optimizers.memalign.utils.token_counter", return_value=50),
        ):
            text = "This is a short text"
            result = truncate_to_token_limit(text, "openai:/gpt-4", model_type="chat")
            assert result == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
