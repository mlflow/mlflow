from unittest.mock import MagicMock, patch

import dspy
import pytest

from mlflow.genai.judges.optimizers.memalign.utils import (
    _count_tokens,
    _create_batches,
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
            "mlflow.genai.judges.optimizers.memalign.utils._create_batches",
            return_value=[[0, 1]],
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
            "mlflow.genai.judges.optimizers.memalign.utils._create_batches",
            return_value=[[0]],
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
    # When LM fails for a batch, distill_guidelines logs error and continues
    with (
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
        ) as mock_construct_lm,
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils._create_batches",
            return_value=[[0]],
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


class TestCountTokens:
    def test_count_tokens_with_litellm(self):
        with (
            patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils.token_counter", return_value=42
            ) as mock_counter,
        ):
            result = _count_tokens("test text", "gpt-4")
            assert result == 42
            mock_counter.assert_called_once_with(model="gpt-4", text="test text")

    def test_count_tokens_without_litellm(self):
        with patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", False):
            # Fallback uses len(text) // 4
            result = _count_tokens("a" * 100, None)
            assert result == 25

    def test_count_tokens_with_none_model(self):
        with patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True):
            # Even if litellm is available, None model uses fallback
            result = _count_tokens("a" * 100, None)
            assert result == 25


class TestCreateBatches:
    def test_create_batches_empty_examples(self):
        with (
            patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", False),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils._get_model_max_input_tokens",
                return_value=10000,
            ),
        ):
            result = _create_batches(
                examples_data=[],
                indices=[],
                judge_instructions="test",
                existing_guidelines=[],
                reflection_lm="openai:/gpt-4",
            )
            assert result == []

    def test_create_batches_single_batch(self):
        with (
            patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", False),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils._get_model_max_input_tokens",
                return_value=100000,
            ),
        ):
            examples = [{"input": "test1"}, {"input": "test2"}, {"input": "test3"}]
            result = _create_batches(
                examples_data=examples,
                indices=[0, 1, 2],
                judge_instructions="test",
                existing_guidelines=[],
                reflection_lm="openai:/gpt-4",
            )
            # All examples should fit in one batch
            assert len(result) == 1
            assert result[0] == [0, 1, 2]

    def test_create_batches_multiple_batches_by_token_limit(self):
        with (
            patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils._get_model_max_input_tokens",
                return_value=10000,
            ),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils.convert_mlflow_uri_to_litellm",
                return_value="gpt-4",
            ),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils.token_counter"
            ) as mock_counter,
        ):
            # Base prompt = 1000 tokens, each example = 3000 tokens
            # Limit = 10000 - 5000 (flex) = 5000 tokens
            # Can fit 1 example per batch: 1000 + 3000 = 4000 < 5000
            # But 2 examples: 1000 + 6000 = 7000 > 5000
            mock_counter.side_effect = [1000, 3000, 3000, 3000]

            examples = [{"input": f"test{i}"} for i in range(3)]
            result = _create_batches(
                examples_data=examples,
                indices=[0, 1, 2],
                judge_instructions="test",
                existing_guidelines=[],
                reflection_lm="openai:/gpt-4",
            )

            assert len(result) == 3
            assert result == [[0], [1], [2]]

    def test_create_batches_multiple_batches_by_max_records(self):
        with (
            patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", False),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils._get_model_max_input_tokens",
                return_value=10000000,
            ),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils._MAX_RECORDS_PER_BATCH", 2
            ),
        ):
            examples = [{"input": f"test{i}"} for i in range(5)]
            result = _create_batches(
                examples_data=examples,
                indices=[0, 1, 2, 3, 4],
                judge_instructions="test",
                existing_guidelines=[],
                reflection_lm="openai:/gpt-4",
            )

            # Max 2 records per batch, so 5 examples -> 3 batches
            assert len(result) == 3
            assert result[0] == [0, 1]
            assert result[1] == [2, 3]
            assert result[2] == [4]

    def test_create_batches_variable_length_examples(self):
        with (
            patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils._get_model_max_input_tokens",
                return_value=10000,
            ),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils.convert_mlflow_uri_to_litellm",
                return_value="gpt-4",
            ),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils.token_counter"
            ) as mock_counter,
        ):
            # Base = 1000, limit = 5000
            # Examples: 500, 500, 500, 3500 tokens
            # Batch 1: 1000 + 500 + 500 + 500 = 2500 (fits)
            # Adding 3500 would make 6000 > 5000, so start new batch
            # Batch 2: 1000 + 3500 = 4500 (fits)
            mock_counter.side_effect = [1000, 500, 500, 500, 3500]

            examples = [
                {"input": "short1"},
                {"input": "short2"},
                {"input": "short3"},
                {"input": "very long example " * 100},
            ]
            result = _create_batches(
                examples_data=examples,
                indices=[0, 1, 2, 3],
                judge_instructions="test",
                existing_guidelines=[],
                reflection_lm="openai:/gpt-4",
            )

            assert len(result) == 2
            assert result[0] == [0, 1, 2]
            assert result[1] == [3]

    def test_create_batches_single_large_example(self):
        with (
            patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils._get_model_max_input_tokens",
                return_value=10000,
            ),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils.convert_mlflow_uri_to_litellm",
                return_value="gpt-4",
            ),
            patch(
                "mlflow.genai.judges.optimizers.memalign.utils.token_counter"
            ) as mock_counter,
        ):
            # Base = 1000, limit = 5000
            # Single example = 6000 tokens (exceeds limit even alone)
            # Still gets added to a batch (we don't skip it)
            mock_counter.side_effect = [1000, 6000]

            examples = [{"input": "huge example"}]
            result = _create_batches(
                examples_data=examples,
                indices=[0],
                judge_instructions="test",
                existing_guidelines=[],
                reflection_lm="openai:/gpt-4",
            )

            # Single example still forms a batch even if over limit
            assert len(result) == 1
            assert result[0] == [0]


def test_distill_guidelines_empty_batches():
    with (
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
        ),
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils._create_batches",
            return_value=[],
        ),
    ):
        example1 = MagicMock(spec=dspy.Example)
        example1.__iter__ = lambda self: iter([("input", "test")])
        example1._trace_id = "trace_1"

        result = distill_guidelines(
            examples=[example1],
            judge_instructions="test",
            reflection_lm="openai:/gpt-4",
            existing_guidelines=[],
        )

        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
