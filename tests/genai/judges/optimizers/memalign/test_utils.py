from unittest.mock import MagicMock, patch

import dspy
import pytest

from mlflow.genai.judges.optimizers.memalign.utils import (
    distill_guidelines,
    get_default_embedding_model,
    retrieve_relevant_examples,
)


def test_get_default_embedding_model():
    assert get_default_embedding_model() == "openai/text-embedding-3-small"


def test_distill_guidelines_empty_examples():
    with patch(
        "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
    ) as mock_construct_lm:
        signature = MagicMock()
        result = distill_guidelines(
            examples=[],
            signature=signature,
            judge_instructions="Test instructions",
            distillation_model="openai:/gpt-4",
            existing_guidelines=[],
        )
        assert result == []
        mock_construct_lm.assert_not_called()


def test_distill_guidelines_with_examples():
    with patch(
        "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
    ) as mock_construct_lm:
        example1 = MagicMock(spec=dspy.Example)
        example1.__iter__ = lambda self: iter([("input", "test input"), ("output", "good")])
        example2 = MagicMock(spec=dspy.Example)
        example2.__iter__ = lambda self: iter([("input", "test input 2"), ("output", "bad")])

        mock_lm = MagicMock()
        mock_lm.return_value = [
            '{"guidelines": [{"guideline_text": "Be concise", "source_ids": [0, 1]}]}'
        ]
        mock_construct_lm.return_value = mock_lm

        signature = MagicMock()

        result = distill_guidelines(
            examples=[example1, example2],
            signature=signature,
            judge_instructions="Evaluate quality",
            distillation_model="openai:/gpt-4",
            existing_guidelines=[],
        )

        assert len(result) == 1
        assert result[0] == "Be concise"
        mock_construct_lm.assert_called_once_with("openai:/gpt-4")


def test_distill_guidelines_filters_existing():
    with patch(
        "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
    ) as mock_construct_lm:
        example1 = MagicMock(spec=dspy.Example)
        example1.__iter__ = lambda self: iter([("input", "test"), ("output", "good")])

        mock_lm = MagicMock()
        mock_lm.return_value = [
            '{"guidelines": [{"guideline_text": "Be concise"}, {"guideline_text": "Be clear"}]}'
        ]
        mock_construct_lm.return_value = mock_lm

        signature = MagicMock()

        result = distill_guidelines(
            examples=[example1],
            signature=signature,
            judge_instructions="Evaluate quality",
            distillation_model="openai:/gpt-4",
            existing_guidelines=["Be concise"],
        )

        assert len(result) == 1
        assert result[0] == "Be clear"


def test_distill_guidelines_handles_error():
    with patch(
        "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
    ) as mock_construct_lm:
        example1 = MagicMock(spec=dspy.Example)
        example1.__iter__ = lambda self: iter([("input", "test"), ("output", "good")])

        mock_lm = MagicMock()
        mock_lm.side_effect = Exception("API Error")
        mock_construct_lm.return_value = mock_lm

        signature = MagicMock()

        result = distill_guidelines(
            examples=[example1],
            signature=signature,
            judge_instructions="Evaluate quality",
            distillation_model="openai:/gpt-4",
            existing_guidelines=[],
        )

        assert result == []


def test_retrieve_relevant_examples_empty():
    result_examples, result_indices = retrieve_relevant_examples(
        search=None,
        examples=[],
        query_kwargs={"inputs": "test"},
        signature=MagicMock(),
    )
    assert result_examples == []
    assert result_indices == []


def test_retrieve_relevant_examples_no_search():
    examples = [MagicMock(), MagicMock()]
    result_examples, result_indices = retrieve_relevant_examples(
        search=None,
        examples=examples,
        query_kwargs={"inputs": "test"},
        signature=MagicMock(),
    )
    assert result_examples == []
    assert result_indices == []


def test_retrieve_relevant_examples_success():
    example1 = MagicMock()
    example2 = MagicMock()
    example3 = MagicMock()
    examples = [example1, example2, example3]

    # Mock search results
    mock_search = MagicMock()
    search_results = MagicMock()
    search_results.indices = [2, 0]  # Return example3 and example1
    mock_search.return_value = search_results

    # Mock signature
    signature = MagicMock()
    signature.input_fields = ["inputs", "outputs"]

    result_examples, result_indices = retrieve_relevant_examples(
        search=mock_search,
        examples=examples,
        query_kwargs={"inputs": "test query", "outputs": "test output"},
        signature=signature,
    )

    assert len(result_examples) == 2
    assert result_examples[0] is example3
    assert result_examples[1] is example1
    assert result_indices == [2, 0]
    mock_search.assert_called_once_with("test query test output")


def test_retrieve_relevant_examples_filters_none_values():
    examples = [MagicMock()]
    mock_search = MagicMock()
    search_results = MagicMock()
    search_results.indices = [0]
    mock_search.return_value = search_results

    signature = MagicMock()
    signature.input_fields = ["inputs", "outputs", "context"]

    retrieve_relevant_examples(
        search=mock_search,
        examples=examples,
        query_kwargs={"inputs": "test", "outputs": None, "context": "ctx"},
        signature=signature,
    )

    # Should only include non-None values
    mock_search.assert_called_once_with("test ctx")


def test_retrieve_relevant_examples_out_of_bounds_raises():
    examples = [MagicMock(), MagicMock()]
    mock_search = MagicMock()
    search_results = MagicMock()
    search_results.indices = [5]  # Out of bounds
    mock_search.return_value = search_results

    signature = MagicMock()
    signature.input_fields = ["inputs"]

    with pytest.raises(IndexError, match="list index out of range"):
        retrieve_relevant_examples(
            search=mock_search,
            examples=examples,
            query_kwargs={"inputs": "test"},
            signature=signature,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
