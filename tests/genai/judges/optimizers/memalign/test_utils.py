import json
from unittest.mock import MagicMock, patch

import dspy
import pytest

import mlflow
from mlflow.genai.judges.optimizers.memalign.utils import (
    Guideline,
    Guidelines,
    _build_strict_response_format,
    _count_tokens,
    _create_batches,
    _extract_json_object,
    distill_guidelines,
    get_default_embedding_model,
    retrieve_relevant_examples,
    truncate_to_token_limit,
    value_to_embedding_text,
)


def test_get_default_embedding_model():
    assert get_default_embedding_model() == "openai:/text-embedding-3-small"


def _iter_object_schemas(schema):
    """Yield every object schema node (top-level and nested $defs) in a JSON schema."""
    yield schema
    yield from schema.get("$defs", {}).values()


@pytest.mark.parametrize("model", [Guideline, Guidelines])
def test_guideline_schemas_satisfy_strict_structured_output(model):
    # Databricks' structured-output endpoint enforces OpenAI strict-schema rules on every
    # object: additionalProperties=false (emitted via extra=forbid) AND every property listed
    # in `required` (so no field may declare a default). Assert both, since either omission
    # produces a BadRequestError from the endpoint.
    schema = model.model_json_schema()
    for obj in _iter_object_schemas(schema):
        if "properties" in obj:
            assert obj.get("additionalProperties") is False
            assert set(obj["required"]) == set(obj["properties"])


def test_guideline_accepts_none_source_trace_ids():
    # source_trace_ids is required (no default) but must remain nullable, since guidelines
    # loaded from persisted memory and filtering logic rely on None being a valid value.
    assert Guideline(guideline_text="x", source_trace_ids=None).source_trace_ids is None


def _walk_schema(node):
    """Yield every dict node in a JSON schema tree."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_schema(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_schema(item)


def test_build_strict_response_format_satisfies_databricks_rules():
    # Databricks' structured-output endpoint enforces the full OpenAI strict-schema contract
    # AND (unlike OpenAI) rejects $ref/$defs indirection. Assert the built response_format:
    #   - is a strict json_schema envelope
    #   - contains no $ref / $defs (fully inlined)
    #   - declares additionalProperties=false and complete `required` on every object
    rf = _build_strict_response_format(Guidelines)

    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]

    for node in _walk_schema(schema):
        assert "$ref" not in node
        assert "$defs" not in node
        if node.get("type") == "object" or "properties" in node:
            assert node.get("additionalProperties") is False
            assert set(node["required"]) == set(node["properties"])


def _assert_objects_forbid_additional_properties(node):
    """Recursively assert every object schema declares additionalProperties=false."""
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            assert node.get("additionalProperties") is False, node
        for value in node.values():
            _assert_objects_forbid_additional_properties(value)
    elif isinstance(node, list):
        for item in node:
            _assert_objects_forbid_additional_properties(item)


# Databricks-served foundation models split into two response_format routing paths in
# litellm: OpenAI-compatible models emit a strict `json_schema`, while Claude models are
# rewritten into a forced tool call whose parameter schema is derived from the same model.
# These tests verify that Guidelines produces a valid payload (every object declaring
# additionalProperties=false) on both paths across a representative model per family.
# NOTE: the version-independent regression guard for the additionalProperties fix is
# test_guideline_schemas_forbid_additional_properties above - recent litellm also injects
# additionalProperties on the json_schema path, so these routing tests alone would not
# catch a removal of extra="forbid" on older litellm. See the Databricks docs:
# https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models
_JSON_SCHEMA_MODELS = [
    "databricks-gpt-5",
    "databricks-gpt-5-mini",
    "databricks-gpt-oss-120b",
    "databricks-gemini-2-5-flash",
    "databricks-qwen35-122b-a10b",
    "databricks-glm-5-2",
    "databricks-gemma-3-12b",
    "databricks-llama-4-maverick",
    "databricks-meta-llama-3-1-8b-instruct",
]
_TOOL_CALL_MODELS = [
    "databricks-claude-sonnet-5",
    "databricks-claude-opus-4-8",
    "databricks-claude-haiku-4-5",
]


@pytest.mark.parametrize("model", _JSON_SCHEMA_MODELS)
def test_guidelines_response_format_json_schema_families(model):
    import litellm

    params = litellm.get_optional_params(
        model=model, custom_llm_provider="databricks", response_format=Guidelines
    )
    response_format = params.get("response_format")
    assert response_format is not None, f"{model} dropped response_format"
    assert response_format["type"] == "json_schema"
    _assert_objects_forbid_additional_properties(response_format["json_schema"]["schema"])


@pytest.mark.parametrize("model", _TOOL_CALL_MODELS)
def test_guidelines_response_format_tool_call_families(model):
    import litellm

    params = litellm.get_optional_params(
        model=model, custom_llm_provider="databricks", response_format=Guidelines
    )
    # Claude routes structured output through a forced tool call rather than json_schema.
    tools = params.get("tools")
    assert tools, f"{model} produced neither response_format nor tools"
    _assert_objects_forbid_additional_properties(tools[0]["function"]["parameters"])


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


def test_distill_guidelines_falls_back_to_unstructured_on_structured_output_error():
    # Some models error on structured-output (response_format) requests. The distiller
    # should retry without response_format rather than dropping the batch.
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

        def lm_side_effect(messages, **kwargs):
            if "response_format" in kwargs:
                raise Exception("INTERNAL_ERROR: invalid response from upstream server")
            # Unstructured fallback returns JSON wrapped in a markdown fence.
            return [
                '```json\n{"guidelines": [{"guideline_text": "Be concise", '
                '"source_trace_ids": [0]}]}\n```'
            ]

        mock_lm = MagicMock(side_effect=lm_side_effect)
        mock_construct_lm.return_value = mock_lm

        result = distill_guidelines(
            examples=[example1],
            judge_instructions="Evaluate quality",
            reflection_lm="databricks:/databricks-gpt-oss-120b",
            existing_guidelines=[],
        )

        assert len(result) == 1
        assert result[0].guideline_text == "Be concise"
        assert result[0].source_trace_ids == ["trace_1"]
        # Structured attempt + unstructured fallback = two calls.
        assert mock_lm.call_count == 2


@pytest.mark.parametrize(
    "response",
    [
        '{"guidelines": []}',
        '```json\n{"guidelines": []}\n```',
        '```\n{"guidelines": []}\n```',
        '  {"guidelines": []}  ',
        'Here are the guidelines:\n```json\n{"guidelines": []}\n```',
        'Sure! {"guidelines": []}',
        '{"guidelines": []}\nLet me know if you need more.',
    ],
)
def test_extract_json_object(response):
    # The unstructured fallback path yields JSON that may be fenced and/or wrapped in
    # prose; the extracted string must be valid JSON with the expected shape.
    assert json.loads(_extract_json_object(response)) == {"guidelines": []}


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
    # Now uses only the first matching field from priority list ("inputs")
    mock_retriever.assert_called_once_with("test query")


def test_retrieve_relevant_examples_uses_first_priority_field():
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
        query_kwargs={"inputs": "test", "outputs": "output_val", "context": "ctx"},
        signature=signature,
    )

    # Should only use "inputs" (first in priority list that exists in input_fields)
    mock_retriever.assert_called_once_with("test")


def test_retrieve_relevant_examples_returns_empty_for_none_value():
    examples = [MagicMock()]
    mock_retriever = MagicMock()

    signature = MagicMock()
    signature.input_fields = ["inputs", "outputs"]

    # When the first priority field value is None, should return empty list
    results = retrieve_relevant_examples(
        retriever=mock_retriever,
        examples=examples,
        query_kwargs={"inputs": None, "outputs": "output_val"},
        signature=signature,
    )

    assert results == []
    mock_retriever.assert_not_called()


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hello world", "hello world"),
        (42, "42"),
        ({"key": "value"}, "{'key': 'value'}"),
        ([1, 2, 3], "[1, 2, 3]"),
        (None, "None"),
    ],
)
def test_value_to_embedding_text_non_trace(value, expected):
    assert value_to_embedding_text(value) == expected


def test_value_to_embedding_text_trace():
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "What is ML?"})
        span.set_outputs({"answer": "ML is machine learning."})

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    result = value_to_embedding_text(trace)
    assert "What is ML?" in result
    assert "ML is machine learning." in result


def test_count_tokens_with_litellm():
    with (
        patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True),
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.token_counter", return_value=42
        ) as mock_counter,
    ):
        result = _count_tokens("test text", "gpt-4")
        assert result == 42
        mock_counter.assert_called_once_with(model="gpt-4", text="test text")


def test_count_tokens_without_litellm():
    with patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", False):
        # Fallback uses len(text) // 4
        result = _count_tokens("a" * 100, None)
        assert result == 25


def test_count_tokens_with_none_model():
    with patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", True):
        # Even if litellm is available, None model uses fallback
        result = _count_tokens("a" * 100, None)
        assert result == 25


def test_create_batches_empty_examples():
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


def test_create_batches_single_batch():
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


def test_create_batches_multiple_batches_by_token_limit():
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
        patch("mlflow.genai.judges.optimizers.memalign.utils.token_counter") as mock_counter,
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


def test_create_batches_multiple_batches_by_max_records():
    with (
        patch("mlflow.genai.judges.optimizers.memalign.utils._LITELLM_AVAILABLE", False),
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils._get_model_max_input_tokens",
            return_value=10000000,
        ),
        patch("mlflow.genai.judges.optimizers.memalign.utils._MAX_RECORDS_PER_BATCH", 2),
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


def test_create_batches_variable_length_examples():
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
        patch("mlflow.genai.judges.optimizers.memalign.utils.token_counter") as mock_counter,
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


def test_create_batches_single_large_example():
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
        patch("mlflow.genai.judges.optimizers.memalign.utils.token_counter") as mock_counter,
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
        patch("mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"),
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
