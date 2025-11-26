"""
Tests for the GenAI schema mapping module.

This module tests the conversion of MLflow span attributes to OpenTelemetry
GenAI semantic conventions and vice versa.
"""

import json

import pytest

from mlflow.tracing.genai_schema import (
    GENAI_MAPPING,
    MLFLOW_SPAN_TYPE_TO_GENAI_OPERATION,
    TOKEN_USAGE_KEY_MAPPING,
    convert_from_genai_schema,
    convert_to_genai_schema,
    get_genai_attribute_keys,
    get_mlflow_attribute_keys,
)


class TestConvertToGenaiSchema:
    """Tests for convert_to_genai_schema function."""

    def test_empty_attrs(self):
        """Test with empty attributes dictionary."""
        result = convert_to_genai_schema({})
        assert result == {}

    def test_none_attrs(self):
        """Test with None attributes."""
        result = convert_to_genai_schema(None)
        assert result == {}

    def test_basic_input_output_mapping(self):
        """Test basic input/output attribute mapping."""
        attrs = {
            "mlflow.spanInputs": "Hello, world!",
            "mlflow.spanOutputs": "Hi there!",
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.input"] == "Hello, world!"
        assert result["gen_ai.response.output"] == "Hi there!"

    def test_trace_input_output_mapping(self):
        """Test trace-level input/output attribute mapping."""
        attrs = {
            "mlflow.traceInputs": "Trace input",
            "mlflow.traceOutputs": "Trace output",
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.input"] == "Trace input"
        assert result["gen_ai.response.output"] == "Trace output"

    def test_token_usage_dict_conversion(self):
        """Test token usage conversion from nested dict."""
        attrs = {
            "mlflow.chat.tokenUsage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.usage.input_tokens"] == 100
        assert result["gen_ai.usage.output_tokens"] == 50
        assert result["gen_ai.usage.total_tokens"] == 150
        assert "mlflow.chat.tokenUsage" not in result

    def test_token_usage_json_string_conversion(self):
        """Test token usage conversion from JSON string."""
        attrs = {
            "mlflow.chat.tokenUsage": json.dumps(
                {
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "total_tokens": 300,
                }
            )
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.usage.input_tokens"] == 200
        assert result["gen_ai.usage.output_tokens"] == 100
        assert result["gen_ai.usage.total_tokens"] == 300

    def test_token_usage_partial(self):
        """Test token usage with partial keys."""
        attrs = {
            "mlflow.chat.tokenUsage": {
                "input_tokens": 50,
                "output_tokens": 25,
            }
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.usage.input_tokens"] == 50
        assert result["gen_ai.usage.output_tokens"] == 25
        assert "gen_ai.usage.total_tokens" not in result

    def test_token_usage_none(self):
        """Test token usage with None value."""
        attrs = {"mlflow.chat.tokenUsage": None}
        result = convert_to_genai_schema(attrs)

        assert "gen_ai.usage.input_tokens" not in result
        assert "gen_ai.usage.output_tokens" not in result
        assert "gen_ai.usage.total_tokens" not in result

    def test_token_usage_invalid_json(self):
        """Test token usage with invalid JSON string."""
        attrs = {"mlflow.chat.tokenUsage": "not valid json"}
        result = convert_to_genai_schema(attrs)

        # Should not crash, just not include token usage
        assert "gen_ai.usage.input_tokens" not in result

    @pytest.mark.parametrize(
        ("mlflow_type", "expected_operation"),
        [
            ("LLM", "text_completion"),
            ("CHAT_MODEL", "chat"),
            ("EMBEDDING", "embeddings"),
            ("RETRIEVER", "retrieval"),
            ("TOOL", "execute_tool"),
            ("AGENT", "invoke_agent"),
            ("CHAIN", "chain"),
            ("RERANKER", "rerank"),
            ("PARSER", "parse"),
            ("GUARDRAIL", "guardrail"),
            ("EVALUATOR", "evaluate"),
            ("UNKNOWN", "unknown"),
        ],
    )
    def test_span_type_conversion(self, mlflow_type, expected_operation):
        """Test span type to GenAI operation name conversion."""
        attrs = {"mlflow.spanType": mlflow_type}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.operation.name"] == expected_operation

    def test_span_type_json_encoded(self):
        """Test span type conversion with JSON-encoded value."""
        attrs = {"mlflow.spanType": json.dumps("LLM")}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.operation.name"] == "text_completion"

    def test_span_type_unknown_value(self):
        """Test span type conversion with unknown value."""
        attrs = {"mlflow.spanType": "CUSTOM_TYPE"}
        result = convert_to_genai_schema(attrs)

        # Should preserve unknown types as-is
        assert result["gen_ai.operation.name"] == "CUSTOM_TYPE"

    def test_model_attributes_mapping(self):
        """Test model-related attribute mappings."""
        attrs = {
            "mlflow.model.name": "gpt-4",
            "mlflow.model.provider": "openai",
            "mlflow.model.temperature": 0.7,
            "mlflow.model.maxTokens": 1000,
            "mlflow.model.topP": 0.9,
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.model"] == "gpt-4"
        assert result["gen_ai.system"] == "openai"
        assert result["gen_ai.request.temperature"] == 0.7
        assert result["gen_ai.request.max_tokens"] == 1000
        assert result["gen_ai.request.top_p"] == 0.9

    def test_chat_tools_mapping(self):
        """Test chat tools attribute mapping."""
        tools = [{"name": "search", "description": "Search tool"}]
        attrs = {"mlflow.chat.tools": tools}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.tools"] == tools

    def test_function_name_mapping(self):
        """Test function name to tool name mapping."""
        attrs = {"mlflow.spanFunctionName": "my_function"}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.tool.name"] == "my_function"

    def test_unmapped_attributes_preserved(self):
        """Test that unmapped attributes are preserved."""
        attrs = {
            "mlflow.spanInputs": "input",
            "custom.attribute": "preserved",
            "another.custom": 123,
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.input"] == "input"
        assert result["custom.attribute"] == "preserved"
        assert result["another.custom"] == 123

    def test_json_encoded_complex_values(self):
        """Test handling of JSON-encoded complex values."""
        complex_input = {"messages": [{"role": "user", "content": "Hello"}]}
        attrs = {"mlflow.spanInputs": json.dumps(complex_input)}
        result = convert_to_genai_schema(attrs)

        # Complex types should be parsed
        assert result["gen_ai.request.input"] == complex_input

    def test_json_encoded_primitive_values(self):
        """Test handling of JSON-encoded primitive values."""
        attrs = {"mlflow.spanInputs": json.dumps("simple string")}
        result = convert_to_genai_schema(attrs)

        # Primitive strings should remain as JSON-encoded
        assert result["gen_ai.request.input"] == json.dumps("simple string")

    def test_mixed_attributes(self):
        """Test with a mix of different attribute types."""
        attrs = {
            "mlflow.spanInputs": "Hello",
            "mlflow.spanOutputs": "World",
            "mlflow.spanType": "LLM",
            "mlflow.chat.tokenUsage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "mlflow.model.name": "gpt-4",
            "custom.field": "value",
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.input"] == "Hello"
        assert result["gen_ai.response.output"] == "World"
        assert result["gen_ai.operation.name"] == "text_completion"
        assert result["gen_ai.usage.input_tokens"] == 10
        assert result["gen_ai.usage.output_tokens"] == 5
        assert result["gen_ai.usage.total_tokens"] == 15
        assert result["gen_ai.request.model"] == "gpt-4"
        assert result["custom.field"] == "value"


class TestConvertFromGenaiSchema:
    """Tests for convert_from_genai_schema function."""

    def test_empty_attrs(self):
        """Test with empty attributes dictionary."""
        result = convert_from_genai_schema({})
        assert result == {}

    def test_none_attrs(self):
        """Test with None attributes."""
        result = convert_from_genai_schema(None)
        assert result == {}

    def test_basic_reverse_mapping(self):
        """Test basic reverse mapping from GenAI to MLflow."""
        attrs = {
            "gen_ai.request.input": "Hello!",
            "gen_ai.response.output": "Hi!",
        }
        result = convert_from_genai_schema(attrs)

        assert result["mlflow.spanInputs"] == "Hello!"
        assert result["mlflow.spanOutputs"] == "Hi!"

    def test_token_usage_aggregation(self):
        """Test token usage values are aggregated into nested dict."""
        attrs = {
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
            "gen_ai.usage.total_tokens": 150,
        }
        result = convert_from_genai_schema(attrs)

        assert "mlflow.chat.tokenUsage" in result
        token_usage = result["mlflow.chat.tokenUsage"]
        assert token_usage["input_tokens"] == 100
        assert token_usage["output_tokens"] == 50
        assert token_usage["total_tokens"] == 150

    def test_unmapped_attributes_preserved(self):
        """Test that unmapped attributes are preserved."""
        attrs = {
            "gen_ai.request.input": "input",
            "custom.attribute": "preserved",
        }
        result = convert_from_genai_schema(attrs)

        assert result["mlflow.spanInputs"] == "input"
        assert result["custom.attribute"] == "preserved"

    def test_roundtrip_conversion(self):
        """Test that converting to GenAI and back preserves data."""
        original = {
            "mlflow.spanInputs": "Hello",
            "mlflow.spanOutputs": "World",
            "mlflow.model.name": "gpt-4",
            "custom.field": "value",
        }

        genai = convert_to_genai_schema(original)
        back = convert_from_genai_schema(genai)

        # Check key mappings are correct
        assert back["mlflow.spanInputs"] == original["mlflow.spanInputs"]
        assert back["mlflow.spanOutputs"] == original["mlflow.spanOutputs"]
        assert back["mlflow.model.name"] == original["mlflow.model.name"]
        assert back["custom.field"] == original["custom.field"]


class TestMappingConstants:
    """Tests for mapping constants."""

    def test_genai_mapping_not_empty(self):
        """Test that GENAI_MAPPING is not empty."""
        assert len(GENAI_MAPPING) > 0

    def test_genai_mapping_keys_are_strings(self):
        """Test that all mapping keys are strings."""
        for key in GENAI_MAPPING.keys():
            assert isinstance(key, str)

    def test_genai_mapping_values_are_strings(self):
        """Test that all mapping values are strings."""
        for value in GENAI_MAPPING.values():
            assert isinstance(value, str)

    def test_span_type_mapping_coverage(self):
        """Test that common span types are covered."""
        expected_types = ["LLM", "CHAT_MODEL", "EMBEDDING", "TOOL", "AGENT"]
        for span_type in expected_types:
            assert span_type in MLFLOW_SPAN_TYPE_TO_GENAI_OPERATION

    def test_token_usage_mapping_complete(self):
        """Test that token usage mapping covers all keys."""
        expected_keys = ["input_tokens", "output_tokens", "total_tokens"]
        for key in expected_keys:
            assert key in TOKEN_USAGE_KEY_MAPPING


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_genai_attribute_keys(self):
        """Test get_genai_attribute_keys returns list."""
        keys = get_genai_attribute_keys()
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert "gen_ai.request.input" in keys
        assert "gen_ai.usage.input_tokens" in keys

    def test_get_mlflow_attribute_keys(self):
        """Test get_mlflow_attribute_keys returns list."""
        keys = get_mlflow_attribute_keys()
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert "mlflow.spanInputs" in keys


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nested_dict_in_value(self):
        """Test handling of nested dictionaries in values."""
        attrs = {
            "mlflow.spanInputs": {
                "messages": [{"role": "user", "content": "nested"}],
                "config": {"temperature": 0.5},
            }
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.input"]["messages"][0]["content"] == "nested"

    def test_list_values(self):
        """Test handling of list values."""
        tools = [{"name": "tool1"}, {"name": "tool2"}]
        attrs = {"mlflow.chat.tools": tools}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.tools"] == tools

    def test_numeric_values(self):
        """Test handling of numeric values."""
        attrs = {
            "mlflow.model.temperature": 0.7,
            "mlflow.model.maxTokens": 1000,
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.temperature"] == 0.7
        assert result["gen_ai.request.max_tokens"] == 1000

    def test_boolean_values(self):
        """Test handling of boolean values."""
        attrs = {"custom.flag": True}
        result = convert_to_genai_schema(attrs)

        assert result["custom.flag"] is True

    def test_none_value_in_attrs(self):
        """Test handling of None values in attributes."""
        attrs = {
            "mlflow.spanInputs": None,
            "mlflow.spanOutputs": "output",
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.input"] is None
        assert result["gen_ai.response.output"] == "output"

    def test_empty_string_value(self):
        """Test handling of empty string values."""
        attrs = {"mlflow.spanInputs": ""}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.input"] == ""

    def test_special_characters_in_values(self):
        """Test handling of special characters in values."""
        attrs = {"mlflow.spanInputs": 'Hello\nWorld\t"quoted"'}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.input"] == 'Hello\nWorld\t"quoted"'

    def test_unicode_values(self):
        """Test handling of unicode values."""
        attrs = {"mlflow.spanInputs": "Hello 世界 🌍"}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.input"] == "Hello 世界 🌍"

    def test_large_token_counts(self):
        """Test handling of large token counts."""
        attrs = {
            "mlflow.chat.tokenUsage": {
                "input_tokens": 1000000,
                "output_tokens": 500000,
                "total_tokens": 1500000,
            }
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.usage.input_tokens"] == 1000000
        assert result["gen_ai.usage.output_tokens"] == 500000
        assert result["gen_ai.usage.total_tokens"] == 1500000

    def test_token_usage_invalid_type(self):
        """Test handling of invalid token usage type."""
        attrs = {"mlflow.chat.tokenUsage": 12345}  # Not a dict or string
        result = convert_to_genai_schema(attrs)

        # Should not crash, just not include token usage
        assert "gen_ai.usage.input_tokens" not in result

    def test_flat_token_usage_keys(self):
        """Test flat token usage keys (alternative format)."""
        attrs = {
            "mlflow.chat.tokenUsage.input_tokens": 100,
            "mlflow.chat.tokenUsage.output_tokens": 50,
            "mlflow.chat.tokenUsage.total_tokens": 150,
        }
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.usage.input_tokens"] == 100
        assert result["gen_ai.usage.output_tokens"] == 50
        assert result["gen_ai.usage.total_tokens"] == 150

    def test_model_id_mapping(self):
        """Test mlflow.modelId mapping to gen_ai.request.model."""
        attrs = {"mlflow.modelId": "gpt-4-turbo"}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.model"] == "gpt-4-turbo"

    def test_message_format_mapping(self):
        """Test mlflow.message.format mapping."""
        attrs = {"mlflow.message.format": "json"}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.format"] == "json"

    def test_model_top_k_mapping(self):
        """Test mlflow.model.topK mapping."""
        attrs = {"mlflow.model.topK": 40}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.top_k"] == 40

    def test_model_stop_sequences_mapping(self):
        """Test mlflow.model.stopSequences mapping."""
        attrs = {"mlflow.model.stopSequences": ["stop1", "stop2"]}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.stop_sequences"] == ["stop1", "stop2"]

    def test_model_frequency_penalty_mapping(self):
        """Test mlflow.model.frequencyPenalty mapping."""
        attrs = {"mlflow.model.frequencyPenalty": 0.5}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.frequency_penalty"] == 0.5

    def test_model_presence_penalty_mapping(self):
        """Test mlflow.model.presencePenalty mapping."""
        attrs = {"mlflow.model.presencePenalty": 0.3}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.presence_penalty"] == 0.3

    def test_response_finish_reason_mapping(self):
        """Test mlflow.response.finishReason mapping."""
        attrs = {"mlflow.response.finishReason": "stop"}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.response.finish_reasons"] == "stop"

    def test_response_id_mapping(self):
        """Test mlflow.response.id mapping."""
        attrs = {"mlflow.response.id": "resp-12345"}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.response.id"] == "resp-12345"

    def test_trace_request_id_mapping(self):
        """Test mlflow.traceRequestId mapping."""
        attrs = {"mlflow.traceRequestId": "req-67890"}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.request.id"] == "req-67890"

    def test_trace_user_mapping(self):
        """Test mlflow.trace.user mapping."""
        attrs = {"mlflow.trace.user": "user123"}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.user.id"] == "user123"

    def test_trace_session_mapping(self):
        """Test mlflow.trace.session mapping."""
        attrs = {"mlflow.trace.session": "session456"}
        result = convert_to_genai_schema(attrs)

        assert result["gen_ai.session.id"] == "session456"

    def test_experiment_id_preserved(self):
        """Test mlflow.experimentId is preserved (not converted)."""
        attrs = {"mlflow.experimentId": "exp-789"}
        result = convert_to_genai_schema(attrs)

        # Should be mapped to mlflow.experiment_id (preserved format)
        assert "mlflow.experiment_id" in result or "mlflow.experimentId" in result

    def test_priority_when_multiple_inputs(self):
        """Test priority when both spanInputs and traceInputs exist."""
        attrs = {
            "mlflow.spanInputs": "span_input",
            "mlflow.traceInputs": "trace_input",
        }
        result = convert_to_genai_schema(attrs)

        # Both map to gen_ai.request.input, span-level takes precedence
        # The last one processed wins (order depends on dict iteration)
        assert result["gen_ai.request.input"] in ["span_input", "trace_input"]

    def test_priority_when_multiple_outputs(self):
        """Test priority when both spanOutputs and traceOutputs exist."""
        attrs = {
            "mlflow.spanOutputs": "span_output",
            "mlflow.traceOutputs": "trace_output",
        }
        result = convert_to_genai_schema(attrs)

        # Both map to gen_ai.response.output
        assert result["gen_ai.response.output"] in ["span_output", "trace_output"]

    def test_priority_when_model_name_and_id(self):
        """Test priority when both model.name and modelId exist."""
        attrs = {
            "mlflow.model.name": "gpt-4",
            "mlflow.modelId": "gpt-4-turbo",
        }
        result = convert_to_genai_schema(attrs)

        # Both map to gen_ai.request.model
        assert result["gen_ai.request.model"] in ["gpt-4", "gpt-4-turbo"]

    def test_all_genai_mappings_convert(self):
        """Test that all mappings in GENAI_MAPPING work correctly."""
        # Skip special cases that need special handling
        skip_keys = {
            "mlflow.spanType",  # Needs conversion logic
            "mlflow.chat.tokenUsage",  # Needs nested handling
            "mlflow.chat.tokenUsage.input_tokens",  # Part of token usage
            "mlflow.chat.tokenUsage.output_tokens",  # Part of token usage
            "mlflow.chat.tokenUsage.total_tokens",  # Part of token usage
        }

        test_values = {
            "mlflow.spanInputs": "test_input",
            "mlflow.spanOutputs": "test_output",
            "mlflow.traceInputs": "trace_input",
            "mlflow.traceOutputs": "trace_output",
            "mlflow.model.name": "test-model",
            "mlflow.model.provider": "test-provider",
            "mlflow.modelId": "model-id",
            "mlflow.chat.tools": [{"name": "tool1"}],
            "mlflow.message.format": "json",
            "mlflow.model.temperature": 0.7,
            "mlflow.model.maxTokens": 1000,
            "mlflow.model.topP": 0.9,
            "mlflow.model.topK": 40,
            "mlflow.model.stopSequences": ["stop"],
            "mlflow.model.frequencyPenalty": 0.5,
            "mlflow.model.presencePenalty": 0.3,
            "mlflow.response.finishReason": "stop",
            "mlflow.response.id": "resp-id",
            "mlflow.spanFunctionName": "func_name",
            "mlflow.traceRequestId": "req-id",
            "mlflow.trace.user": "user123",
            "mlflow.trace.session": "session456",
        }

        for mlflow_key, test_value in test_values.items():
            if mlflow_key in skip_keys:
                continue

            attrs = {mlflow_key: test_value}
            result = convert_to_genai_schema(attrs)

            genai_key = GENAI_MAPPING[mlflow_key]
            assert genai_key in result, f"Mapping {mlflow_key} -> {genai_key} failed"
            assert result[genai_key] == test_value, f"Value mismatch for {mlflow_key}"
