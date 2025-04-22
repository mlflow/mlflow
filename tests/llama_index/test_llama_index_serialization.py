import json
from collections import Counter, deque
from unittest import mock

import pytest
from llama_index.core import PromptTemplate, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from mlflow.llama_index.serialize_objects import (
    _construct_prompt_template_object,
    _get_object_import_path,
    _sanitize_api_key,
    deserialize_settings,
    object_to_dict,
    serialize_settings,
)


@pytest.fixture
def mock_logger():
    with mock.patch("mlflow.llama_index.serialize_objects._logger") as mock_logger:
        yield mock_logger


def test_get_object_import_path_class_instantiated():
    # The rationale for using collections as the library to test
    # import path is it's very stable and unlikely to change.
    expected_path = "collections.deque"
    assert _get_object_import_path(deque()) == expected_path


def test_get_object_import_path_class_not_instantiated():
    expected_path = "collections.Counter"
    assert _get_object_import_path(Counter) == expected_path


def test_object_is_class_do_validate_passes():
    expected_path = "collections.Counter"
    assert _get_object_import_path(Counter) == expected_path


def test_object_is_class_do_validate_raises():
    class CustomClass:
        pass

    with pytest.raises(ValueError, match="does not have"):
        _get_object_import_path(CustomClass)


def test_sanitize_api_key_keys_present():
    data = {"openai_api_key": "sk-123456", "api_key": "sk-abcdef", "other_key": "value"}
    sanitized_data = _sanitize_api_key(data)
    assert "openai_api_key" not in sanitized_data
    assert "api_key" not in sanitized_data
    assert "other_key" in sanitized_data
    assert sanitized_data["other_key"] == "value"


def test_sanitize_api_key_keys_not_present():
    data = {"some_key": "some_value", "another_key": "another_value"}
    sanitized_data = _sanitize_api_key(data)
    assert "some_key" in sanitized_data
    assert "another_key" in sanitized_data
    assert sanitized_data["some_key"] == "some_value"
    assert sanitized_data["another_key"] == "another_value"


def test_object_to_dict_no_required_param():
    o = OpenAI()
    result = object_to_dict(o)
    assert result["object_constructor"] == "llama_index.llms.openai.base.OpenAI"
    expected_kwargs = {k: v for k, v in o.to_dict().items() if k not in {"class_name", "api_key"}}
    assert result["object_kwargs"] == expected_kwargs


def test_object_to_dict_one_required_param():
    o = OpenAIEmbedding()
    result = object_to_dict(o)
    assert result["object_constructor"] == "llama_index.embeddings.openai.base.OpenAIEmbedding"
    expected_kwargs = {k: v for k, v in o.to_dict().items() if k not in {"class_name", "api_key"}}
    assert result["object_kwargs"] == expected_kwargs


def test_construct_prompt_template_object_success(qa_prompt_template):
    kwargs = qa_prompt_template.dict()
    observed = _construct_prompt_template_object(PromptTemplate, kwargs)
    assert observed == qa_prompt_template


def test_construct_prompt_template_object_no_template_kwarg():
    kwargs = {}
    with pytest.raises(ValueError, match="'template' is a required kwargs and is not present"):
        _construct_prompt_template_object(PromptTemplate, kwargs)


def test_settings_serialization_full_object(tmp_path, settings):
    path = tmp_path / "serialized_settings.json"
    serialize_settings(path)

    with open(path) as f:
        objects = json.load(f)

    assert len(set(objects.keys()) - set(settings.__dict__.keys())) == 0


def _assert_equal(settings_obj, deserialized_obj):
    if isinstance(settings_obj, list):
        assert len(settings_obj) == len(deserialized_obj)
        for i in range(len(settings_obj)):
            _assert_equal(settings_obj[i], deserialized_obj[i])
    else:
        for k, v in settings_obj.__dict__.items():
            if k != "callback_manager":
                assert getattr(deserialized_obj, k) == v
            else:
                assert getattr(deserialized_obj, k) is not None


def test_settings_serde(tmp_path, settings, mock_logger):
    path = tmp_path / "serialized_settings.json"
    _llm = settings.llm
    assert settings.llm.api_key == "test"
    _embed_model = settings.embed_model
    _node_parser = settings.node_parser
    _prompt_helper = settings.prompt_helper
    _transformations = settings.transformations

    serialize_settings(path)

    assert mock_logger.info.call_count == 2  # 1 for API key, 1 for unsupported objects
    log_message = mock_logger.info.call_args[0][0]
    assert log_message.startswith("The following objects in Settings are not supported")
    assert " - function for Settings.tokenizer" in log_message
    assert " - CallbackManager for Settings.callback_manager" in log_message

    for k in Settings.__dict__.keys():
        setattr(Settings, k, None)

    deserialize_settings(path)

    assert Settings is not None
    # Token is automatically applied from environment vars
    _assert_equal(Settings.llm, _llm)
    _assert_equal(Settings.embed_model, _embed_model)
    assert Settings.callback_manager is not None  # Auto-generated from defaults
    assert Settings.tokenizer is not None  # Auto-generated from defaults
    _assert_equal(Settings.node_parser, _node_parser)
    _assert_equal(Settings.prompt_helper, _prompt_helper)
    _assert_equal(Settings.transformations, _transformations)
