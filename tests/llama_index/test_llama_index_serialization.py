import json
from collections import Counter, deque

import pytest
from llama_index.core import PromptTemplate

from mlflow.llama_index.serialize_objects import (
    _construct_prompt_template_object,
    _deserialize_dict_of_objects,
    _get_object_import_path,
    _sanitize_api_key,
    deserialize_settings,
    object_to_dict,
    serialize_settings,
)

from tests.llama_index._llama_index_test_fixtures import (
    embed_model,  # noqa: F401
    llm,  # noqa: F401
    qa_prompt_template,  # noqa: F401
    settings,  # noqa: F401
    single_index,  # noqa: F401
)


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
    assert _get_object_import_path(Counter, do_validate_import=True) == expected_path


def test_object_is_class_do_validate_raises():
    class CustomClass:
        pass

    with pytest.raises(ValueError, match="does not have"):
        _get_object_import_path(CustomClass, do_validate_import=True)


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


def test_object_to_dict_no_required_param(llm):
    o = llm
    result = object_to_dict(o)
    assert (
        result["object_constructor"] == "tests.llama_index._llama_index_test_fixtures.MockChatLLM"
    )
    expected_kwargs = {k: v for k, v in o.to_dict().items() if k != "class_name"}
    assert result["object_kwargs"] == expected_kwargs


def test_object_to_dict_one_required_param(embed_model):
    o = embed_model
    result = object_to_dict(o)
    assert (
        result["object_constructor"] == "llama_index.core.embeddings.mock_embed_model.MockEmbedding"
    )
    expected_kwargs = {k: v for k, v in o.to_dict().items() if k != "class_name"}
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


def test_deserialize_dict_of_objects(tmp_path, settings):
    path = tmp_path / "serialized_settings.json"
    serialize_settings(path)
    observed = _deserialize_dict_of_objects(path)

    assert len(set(observed.keys()) - set(settings.__dict__.keys())) == 0
    assert observed["_llm"] == settings.llm
    assert observed["_embed_model"] == settings.embed_model
    assert "_callback_manager" not in observed.keys()
    assert "_tokenizer" not in observed.keys()  # TODO
    assert observed["_node_parser"] == settings.node_parser
    assert observed["_prompt_helper"] == settings.prompt_helper
    assert observed["_transformations"] == settings.transformations


def test_settings_serde(tmp_path, settings):
    path = tmp_path / "serialized_settings.json"
    _llm = settings.llm
    _embed_model = settings.embed_model
    _callback_manager = settings.callback_manager
    _tokenizer = settings.tokenizer
    _node_parser = settings.node_parser
    _prompt_helper = settings.prompt_helper
    _transformations = settings.transformations

    serialize_settings(path)
    del settings

    deserialize_settings(path)

    from llama_index.core import Settings

    assert Settings is not None
    assert Settings.llm == _llm
    assert Settings.embed_model == _embed_model
    assert Settings.callback_manager == _callback_manager  # Auto-generated from defaults
    assert Settings.tokenizer == _tokenizer  # Auto-generated from defaults
    assert Settings.node_parser == _node_parser
    assert Settings.prompt_helper == _prompt_helper
    assert Settings.transformations == _transformations
