import json
from collections import Counter, deque
from functools import wraps

import pytest
from llama_index.core import PromptTemplate

from mlflow.llama_index.serialize_objects import (
    _construct_prompt_template_object,
    _deserialize_json_to_dict_of_objects,
    _extract_constructor_from_object,
    _get_dict_method_if_exists,
    _get_kwargs,
    _get_object_import_path,
    _has_arg_unpacking,
    _has_kwarg_unpacking,
    _sanitize_kwargs,
    deserialize_json_to_engine_kwargs,
    deserialize_json_to_settings,
    object_to_dict,
    serialize_engine_kwargs_to_json,
    serialize_settings_to_json,
)

from tests.llama_index._llama_index_test_fixtures import (
    qa_prompt_template,
)


def test_extract_constructor_from_object_non_constructor_with_init():
    class TestClass:
        def __init__(self):
            self.value = 10

    init_method_from_obj = _extract_constructor_from_object(TestClass())
    init_method_from_class = TestClass().__class__

    assert init_method_from_obj.__qualname__ == init_method_from_class.__qualname__
    assert init_method_from_obj.__module__ == init_method_from_class.__module__
    assert _get_kwargs(init_method_from_obj) == _get_kwargs(init_method_from_class)


def test_get_dict_method_if_exists_passes_to_dict():
    expected = {"example": "example"}

    class ToDict:
        def to_dict(self):
            return expected

    observed = _get_dict_method_if_exists(ToDict())
    assert observed == expected


def test_get_dict_method_if_exists_passes_dict():
    expected = {"example": "example"}

    class _Dict:
        def dict(self):
            return expected

    observed = _get_dict_method_if_exists(_Dict())
    assert observed == expected


def test_get_dict_method_if_exists_fails():
    class NoSerialization:
        pass

    with pytest.raises(AttributeError, match="does not have a supported"):
        _ = _get_dict_method_if_exists(NoSerialization())


def test_get_kwargs_no_args():
    class NoArgs:
        def __init__(self):
            pass

    assert _get_kwargs(NoArgs.__init__) == ([], [])


def test_get_kwargs_one_required_one_optional():
    def func(x, y=1):
        pass

    assert _get_kwargs(func) == (["x"], ["y"])


def test_get_kwargs_two_required_two_optional():
    class TwoRequiredTwoOptional:
        def __init__(self, a, b, c=3, d=4):
            pass

    assert _get_kwargs(TwoRequiredTwoOptional.__init__) == (["a", "b"], ["c", "d"])


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


def test_has_arg_unpacking_true():
    def func(x, *args):
        pass

    assert _has_arg_unpacking(func)


def test_has_arg_unpacking_false():
    def func(x, y=2):
        pass

    assert not _has_arg_unpacking(func)


def test_has_kwarg_unpacking_true():
    def func(x, **kwargs):
        pass

    assert _has_kwarg_unpacking(func)


def test_has_kwarg_unpacking_false():
    def func(x, y=2):
        pass

    assert not _has_kwarg_unpacking(func)


def test_decorated_with_unpacking():
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapped

    @decorator
    def func(x, *args, **kwargs):
        pass

    assert _has_arg_unpacking(func)
    assert _has_kwarg_unpacking(func)


def test_sanitize_kwargs_no_required_kwargs(llm):
    assert _sanitize_kwargs(llm, {}) == {}


def test_sanitize_kwargs_yes_required_kwargs(embed_model):
    kwargs = {"embed_dim": 1}
    assert _sanitize_kwargs(embed_model, kwargs) == kwargs


def test_sanitize_kwargs_missing_required_kwargs(embed_model):
    kwargs = {}
    with pytest.raises(
        ValueError, match="the following required kwargs were missing: {'embed_dim'}"
    ):
        _ = _sanitize_kwargs(embed_model, kwargs)


def test_sanitize_kwargs_extra_kwargs(embed_model):
    kwargs = {"embed_dim": 1}
    assert _sanitize_kwargs(embed_model, kwargs | {"extra": 1}) == kwargs


def test_object_to_dict_no_required_param(llm):
    o = llm
    result = object_to_dict(o)
    assert (
        result["object_constructor"] == "llama_index.core.llms.mock.MockLLM"
    )  # Expected import path
    expected_kwargs = {k: v for k, v in o.to_dict().items() if k != "class_name"}
    assert result["object_kwargs"] == expected_kwargs


def test_object_to_dict_one_required_param(embed_model):
    o = embed_model
    result = object_to_dict(o)
    assert (
        result["object_constructor"] == "llama_index.core.embeddings.mock_embed_model.MockEmbedding"
    )  # Expected import path
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
    serialize_settings_to_json(settings, path)

    with open(path) as f:
        objects = json.load(f)

    assert len(set(objects.keys()) - set(settings.__dict__.keys())) == 0


def test_deserialize_json_to_dict_of_objects(tmp_path, settings):
    path = tmp_path / "serialized_settings.json"
    serialize_settings_to_json(settings, path)
    observed = _deserialize_json_to_dict_of_objects(path)

    assert len(set(observed.keys()) - set(settings.__dict__.keys())) == 0
    assert observed["_llm"] == settings.llm
    assert observed["_embed_model"] == settings.embed_model
    assert "_callback_manager" not in observed.keys()
    assert "_tokenizer" not in observed.keys()  # TODO:
    assert observed["_node_parser"] == settings.node_parser
    assert observed["_prompt_helper"] == settings.prompt_helper
    assert observed["_transformations"] == settings.transformations


def test_settings_serde(tmp_path, settings):
    path = tmp_path / "serialized_settings.json"
    serialize_settings_to_json(settings, path)
    observed = deserialize_json_to_settings(path)

    assert observed is not None


def test_engine_kwargs_serde(tmp_path, single_index):
    engine_kwargs = {
        "temperature": 0.0,
        "max_tokens": 100,
        "qa_prompt_template": qa_prompt_template,
    }
    path = tmp_path / "serialized_settings.json"
    serialize_engine_kwargs_to_json(engine_kwargs, path)
    observed = deserialize_json_to_engine_kwargs(path)

    chat_engine = single_index.as_chat_engine(**observed)
    assert chat_engine.chat("Spell llamaindex") != ""
