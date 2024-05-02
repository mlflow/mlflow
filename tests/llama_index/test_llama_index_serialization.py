import json
from collections import Counter, deque

import pytest
import tiktoken
from llama_index.core import (
    Settings,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter

from mlflow.llama_index.serialize_objects import (
    _extract_constructor_from_object,
    _get_dict_method_if_exists,
    _get_kwargs,
    _get_object_import_path,
    _sanitize_kwargs,
    object_to_dict,
    serialize_settings_to_json,
)

llm = MockLLM()
embed_model = MockEmbedding(embed_dim=1)


def test_extract_constructor_from_object_non_constructor_with_init():
    class TestClass:
        def __init__(self):
            self.value = 10

    init_method_from_obj = _extract_constructor_from_object(TestClass())
    init_method_from_class = TestClass().__class__

    assert init_method_from_obj.__qualname__ == init_method_from_class.__qualname__
    assert init_method_from_obj.__module__ == init_method_from_class.__module__
    assert _get_kwargs(init_method_from_obj) == _get_kwargs(init_method_from_class)


# TODO:
# def test_extract_constructor_from_object_raises():
#     class Empty:
#         pass

#     with pytest.raises(AttributeError) as exc_info:
#         _extract_constructor_from_object(Empty())
#     assert "cannot be converted to constructor" in str(exc_info.value)


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
    # NB: the rationale for using collections as the library to test
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


def test_sanitize_kwargs_no_required_kwargs():
    assert _sanitize_kwargs(llm, {}) == {}


def test_sanitize_kwargs_yes_required_kwargs():
    kwargs = {"embed_dim": 1}
    assert _sanitize_kwargs(embed_model, kwargs) == kwargs


def test_sanitize_kwargs_missing_required_kwargs():
    kwargs = {}
    with pytest.raises(
        ValueError, match="the following required kwargs were missing: {'embed_dim'}"
    ):
        _ = _sanitize_kwargs(embed_model, kwargs)


def test_sanitize_kwargs_extra_kwargs():
    kwargs = {"embed_dim": 1}
    assert _sanitize_kwargs(embed_model, kwargs | {"extra": 1}) == kwargs


def test_object_to_dict_no_required_param():
    o = MockLLM()
    result = object_to_dict(o)
    assert (
        result["object_constructor"] == "llama_index.core.llms.mock.MockLLM"
    )  # Expected import path
    expected_kwargs = {k: v for k, v in o.to_dict().items() if k != "class_name"}
    assert result["object_kwargs"] == expected_kwargs


def test_object_to_dict_one_required_param():
    o = MockEmbedding(embed_dim=1)
    result = object_to_dict(o)
    assert (
        result["object_constructor"] == "llama_index.core.embeddings.mock_embed_model.MockEmbedding"
    )  # Expected import path
    expected_kwargs = {k: v for k, v in o.to_dict().items() if k != "class_name"}
    assert result["object_kwargs"] == expected_kwargs


def test_settings_serialization_full_object(tmp_path):
    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=1)
    Settings.callback_manager = CallbackManager([TokenCountingHandler()])
    # TODO: use something other than openai
    Settings._tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    Settings._node_parser = SentenceSplitter(chunk_size=1024)
    Settings.context_window = 4096  # this enters the _prompt_helper field
    Settings._transformations = [SentenceSplitter(chunk_size=1024)]

    # Validate that the object is populated
    for k in Settings.__dict__.keys():
        assert Settings.__dict__[k] is not None

    path = tmp_path / "serialized_settings.json"
    serialize_settings_to_json(Settings, path)

    with open(path) as f:
        objects = json.load(f)

    assert len(set(objects.keys()) - set(Settings.__dict__.keys())) == 0
    assert "_transformations" not in objects.keys()
    assert "_callback_manager" not in objects.keys()
