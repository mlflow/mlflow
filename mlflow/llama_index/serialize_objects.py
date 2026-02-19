import importlib
import inspect
import json
import logging
from typing import Any, Callable

from llama_index.core import PromptTemplate
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import BaseComponent

_logger = logging.getLogger(__name__)


def _get_object_import_path(o: object) -> str:
    if not inspect.isclass(o):
        o = o.__class__

    module_name = inspect.getmodule(o).__name__
    class_name = o.__qualname__

    # Validate the import
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise ValueError(f"Module {module} does not have {class_name}")

    return f"{module_name}.{class_name}"


def _sanitize_api_key(object_as_dict: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in object_as_dict.items() if "api_key" not in k.lower()}


def object_to_dict(o: object):
    if isinstance(o, (list, tuple)):
        return [object_to_dict(v) for v in o]

    if isinstance(o, BaseComponent):
        # we can't serialize callables in the model fields
        callable_fields = set()
        # Access model_fields from the class to avoid pydantic deprecation warning
        fields = (
            o.__class__.model_fields if hasattr(o.__class__, "model_fields") else o.model_fields
        )
        for k, v in fields.items():
            field_val = getattr(o, k, None)
            # Exclude all callable fields, including those with default values
            # to prevent serialization issues in llama_index
            if callable(field_val):
                callable_fields.add(k)
        # exclude default values from serialization to avoid
        # unnecessary clutter in the serialized object
        o_state_as_dict = o.to_dict(exclude=callable_fields)

        if o_state_as_dict != {}:
            o_state_as_dict = _sanitize_api_key(o_state_as_dict)
            o_state_as_dict.pop("class_name")
        else:
            return o_state_as_dict

        return {
            "object_constructor": _get_object_import_path(o),
            "object_kwargs": o_state_as_dict,
        }
    else:
        return None


def _construct_prompt_template_object(
    constructor: Callable[..., PromptTemplate], kwargs: dict[str, Any]
) -> PromptTemplate:
    """Construct a PromptTemplate object based on the constructor and kwargs.

    This method is necessary because the `template_vars` cannot be passed directly to the
    constructor and needs to be set on an instantiated object.
    """
    if template := kwargs.pop("template", None):
        prompt_template = constructor(template)
        for k, v in kwargs.items():
            setattr(prompt_template, k, v)

        return prompt_template
    else:
        raise ValueError(
            "'template' is a required kwargs and is not present in the prompt template kwargs."
        )


def dict_to_object(object_representation: dict[str, Any]) -> object:
    if "object_constructor" not in object_representation:
        raise ValueError("'object_constructor' key not found in dict.")
    if "object_kwargs" not in object_representation:
        raise ValueError("'object_kwargs' key not found in dict.")

    constructor_str = object_representation["object_constructor"]
    kwargs = object_representation["object_kwargs"]

    import_path, class_name = constructor_str.rsplit(".", 1)
    module = importlib.import_module(import_path)

    if isinstance(module, PromptTemplate):
        return _construct_prompt_template_object(module, kwargs)
    else:
        object_class = getattr(module, class_name)

        # Many embeddings model accepts parameter `model`, while BaseEmbedding accepts `model_name`.
        # Both parameters will be serialized as kwargs, but passing both to the constructor will
        # raise duplicate argument error. Some class like OpenAIEmbedding handles this in its
        # constructor, but not all integrations do. Therefore, we have to handle it here.
        # E.g. https://github.com/run-llama/llama_index/blob/2b18eb4654b14c68d63f6239cddb10740668fbc8/llama-index-integrations/embeddings/llama-index-embeddings-openai/llama_index/embeddings/openai/base.py#L316-L320
        if (
            issubclass(object_class, BaseEmbedding)
            and (model := kwargs.get("model"))
            and (model_name := kwargs.get("model_name"))
            and model == model_name
        ):
            kwargs.pop("model_name")

        return object_class.from_dict(kwargs)


def _deserialize_dict_of_objects(path: str) -> dict[str, Any]:
    with open(path) as f:
        to_deserialize = json.load(f)

        output = {}
        for k, v in to_deserialize.items():
            if isinstance(v, list):
                output.update({k: [dict_to_object(vv) for vv in v]})
            else:
                output.update({k: dict_to_object(v)})

        return output


def serialize_settings(path: str) -> None:
    """Serialize the global LlamaIndex Settings object to a JSON file at the given path."""
    from llama_index.core import Settings

    _logger.info(
        "API key(s) will be removed from the global Settings object during serialization "
        "to protect against key leakage. At inference time, the key(s) must be passed as "
        "environment variables."
    )

    to_serialize = {}
    unsupported_objects = []

    for k, v in Settings.__dict__.items():
        if v is None:
            continue

        # Setting.callback_manager is default to an empty CallbackManager instance.
        if (k == "_callback_manager") and isinstance(v, CallbackManager) and v.handlers == []:
            continue

        def _convert(obj):
            object_json = object_to_dict(obj)
            if object_json is None:
                prop_name = k.removeprefix("_")
                unsupported_objects.append((prop_name, v))
            return object_json

        if isinstance(v, list):
            to_serialize[k] = [_convert(obj) for obj in v if v is not None]
        else:
            if (object_json := _convert(v)) and (object_json is not None):
                to_serialize[k] = object_json

    if unsupported_objects:
        msg = (
            "The following objects in Settings are not supported for serialization and will not "
            "be logged with your model. MLflow only supports serialization of objects that inherit "
            "from llama_index.core.schema.BaseComponent.\n"
        )
        msg += "\n".join(f" - {type(v).__name__} for Settings.{k}" for k, v in unsupported_objects)
        _logger.info(msg)

    with open(path, "w") as f:
        json.dump(to_serialize, f, indent=2)


def deserialize_settings(path: str):
    """Deserialize the global LlamaIndex Settings object from a JSON file at the given path."""
    settings_dict = _deserialize_dict_of_objects(path)

    from llama_index.core import Settings

    for k, v in settings_dict.items():
        # To use the property setter rather than directly setting the private attribute e.g. _llm
        k = k.removeprefix("_")
        setattr(Settings, k, v)
