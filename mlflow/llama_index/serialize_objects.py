import importlib
import inspect
import json
import logging
from typing import Any, Callable, Dict

import llama_index
from llama_index.core import PromptTemplate

from mlflow.exceptions import INTERNAL_ERROR, MlflowException

_logger = logging.getLogger(__name__)

# TODO: add versioning to the map
# TODO: think about hierarchy of objects
OBJECT_DICT_METHOD_MAP = {
    llama_index.core.base.llms.base.BaseLLM: ("to_dict", "from_dict"),
    llama_index.core.base.embeddings.base.BaseEmbedding: ("to_dict", "from_dict"),
    llama_index.core.node_parser.interface.NodeParser: ("to_dict", "from_dict"),
    llama_index.core.indices.prompt_helper.PromptHelper: ("to_dict", "from_dict"),
}


def _object_to_dict(obj: object) -> Dict[str, any]:
    for k, v in OBJECT_DICT_METHOD_MAP.items():
        if isinstance(obj, k):
            if not hasattr(obj, v[0]):
                raise MlflowException(
                    f"Failed to deserialize object {obj}. This is likely an unsupported "
                    "object in the MLflow Llama-Index flavor.",
                    error_code=INTERNAL_ERROR,
                )

            return getattr(obj, v[0])()

    return {}


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


def _sanitize_api_key(object_as_dict: Dict[str, str]) -> Dict[str, str]:
    keys_to_remove = [k for k in object_as_dict.keys() if "api_key" in k.lower()]

    for k in keys_to_remove:
        if object_as_dict.pop(k, None):
            _logger.info(
                "API key removed from object serialization. At inference time,"
                " the key must be passed as an environment variable or via inference"
                " parameters."
            )

    return object_as_dict


def object_to_dict(o: object) -> None:
    o_state_as_dict = _object_to_dict(o)

    if o_state_as_dict != {}:
        o_state_as_dict = _sanitize_api_key(o_state_as_dict)
        o_state_as_dict.pop("class_name")
    else:
        _logger.warning(f"Skipping serialization of {o} because...")
        return o_state_as_dict

    return {
        "object_constructor": _get_object_import_path(o),
        "object_kwargs": o_state_as_dict,
    }


def _construct_prompt_template_object(
    constructor: Callable, kwargs: Dict[str, any]
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


def dict_to_object(object_representation: Dict[str, Any]) -> object:
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

        for k, v in OBJECT_DICT_METHOD_MAP.items():
            if isinstance(object_class, k):
                if not hasattr(object_class, v[1]):
                    raise AttributeError(
                        f"Object {object_class} was inferred to be of type {k} but does not "
                        "have a {v[1]} method. Ensure that `OBJECT_DICT_METHOD_MAP` is "
                        "correct and the object is of type {k}."
                    )

                return object_class.from_dict(kwargs)

        return object_class(**kwargs)


def _deserialize_dict_of_objects(path: str) -> Dict[str, any]:
    with open(path) as f:
        to_deserialize = json.load(f)

        output = {}
        for k, v in to_deserialize.items():
            if isinstance(v, list):
                output.update({k: [dict_to_object(vv) for vv in v]})
            else:
                output.update({k: dict_to_object(v)})

        return output


def _serialize_dict_of_objects(dict_of_objects: Dict[str, object], path: str) -> None:
    to_serialize = {}

    for k, v in dict_of_objects.items():
        object_json = [object_to_dict(vv) for vv in v] if isinstance(v, list) else object_to_dict(v)

        if object_json == {}:
            _logger.info(
                f"{k} serialization is not supported. It will not be logged with your model"
            )
        else:
            to_serialize.update({k: object_json})

    with open(path, "w") as f:
        json.dump(to_serialize, f, indent=2)


def serialize_settings(path: str) -> None:
    from llama_index.core import Settings

    _serialize_dict_of_objects(Settings.__dict__, path)


def deserialize_settings(path: str):
    settings_dict = _deserialize_dict_of_objects(path)

    from llama_index.core import Settings

    for k, v in settings_dict.items():
        setattr(Settings, k, v)

    return Settings
