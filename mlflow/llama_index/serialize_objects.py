import importlib
import inspect
import json
import logging
from typing import Callable, Dict, Set, Tuple

from llama_index.core import PromptTemplate, Settings

_logger = logging.getLogger(__name__)

_SUPPORTED_SERIALIZATION_METHODS = ["to_dict", "dict"]


def _extract_constructor_from_object(o: object) -> Callable:
    if hasattr(o, "__class__"):
        for cls in inspect.getmro(o.__class__):
            if hasattr(cls, "__init__"):
                return cls
        raise AttributeError(
            f"Class {o.__class} does not have a __init__ method and thereby cannot "
            "be concerted to a callable."
        )
    else:
        raise AttributeError(
            f"Object {o} does not have __class__ attribute and thereby cannot be "
            "converted to callable."
        )


def _get_dict_method_if_exists(o: object) -> Dict[str, any]:
    for method in _SUPPORTED_SERIALIZATION_METHODS:
        if hasattr(o, method):
            return getattr(o, method)()

    raise AttributeError(f"Object {o} does not have a supported deserialization method.")


def _get_kwargs(c: Callable) -> Tuple[Set]:
    params = inspect.signature(c).parameters
    required_kwargs, optional_kwargs = [], []

    for name, param in params.items():
        if name != "self":
            if param.default is inspect.Parameter.empty:
                required_kwargs.append(name)
            else:
                optional_kwargs.append(name)

    return (required_kwargs, optional_kwargs)


def _get_object_import_path(o: object, do_validate_import: bool = False) -> str:
    """Return an import path to the object."""
    # TODO: class_name is in the to_dict() payload. If we can traverse the dependency
    # tree to get an object constructor by class name, that might be more robust given
    # import paths can change
    if not inspect.isclass(o):
        o = o.__class__

    module_name = inspect.getmodule(o).__name__
    class_name = o.__qualname__

    if do_validate_import:
        module = importlib.import_module(module_name)

        if not hasattr(module, class_name):
            raise ValueError(f"Module {module} does not have {class_name}")

    return f"{module_name}.{class_name}"


def _sanitize_api_key(d: Dict[str, str]) -> Dict[str, str]:
    # TODO: drop API key and make a note that it must be in an enviornment variable
    return d


def _has_arg_unpacking(c: Callable) -> bool:
    """
    If the passed callable is a wrapped function, the kwargs unpack will operate on the
    wrapping function.
    """
    params = inspect.signature(c).parameters.values()
    return any(p.kind == p.VAR_POSITIONAL for p in params)


def _has_kwarg_unpacking(c: Callable) -> bool:
    """
    If the passed callable is a wrapped function, the kwargs unpack will operate on the
    wrapping function.
    """
    params = inspect.signature(c).parameters.values()
    return any(p.kind == p.VAR_KEYWORD for p in params)


def _sanitize_kwargs(o: object, o_state_as_dict: Dict[str, any]) -> Dict[str, any]:
    """
    Sanitize the object kwargs by...
    1. Asserting that all required kwargs exist in the object state dict.
    2. Dropping all state that is not an argument in the object constructor signature.
    """
    o_callable = _extract_constructor_from_object(o)
    required_kwargs, optional_kwargs = _get_kwargs(o_callable)
    missing_kwargs = set(required_kwargs) - set(o_state_as_dict.keys())
    if len(missing_kwargs) > 0:
        raise ValueError(
            f"When trying to validate {o.__class__} payload, the following required kwargs "
            f"were missing: {missing_kwargs}. It is possible that the incorrect constructor "
            f"{o_callable.__class__} is being used for the inferred kwargs."
        )

    if _has_kwarg_unpacking(o_callable) or _has_arg_unpacking(o_callable):
        # If there is argument unpacking, we can't condense the payload to kwargs in the constructor
        return o_state_as_dict
    else:
        return {k: v for k, v in o_state_as_dict.items() if k in required_kwargs + optional_kwargs}


def object_to_dict(o: object) -> None:
    try:
        o_state_as_dict = _get_dict_method_if_exists(o)
    except AttributeError as e:
        if "does not have a supported deserialization method" in str(e):
            _logger.info(str(e))
            return {}
        else:
            raise

    return {
        "object_constructor": _get_object_import_path(o, do_validate_import=True),
        "object_kwargs": _sanitize_kwargs(o, o_state_as_dict),
    }


def _construct_prompt_template_object(c: Callable, kwargs: Dict[str, any]) -> PromptTemplate:
    """Construct a PromptTemplate object based on the constructor and kwargs.

    This method is necessary because the `template_vars` cannot be passed directly to the
    constructor and needs to be set on an instantiated object.
    """
    if template := kwargs.pop("template", None):
        prompt_template = c(template)
        for k, v in kwargs.items():
            setattr(prompt_template, k, v)

        return prompt_template
    else:
        raise ValueError(
            "'template' is a required kwargs and is not present in the prompt " "template kwargs."
        )


def dict_to_object(d: Dict[str, str]) -> object:
    if "object_constructor" not in d:
        raise ValueError("'object_constructor' key not found in dict.")
    if "object_kwargs" not in d:
        raise ValueError("'object_kwargs' key not found in dict.")

    constructor = d["object_constructor"]
    kwargs = d["object_kwargs"]

    import_path, class_name = constructor.rsplit(".", 1)
    module = importlib.import_module(import_path)

    if isinstance(module, PromptTemplate):
        return _construct_prompt_template_object(module, kwargs)
    else:
        # Generic object that can be constructed from class __init__ method
        callable = getattr(module, class_name)
        return callable(**kwargs)


def _deserialize_json_to_dict_of_objects(path: str) -> Dict[str, any]:
    with open(path) as f:
        to_deserialize = json.load(f)

        output = {}
        for k, v in to_deserialize.items():
            if isinstance(v, list):
                output.update({k: [dict_to_object(vv) for vv in v]})
            else:
                output.update({k: dict_to_object(v)})

        return output


def _serialize_dict_of_objects_to_json(dict_of_objects: Dict[str, object], path: str) -> None:
    to_serialize = {}

    for k, v in dict_of_objects.items():
        if isinstance(v, list):  # noqa
            object_json = [object_to_dict(vv) for vv in v]
        else:
            object_json = object_to_dict(v)

        if object_json == {}:
            _logger.info(
                f"{k} serialization is not supported. It will not be logged with your model"
            )
        else:
            to_serialize.update({k: object_json})

    with open(path, "w") as f:
        json.dump(to_serialize, f, indent=2)


def serialize_settings_to_json(settings: Settings, path: str) -> None:
    _serialize_dict_of_objects_to_json(settings.__dict__, path)


def deserialize_json_to_settings(path: str) -> Settings:
    settings_dict = _deserialize_json_to_dict_of_objects(path)

    for k, v in settings_dict.items():
        setattr(Settings, k, v)

    return Settings


def serialize_engine_kwargs_to_json(engine_kwargs: Dict[str, object], path: str) -> None:
    _serialize_dict_of_objects_to_json(engine_kwargs, path)


def deserialize_json_to_engine_kwargs(path: str) -> Dict[str, object]:
    return _deserialize_json_to_dict_of_objects(path)
