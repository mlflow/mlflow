import importlib


def string_is_plugin_obj(string: str) -> bool:
    """
    Returns True if the given string refers ti
    a plugin object in the format of ``plugin:module:obj``
    """
    return string.startswith("plugin:") and len(string.split(":")) == 3


def import_plugin_obj(string: str):
    """
    Import a plugin object from a string in the format of ``plugin:module:obj``
    """
    _, module, obj_name = string.split(":")
    mod = importlib.import_module(module)
    return getattr(mod, obj_name)
