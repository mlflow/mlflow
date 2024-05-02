import importlib


def string_is_plugin_obj(string: str) -> bool:
    return string.startswith("plugin:") and len(string.split(":")) == 3


def import_plugin_obj(string: str):
    _, module, obj = string.split(":")
    mod = importlib.import_module(module)
    return getattr(mod, obj)
