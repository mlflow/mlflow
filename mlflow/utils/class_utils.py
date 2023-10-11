import dis
import importlib
import inspect


def _get_class_from_string(fully_qualified_class_name):
    module, class_name = fully_qualified_class_name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), class_name)


def nameof(_) -> str:
    """
    Returns the name of a variable, attribute, or class
    """
    frame = inspect.currentframe().f_back
    code = frame.f_code
    return list(
        filter(lambda f: f.opname.startswith("LOAD_") and f.argval, dis.get_instructions(code))
    )[-1].argval
