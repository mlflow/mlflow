from typing import Any

import pydantic
from packaging.version import Version
from pydantic import BaseModel

IS_PYDANTIC_V2_OR_NEWER = Version(pydantic.VERSION).major >= 2

if IS_PYDANTIC_V2_OR_NEWER:
    from pydantic import field_validator as py_validator

    def validator(field, pre=True):
        mode = "before" if pre else "after"
        def decorator(func):
            return py_validator(field, mode=mode)(func)
        return decorator
else:
    from pydantic import validator as py_validator

    def validator(field, pre=True):
        def decorator(func):
            return py_validator(field, pre=pre)(func)
        return decorator


def model_dump_compat(pydantic_model: BaseModel, **kwargs: Any) -> dict[str, Any]:
    """
    Dump the Pydantic model to dictionary, in a compatible way for Pydantic v1 and v2.

    Args:
        pydantic_model: The Pydantic model to dump.
        kwargs: Additional arguments to pass to the dump method.
    """
    return (
        pydantic_model.model_dump(**kwargs)
        if IS_PYDANTIC_V2_OR_NEWER
        else pydantic_model.dict(**kwargs)
    )
