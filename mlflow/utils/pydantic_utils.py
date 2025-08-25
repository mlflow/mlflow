from typing import Any, Callable

import pydantic
from packaging.version import Version
from pydantic import BaseModel

IS_PYDANTIC_V2_OR_NEWER = Version(pydantic.VERSION).major >= 2


def field_validator(field: str, mode: str = "before"):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if IS_PYDANTIC_V2_OR_NEWER:
            from pydantic import field_validator as pydantic_field_validator

            return pydantic_field_validator(field, mode=mode)(func)
        else:
            from pydantic import validator as pydantic_field_validator

            return pydantic_field_validator(field, pre=mode == "before")(func)

    return decorator


def model_validator(mode: str, skip_on_failure: bool = False):
    """
    A wrapper for Pydantic model validator that is compatible with Pydantic v1 and v2.
    Note that the `skip_on_failure` argument is only available in Pydantic v1.

    To use this decorator, the function must be of below signature:
    def func(cls, values: Any) -> Any:
        ...
    where `cls` is the Pydantic model class and `values` is the dictionary of values.

    For Pydantic v2 BaseModel, import `model_validator` from pydantic directly.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if IS_PYDANTIC_V2_OR_NEWER:
            from pydantic import model_validator as pydantic_model_validator

            if mode == "after":

                def wrapper(self):
                    return func(type(self), self)

                return pydantic_model_validator(mode=mode)(wrapper)
            else:
                return pydantic_model_validator(mode=mode)(func)
        else:
            from pydantic import root_validator

            return root_validator(pre=mode == "before", skip_on_failure=skip_on_failure)(func)

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
