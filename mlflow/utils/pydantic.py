from typing import Callable
from packaging import version
import pydantic

IS_PYDANTIC_V2 = version.parse(pydantic.version.VERSION) >= version.parse("2.0")


def pydantic_field_validator(field_name: str):
    def decorator(func: Callable) -> Callable:
        if IS_PYDANTIC_V2:
            from pydantic import field_validator

            return field_validator(field_name, mode="before")(func)
        else:
            from pydantic import validator

            return validator(field_name, pre=True)(func)

    return decorator
