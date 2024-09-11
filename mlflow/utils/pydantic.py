from typing import Callable

import pydantic
from packaging import version

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


def pydantic_model_validator(mode: str):
    def decorator(func: Callable) -> Callable:
        if IS_PYDANTIC_V2:
            from pydantic import model_validator

            return model_validator(mode=mode)(func)
        else:
            from pydantic import root_validator

            pre = mode == "pre"
            return root_validator(pre=pre)(func)

    return decorator
