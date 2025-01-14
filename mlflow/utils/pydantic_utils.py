from typing import Any

import pydantic
from packaging.version import Version
from pydantic import BaseModel

IS_PYDANTIC_V2_OR_NEWER = Version(pydantic.VERSION).major >= 2


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
