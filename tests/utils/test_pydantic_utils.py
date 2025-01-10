"""
NB: This test file is executed with both pydantic v1 and v2 in the CI.
"""

import pytest
from pydantic import BaseModel

from mlflow.utils.pydantic import IS_PYDANTIC_V2_OR_NEWER, model_dump_compat, model_validate_compat


class MyModel(BaseModel):
    name: str
    age: int = 30


def test_model_dump_compat():
    model = MyModel(name="John", age=30)
    assert model_dump_compat(model) == {"name": "John", "age": 30}

    model = MyModel(name="John")
    assert model_dump_compat(model, exclude_unset=True) == {"name": "John"}


def test_model_validate_compat():
    assert model_validate_compat(MyModel, {"name": "John", "age": "30"}) == MyModel(
        name="John", age=30
    )

    with pytest.raises(ValueError, match="1 validation error for MyModel"):
        model_validate_compat(MyModel, "invalid")

    # "strict" parameter is not supported in Pydantic v1
    if IS_PYDANTIC_V2_OR_NEWER:
        with pytest.raises(ValueError, match="1 validation error for MyModel"):
            model_validate_compat(MyModel, {"name": "John", "age": "30"}, strict=True)
