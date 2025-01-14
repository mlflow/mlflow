"""
NB: This test file is executed with both pydantic v1 and v2 in the CI.
"""

from pydantic import BaseModel

from mlflow.utils.pydantic_utils import model_dump_compat


class MyModel(BaseModel):
    name: str
    age: int = 30


def test_model_dump_compat():
    model = MyModel(name="John", age=30)
    assert model_dump_compat(model) == {"name": "John", "age": 30}

    model = MyModel(name="John")
    assert model_dump_compat(model, exclude_unset=True) == {"name": "John"}
