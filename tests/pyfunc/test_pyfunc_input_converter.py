from dataclasses import asdict, dataclass
from typing import List, Optional

import pandas as pd
import pytest

from mlflow.models.rag_signatures import ChatCompletionRequest
from mlflow.pyfunc.utils.input_converter import _hydrate_dataclass


def test_hydrate_dataclass_input_no_dataclass():
    # Define a class that is not a dataclass
    class NotADataclass:
        pass

    # Create some dummy data as a pandas df
    data = {"a": 1, "b": 2}
    df = pd.DataFrame(data, index=[0])

    # Check that an error is raised when trying to hydrate the dataclass
    with pytest.raises(ValueError, match="NotADataclass is not a dataclass"):
        _hydrate_dataclass(NotADataclass, df.iloc[0])


def test_hydrate_dataclass_simple():
    # Define a dataclass
    @dataclass
    class MyDataclass:
        a: int
        b: int

    # Create some dummy data as a pandas df
    df = pd.DataFrame({"a": [1], "b": [2]})

    # Check that the dataclass is hydrated
    result = _hydrate_dataclass(MyDataclass, df.iloc[0])
    assert result == MyDataclass(a=1, b=2)


def test_hydrate_dataclass_complex():
    # Define a more complex dataclass
    @dataclass
    class MyDataclass:
        a: int
        b: int

    @dataclass
    class MyListDataclass:
        c: List[MyDataclass]

    # Create some dummy data as a pandas df
    df = pd.DataFrame({"c": [[{"a": 1, "b": 2}, {"a": 3, "b": 4}]]})

    # Check that the dataclass is hydrated
    result = _hydrate_dataclass(MyListDataclass, df.iloc[0])
    assert result == MyListDataclass(c=[MyDataclass(a=1, b=2), MyDataclass(a=3, b=4)])


@dataclass
class CustomInput:
    id: int = 0


@dataclass
class FlexibleChatCompletionRequest(ChatCompletionRequest):
    custom_input: Optional[CustomInput] = None


def test_hydrate_child_dataclass():
    result = _hydrate_dataclass(
        FlexibleChatCompletionRequest,
        asdict(FlexibleChatCompletionRequest(custom_input=CustomInput())),
    )
    assert result == FlexibleChatCompletionRequest(custom_input=CustomInput())


def test_hydrate_optional_dataclass():
    result = _hydrate_dataclass(
        FlexibleChatCompletionRequest,
        asdict(FlexibleChatCompletionRequest(custom_input=None)),
    )
    assert result == FlexibleChatCompletionRequest(custom_input=None)
