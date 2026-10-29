import pandas as pd
import pytest

from mlflow.dspy.wrapper import DspyModelWrapper


class DummyDSPyModel:
    """Minimal callable DSPy-like model for testing."""
    def __call__(self, **kwargs):
        return kwargs


def test_dspy_wrapper_accepts_single_row_dataframe():
    wrapper = DspyModelWrapper(
        model=DummyDSPyModel(),
        dspy_settings={},
    )

    df = pd.DataFrame([{
        "prompt": "hello",
        "response": "world",
        "violation_rule": "test",
    }])

    output = wrapper.predict(df)

    assert output == {
        "prompt": "hello",
        "response": "world",
        "violation_rule": "test",
    }


def test_dspy_wrapper_rejects_multi_row_dataframe():
    wrapper = DspyModelWrapper(
        model=DummyDSPyModel(),
        dspy_settings={},
    )

    df = pd.DataFrame([
        {"prompt": "a"},
        {"prompt": "b"},
    ])

    with pytest.raises(Exception):
        wrapper.predict(df)
