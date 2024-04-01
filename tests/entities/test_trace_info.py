import pytest

from mlflow.entities import TraceInfo
from mlflow.exceptions import MlflowException


def test_validate_tag_key_value():
    TraceInfo.validate_tag_key_value(key="a", value="b")

    with pytest.raises(MlflowException, match=r"A key for a trace tag must be a non-empty string."):
        TraceInfo.validate_tag_key_value(key="", value="b")

    with pytest.raises(MlflowException, match=r"A key for a trace tag must be a non-empty string."):
        TraceInfo.validate_tag_key_value(key=1, value="b")

    with pytest.raises(MlflowException, match=r"A value for a trace tag must be a string."):
        TraceInfo.validate_tag_key_value(key="a", value=1)

    with pytest.raises(
        MlflowException, match=r"A key for a trace tag exceeds the maximum allowed "
    ):
        TraceInfo.validate_tag_key_value(key="a" * 251, value="b")

    with pytest.raises(
        MlflowException, match=r"A value for a trace tag exceeds the maximum allowed "
    ):
        TraceInfo.validate_tag_key_value(key="a", value="b" * 251)
