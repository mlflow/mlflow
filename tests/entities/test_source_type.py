import pytest

from mlflow.entities import SourceType
from mlflow.exceptions import MlflowException


def test_an_unknown_source_type_raises_the_named_mlflow_exception():
    with pytest.raises(MlflowException, match="Could not get run status corresponding"):
        SourceType.from_string("NOT_A_SOURCE")


def test_an_unknown_source_type_code_raises_the_named_mlflow_exception():
    with pytest.raises(MlflowException, match="Could not get string corresponding"):
        SourceType.to_string(-1)


def test_the_round_trip_still_works():
    for source_type in SourceType.SOURCETYPE_TO_STRING:
        assert SourceType.from_string(SourceType.to_string(source_type)) == source_type
