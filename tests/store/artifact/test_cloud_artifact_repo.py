import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.cloud_artifact_repo import _readable_size, _validate_chunk_size_aws


@pytest.mark.parametrize(
    ("size", "size_str"), [(5 * 1024**2, "5.00 MB"), (712.345 * 1024**2, "712.35 MB")]
)
def test_readable_size(size, size_str):
    assert _readable_size(size) == size_str


def test_chunk_size_validation_failure():
    with pytest.raises(MlflowException, match="Multipart chunk size"):
        _validate_chunk_size_aws(5 * 1024**2 - 1)
    with pytest.raises(MlflowException, match="Multipart chunk size"):
        _validate_chunk_size_aws(5 * 1024**3 + 1)
