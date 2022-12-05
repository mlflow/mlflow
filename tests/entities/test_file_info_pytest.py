import pytest

from mlflow.entities import FileInfo
from tests.helper_functions import random_str, random_int


def _check(fi, path, is_dir, size_in_bytes):
    assert isinstance(fi, FileInfo)
    assert fi.path == path
    assert fi.is_dir == is_dir
    assert fi.file_size == size_in_bytes


path = random_str(random_int(10, 50))
is_dir = random_int(10, 2500) % 2 == 0
size_in_bytes = random_int(1, 10000)


@pytest.mark.parametrize(
    "fi",
    [
        FileInfo(path, is_dir, size_in_bytes),
        FileInfo.from_proto(FileInfo(path, is_dir, size_in_bytes).to_proto()),
        FileInfo.from_dictionary({"path": path, "is_dir": is_dir, "file_size": size_in_bytes}),
    ],
)
def test_creation_and_hydration(fi):
    # path = random_str(random_int(10, 50))
    # is_dir = random_int(10, 2500) % 2 == 0
    # size_in_bytes = random_int(1, 10000)
    # fi1 = FileInfo(path, is_dir, size_in_bytes)
    _check(fi, path, is_dir, size_in_bytes)

    as_dict = {"path": path, "is_dir": is_dir, "file_size": size_in_bytes}
    assert dict(fi) == as_dict

    # proto = fi1.to_proto()
    # fi2 = FileInfo.from_proto(proto)
    # _check(fi2, path, is_dir, size_in_bytes)

    # fi3 = FileInfo.from_dictionary(as_dict)
    # _check(fi3, path, is_dir, size_in_bytes)
