import json

from mlflow.entities import FileInfo
from mlflow.store.cli import _file_infos_to_json


def test_file_info_to_json():
    file_infos = [
        FileInfo("/my/file", False, 123),
        FileInfo("/my/dir", True, None),
    ]
    info_str = _file_infos_to_json(file_infos)
    assert json.loads(info_str) == [{
        "path": "/my/file",
        "is_dir": False,
        "file_size": "123",
    }, {
        "path": "/my/dir",
        "is_dir": True,
    }]
