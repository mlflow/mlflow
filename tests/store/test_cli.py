from click.testing import CliRunner
from mock import mock

from mlflow.entities import FileInfo
from mlflow.store.cli import _file_infos_to_json

def test_file_info_to_json():
  file_infos = [
    FileInfo("/my/file", False, 123),
    FileInfo("/my/dir", True, None),
  ]
  info_str = _file_infos_to_json(file_infos)
  assert info_str == (
    '[{\n' +
    '  "path": "/my/file",\n' +
    '  "is_dir": false,\n' +
    '  "file_size": "123"\n' +
    '}, {\n' +
    '  "path": "/my/dir",\n' +
    '  "is_dir": true\n' +
    '}]'
  )
