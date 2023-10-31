import subprocess
import sys
from unittest import mock

import pytest

from mlflow.utils import request_utils


def test_request_utils_does_not_import_mlflow(tmp_path):
    file_content = f"""
import importlib.util
import os
import sys

file_path = r"{request_utils.__file__}"
module_name = "mlflow.utils.request_utils"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

assert "mlflow" not in sys.modules
assert "mlflow.utils.request_utils" in sys.modules
"""
    test_file = tmp_path.joinpath("test_request_utils_does_not_import_mlflow.py")
    test_file.write_text(file_content)

    subprocess.run([sys.executable, str(test_file)], check=True)


class IncompleteResponse:
    def __init__(self):
        self.headers = {"Content-Length": "100"}
        raw = mock.MagicMock()
        raw.tell.return_value = 50
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_download_chunk_incomplete_read(tmp_path):
    with mock.patch.object(
        request_utils, "cloud_storage_http_request", return_value=IncompleteResponse()
    ):
        download_path = tmp_path / "chunk"
        download_path.touch()
        with pytest.raises(IOError, match="Incomplete read"):
            request_utils.download_chunk(
                range_start=0,
                range_end=999,
                headers={},
                download_path=download_path,
                http_uri="https://example.com",
            )
