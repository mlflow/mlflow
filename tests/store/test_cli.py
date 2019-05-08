import json
import os
import posixpath

import mlflow
import mlflow.pyfunc
from mlflow.entities import FileInfo
from mlflow.store.cli import _file_infos_to_json
from mlflow.utils.file_utils import TempDir
from subprocess import Popen, STDOUT, PIPE


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


def test_download_artifacts_from_uri():
    with mlflow.start_run() as run:
        with TempDir() as tmp:
            local_path = tmp.path("test")
            with open(local_path, "w") as f:
                f.write("test")
            mlflow.log_artifact(local_path, "test")
    command = ["mlflow", "artifacts", "download-from-uri", "-a"]
    # Test with run uri
    run_uri = "runs:/{run_id}/test".format(run_id=run.info.run_id)
    actual_uri = posixpath.join(run.info.artifact_uri, "test")
    for uri in (run_uri, actual_uri):
        p = Popen(command + [uri], stdout=PIPE,
                  stderr=STDOUT)
        output = p.stdout.readlines()
        downloaded_file_path = output[-1].strip()
        downloaded_file = os.listdir(downloaded_file_path)[0]
        with open(os.path.join(downloaded_file_path, downloaded_file), "r") as f:
            assert f.read() == "test"
