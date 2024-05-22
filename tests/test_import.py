import os
import shutil
import subprocess
import sys
import uuid

import pytest

from mlflow.utils.os import is_windows


@pytest.mark.skipif(is_windows(), reason="This test fails on Windows")
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is required to run this test")
def test_import_mlflow(tmp_path):
    tmp_script = tmp_path.joinpath("test.sh")
    uid = uuid.uuid4().hex
    tmp_script.write_text(
        f"""
set -ex

# Install mlflow without extra dependencies
pip install -e .

# Move to /tmp/{uid} which should only contain this shell script
cd /tmp/{uid}

# Ensure mlflow can be imported
python -c 'import mlflow'

# Ensure importing mlflow does not create an mlruns directory
if [ -d "./mlruns" ]; then
    exit 1
fi
"""
    )
    tmp_script.chmod(0o777)
    workdir = "/app"
    python_ver = ".".join(map(str, sys.version_info[:2]))
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-w",
            workdir,
            "-v",
            f"{os.getcwd()}:{workdir}",
            "-v",
            f"{tmp_path}:/tmp/{uid}",
            f"python:{python_ver}",
            "bash",
            "-c",
            f"/tmp/{uid}/{tmp_script.name}",
        ],
        check=True,
        text=True,
    )
