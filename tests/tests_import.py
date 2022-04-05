import os
import subprocess
import uuid


def test_import_mlflow(tmp_path):
    tmp_script = tmp_path.joinpath("test.sh")
    uid = uuid.uuid4().hex
    tmp_script.write_text(
        """
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
""".format(
            uid=uid
        )
    )
    tmp_script.chmod(0o777)
    workdir = "/app"
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
            "python:3.7",
            "bash",
            "-c",
            f"/tmp/{uid}/{tmp_script.name}",
        ],
        check=True,
        text=True,
    )
