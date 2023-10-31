import subprocess
import sys

import mlflow


def test_mlflow_dev_is_installable():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            f"mlflow=={mlflow.__version__}",
        ],
        check=True,
    )
