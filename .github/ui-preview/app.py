import logging
import os
import subprocess
import sys
from pathlib import Path

import mlflow.server
from mlflow.demo import generate_all_demos

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def setup():
    # Extract UI build assets into the mlflow package's expected location
    tar_path = Path(__file__).parent.resolve() / "build.tar.gz"
    target_dir = Path(mlflow.server.__file__).parent / "js"
    target_dir.mkdir(parents=True, exist_ok=True)

    _logger.info("Extracting UI assets to %s", target_dir)
    subprocess.check_call(["tar", "xzf", tar_path, "-C", target_dir])

    # Generate demo data
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    _logger.info("Generating demo data...")
    generate_all_demos()
    _logger.info("Demo data generated.")


def main():
    setup()

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--default-artifact-root",
        "./mlartifacts",
        "--serve-artifacts",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--workers",
        "1",
    ]
    _logger.info("Starting MLflow server: %s", " ".join(cmd))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
