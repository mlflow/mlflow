"""
Benchmark for artifact upload and download performance.
"""
import hashlib
import json
import os
import pathlib
import tempfile

import pandas as pd
import psutil

import mlflow
from mlflow.environment_variables import (
    MLFLOW_ENABLE_MULTIPART_DOWNLOAD,
    MLFLOW_ENABLE_MULTIPART_UPLOAD,
)
from mlflow.utils.time import Timer

GiB = 1024**3


def show_system_info():
    svmem = psutil.virtual_memory()
    info = json.dumps(
        {
            "MLflow version": mlflow.__version__,
            "MPU enabled": MLFLOW_ENABLE_MULTIPART_DOWNLOAD.get(),
            "MPD enabled": MLFLOW_ENABLE_MULTIPART_UPLOAD.get(),
            "CPU count": psutil.cpu_count(),
            "Memory total [GiB]": svmem.total // GiB,
            "Memory used [GiB]": svmem.used // GiB,
            "Memory available [GiB]": svmem.available // GiB,
        },
        indent=2,
    )
    max_len = max(map(len, info.splitlines()))
    print("=" * max_len)
    print(info)
    print("=" * max_len)


def md5_checksum(path):
    file_hash = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def upload_and_download(file_size, num_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        # Prepare files
        src_dir = tmpdir / "src"
        src_dir.mkdir()
        files = {}
        for i in range(num_files):
            f = src_dir / str(i)
            f.write_bytes(os.urandom(file_size))
            files[f.name] = f

        # Upload
        with mlflow.start_run() as run:
            with Timer() as t_upload:
                mlflow.log_artifacts(str(src_dir))

        # Download
        dst_dir = tmpdir / "dst"
        dst_dir.mkdir()
        with Timer() as t_download:
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"{run.info.artifact_uri}/", dst_path=dst_dir
            )

        # Verify checksums
        for f in dst_dir.rglob("*"):
            if f.is_dir():
                continue
            assert md5_checksum(f) == md5_checksum(files[f.name]), f"Checksum mismatch for {f}"

        return t_upload.elapsed, t_download.elapsed


def main():
    # Uncomment the following lines if you're running this script outside of Databricks
    # using a personal access token:
    # mlflow.set_tracking_uri("databricks")
    # mlflow.set_experiment("/Users/<username>/benchmark")

    FILE_SIZE = 1 * GiB
    NUM_FILES = 2
    NUM_ATTEMPTS = 3

    show_system_info()
    stats = []
    for i in range(NUM_ATTEMPTS):
        print(f"Running {i + 1} / {NUM_ATTEMPTS}")
        stats.append(upload_and_download(FILE_SIZE, NUM_FILES))

    df = pd.DataFrame(stats, columns=["upload [s]", "download [s]"])
    # show mean, min, max in markdown table
    print(df.aggregate(["count", "mean", "min", "max"]).to_markdown())


if __name__ == "__main__":
    main()
