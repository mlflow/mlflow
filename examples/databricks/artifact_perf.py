"""
A simple script to benchmark MLflow's artifact upload/download performance.
"""
import tempfile
import contextlib
import pathlib
import hashlib
import os
import psutil
import time

import mlflow


def show_env_info():
    svmem = psutil.virtual_memory()
    print("=" * 50)
    print(f"CPU count: {psutil.cpu_count()}")
    total = svmem.total // (1024**3)
    available = svmem.available // (1024**3)
    print(f"Memory (available/total): {available}/{total} GB")
    print(f"MLflow version: {mlflow.__version__}")
    print("=" * 50)


def md5_checksum(path):
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


@contextlib.contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def main():
    KiB = 1024
    MiB = KiB * KiB
    GiB = MiB * KiB

    # Uncomment the following lines if you're running this script outside of Databricks
    # using a personal access token:
    # mlflow.set_tracking_uri("databricks")
    # mlflow.set_experiment("/Users/<username>/benchmark")

    show_env_info()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        f = tmpdir / "large_file"
        file_size = 1 * GiB
        f.write_bytes(os.urandom(file_size))

        # Upload
        with mlflow.start_run():
            with timer() as t:
                mlflow.log_artifact(f)
                print(f"Upload took {t():.3f} seconds")
            artifact_uri = mlflow.get_artifact_uri(f.name)

        # Download
        dst_dir = tmpdir / "dst"
        with timer() as t:
            dst_path = mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri, dst_path=dst_dir
            )
            print(f"Download took {t():.3f} seconds")

        assert md5_checksum(f) == md5_checksum(dst_path), "File checksums do not match"


if __name__ == "__main__":
    main()
