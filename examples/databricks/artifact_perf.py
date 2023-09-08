"""
A simple script to benchmark MLflow's artifact upload/download performance.
"""
import contextlib
import hashlib
import os
import pathlib
import tempfile
import time

import psutil

import mlflow
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils.rest_utils import http_request
from mlflow.utils.databricks_utils import get_databricks_host_creds


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
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/jerry.liang@databricks.com/benchmark")

    host_creds = get_databricks_host_creds()
    table_id = "99fa55e5-ad56-4272-94c7-7e90888d5398"
    storage_url = "s3://us-west-2-extstaging-managed-catalog-test-bucket-1/19a85dee-54bc-43a2-87ab-023d0ec16013/tables/99fa55e5-ad56-4272-94c7-7e90888d5398"
    resp = http_request(
        host_creds,
        endpoint="/api/2.0/unity-catalog/temporary-table-credentials",
        method="POST",
        json={
            "operation": "READ_WRITE",
            "name": "main.default.jerry_broken_table",
            "table_id": table_id,
        },
    )
    aws_temp_creds = resp.json()["aws_temp_credentials"]

    s3_repo = S3ArtifactRepository(
        artifact_uri=os.path.join(storage_url, "model"), **aws_temp_creds
    )

    show_env_info()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        f = tmpdir / "large_file"
        file_size = 30 * MiB
        f.write_bytes(os.urandom(file_size))

        # Upload
        with mlflow.start_run():
            with timer() as t:
                s3_repo.log_artifact(f)
                print(f"Upload took {t():.3f} seconds")
            # artifact_uri = mlflow.get_artifact_uri(f.name)

        # Download
        dst_dir = tmpdir / "dst"
        os.mkdir(dst_dir)
        with timer() as t:
            dst_path = s3_repo.download_artifacts(artifact_path="large_file", dst_path=dst_dir)
            print(f"Download took {t():.3f} seconds")

        assert md5_checksum(f) == md5_checksum(dst_path), "File checksums do not match"


if __name__ == "__main__":
    main()
