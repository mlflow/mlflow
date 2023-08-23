import os
import random
import shutil
import sys
import tempfile

from mlflow import (
    MlflowClient,
    active_run,
    get_artifact_uri,
    get_tracking_uri,
    log_artifact,
    log_artifacts,
    log_metric,
    log_param,
)

if __name__ == "__main__":
    print(f"Running {sys.argv[0]} with tracking URI {get_tracking_uri()}")
    log_param("param1", 5)
    log_metric("foo", 5)
    log_metric("foo", 6)
    log_metric("foo", 7)
    log_metric("random_int", random.randint(0, 100))
    run_id = active_run().info.run_id
    # Get run metadata & data from the tracking server
    service = MlflowClient()
    run = service.get_run(run_id)
    print(f"Metadata & data for run with UUID {run_id}: {run}")
    local_dir = tempfile.mkdtemp()
    message = "test artifact written during run {} within artifact URI {}\n".format(
        active_run().info.run_id,
        get_artifact_uri(),
    )
    try:
        file_path = os.path.join(local_dir, "some_output_file.txt")
        with open(file_path, "w") as handle:
            handle.write(message)
        log_artifacts(local_dir, "some_subdir")
        log_artifact(file_path, "another_dir")
    finally:
        shutil.rmtree(local_dir)
