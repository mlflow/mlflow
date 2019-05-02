from __future__ import print_function

import os
import shutil
import sys
import random
import tempfile

import mlflow
from mlflow import log_metric, log_param, log_artifacts, get_artifact_uri, active_run,\
    get_tracking_uri, log_artifact

if __name__ == "__main__":
    print("Running {} with tracking URI {}".format(sys.argv[0], get_tracking_uri()))
    log_param("param1", 5)
    log_metric("foo", 5)
    log_metric("foo", 6)
    log_metric("foo", 7)
    log_metric("random_int", random.randint(0, 100))
    run_id = active_run().info.run_id
    # Get run metadata & data from the tracking server
    service = mlflow.tracking.MlflowClient()
    run = service.get_run(run_id)
    print("Metadata & data for run with UUID %s: %s" % (run_id, run))
    local_dir = tempfile.mkdtemp()
    message = "test artifact written during run %s within artifact URI %s\n" \
              % (active_run().info.run_id, get_artifact_uri())
    try:
        file_path = os.path.join(local_dir, "some_output_file.txt")
        with open(file_path, "w") as handle:
            handle.write(message)
        log_artifacts(local_dir, "some_subdir")
        log_artifact(file_path, "another_dir")
    finally:
        shutil.rmtree(local_dir)
