import mlflow
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback
import tempfile
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--tracking-uri", required=True)
parser.add_argument("--file-size-mib", required=True, default=1)
parser.add_argument("--num-concurrent-requests", required=True, default=5)
args = parser.parse_args()

tracking_uri = args.tracking_uri
file_size_mib = int(args.file_size_mib)
num_concurrent_requests = int(args.num_concurrent_requests)


def task(idx):
    idx = str(idx)
    logger.info(idx + " start")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment_name = "stream-s3"
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment = client.create_experiment(experiment_name)
    logger.info(idx + " created experiment")

    experiment_id = experiment.experiment_id
    run = client.create_run(experiment_id)
    logger.info(idx + " created run")

    logger.info(idx + " uploading artifact")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "temp.txt")
        with open(tmp_path, "w") as f:
            # Create a large text file
            f.write("a" * file_size_mib * 1024 ** 2)  # 100 MiB

        client.log_artifact(run.info.run_id, tmp_path, "test")
    logger.info(idx + " uploaded artifact")
    client.set_terminated(run.info.run_id, status="FINISHED")
    logger.info(idx + " downloading artifact")
    client.download_artifacts(run.info.run_id, path="test")
    logger.info(idx + " downloaded artifact")
    logger.info(idx + " end")
    return run.info.run_id


with ThreadPoolExecutor() as pool:
    futures = {idx: pool.submit(task, idx) for idx in range(num_concurrent_requests)}
    failed = []
    for idx, f in futures.items():
        try:
            f.result()
        except:
            failed.append(idx)
            print(traceback.format_exc())
    print(failed)
