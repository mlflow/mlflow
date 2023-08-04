from collections import namedtuple
import os
import requests
import signal
import sys
import time
from tests.helper_functions import _start_scoring_proc, _get_mlflow_home
from sentence_transformers import SentenceTransformer
import transformers

import mlflow

ServerInfo = namedtuple("ServerInfo", ["pid", "url"])


def log_sentence_transformers_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    artifact_path = "gen_model"

    with mlflow.start_run():
        mlflow.sentence_transformers.log_model(
            model,
            artifact_path=artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)
    return model_uri


def log_completions_transformers_model():
    # NB: This is one of the smallest / fastest text2text generation models in transformers.
    # It is being used here simply due to its size and to verify serving capabilities with
    # a fast and responsive model architecture.
    architecture = "lordtt13/emo-mobilebert"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(architecture)
    pipe = transformers.pipeline(task="text-classification", model=model, tokenizer=tokenizer)
    inference_params = {"top_k": 1}

    signature_with_params = mlflow.models.infer_signature(
        ["test1", "test2"],
        mlflow.transformers.generate_signature_output(pipe, ["test3"]),
        inference_params,
    )

    artifact_path = "emo-bert"

    with mlflow.start_run():
        mlflow.transformers.log_model(
            pipe,
            inference_params=inference_params,
            signature=signature_with_params,
            artifact_path=artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)
    return model_uri


def start_mlflow_server(port, model_uri):
    server_url = f"http://127.0.0.1:{port}"

    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    env.update(MLFLOW_TRACKING_URI=mlflow.get_tracking_uri())
    env.update(MLFLOW_HOME=_get_mlflow_home())
    scoring_cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "-p",
        str(port),
        "--install-mlflow",
        "--no-conda",
    ]

    server_pid = _start_scoring_proc(cmd=scoring_cmd, env=env, stdout=sys.stdout, stderr=sys.stdout)

    ping_status = None
    for i in range(120):
        time.sleep(1)
        try:
            ping_status = requests.get(url=f"{server_url}/ping")
            if ping_status.status_code == 200:
                break
        except Exception:
            pass
    if ping_status is None or ping_status.status_code != 200:
        raise Exception("Could not start mlflow serving instance.")

    return ServerInfo(pid=server_pid, url=server_url)


def stop_mlflow_server(server_pid):
    process_group = os.getpgid(server_pid.pid)
    os.killpg(process_group, signal.SIGTERM)
