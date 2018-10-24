CONDA_ENV_NAME = "py2_conda_env"

SCORE_SRC = """\
import os
import multiprocessing
from subprocess import check_output, Popen 

from azureml.core.model import Model


def get_conda_bin_path():
    conda_symlink_cmd = "which conda"
    symlink_path = check_output(conda_symlink_cmd.split(" ")).decode("utf-8").rstrip()
    conda_full_path_cmd = "realpath {{symlink_path}}".format(symlink_path=symlink_path)
    full_path = check_output(conda_full_path_cmd.split(" ")).decode("utf-8").rstrip()
    return os.path.dirname(full_path)


def init():
    global py2_server_url
    py2_server_url = "127.0.0.1:8080"

    model_path = Model.get_model_path(model_name="{model_name}", version={model_version})
    conda_bin_path = get_conda_bin_path()
    conda_activate_path = os.path.join(conda_bin_path, "activate")

    num_gunicorn_workers = multiprocessing.cpu_count()
    gunicorn_target = (
        'mlflow.sagemaker.container.scoring_server.wsgi:app(model_path=\"{{model_path}}\")'.format(
            model_path=model_path))

    bash_cmds = [
        "source {{conda_activate_path}} {conda_env_name}".format(
            conda_activate_path=conda_activate_path),
        ("gunicorn --timeout 60 -k gevent -b {{gunicorn_bind_address}}"
         " -w {{num_gunicorn_workers}} '{{gunicorn_target}}'".format(
             gunicorn_bind_address=py2_server_url,
             num_gunicorn_workers=num_gunicorn_workers,
             gunicorn_target=gunicorn_target))
    ]
    cmd = ["/bin/bash", "-c", " && ".join(bash_cmds)]
    proc = Popen(cmd)


def run(input_json):
    import requests

    response = requests.post(
        url=py2_server_url,
        headers={{"Content-type": "application/json"}},
        data=input_json)
    return response.text
"""
