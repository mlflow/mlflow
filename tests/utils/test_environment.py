import os

from mlflow.utils.environment import _mlflow_conda_env

import mlflow.utils.environment as env
import mlflow.projects


def test_list_environments(tmpdir):

    envs = env._get_mlflow_environments()
    print('envs', envs)
    filename = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(filename, additional_conda_deps=[],
                      additional_pip_deps=[])
    mlflow.projects._get_or_create_conda_env(filename)
    name = mlflow.projects._get_conda_env_name(filename)
    assert set(env._get_mlflow_environments()) == set(envs + [name])
    env._clear_conda_env_cache(remove_envs=[filename], force=True)
    assert env._get_mlflow_environments() == envs


def test_save(tmpdir):
    filename = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(filename, additional_conda_deps=["conda-dep-1=0.0.1", "conda-dep-2"],
                      additional_pip_deps=["pip-dep-1", "pip-dep2==0.1.0"])
    print("")
    print("env start")
    with open(filename) as f:
        print(f.read())
    print("env end")
