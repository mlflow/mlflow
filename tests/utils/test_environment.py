import os

from mlflow.utils.environment import _mlflow_conda_env


def test_save(tmpdir):
    filename = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(filename, additional_conda_deps=["conda-dep-1=0.0.1", "conda-dep-2"],
                      additional_pip_deps=["pip-dep-1", "pip-dep2==0.1.0"])
    print("")
    print("env start")
    with open(filename) as f:
        print(f.read())
    print("env end")
