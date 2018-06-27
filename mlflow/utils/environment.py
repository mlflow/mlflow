from mlflow.utils import PYTHON_VERSION

_conda_header = """name: mlflow-env
channels:
  - anaconda
  - defaults
dependencies:"""


def _mlflow_conda_env(path, additional_conda_deps=None, additional_pip_deps=None):
    conda_deps = ["python={}".format(PYTHON_VERSION)]
    if additional_conda_deps:
        conda_deps += additional_conda_deps
    pip_deps = additional_pip_deps
    with open(path, "w") as f:
        f.write(_conda_header)
        prefix = "\n  - "
        f.write(prefix + prefix.join(conda_deps))
        if pip_deps:
            f.write(prefix + "pip:")
            prefix = "\n    - "
            f.write(prefix + prefix.join(pip_deps))
        f.write("\n")
    with open(path, "r") as f:
        print(f.read())
