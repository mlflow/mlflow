# MLflow Skinny

`mlflow-skinny` a lightweight version of MLflow that is designed to be used in environments where you want to minimize the size of the package.

## Core Files

| File               | Description                                                                     |
| ------------------ | ------------------------------------------------------------------------------- |
| `mlflow`           | A symlink that points to the `mlflow` directory in the root of the repository.  |
| `pyproject.toml`   | The package metadata. Autogenerate by [`dev/pyproject.py`](../dev/pyproject.py) |
| `README_SKINNY.md` | The package description. Autogenerate by [`dev/skinny.py`](../dev/pyproject.py) |

## Installation

```sh
# If you have a local clone of the repository
pip install ./libs/skinny

# If you want to install the latest version from GitHub
pip install git+https://github.com/mlflow/mlflow.git#subdirectory=libs/skinny
```
