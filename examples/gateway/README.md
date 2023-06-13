# MLflow Gateway

## Installation

From PyPI:

```shell
pip install 'mlflow[gateway]'
```

From the repository:

```shell
pip install -e '.[gateway]'
```

## Running the service

```shell
mlflow gateway start --config-path examples/gateway/config.yaml
```

## Interactive API docs

Navigate to http://127.0.0.1:5000/docs.

## Make requests

```sh
python examples/gateway/request.py
```
