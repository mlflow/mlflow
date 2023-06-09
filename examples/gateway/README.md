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

## Making requests

```shell
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/gateway/routes/
```
