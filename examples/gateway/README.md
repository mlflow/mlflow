# MLflow Model Gateway

## Installation

From PyPI:

```shell
pip install 'mlflow[gateway]'
```

From the source code:

```shell
pip install -e '.[gateway]'
```

## Starting the gateway service

```shell
mlflow gateway start --config-path gateway.yml
```

## Updating the gateway service

```shell
mlflow gateway update --config-path gateway.yml
```
