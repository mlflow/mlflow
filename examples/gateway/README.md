# MLflow Gateway

## Installation

```sh
# From PyPI
pip install 'mlflow[gateway]'

# From the repository
pip install -e '.[gateway]'
```

## Setting an OpenAI API Key

This example requires an [OpenAI API key](https://platform.openai.com/account/api-keys):

```sh
export OPENAI_API_KEY=...
```

## Running the Gateway Service

```sh
mlflow gateway start --config-path examples/gateway/openai.yaml
```

## Interactive API documentation

Navigate to http://127.0.0.1:5000/docs.

## Sending Requests

```sh
python examples/gateway/request.py
```
