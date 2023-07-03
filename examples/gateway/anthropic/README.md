# Getting Started with MLflow AI Gateway for Anthropic

This guide is designed to assist you in setting up and using MLflow AI Gateway for Anthropic, focusing particularly on establishing a single route for completions. Let's get started.

## Step 1: Installing the MLflow AI Gateway

You can conveniently install the MLflow AI Gateway from PyPI. Open your terminal and input the following pip command:

```sh
# Installation from PyPI
pip install 'mlflow[gateway]'
```

For those interested in development or in using the most recent build of MLflow AI Gateway, you can install it from a fork of the repository:

```sh
# Installation from the repository
pip install -e '.[gateway]'
```

## Step 2: Configuring the Anthropic Endpoint in the MLflow Configuration File

To set up your MLflow configuration file, include a single route for the completions endpoint as follows:

```yaml
routes:
  - name: completions-claude
    route_type: llm/v1/completions
    model:
      provider: anthropic
      name: claude-1.3-100k
      config:
        anthropic_api_base: https://api.anthropic.com/v1
        anthropic_api_key: $ANTHROPIC_API_KEY
```

Please replace `$ANTHROPIC_API_KEY` with your actual Anthropic API Key, which you will generate in the next step.

## Step 3: Obtaining and Setting the Anthropic API Key

To obtain an Anthropic API key, you need to create an account and subscribe to the service at [Anthropic](https://docs.anthropic.com/claude/docs/getting-access-to-claude).

After obtaining the key, you can export it to your environment variables. Make sure to replace the '...' with your actual API key:

```sh
export ANTHROPIC_API_KEY=...
```

## Step 4: Starting the MLflow Gateway Service

With the MLflow configuration file properly set and the Anthropic API key in place, you are now ready to start the MLflow Gateway service. Replace `anthropic.yaml` with the actual path to your MLflow configuration file:

```sh
mlflow gateway start --config-path examples/gateway/anthropic.yaml
```

## Step 5: Accessing the Interactive API Documentation

Once your MLflow Gateway Service is up and running, you can access the interactive API documentation by navigating to the following URL:

http://127.0.0.1:5000/docs

## Step 6: Sending Test Requests

Upon successful setup of the MLflow Gateway Service, you can send a test request using the provided Python script.

```sh
python examples/gateway/anthropic/anthropic_example.py
```
