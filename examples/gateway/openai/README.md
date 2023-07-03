# Getting Started with MLflow AI Gateway for OpenAI

This guide will walk you through the installation and basic setup of MLflow AI Gateway for OpenAI, focusing on establishing a single route. Let's get started.

## Step 1: Installing the MLflow AI Gateway

MLflow AI Gateway is best installed from PyPI. Open your terminal and use the following pip command:

```sh
# Installation from PyPI
pip install 'mlflow[gateway]'
```

For those interested in development or in using the most recent build of MLflow AI Gateway, you may choose to install from the fork of the repository:

```sh
# Installation from the repository
pip install -e '.[gateway]'
```

## Step 2: Configuring OpenAI Endpoint in the MLflow Configuration File

Create or modify your MLflow configuration file. It should include a single route, for instance, to the `completions` endpoint. The structure of the configuration file should look like this:

```yaml
routes:
  - name: completions
    route_type: llm/v1/completions
    model:
      provider: openai
      name: gpt-4
      config:
        openai_api_base: https://api.openai.com/v1
        openai_api_key: $OPENAI_API_KEY
```

Replace `$OPENAI_API_KEY` with your actual OpenAI API Key, which you will set in the next step.

## Step 3: Setting the OpenAI API Key

An OpenAI API key is required for the configuration. If you haven't already, obtain an [OpenAI API key](https://platform.openai.com/account/api-keys).

With the key, export it to your environment variables. Replace the '...' with your actual API key:

```sh
export OPENAI_API_KEY=...
```

## Step 4: Starting the MLflow Gateway Service

With the MLflow configuration file in place and the OpenAI API key set, you can now start the MLflow Gateway service. Replace `openai.yaml` with the actual path to your MLflow configuration file:

```sh
mlflow gateway start --config-path examples/gateway/openai.yaml
```

## Step 5: Accessing the Interactive API Documentation

With the MLflow Gateway Service up and running, access its interactive API documentation by navigating to the following URL:

http://127.0.0.1:5000/docs

## Step 6: Sending Test Requests

After successfully setting up the MLflow Gateway Service, you can send a test request using the provided Python script. Replace `request.py` with the actual path to your test script:

```sh
python examples/gateway/openai/openai_example.py
```
