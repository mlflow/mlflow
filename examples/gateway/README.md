# MLflow AI Gateway

The examples provided within this directory show how to get started with individual providers and at least
one of the supported endpoint types. When configuring an instance of the MLflow AI Gateway, multiple providers,
instances of endpoint types, and model versions can be specified for each query endpoint on the server.

## Example configuration files

Within this directory are example config files for each of the supported providers. If using these as a guide
for configuring a large number of endpoints, ensure that the placeholder names (i.e., "completions", "chat", "embeddings")
are modified to prevent collisions. These names are provided for clarity only for the examples and real-world
use cases should define a relevant and meaningful endpoint name to eliminate ambiguity and minimize the chances of name collisions.

# Getting Started with MLflow AI Gateway for OpenAI

This guide will walk you through the installation and basic setup of the MLflow AI Gateway.
Within sub directories of this examples section, you can find specific executable examples
that can be used to validate a given provider's configuration through the MLflow AI Gateway.
Let's get started.

## Step 1: Installing the MLflow AI Gateway

The MLflow AI Gateway is best installed from PyPI. Open your terminal and use the following pip command:

```sh
# Installation from PyPI
pip install 'mlflow[genai]'
```

For those interested in development or in using the most recent build of the MLflow AI Gateway, you may choose to install from the fork of the repository:

```sh
# Installation from the repository
pip install -e '.[genai]'
```

## Step 2: Configuring Endpoints

Each provider has a distinct set of allowable endpoint types (i.e., chat, completions, etc) and
specific requirements for the initialization of the endpoints to interface with their services.
For full examples of configurations and supported endpoint types, see:

- [OpenAI](openai/config.yaml)
- [MosaicML](mosaicml/config.yaml)
- [Anthropic](anthropic/config.yaml)
- [Cohere](cohere/config.yaml)
- [AI21 Labs](ai21labs/config.yaml)
- [PaLM](palm/config.yaml)
- [AzureOpenAI](azure_openai/config.yaml)
- [Mistral](mistral/config.yaml)
- [TogetherAI](togetherai/config.yaml)

## Step 3: Setting Access Keys

See information on specific methods of obtaining and setting the access keys within the provider-specific documentation within this directory.

## Step 4: Starting the MLflow AI Gateway

With the MLflow configuration file in place and access key(s) set, you can now start the MLflow AI Gateway.
Replace `<provider>` with the actual path to the MLflow configuration file for the provider of your choice:

```sh
mlflow gateway start --config-path examples/gateway/<provider>/config.yaml --port 7000

# For example:
mlflow gateway start --config-path examples/gateway/openai/config.yaml --port 7000
```

## Step 5: Accessing the Interactive API Documentation

With the MLflow AI Gateway up and running, access its interactive API documentation by navigating to the following URL:

http://127.0.0.1:7000/docs

## Step 6: Sending Test Requests

After successfully setting up the MLflow AI Gateway, you can send a test request using the provided Python script.
Replace <provider> with the name of the provider example test script that you'd like to use:

```sh
python examples/gateway/<provider>/example.py
```
