# MLflow Trace: An Open-Source SDK for Observability and Monitoring GenAI Applications🔍

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)

![Trace Hero](https://mlflow.org/docs/latest/assets/images/tracing-top-dcca046565ab33be6afe0447dd328c22.gif)

MLflow Trace is an open-source, lightweight Python package that only includes the minimum set of dependencies and functionality
to instrument your code/models/agents with [MLflow Tracing](https://mlflow.org/docs/latest/tracing). This package is designed to be
used in environments where you want to minimize the size of your dependencies, such as in production or serverless environments.

## ✨ Features

- [Automatic Tracing](https://mlflow.org/docs/latest/tracing/integrations/) for AI libraries (OpenAI, LangChain, DSPy, Anthropic, etc...). Follow the link for the full list of supported libraries.
- [Manual instrumentation APIs](https://mlflow.org/docs/latest/tracing/api/manual-instrumentation) such as `@trace` decorator.
- [Production Monitoring](https://mlflow.org/docs/latest/tracing/production)
- Other tracing APIs such as `mlflow.set_trace_tag`, `mlflow.search_traces`, etc.
- Authentication

## 🌐 Choose Backend

The MLflow Trace package is designed to work with te remote hosted MLflow server as a backend. This allows you to log your traces to a central location, making it easier to manage and analyze your traces. There are several different options for hosting your MLflow server, including:

- [Databricks](https://docs.databricks.com/machine-learning/mlflow/managed-mlflow.html) - Databricks offers a FREE, fully managed MLflow server as a part of their platform. This is the easiest way to get started with MLflow tracing, without having to set up any infrastructure.
- [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/) - MLflow on Amazon SageMaker is a fully managed service offer by AWS, including tracing and other MLflow features such as model registry.
- [Nebius](https://nebius.com/) - Nebius, a cutting-edge cloud platform for GenAI explorers, offers a fully managed MLflow server.
- [Self-hosting](https://mlflow.org/docs/latest/tracking/#tracking_setup) - MLflow is a fully open-source project, allowing you to self-host your own MLflow server and keep your data private. This is a great option if you want to have full control over your data and infrastructure.

## 🚀 Getting Started

### Installation

To install the MLflow Python package, run the following command:

```bash
pip install mlflow-trace
```

### Connect to the MLflow Server

To connect to your MLflow server to log your traces, set the `MLFLOW_TRACKING_URI` environment variable or use the `mlflow.set_tracking_uri` function:

```python
import mlflow

mlflow.set_tracking_uri("databricks")
# Specify the experiment to log the traces to
mlflow.set_experiment("/Path/To/Experiment")
```

### Start Logging Traces

```python
import openai

client = openai.OpenAI(api_key="<your-api-key>")

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Call the OpenAI API as usual
response = client.chat.completions.create(
    model="gpt-o1-mini",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)
```

## 📘 Documentation

Official documentation for MLflow can be found at [here](https://mlflow.org/docs/latest/index.html).

## 🛑 Features _Not_ Included

The following MLflow features are not included in this package.

- MLflow tracking server and UI.
- MLflow's other tracking capabilities such as Runs, Model Registry, Projects, etc.
- Evaluate models/agents and log evaluation results.

To leverage the full feature set of MLflow, install the full package by running `pip install mlflow`.
