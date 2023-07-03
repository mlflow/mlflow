
# MLflow AI Gateway Guide to using Cohere 

## Installing the MLflow AI Gateway

First, we need to install the MLflow AI Gateway. The recommended way is to install it from PyPI:

```sh
pip install 'mlflow[gateway]'
```

If you're interested in development work or wish to use the latest build of MLflow AI Gateway, you can install from a fork of the repository:

```sh
pip install -e '.[gateway]'
```

## Setting up Cohere in the Configuration File

Now, we need to configure the Cohere endpoint and the model we want to use in the configuration file. Here's an example:

```yaml
routes:
  - name: completions
    route_type: llm/v1/completions
    model:
      provider: cohere
      name: command
      config:
        api_key: $COHERE_API_KEY

  - name: embeddings
    route_type: llm/v1/embeddings
    model:
      provider: cohere
      name: embed-english-light-v2.0
      config:
        api_key: $COHERE_API_KEY
```

This configuration file specifies two routes: 'completions' and 'embeddings', both using Cohere's models 'command' and 'embed-english-light-v2.0', respectively.

## Setting a Cohere API Key

This example requires a [Cohere API key](https://docs.cohere.com/docs/going-live):

```sh
export COHERE_API_KEY=...
```

## Running the Gateway Service

With everything set up, you can now start the MLflow Gateway Service. To do this, you'll use the following command, replacing 'examples/gateway/cohere.yaml' with the path to your configuration file:

```sh
mlflow gateway start --config-path examples/gateway/cohere.yaml
```

## Interactive API documentation

Once the Gateway Service is running, you can view and interact with the API documentation by navigating to http://127.0.0.1:5000/docs in your web browser.

## Sending Requests

Now, you're ready to send requests to your Cohere endpoint. Here's an example of how you can use Python to send a request:

```sh
python examples/gateway/cohere/cohere_example.py
```
