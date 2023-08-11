# Guide to using an MLflow served model with the MLflow AI Gateway
In order to utilize the MLflow AI Gateway with the MLflow Model serving service, a few steps must be taken 
in addition to those for configuring access to SaaS models (such as Anthropic and OpenAI). The first and most obvious 
step that must be taken prior to interfacing with an Mlflow served model is that a model needs to be logged to the 
MLflow tracking server. 

An important consideration for deciding whether to interface the MLflow AI Gateway with a specific model is to evaluate the PyFunc interface that the model will 
return after being called for inference. Due to the fact that the AI Gateway defines a specific response signature, expectations for each route type's payload contents 
must be met in order for a route to be valid. 

For example, an embeddings route (llm/v1/embeddings route type) is designed to return embeddings data as a collection (a list) of floats that correspond to each of the 
input strings that are sent for embeddings inference to a service. The expectation that the embeddings route definition has is that the data is in a particular format. Specifically one that 
is capable of having the embeddings data extractable from a service response. Therefore, an MLflow model that returns data in the format below is perfectly valid. 

```json
{"predictions": [
  [0.0, 0.1], 
  [1.0, 0.0]
]
}
```
However, a return value from a serving endpoint via a custom PyFunc of the form below will not work. 
```json
{"predictions": [
  {"embedding": [0.0, 0.1]}, 
  {"embedding": [1.0, 0.0]}
]
}
```

It is important to note that the MLflow AI Gateway does not perform validation on a configured route until the point of querying. Creating a route that interfaces with the 
MLflow model server that is returning a payload that is incompatible with the configured route type definition will raise 502 exceptions only when queried. 

> **NOTE:** It is important to validate the output response of a model served by MLflow to ensure compatibility with the MLflow AI Gateway route definitions. Not all model outputs are compatible with given route types.

## Creating and logging an embeddings model
To start, we need a model that is capable of generating embeddings. For this example, we'll use 
the `sentence_transformers` library and the corresponding MLflow flavor. 

```python
from sentence_transformers import SentenceTransformer
import mlflow


model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
artifact_path = "embeddings_model"

with mlflow.start_run():
    mlflow.sentence_transformers.log_model(
        model,
        artifact_path=artifact_path,
    )
    model_uri = mlflow.get_artifact_uri(artifact_path)

# Generate the cli command for starting a local MLflow Model Serving endpoint for this embeddings model
print(f"mlflow models serve -m {model_uri} -h 127.0.0.1 -p 9020 --no-conda")
```

Copy the output from the print statement to the clipboard.

## Starting the model server for the embeddings model
With the printed string from running the above command copied to the clipboard, open a new terminal 
and paste the string. Leave the terminal window open and running. 

```commandline
mlflow models serve -m file:///Users/me/demos/mlruns/0/2bfcdcb66eaf4c88abe8e0c7bcab639e/artifacts/embeddings_model -h 127.0.0.1 -p 9020 --no-conda
```

## Update the config.yaml to add a new embeddings route
After assigning a valid port and ensuring that the model server starts correctly:

```commandline
2023/08/08 17:36:44 INFO mlflow.models.flavor_backend_registry: Selected backend for flavor 'python_function'
2023/08/08 17:36:44 INFO mlflow.pyfunc.backend: === Running command 'exec gunicorn --timeout=60 -b 127.0.0.1:9020 -w 1 ${GUNICORN_CMD_ARGS} -- mlflow.pyfunc.scoring_server.wsgi:app'
[2023-08-08 17:36:45 -0400] [54917] [INFO] Starting gunicorn 20.1.0
[2023-08-08 17:36:45 -0400] [54917] [INFO] Listening at: http://127.0.0.1:9020 (54917)
[2023-08-08 17:36:45 -0400] [54917] [INFO] Using worker: sync
[2023-08-08 17:36:45 -0400] [54919] [INFO] Booting worker with pid: 54919
```

The flask app is ready to receive traffic. 

Update the MLflow AI Gateway configuration file (config.yaml) with the new route:

```yaml
routes:
  - name: embeddings
    route_type: llm/v1/embeddings
    model:
      provider: mlflow-model-serving
      name: sentence-transformer
      config:
        model_server_url: http://127.0.0.1:9020
```

The key component here is the ``model_server_url``. For serving an MLflow LLM, this url must match to the service that you are specifying for the 
Model Serving server. 

> **NOTE:** The MLflow Model Server does not have to be running in order to update the configuration file or to start the MLflow AI Gateway. In order to respond to submitted queries, it is required to be running.


## Creating and logging a fill mask model
To support an additional route for generating a mask fill response from masked input text, we need to log an appropriate model. 
For this tutorial example, we'll use a `transformers` `Pipeline` wrapping a `BertForMaskedLM` torch model and will log this pipeline using the MLflow `transformers` flavor.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import mlflow


lm_architecture = "bert-base-cased"
artifact_path = "mask_fill_model"

tokenizer = AutoTokenizer.from_pretrained(lm_architecture)
model = AutoModelForMaskedLM.from_pretrained(lm_architecture)

components = {
    "model": model,
    "tokenizer": tokenizer
}

with mlflow.start_run():
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path=artifact_path,
    )
    model_uri = mlflow.get_artifact_uri(artifact_path)

# Generate the cli command for starting a local MLflow Model Serving endpoint for this fill mask model
print(f"mlflow models serve -m {model_uri} -h 127.0.0.1 -p 9010 --no-conda")
```

## Starting the model server for the fill mask model
Using the command printed to stdout from above, open a new terminal (do not close the terminal that is currently running the embeddings model being served!) 
and paste the command.

```commandline
mlflow models serve -m file:///Users/me/demos/mlruns/0/bc8bdb7fb90c406eb95603a97742cef8/artifacts/mask_fill_model -h 127.0.0.1 -p 9010 --no-conda
```

## Update the config.yaml to add a new completions route
Ensure that the Mlflow serving endpoint starts and is ready for traffic.

```commandline
2023/08/08 17:39:14 INFO mlflow.models.flavor_backend_registry: Selected backend for flavor 'python_function'
2023/08/08 17:39:14 INFO mlflow.pyfunc.backend: === Running command 'exec gunicorn --timeout=60 -b 127.0.0.1:9010 -w 1 ${GUNICORN_CMD_ARGS} -- mlflow.pyfunc.scoring_server.wsgi:app'
[2023-08-08 17:39:15 -0400] [55722] [INFO] Starting gunicorn 20.1.0
[2023-08-08 17:39:15 -0400] [55722] [INFO] Listening at: http://127.0.0.1:9010 (55722)
[2023-08-08 17:39:15 -0400] [55722] [INFO] Using worker: sync
[2023-08-08 17:39:15 -0400] [55722] [INFO] Booting worker with pid: 55724
```

Add the entry to the MLflow AI Gateway configuration file (the full file is shown below):

```yaml
routes:
  - name: embeddings
    route_type: llm/v1/embeddings
    model:
      provider: mlflow-model-serving
      name: sentence-transformer
      config:
        model_server_url: http://127.0.0.1:9020
  - name: fillmask
    route_type: llm/v1/completions
    model:
      provider: mlflow-model-serving
      name: fill-mask
      config:
        model_server_url: http://127.0.0.1:9010
```

## Start the Mlflow AI Gateway server
Now that both routes are defined within the configuration YAML file and both servers are ready to receive queries, we can start the MLflow AI Gateway server.

```sh
mlflow gateway start --config-path examples/gateway/mlflow_serving/config.yaml
```

## Query the AI Gateway
See the included example.py script within this directory to see how to query these two models that are being served.