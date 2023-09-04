This document introduces the MLflow Gateway. 

# Introduciton
The MLflow Gateway is a tool designed to streamline the usage and management of various large language model (LLM) providers, such as OpenAI and Anthropic. 

* Benefits (bullet)
    - High-level interface: complexity of model routing and serving is abstracted from the developer. All you have to do is build the best model possible!
    - Unified endpoint: all routes leverage the same user-specified endpoint, so switching between routes simply invovles changing the route `name`.
    - Centralized management of API keys: keys are stored in a single centralized location, thereby minimizing your security risks. 
        - TODO: how to specify custom key storage? 
    - Realtime configuration changes: changes made to the route configuration are applied to a running gateway in real time.
    - TODO: performance stuff
    - Hosted documentation: FastAPI documentation is automatically created at `http://my.gateway:9000/docs`, which helps you quickly understand how to query your routes. 

# Quickstart
Follow the below steps to create an OpenAI competions model up and running within an MLflow Gateway instance.
TODO: maybe make this a free model?
### Install
`pip install 'mlflow[gateway]'` - CLI

### Create a Route
Routes are created in `.yaml` configurations files. Below is an example route for an OpenAI 
`gpt-3.5-turbo` completions model. 

```
routes:
  - name: completions
    route_type: llm/v1/completions
    model:
      provider: openai
      name: gpt-3.5-turbo
      config:
        openai_api_key: $OPENAI_API_KEY
```

### Authenticate your Route's Model
Here we are using an environment variable called `OPENAI_API_KEY`. To make the token accessible to MLflow Gateway, 
create an environment variable storing that value.

Mac/Linux
`export OPENAI_API_KEY={TOKEN_VALUE}`

Windows
`setx OPENAI_API_KEY "{TOKEN_VALUE}"`

### Start the Gateway
To start the gateway at `http://localhost:5000`, run the following command.

`mlflow gateway start --config-path /path/to/config.yaml`

### Query the Gateway
With the gateway up and running, you can now interact with the served model via the fluent API, 
client API, or routes REST API. Below we show an exmaple of querying the model via the fluent API.

```
from mlflow.gateway import query, set_gateway_uri

set_gateway_uri(gateway_uri="http://localhost:5000")

response = query(
    "chat",
    {"messages": [{"role": "user", "content": "What is the purpose of life?"}]},
)

print(response)
```

# Route Configuration Deep Dive
### Routes
Routes are proxy endpoints for the user. They forward requests to an underlying provider model, such as OpenAI's `gpt-3.5-turbo`.

TODO: Routes can be defined via `.yaml` files, the mlflow Client, etc. 

Routes have 3 parameters:
1. name (required string): the unique identifier for the route. 
2. type (required string): the type of language model. Currently, MLflow Gatway supports `llm/v1/completions` for text completion operations, `llm/v1/embeddings` for text embeddings, and `llm/v1/chat` for chat operations.
3. model (required map): the model to which a route will forward requests. The model contains the following details:
    3.1. provider (required string): the name of the provider for this model. Currently, Mlflow Gateway supports: `anthropic`, `azure` / `azuread`, `cohere`, `mlflow-model-serving`, `openai`,
    3.2. name (optional string): the name of the model to use. For example, `gpt-3.5-turbo` for OpenAI’s GPT-3.5-Turbo model. 
    3.3. config (optional map): additional model-specific configuration details.

TODO For complete code examples of the above model configurations, please visit HERE.
* TODO: build code examples for each unqiue provider config. Components include the follow for each of the below 1. working example, 2. Every model specific config (python dicts), 3. Links to the documentation - https://mlflow.org/docs/latest/gateway/index.html#completions 
* Types
    1. Completions
    2. Embeddings
    3. Chat
* Methods
    1. `.yaml`
    2. MLflow Client
    3. etc.


TODO Routes Deep Dive
* How are requests routed?
* How is the request interface standardized?
* How is authentication handled?
* How does it scale? What bottlenecks will be found?

# Serving Deep Dive
TODO
* What can you serve on? (local, clouds, etc.)

### Serving a Custom Model
TODO code example


# Querying Deep Dive
Querying refers to sending text to a MLflow Gateway route and receiving a response. Query parameters depend on the model type. You can query a route via the following:
1. `Fluent API`: copy this: https://mlflow.org/docs/latest/gateway/index.html#fluent-api
2. `Client API`: copy this: https://mlflow.org/docs/latest/gateway/index.html#client-api

# Additional Examples
TODO 
* Langchain
* Security?

