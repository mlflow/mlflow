## Example endpoint configuration for plugin provider

To see an example of specifying both the completions and the embeddings endpoints for a plugin provider, 
see [the configuration](config.yaml) YAML file.

We implement our plugin provider under `foo/` folder. It implements the completions method.

This configuration file specifies one endpoint: 'completions', using the model 'foo-model-0.1.2'.

## Setting up the server

First, install the provider:

```sh
pip install -e ./foo-provider
```

Then, start the server:

```sh
FOO_API_KEY=my-key mlflow deployments start-server --config-path config.yaml --workers 1
```

Then, we can query the server:

```sh
curl -X POST 127.0.0.1:5000/endpoints/completions/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'

# output
# {
#   "id":"id-123","object":"text_completion","created":1714717112,"model":"foo-model-0.1.2",
#   "choices":[{"index":0,"text":"This is a response from FooProvider","finish_reason":null}],
#   "usage":{"prompt_tokens":12,"completion_tokens":34,"total_tokens":46}
# }
```
