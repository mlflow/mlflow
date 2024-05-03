## Example endpoint configuration for plugin provider

To see an example of specifying the completions endpoint for a plugin provider, 
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
FOO_API_KEY=my-key mlflow deployments start-server --config-path config.yaml --port 7000
```
