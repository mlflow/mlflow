## Example endpoint configuration for plugin provider

To see an example of specifying the chat endpoint for a plugin provider,
see [the configuration](config.yaml) YAML file.

We implement our plugin provider package `foo` under `./foo-provider` folder. It implements the chat method.

This configuration file specifies one endpoint: 'chat', using the model 'foo-model-0.1.2'.

## Setting up the server

First, install the provider package `foo`:

```sh
pip install -e ./foo-provider
```

Then, start the server:

```sh
FOO_API_KEY=my-key mlflow deployments start-server --config-path config.yaml --port 7000
```

To clean up the installed package after the example, run

```sh
pip uninstall foo
```
