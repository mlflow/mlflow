## Example endpoint configuration for plugin provider

To see an example of specifying the chat endpoint for a plugin provider,
see [the configuration](config.yaml) YAML file.

We implement our plugin provider package `my_llm` under `./my-llm` folder. It implements the chat method.

This configuration file specifies one endpoint: 'chat', using the model 'my-model-0.1.2'.

## Setting up the server

First, install the provider package `my_llm`:

```sh
pip install -e ./my-llm
```

Then, start the server:

```sh
MY_LLM_API_KEY=some-api-key mlflow gateway start --config-path config.yaml --port 7000
```

To clean up the installed package after the example, run

```sh
pip uninstall my_llm
```
