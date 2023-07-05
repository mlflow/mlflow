## Example route configuration for OpenAI

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

## Setting the OpenAI API Key

An OpenAI API key is required for the configuration. If you haven't already, obtain an [OpenAI API key](https://platform.openai.com/account/api-keys).

With the key, export it to your environment variables. Replace the '...' with your actual API key:

```sh
export OPENAI_API_KEY=...
```
