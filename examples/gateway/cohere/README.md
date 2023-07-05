## Example route configuration for Cohere

Here's an example of specifying both the completions and the embeddings routes for Cohere:

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
