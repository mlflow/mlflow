## Example endpoint configuration for Cohere

To see an example of specifying both the completions and the embeddings endpoints for Cohere, see [the configuration](config.yaml) YAML file.

This configuration file specifies two endpoints: 'completions' and 'embeddings', both using Cohere's models 'command' and 'embed-english-light-v2.0', respectively.

## Setting a Cohere API Key

This example requires a [Cohere API key](https://docs.cohere.com/docs/going-live):

```sh
export COHERE_API_KEY=...
```
