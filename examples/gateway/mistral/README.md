## Example endpoint configuration for Mistral

To see an example of specifying both the completions and the embeddings endpoints for Mistral, see [the configuration](config.yaml) YAML file.

This configuration file specifies two endpoints: 'completions' and 'embeddings', both using Mistral's models 'mistral-tiny' and 'mistral-embed', respectively.

## Setting a Mistral API Key

This example requires a [Mistral API key](https://docs.mistral.ai/):

```sh
export MISTRAL_API_KEY=...
```
