## Example endpoint configuration for TogetherAI

To see an example of specifying both the completions and the embeddings endpoints for TogetherAI, see [the configuration](config.yaml) YAML file.

This configuration file specifies two endpoints: 'completions' and 'embeddings', both using TogetherAI's provided models 'mistralai/Mixtral-8x7B-v0.1' and 'togethercomputer/m2-bert-80M-8k-retrieval', respectively.

## Setting a Mistral API Key

This example requires a [TogetherAI API key](https://docs.together.ai/docs/):

```sh
export TOGETHERAI_API_KEY=...
```
