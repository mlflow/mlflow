## Example route configuration for Clarifai

To see an example of specifying both the completions and the embeddings routes for Clarifai, see [the configuration](config.yaml) YAML file.

This configuration file specifies two routes: 'completions' and 'embeddings', both using models 'mistral-7B-Instruct' and 'multimodal-clip-embed' hosted in Clarifai Platform, respectively.

## Setting a Clarifai PAT Key

This example requires a [Clarifai PAT key](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/):

```sh
export CLARIFAI_PAT_KEY=...
```

### About
Explore more Clarifai hosted LLMs [here](https://clarifai.com/explore/models).