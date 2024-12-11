## Example endpoint configuration for PaLM

To see an example of specifying both the completions and the embeddings endpoints for PaLM, see [the configuration](config.yaml) YAML file.

This configuration file specifies three endpoints: 'completions', 'embeddings', and 'chat', using PaLM's models 'text-bison-001', 'embedding-gecko-001', and 'chat-bison-001', respectively.

## Setting a PaLM API Key

This example requires a [PaLM API key](https://developers.generativeai.google/tutorials/setup):

```sh
export PALM_API_KEY=...
```
