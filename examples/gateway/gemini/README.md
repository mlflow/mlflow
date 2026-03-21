## Example endpoint configuration for GEMINI

To see an example of specifying both the completions and embeddings endpoints for Gemini, see [the configuration](config.yaml) YAML file.

This configuration file specifies three endpoints: 'completions', 'embeddings', and 'chat', using Gemini's model gemini-2.0-flash for completions and chat and gemini-embedding-exp-03-07 for embeddings.

## Setting a GEMINI API Key

This example requires a [GEMINI API key](https://ai.google.dev/gemini-api/docs/api-key):

```sh
export GEMINI_API_KEY=...
```
