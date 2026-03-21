## Example endpoint configuration for MosaicML

To see an example of specifying both the completions and the embeddings endpoints for MosaicML, see [the configuration](config.yaml) YAML file.

This configuration file specifies three endpoints: 'completions', 'embeddings', and 'chat', using MosaicML's models 'mpt-7b-instruct', 'instructor-xl', and 'llama2-70b-chat', respectively.

## Setting a MosaicML API Key

This example requires a [MosaicML API key](https://docs.mosaicml.com/en/latest/getting_started.html):

```sh
export MOSAICML_API_KEY=...
```
