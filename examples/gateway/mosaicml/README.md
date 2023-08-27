## Example route configuration for MosaicML

To see an example of specifying both the completions and the embeddings routes for MosaicML, see [the configuration](config.yaml) YAML file.

This configuration file specifies two routes: 'completions' and 'embeddings', both using MosaicML's models 'mpt-7b-instruct' and 'instructor-xl', respectively.

## Setting a MosaicML API Key

This example requires a [MosaicML API key](https://docs.mosaicml.com/en/latest/getting_started.html):

```sh
export MOSAICML_API_KEY=...
```
