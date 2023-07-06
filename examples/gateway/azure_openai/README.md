## Example route configuration for AzureOpenAI

The following example configuration shows the 3 supported routes for AzureOpenAI: chat, completions, and embeddings.
Additionally, it illustrates the two separate api types that are supported for this service.

- `azure` api type: uses a generated token that is applied by setting the API token key directly to an environment variable
- `azuread` api type: uses Azure Active Directory for supplying the active directory key to be used to an environment variable

Depending on how your users will be interacting with the MLflow AI Gateway, a single access paradigm (either `azure` **or** `azuread` is recommended, not a mix of both).

See the [AzureOpenAI configuration](config.yaml) YAML file for example configurations showing all supported route types and the different token access types.

## Setting the AzureOpenAI API Key

In order to get access to the Azure OpenAI service, [see the documentation](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service) guidance in the cognitive services portal.
With the key, export it to your environment variables.

Replace the '...' with your actual API key:

```sh
export OPENAI_API_KEY=...
```

## Validating the Azure OpenAI route

See the [OpenAI Example](../openai/openai_example.py) for testing the Azure OpenAI routes. The usage is identical to the standard OpenAI integration from an API perspective.
