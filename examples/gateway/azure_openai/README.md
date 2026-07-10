## Example endpoint configuration for Azure OpenAI

The following example configuration shows the 3 supported endpoints for Azure OpenAI: chat, completions, and embeddings.
Additionally, it illustrates the separate access paradigms that are supported for this service.

- `azure` api type: uses a generated token that is applied by setting the API token key directly to an environment variable
- `azuread` api type with `openai_api_key`: uses a pre-fetched Microsoft Entra ID (formerly Azure Active Directory) access token supplied via an environment variable
- `azuread` api type without `openai_api_key`: acquires Microsoft Entra ID tokens at request time via [azure-identity](https://pypi.org/project/azure-identity/) (`pip install azure-identity`). With no service principal fields, `DefaultAzureCredential` is used (managed identity, Azure CLI login, or the `AZURE_CLIENT_ID`/`AZURE_TENANT_ID`/`AZURE_CLIENT_SECRET` environment variables); with `openai_ad_client_id`, `openai_ad_tenant_id`, and `openai_ad_client_secret` set, that service principal is used via `ClientSecretCredential`

Depending on how your users will be interacting with the MLflow AI Gateway, a single access paradigm is recommended, not a mix of several.

See the [Azure OpenAI configuration](config.yaml) YAML file for example configurations showing all supported endpoint types and the different token access types.

## Setting the Azure OpenAI API Key

In order to get access to the Azure OpenAI service, [see the documentation](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service) guidance in the cognitive services portal.
With the key, export it to your environment variables.

Replace the '...' with your actual API key:

```sh
export OPENAI_API_KEY=...
```

## Validating the Azure OpenAI endpoint

See the [OpenAI Example](../openai/example.py) for testing the Azure OpenAI endpoints. The usage is identical to the standard OpenAI integration from an API perspective.
