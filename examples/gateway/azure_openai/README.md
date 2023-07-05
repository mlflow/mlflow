## Example route configuration for AzureOpenAI

The following example configuration shows the 3 supported routes for AzureOpenAI: chat, completions, and embeddings.
Additionally, it illustrates the two separate api types that are supported for this service.

- _"azure"_ api type: uses a generated token that is applied by setting the API token key directly to an environment variable
- _"azuread"_ api type: uses Azure Active Directory for supplying the active directory key to be used to an environment variable

Depending on how your users will be interacting with the MLflow AI Gateway, a single access paradigm (either "azure" **or** "azuread" is recommended, not a mix of both as shown for example purposes below.)

```yaml
routes:
  - name: chat
    route_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-35-turbo
      config:
        openai_api_type: "azure"
        openai_api_key: $OPENAI_API_KEY
        openai_deployment_name: "{your_deployment_name}"
        openai_api_base: "https://{your_resource_name}-azureopenai.openai.azure.com/"
        openai_api_version: "2023-05-15"

  - name: completions
    route_type: llm/v1/completions
    model:
      provider: openai
      name: gpt-35-turbo
      config:
        openai_api_type: "azuread"
        openai_api_key: $AZURE_AAD_TOKEN
        openai_deployment_name: "{your_deployment_name}"
        openai_api_base: "https://{your_resource_name}-azureopenai.openai.azure.com/"
        openai_api_version: "2023-05-15"

  - name: embeddings
    route_type: llm/v1/embeddings
    model:
      provider: openai
      name: text-embedding-ada-002
      config:
        openai_api_type: "azure"
        openai_api_key: $OPENAI_API_KEY
        openai_deployment_name: "{your_deployment_name}"
        openai_api_base: "https://{your_resource_name}-azureopenai.openai.azure.com/"
        openai_api_version: "2023-05-15"
```

## Setting the AzureOpenAI API Key

In order to get access to the Azure OpenAI service, [see the documentation](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service) guidance in the cognitive services portal.
With the key, export it to your environment variables.

With the key, export it to your environment variables. Replace the '...' with your actual API key:

```sh
export OPENAI_API_KEY=...
```

## Validating the Azure OpenAI route

See the [OpenAI Example](../openai/openai_example.py) for testing the Azure OpenAI routes. The usage is identical to the standard OpenAI integration.
