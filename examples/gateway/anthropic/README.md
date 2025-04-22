## Example endpoint configuration for Anthropic

To set up your MLflow configuration file, include a single endpoint for the completions endpoint as shown in the [anthropic configuration](config.yaml) YAML file.

## Obtaining and Setting the Anthropic API Key

To obtain an Anthropic API key, you need to create an account and subscribe to the service at [Anthropic](https://docs.anthropic.com/claude/docs/getting-access-to-claude).

After obtaining the key, you can export it to your environment variables. Make sure to replace the '...' with your actual API key:

```sh
export ANTHROPIC_API_KEY=...
```
