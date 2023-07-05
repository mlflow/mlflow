## Example route configuration for Anthropic

To set up your MLflow configuration file, include a single route for the completions endpoint as follows:

```yaml
routes:
  - name: completions-claude
    route_type: llm/v1/completions
    model:
      provider: anthropic
      name: claude-1.3-100k
      config:
        anthropic_api_base: https://api.anthropic.com/v1
        anthropic_api_key: $ANTHROPIC_API_KEY
```

Please replace `$ANTHROPIC_API_KEY` with your actual Anthropic API Key, which you will generate in the next step.

## Obtaining and Setting the Anthropic API Key

To obtain an Anthropic API key, you need to create an account and subscribe to the service at [Anthropic](https://docs.anthropic.com/claude/docs/getting-access-to-claude).

After obtaining the key, you can export it to your environment variables. Make sure to replace the '...' with your actual API key:

```sh
export ANTHROPIC_API_KEY=...
```
