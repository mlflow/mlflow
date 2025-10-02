"""Claude Code integration for MLflow.

This module provides automatic tracing of Claude Code conversations to MLflow.

Usage:
    mlflow autolog claude [directory] [options]

After setup, use the regular 'claude' command and traces will be automatically captured.

To enable tracing for the Claude Agent SDK, use `mlflow.anthropic.autolog()`.

Example:

```python
import mlflow.anthropic

mlflow.anthropic.autolog()

async with ClaudeSDKClient() as client:
    await client.query("What is the capital of France?")

    async for message in client.receive_response():
        print(message)
```
"""
