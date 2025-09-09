---
mode: "agent"
description: "Delegate a task to Copilot"
model: GPT-4.1
tools: [create_pull_request_with_copilot]
---

Your task is to delegate the given request to Copilot using the `create_pull_request_with_copilot` tool. The pull request must be created in the [`mlflow/mlflow`](https://github.com/mlflow/mlflow) repository. The `base_ref` parameter must be unspecified to use the default branch.
