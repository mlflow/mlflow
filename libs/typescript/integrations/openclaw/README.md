<h1 align="center" style="border-bottom: none">
    <a href="https://mlflow.org/">
        <img alt="MLflow logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
    </a>
</h1>
<h2 align="center" style="border-bottom: none">🦞 OpenClaw MLflow Observability Plugin</h2>

<div align="center">

[![NPM](https://img.shields.io/npm/v/@mlflow/mlflow-openclaw)](https://www.npmjs.com/package/@mlflow/mlflow-openclaw)
[![License](https://img.shields.io/github/license/mlflow/mlflow)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
<a href="https://twitter.com/intent/follow?screen_name=mlflow" target="_blank">
<img src="https://img.shields.io/twitter/follow/mlflow?logo=X&color=%20%23f5f5f5"
      alt="follow on X(Twitter)"></a>
<a href="https://www.linkedin.com/company/mlflow-org/" target="_blank">
<img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff"
      alt="follow on LinkedIn"></a>

</div>

MLflow integration for [OpenClaw](https://github.com/openclaw/openclaw) for [observability](https://mlflow.org/docs/latest/genai/tracing/), [evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/), and [monitoring](https://mlflow.org/docs/latest/genai/governance/ai-gateway/). This plugin automatically traces OpenClaw agent executions in MLflow, capturing LLM calls, tool invocations, and sub-agent spans in a hierarchical trace structure.

<p align="center">
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/master/libs/typescript/integrations/openclaw/dashboard-screenshot.png" alt="OpenClaw MLflow Integration" width="700" style="border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);" />
</p>

## Key Benefits

- 🌐 **Open Source**: MLflow is 100% open source and governed by the Linux Foundation, rooted in the same philosophy as OpenClaw.
- 🛡️ **You Own Your Data**: MLflow is self-hosted. Trace data from OpenClaw stays on your infrastructure and never leaves it.
- 🔀 **Vendor Neutral**: MLflow works with any LLM provider or agent framework, with no vendor lock-in.

## Setup

### 1. Install the Plugin

```bash
openclaw plugins install @mlflow/mlflow-openclaw
```

### 2. Start the MLflow Server

Start the MLflow server (self-hosting) following the [instructions](https://mlflow.org/docs/latest/genai/getting-started/connect-environment/). Alternatively, use a managed MLflow service if you prefer not to self-host.

### 3. Configure the Plugin

```
openclaw mlflow configure
```

The plugin will prompt you for the MLflow tracking URI and experiment ID. You can [create an experiment](https://mlflow.org/docs/latest/genai/tracing/quickstart/#create-a-mlflow-experiment) from the MLflow UI.

```
~$ openclaw mlflow configure

🦞 OpenClaw 2026.3.13 (61d171a) — Automation with claws: minimal fuss, maximal pinch.

┌  MLflow Tracing configuration
│
◆  MLflow Tracking URI
│  http://localhost:5000
└
◇  Experiment ID
│  2
```

### 4. Check the Status

Verify the configuration by running the following command:

```bash
openclaw mlflow status
```

If the configuration is successful, you should see the effective configuration in the output.

### 5. Talk to OpenClaw

Run or restart the OpenClaw gateway to apply the configuration.

```bash
openclaw gateway run  # or openclaw gateway restart
openclaw message send "Hello, Lobster!"
```

Visit the MLflow UI (e.g. http://localhost:5000) to see the trace.

<p align="center">
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/master/libs/typescript/integrations/openclaw/trace-screenshot.png" alt="MLflow UI" width="700" style="border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);" />
</p>

## Configuration

### Environment Variables

Tracking URI and experiment ID can also be set through environment variables:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_ID=<your-experiment-id>
```

### Plugin Allowlist

OpenClaw shows a warning when a community plugin is installed but not declared in the [plugin allowlist](https://docs.openclaw.ai/tools/plugin#config). Add `mlflow-openclaw` to the plugin allowlist in your `openclaw.json` file to suppress the warning.

```
{
    "plugins": {
        "allow": ["mlflow-openclaw"]
    }
}
```

## What Gets Traced

The plugin creates a span hierarchy for each agent session:

```
AGENT (openclaw_agent)              ← root span
├── LLM (llm_call)                  ← each LLM interaction
├── TOOL (tool_<name>)              ← each tool invocation
├── AGENT (subagent_<label>)        ← sub-agent executions
└── ...
```

## Development

```bash
# Type-check
npm run typecheck

# Test
npm test

# Format
npm run format

# Lint
npm run lint
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt) file for details.
