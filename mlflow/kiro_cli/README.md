# MLflow Kiro CLI Integration

This module provides automatic tracing integration between Kiro CLI and MLflow.

## Module Structure

- **`config.py`** - Configuration management (settings files, environment variables)
- **`hooks.py`** - Kiro CLI hook setup and management
- **`cli.py`** - MLflow CLI commands (`mlflow autolog kiro-cli`)
- **`tracing.py`** - Core tracing logic and processors

## Installation

```bash
pip install mlflow
```

## Usage

Set up Kiro CLI tracing in any project directory:

```bash
# Set up tracing in current directory
mlflow autolog kiro-cli

# Set up tracing in specific directory
mlflow autolog kiro-cli -d ~/my-project

# Set up with custom tracking URI
mlflow autolog kiro-cli -u file://./custom-mlruns
mlflow autolog kiro-cli -u sqlite:///mlflow.db

# Set up with Databricks
mlflow autolog kiro-cli -u databricks -e 123456789

# Check status
mlflow autolog kiro-cli --status

# Disable tracing
mlflow autolog kiro-cli --disable
```

## How it Works

1. **Setup**: The `mlflow autolog kiro-cli` command configures Kiro CLI hooks in `.kiro/agents/kiro_default.json` and stores MLflow environment variables in `.kiro/settings.json`
2. **Automatic Tracing**: When you use the `kiro` command in the configured directory, your conversations are automatically traced to MLflow
3. **View Traces**: Use `mlflow server` to view your conversation traces

### How Traces Are Emitted

The integration wires five Kiro CLI hook events into the agent config:

- **agentSpawn**: Logs session start (diagnostic only)
- **userPromptSubmit**: Logs user prompt (diagnostic only)
- **preToolUse**: Logs tool invocation (diagnostic only)
- **postToolUse**: Logs tool result (diagnostic only)
- **stop**: Reads the session transcript and emits the MLflow trace

The `stop` hook is the only hook that emits a trace. It reads the session transcript files from `~/.kiro/sessions/cli/` (a `.jsonl` per-message stream and a `.json` aggregate), parses the most recent turn, and builds a span tree:

```
kiro_cli_conversation  (AGENT root span)
  └── turn  (CHAIN child span with token usage and metering)
        ├── llm             (LLM grandchild — one per text response)
        ├── tool_<name>     (TOOL grandchild — one per tool call)
        └── ...
```

Each trace captures user prompts, assistant responses, tool usage, token counts, metering credits, context usage percentage, session metadata, and timing information.

## Configuration

The setup creates two types of configuration:

### Kiro CLI Hooks (`.kiro/agents/kiro_default.json`)

Hook entries are added under the five event keys (`agentSpawn`, `userPromptSubmit`, `preToolUse`, `postToolUse`, `stop`). Existing user hooks are preserved alongside the MLflow entries.

### Environment Variables (`.kiro/settings.json`)

- `MLFLOW_KIRO_CLI_TRACING_ENABLED=true`: Enables tracing
- `MLFLOW_TRACKING_URI`: Where to store traces (defaults to local storage)
- `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`: Which experiment to use

Settings-file values take precedence over OS environment variables.

## Examples

### Basic Local Setup

```bash
mlflow autolog kiro-cli
kiro chat "help me write a function"
mlflow server
```

### Databricks Integration

```bash
mlflow autolog kiro-cli -u databricks -e 123456789
kiro chat "analyze this data"
# View traces in Databricks
```

### Custom Project Setup

```bash
mlflow autolog kiro-cli -d ~/my-ai-project -u sqlite:///mlflow.db -n "My AI Project"
cd ~/my-ai-project
kiro chat "refactor this code"
mlflow server --backend-store-uri sqlite:///mlflow.db
```

## Troubleshooting

### Check Status

```bash
mlflow autolog kiro-cli --status
```

### Disable Tracing

```bash
mlflow autolog kiro-cli --disable
```

### View Tracing Logs

Diagnostic log entries are written to `.kiro/mlflow/kiro_tracing.log` in the project directory:

```bash
tail -f .kiro/mlflow/kiro_tracing.log
```

### View Raw Configuration

The hook configuration is stored in `.kiro/agents/kiro_default.json`:

```bash
cat .kiro/agents/kiro_default.json
```

The environment configuration is stored in `.kiro/settings.json`:

```bash
cat .kiro/settings.json
```

## Known Limitations

1. **Project-scoped only.** v1 does not support user-global installation under `~/.kiro/`. Users must run `mlflow autolog kiro-cli` in each project directory.
2. **One trace per turn.** The `stop` hook fires after every turn, not only at session end. Each turn becomes its own trace. Users expecting one trace per whole session should be aware.
3. **Token counts may be zero.** Kiro CLI does not always populate `input_token_count` and `output_token_count` (especially for tool-only turns). When both are zero, the `CHAT_USAGE` attribute is omitted. The `metering_usage` credit count is attached whenever present.
4. **Assistant message timestamps are reconstructed.** Only `Prompt` records carry wall-clock timestamps. Per-message grandchild span durations are proportionally allocated within the turn and are not authoritative.
5. **Only the `kiro_default` agent is supported.** Custom agents defined in other `.kiro/agents/*.json` files are not wired by v1.
6. **Hook command uses the `mlflow` binary on PATH.** If `mlflow` is installed in a virtualenv that is not active when Kiro CLI launches, the hook will fail. The `uv run mlflow` prefix (when `UV` is set at enable time) partially mitigates this for `uv`-managed projects.

## Requirements

- Python 3.10+ (required by MLflow)
- MLflow installed (`pip install mlflow`)
- Kiro CLI installed
