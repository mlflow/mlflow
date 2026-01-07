# MLflow Environment Setup Guide

Complete guide for setting up MLflow environment before agent evaluation.

## Table of Contents

1. [Step 1: Install MLflow](#step-1-install-mlflow)
2. [Step 2: Configure Environment](#step-2-configure-environment)
3. [Step 3: Integrate MLflow Tracing](#step-3-integrate-mlflow-tracing)

## Overview

Before evaluation, complete these setup steps in order.

## Step 1: Install MLflow

Check if MLflow >=3.8.0 is installed:

```bash
uv run mlflow --version
```

If not installed or version too old:

```bash
uv pip install mlflow>=3.8.0
```

## Step 2: Configure Environment

### Step 2.1: Set Tracking URI

Ask user: "Where is your MLflow tracking server? (Databricks, local, or other)"

```bash
# For Databricks
export MLFLOW_TRACKING_URI="databricks://DEFAULT"

# For local server (start server first if needed: mlflow server --host 127.0.0.1 --port 5050 &)
export MLFLOW_TRACKING_URI="http://127.0.0.1:5050"

# For other
export MLFLOW_TRACKING_URI="<user_provided_uri>"
```

### Step 2.2: Find or Create Experiment

Search for existing experiments before asking user:

```bash
# List all experiments
uv run mlflow experiments search --view all
```

After reviewing results:

- If suitable experiments exist → Ask user which one to use
- If no experiments exist → Ask user what name for new experiment

**Option A: Use Existing Experiment**

```bash
# Set the experiment ID user selected
export MLFLOW_EXPERIMENT_ID="<user_selected_id>"

# Verify it works
uv run mlflow experiments get --experiment-id "$MLFLOW_EXPERIMENT_ID"
```

**Option B: Create New Experiment** (only after user confirms name)

```bash
# For Databricks - must use /Users/<username>/<name> format
uv run mlflow experiments create -n "/Users/$(whoami)/<user_provided_name>"

# For local - simple name works
uv run mlflow experiments create -n "<user_provided_name>"

# Set the experiment ID from command output
export MLFLOW_EXPERIMENT_ID="<id_from_create_output>"
```

### Step 2.3: Persist Configuration

Add to .env file (recommended):

```bash
echo "MLFLOW_TRACKING_URI=\"$MLFLOW_TRACKING_URI\"" >> .env
echo "MLFLOW_EXPERIMENT_ID=\"$MLFLOW_EXPERIMENT_ID\"" >> .env
```

## Step 3: Integrate MLflow Tracing

⚠️ **Tracing must work before evaluation.** If tracing fails, stop and troubleshoot before proceeding.

Complete these steps in order:

### Step 3.1: Enable Autolog

Add autolog for your agent's library (LangChain, LangGraph, OpenAI, etc.):

```python
import mlflow

mlflow.langchain.autolog()  # Place in __init__.py before agent imports
```

### Step 3.2: Add @mlflow.trace Decorators

Decorate all entry point functions:

```python
import mlflow


@mlflow.trace  # <-- ADD THIS
def run_agent(query: str, llm_provider: LLMProvider) -> str:
    # Agent code here
    ...
```

Verify decorators present:

```bash
grep -B 2 "def run_agent\|def stream_agent" src/*/agent/*.py
```

### Step 3.3: Capture Session ID (Optional)

If agent supports conversations, capture session_id:

```python
@mlflow.trace
def run_agent(query: str, session_id: str | None = None) -> str:
    if session_id is None:
        session_id = str(uuid.uuid4())

    trace_id = mlflow.get_last_active_trace_id()
    if trace_id:
        mlflow.set_trace_tag(trace_id, "session_id", session_id)

    # Rest of function...
```

### Step 3.4: Verify Complete Tracing

**Stage 1: Static Code Check** (no auth required - fast):

```bash
# Check that autolog is called
grep -r "mlflow\..*\.autolog()" src/

# Check that @mlflow.trace decorators are present on entry points
grep -B 2 "@mlflow.trace" src/
```

Verify you see:

- ✓ Autolog import and call (e.g., `mlflow.langchain.autolog()`)
- ✓ `@mlflow.trace` decorator before agent entry point functions

**Stage 2: Runtime Test** (requires auth & LLM - blocking):

```bash
# Run agent with a test query
<your_agent_run_command> "test query"

# Check if trace was created
uv run python -c "import mlflow; trace_id = mlflow.get_last_active_trace_id(); print(f'Trace ID: {trace_id}' if trace_id else 'NO TRACE CAPTURED!')"
```

If no trace is captured, stop and work with user to fix:

- MLflow tracing integration
- Authentication issues
- LLM configuration problems

**Checkpoint - verify before proceeding:**

- [ ] Autolog present and called before agent imports
- [ ] @mlflow.trace decorators on entry points
- [ ] Test run creates a trace (trace ID is not None)
- [ ] Trace visible in MLflow UI (if applicable)

For detailed tracing setup, see `references/tracing-integration.md`.
