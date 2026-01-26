# MLflow Tracing Integration

Complete guide for integrating MLflow tracing with your agent. This is the authoritative source for all tracing implementation - follow these instructions after completing environment setup in setup-guide.md.

**Prerequisites**: Complete setup-guide.md Steps 1-2 (MLflow install + environment configuration)

## Quick Start

Three steps to integrate tracing:

1. **Enable autolog** - Add `mlflow.<library>.autolog()` BEFORE importing agent code
2. **Decorate entry points** - Add `@mlflow.trace` to agent's main functions
3. **Verify** - Run test query and check trace is captured

**Minimum implementation:**
```python
import mlflow
mlflow.langchain.autolog()  # Before imports

from my_agent import agent

@mlflow.trace
def run_agent(query: str) -> str:
    return agent.run(query)
```

See sections below for detailed instructions and verification steps.

## Documentation Access Protocol

**MANDATORY: Follow the documentation protocol to read MLflow documentation before implementing:**

```bash
# Query llms.txt for tracing documentation
curl https://mlflow.org/docs/latest/llms.txt | grep -A 20 "tracing"
```

Or use WebFetch:
- Start: `https://mlflow.org/docs/latest/llms.txt`
- Query for: "MLflow tracing documentation", "autolog setup", "trace decorators"
- Follow referenced URLs for detailed guides

## Key Rules for Agent Evaluation

1. **Enable Autolog FIRST** - Call `mlflow.{library}.autolog()` before importing agent code
   - Captures internal library calls automatically
   - Supported: `langchain`, `langgraph`, `openai`, `anthropic`, etc.

2. **Add @mlflow.trace to Entry Points** - Decorate agent's main functions
   - Creates top-level span in trace hierarchy
   - Example: `@mlflow.trace` on `run_agent()`, `process_query()`, etc.

3. **Enable Session Tracking for Multi-Turn** - Group conversations by session
   ```python
   trace_id = mlflow.get_last_active_trace_id()
   mlflow.set_trace_tag(trace_id, "session_id", session_id)
   ```

4. **Verify Trace Creation** - Test run should create traces with non-None trace_id
   ```bash
   # Check traces exist
   uv run mlflow traces search --experiment-id $MLFLOW_EXPERIMENT_ID
   ```

5. **Tracing Must Work Before Evaluation** - If traces aren't created, stop and troubleshoot

## Minimal Example

```python
# step 1: Enable autolog BEFORE imports
import mlflow
mlflow.langchain.autolog()  # Or langgraph, openai, etc. Use the documentation protocol to find the integration for different libraries.

# step 2: Import agent code
from my_agent import agent

# step 3: Add @mlflow.trace decorator
@mlflow.trace
def run_agent(query: str, session_id: str = None) -> str:
    """Agent entry point with tracing."""
    result = agent.run(query)

    # step 4 (optional): Track session for multi-turn
    if session_id:
        trace_id = mlflow.get_last_active_trace_id()
        if trace_id:
            mlflow.set_trace_tag(trace_id, "session_id", session_id)

    return result
```

## ⚠️ Critical Verification Checklist

After implementing tracing, verify these requirements **IN ORDER**:

**Quick verification:** Edit and run `scripts/validate_agent_tracing.py` - it checks all items below automatically.

**Manual verification** (if needed):

### 1. Autolog Enabled
```bash
# Find autolog call
grep -r "mlflow.*autolog" .
```
**Expected**: Find autolog() call in initialization file (main.py, __init__.py, app.py, etc.)

### 2. Import Order Correct
**Check**: Autolog call appears BEFORE any agent/library imports in the file
**Expected**: The line with `mlflow.autolog()` comes before any `from your_agent import ...` statements

### 3. Entry Points Decorated
```bash
# Find trace decorators
grep -r "@mlflow.trace" .
```
**Expected**: Find @mlflow.trace on agent's main functions

### 4. Traces Created
```bash
# Run agent with test input
uv run python -c "from my_agent import run_agent; run_agent('test query')"

# Check trace was created
uv run mlflow traces search --experiment-id $MLFLOW_EXPERIMENT_ID --extract-fields info.trace_id
```
**Expected**: Non-empty trace_id returned

### 5. Trace Structure Complete
```bash
# View trace details
uv run mlflow traces get <trace_id>
```
**Expected**:
- Top-level span with your function name
- Child spans showing internal library calls (if autolog enabled)
- Session tags (if multi-turn agent)

**If ANY check fails**: Stop and troubleshoot before proceeding to evaluation.

## Common Issues

**Traces not created**:
- Check autolog is called before imports
- Verify decorator is @mlflow.trace (not @trace or @mlflow.trace_span)
- Ensure MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID are set

**Empty traces** (no child spans):
- Autolog may not support your library version
- Check MLflow docs for supported library versions
- Verify autolog is called before library imports

**Session tracking not working**:
- Verify `trace_id = mlflow.get_last_active_trace_id()` is called inside traced function
- Check `mlflow.set_trace_tag(trace_id, key, value)` has correct parameter order

For detailed troubleshooting, see `troubleshooting.md`.

## Tracing Integration Complete

After completing all steps above, verify:

- [ ] Test run creates traces with non-None trace_id (verified with validate_agent_tracing.py)
- [ ] Traces visible in MLflow UI or via `mlflow traces search`
- [ ] Trace hierarchy includes both @mlflow.trace spans and autolog spans
