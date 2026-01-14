# MLflow Tracing Integration Guide

Complete guide for integrating MLflow tracing with your agent.

## Table of Contents

1. [Overview](#overview)
2. [Documentation-First Implementation Protocol](#-critical-documentation-first-implementation-protocol)
3. [Step 4.1: Enable Autolog for the Library](#step-41-enable-autolog-for-the-library)
4. [Step 4.2: Add @trace Decorators to Entry Points](#step-42-add-trace-decorators-to-entry-points)
5. [Step 4.2.5: Capture Session ID for Conversation Grouping](#step-425-capture-session-id-for-conversation-grouping)
6. [Implementation Verification Against Documentation](#-implementation-verification-against-documentation)
7. [Step 4.3: Verify Complete Trace Structure](#step-43-verify-complete-trace-structure)
8. [Troubleshooting](#troubleshooting)

## Overview

**CRITICAL**: The agent needs to be integrated with MLflow tracing. DO NOT MOVE FORWARD IF TRACING DOES NOT WORK.

MLflow tracing requires **BOTH** autolog and decorators to capture complete traces:

- **Autolog**: Captures internal library calls (LangChain, LangGraph, OpenAI, etc.)
- **@mlflow.trace decorator**: Captures the top-level function call as a span

Together, they create a complete trace hierarchy showing both your function and the internal library operations.

Complete these steps **IN ORDER**.

## ⚠️ CRITICAL: Documentation-First Implementation Protocol

**MANDATORY: Read BEFORE implementing tracing code**

This section enforces the **Documentation Access Protocol** during implementation to prevent common mistakes like:
- ❌ Guessing API names (e.g., `mlflow.set_span_attribute()` - doesn't exist)
- ❌ Assuming method signatures without verification
- ❌ Skipping documentation examples

### Pre-Implementation Checklist

**BEFORE writing ANY tracing code**, complete these steps IN ORDER:

#### Step 1: Identify Your Implementation Task

Which tracing features are you implementing? Check all that apply:

- [ ] **Autolog setup** → Read [Step 4.1](#step-41-enable-autolog-for-the-library) (lines 27-97)
- [ ] **@mlflow.trace decorators** → Read [Step 4.2](#step-42-add-trace-decorators-to-entry-points) (lines 98-188)
- [ ] **Session ID tracking** → Read [Step 4.2.5](#step-425-capture-session-id-for-conversation-grouping) (lines 189-283)

#### Step 2: Read Complete Documentation Section

For EACH feature you checked above:

1. **Read the ENTIRE section** - don't skip examples
2. **Locate code examples** showing exact API usage
3. **Note the EXACT function/method names** used

**Common APIs by feature**:

| Feature | Correct APIs | Wrong APIs (DON'T use these) |
|---------|-------------|------------------------------|
| Autolog | `mlflow.langchain.autolog()` | `mlflow.autolog_langchain()` ❌ |
| Decorators | `@mlflow.trace` | `@trace`, `@mlflow.trace_span` ❌ |
| Session tracking | `mlflow.get_last_active_trace_id()`, `mlflow.set_trace_tag(trace_id, key, value)` | `mlflow.set_span_attribute()`, `mlflow.add_span_attribute()` ❌ |

#### Step 3: Verify Understanding

Before coding, answer these questions:

1. **Can you name the exact MLflow functions/methods you'll use?** (Write them down)
2. **Can you describe the parameter order for each API call?** (e.g., `set_trace_tag` takes trace_id, key, value)
3. **Can you explain the pattern from the documentation example?** (e.g., "Get trace_id first, then set tag")

**If you answered NO to any question**: Re-read the documentation section.

**If you answered YES to all questions**: Proceed to implementation.

---

### During Implementation: Documentation Side-By-Side

**Best practice**: Keep the documentation section OPEN in a side window while coding.

- **Left side**: Your code editor
- **Right side**: This documentation file at the relevant section
- **Verify**: Each line you write matches the documented pattern

**DO NOT**:
- ❌ Close the documentation and code from memory
- ❌ Guess API names or parameters
- ❌ Assume similarity to other libraries

---

## Step 4.1: Enable Autolog for the Library

If the agent uses a popular library (e.g., LangGraph, LangChain, Strands), enable autolog to automatically trace library calls.

### Identify Library

Determine which library your agent uses:

- LangChain
- LangGraph
- OpenAI
- Other supported libraries (check MLflow documentation)

### Add Autolog Call

Add the appropriate autolog call based on your library. Consult MLflow documentation for the exact call:

**LangChain/LangGraph:**

```python
import mlflow

mlflow.langchain.autolog()
```

**OpenAI:**

```python
import mlflow

mlflow.openai.autolog()
```

### Placement

Place the autolog call in your initialization code, typically in:

- `main.py`
- `__init__.py`
- Application startup code

**IMPORTANT**: Call autolog **before** importing or initializing the agent library.

**Example** (`src/myagent/__init__.py`):

```python
import mlflow

# Enable autolog FIRST
mlflow.langchain.autolog()

# Then import agent components
from .agent import run_agent, stream_agent

__all__ = ["run_agent", "stream_agent"]
```

### Verification

Check that the autolog call is present:

```bash
grep -r "mlflow.*autolog" src/
```

You should see output like:

```
src/myagent/__init__.py:mlflow.langchain.autolog()
```

## Step 4.2: Add @trace Decorators to Entry Points

**CRITICAL STEP - DO NOT SKIP**: You must decorate all entry point functions with `@mlflow.trace`.

### What are Entry Points?

Entry points are the main functions that serve as the interface to your agent. These are typically:

- `run_agent()`
- `stream_agent()`
- `handle_request()`
- `process_query()`
- `chat()`
- `query()`

### Identify All Entry Points

1. **Read the codebase** to find functions that external code calls to invoke the agent
2. **Look for functions that**:

   - Accept user queries or requests
   - Return agent responses
   - Are exported in `__all__`
   - Are called from API endpoints or CLI commands

3. **Use grep to find candidates**:
   ```bash
   grep -rn "def run_\|def stream_\|def handle_\|def process_\|def chat\|def query" src/
   ```

### Add @mlflow.trace Decorator

For **EACH** entry point function:

1. **Import mlflow** at the top of the file:

   ```python
   import mlflow
   ```

2. **Add the decorator** directly above the function definition:
   ```python
   @mlflow.trace
   def run_agent(query: str, llm_provider: LLMProvider) -> str:
       # Agent code here
       ...
   ```

**Complete Example:**

```python
import mlflow
from .llm import LLMProvider


@mlflow.trace  # <-- ADD THIS
def run_agent(query: str, llm_provider: LLMProvider) -> str:
    """Run the agent on a query."""
    # Agent implementation
    result = agent.invoke({"query": query})
    return result["output"]


@mlflow.trace  # <-- ADD THIS TOO
def stream_agent(query: str, llm_provider: LLMProvider):
    """Stream agent responses."""
    # Streaming implementation
    for chunk in agent.stream({"query": query}):
        yield chunk
```

### Verify Decorators are Present

Use grep to confirm all entry points have the decorator:

```bash
grep -B 2 "def run_agent\|def stream_agent\|def handle_request" src/*/agent/*.py
```

**Expected output:**

```
src/myagent/agent/graph.py--@mlflow.trace
src/myagent/agent/graph.py-def run_agent(query: str, llm_provider: LLMProvider) -> str:
--
src/myagent/agent/graph.py--@mlflow.trace
src/myagent/agent/graph.py-def stream_agent(query: str, llm_provider: LLMProvider):
```

You should see `@mlflow.trace` above each entry point function.

## Step 4.2.5: Capture Session ID for Conversation Grouping

**IMPORTANT**: If the agent supports sessions or conversations, capture the session_id in traces to enable grouping related queries.

### When to Use Session Tracking

Use session tracking if your agent:

- Supports multi-turn conversations
- Maintains chat history
- Has a session or conversation ID in its interface
- Can handle follow-up questions

**Skip this step** if your agent only handles independent, single-turn queries.

### Check if Agent Uses Session ID

Search the codebase for session-related variables:

```bash
grep -r "session_id\|session_ID\|conversation_id" src/
```

Look for:

- Function parameters named `session_id`, `conversation_id`, etc.
- Session state in agent configuration or state classes
- Chat/conversation modes that maintain context

### Implementation Pattern

If session_id exists in the code but **NOT** in traces, add session tracking to entry points.

**Pattern**: After the `@mlflow.trace` decorator, capture the session_id:

```python
import mlflow
import uuid


@mlflow.trace
def run_agent(
    query: str, llm_provider: LLMProvider, session_id: str | None = None
) -> str:
    # Generate session_id if not provided
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Capture session_id in trace for conversation grouping
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id:
        mlflow.set_trace_tag(trace_id, "session_id", session_id)

    # Rest of agent implementation
    result = agent.invoke({"query": query})
    return result["output"]
```

**Key points**:

1. Add `session_id` parameter (optional, with default)
2. Generate session_id if not provided (for backward compatibility)
3. Get the active trace ID with `mlflow.get_last_active_trace_id()`
4. Tag the trace with `mlflow.set_trace_tag(trace_id, "session_id", session_id)`

### Verify Session ID Capture

Test with an explicit session_id:

```python
import mlflow
from mlflow import MlflowClient

# Run agent with test session
response = run_agent("test query", provider, session_id="test-session-123")

# Verify trace has session tag
trace_id = mlflow.get_last_active_trace_id()
client = MlflowClient()
trace = client.get_trace(trace_id)

# Check for session_id in tags
assert "session_id" in trace.info.tags
assert trace.info.tags["session_id"] == "test-session-123"

print("✓ Session ID captured successfully!")
```

### Benefits of Session Tracking

- **Filter traces by conversation**: `mlflow traces search --filter "tags.session_id = '<id>'"`
- **Group multi-turn interactions** for analysis
- **Track conversation-level metrics** (e.g., queries per session, session duration)
- **Analyze conversation patterns** and user behavior

## ✓ Implementation Verification Against Documentation

**MANDATORY: Run AFTER implementing tracing code, BEFORE runtime testing**

This checkpoint catches common implementation errors by comparing your code against documented patterns.

### Automated Validation (REQUIRED)

Run the static validation script to automatically check your implementation:

```bash
uv run python scripts/validate_tracing_static.py
```

**This script verifies**:
- ✓ Autolog call present and placed before agent imports
- ✓ @mlflow.trace decorators on all entry points
- ✓ Session ID capture code present (if applicable)
- ✓ Correct API names used

**Expected output** (if implementation is correct):

```
Checking for autolog...
✓ Found autolog: mlflow.langchain.autolog() in src/myagent/__init__.py

Checking for @mlflow.trace decorators...
✓ Found decorator on: run_agent (src/myagent/agent/graph.py)
✓ Found decorator on: stream_agent (src/myagent/agent/graph.py)

Checking for session tracking...
✓ Found session_id capture in: run_agent
✓ Uses correct API: mlflow.get_last_active_trace_id()
✓ Uses correct API: mlflow.set_trace_tag()

============================================================
✓ VALIDATION PASSED - Implementation matches documentation
============================================================
```

**If validation FAILS**, you will see errors like:

```
❌ ERROR: Autolog not found in initialization code
❌ ERROR: Missing @mlflow.trace decorator on: run_agent
❌ ERROR: Found invalid API: mlflow.set_span_attribute() (should be: mlflow.set_trace_tag())
```

**Action required if validation fails**:
1. Review the error messages
2. Go back to the relevant documentation section
3. Fix your code to match the documented pattern
4. Re-run validation script
5. Repeat until validation passes

---

### Manual Verification Checklist

After validation script passes, manually verify these patterns:

#### 1. Autolog Implementation

**Check your code**:
```bash
grep -B 2 -A 2 "autolog()" src/
```

**Compare against documentation pattern** (Step 4.1, line 49):
```python
import mlflow
mlflow.langchain.autolog()  # Or mlflow.openai.autolog(), etc.
```

**Verify**:
- [ ] Autolog called in initialization file (__init__.py or before agent imports)
- [ ] Library name matches your agent (langchain, openai, etc.)
- [ ] Called BEFORE agent components are imported

#### 2. Decorator Implementation

**Check your code**:
```bash
grep -B 1 "def run_agent\|def stream_agent" src/*/agent/*.py
```

**Compare against documentation pattern** (Step 4.2, line 141):
```python
@mlflow.trace
def run_agent(query: str, llm_provider: LLMProvider) -> str:
    # Implementation
```

**Verify**:
- [ ] `@mlflow.trace` decorator present (NOT `@trace` or `@mlflow.trace_span`)
- [ ] Decorator placed directly above function definition (no blank lines)
- [ ] All entry points decorated (run_agent, stream_agent, etc.)

#### 3. Session Tracking Implementation (if applicable)

**Check your code**:
```bash
grep -A 5 "get_last_active_trace_id\|set_trace_tag" src/
```

**Compare against documentation pattern** (Step 4.2.5, lines 237-240):
```python
trace_id = mlflow.get_last_active_trace_id()
if trace_id:
    mlflow.set_trace_tag(trace_id, "session_id", session_id)
```

**Verify**:
- [ ] Uses `mlflow.get_last_active_trace_id()` (NOT `set_span_attribute` or `add_span_attribute`)
- [ ] Uses `mlflow.set_trace_tag(trace_id, key, value)` with 3 parameters
- [ ] Has null check `if trace_id:` before setting tag
- [ ] Sets tag with key "session_id" and the session_id value

---

### Common Errors and Fixes

| Error | Why It's Wrong | Correct Approach |
|-------|----------------|------------------|
| `mlflow.set_span_attribute()` | This API doesn't exist in MLflow | Use `mlflow.set_trace_tag(trace_id, key, value)` |
| `mlflow.add_span_attribute()` | This API doesn't exist | Use `mlflow.set_trace_tag(trace_id, key, value)` |
| `@trace` decorator | Wrong import/name | Use `@mlflow.trace` |
| Autolog after imports | Won't capture library calls | Move autolog BEFORE agent imports |
| Missing trace_id check | Fails if no active trace | Add `if trace_id:` before `set_trace_tag` |

---

### Final Verification Gate

**DO NOT PROCEED to runtime testing** until:

- [ ] Static validation script passes ✓
- [ ] Manual verification checklist complete ✓
- [ ] No errors in comparison with documentation patterns ✓
- [ ] All API names match documented examples exactly ✓

**If all checks pass**: Proceed to [Step 4.3: Verify Complete Trace Structure](#step-43-verify-complete-trace-structure)

**If any check fails**: Return to the relevant documentation section, fix implementation, re-verify

---

## Step 4.3: Verify Complete Trace Structure

Now verify that **BOTH** autolog and decorators are working together.

### Using the Validation Script

The easiest way to verify tracing is to use the validation script:

```bash
uv run python scripts/validate_tracing_runtime.py  # Auto-detects module and entry point
# Optional overrides:
uv run python scripts/validate_tracing_runtime.py --module my_agent.agent --entry-point run_agent
```

**Auto-detection**: The script will automatically detect your agent module and entry point. If detection fails, provide them explicitly with --module and --entry-point flags.

This script will:

- Check environment variables
- Verify autolog is enabled
- Find entry points and check decorators
- Run a test query
- Verify complete trace structure
- Check session ID capture
- Print a comprehensive PASS/FAIL report

### Manual Verification

If you prefer manual verification:

1. **Invoke the agent** with a sample input:

   ```python
   import mlflow
   from myagent import run_agent

   response = run_agent("What is MLflow?", llm_provider)
   trace_id = mlflow.get_last_active_trace_id()
   print(f"Trace ID: {trace_id}")
   ```

2. **Retrieve the trace**:

   ```python
   from mlflow import MlflowClient

   client = MlflowClient()
   trace = client.get_trace(trace_id)
   ```

3. **Verify the trace hierarchy** shows ALL levels:
   ```
   ✓ Top-level span: Function decorated with @trace (e.g., "run_agent")
     └── ✓ Library spans: LangChain/LangGraph (from autolog)
         └── Agent nodes
             └── LLM calls
   ```

**Expected structure**:

- **Top-level span**: Your function name (e.g., "run_agent") - from `@mlflow.trace`
- **Child spans**: Library operations (e.g., "LangChain", "Agent", "Tool") - from autolog
- **Nested spans**: Individual operations (LLM calls, tool calls, etc.)

**If you don't see the top-level span** with your function name, the `@trace` decorator is missing or not working.

**If you don't see library spans**, autolog is not enabled or not working.

### Print Trace Hierarchy

```python
def print_trace_hierarchy(spans, indent=0):
    """Recursively print trace span hierarchy."""
    for span in spans:
        prefix = "  " * indent
        print(f"{prefix}- {span.name} ({span.span_type})")
        if hasattr(span, "spans") and span.spans:
            print_trace_hierarchy(span.spans, indent + 1)


# Print the hierarchy
print("Trace structure:")
print_trace_hierarchy(trace.data.spans)
```

### Example Output

**Good - Complete hierarchy:**

```
Trace structure:
- run_agent (FUNCTION)
  - LangChainRunnableSequence (CHAIN)
    - AgentExecutor (AGENT)
      - call_tool (TOOL)
        - search_docs (FUNCTION)
      - ChatOpenAI (LLM)
```

**Bad - Missing library spans:**

```
Trace structure:
- run_agent (FUNCTION)
  [No child spans - autolog not working!]
```

**Bad - Missing top-level span:**

```
Trace structure:
- LangChainRunnableSequence (CHAIN)
  [Top-level function span missing - @trace decorator not working!]
```

## CHECKPOINT

**DO NOT PROCEED** until you can confirm:

- [ ] Autolog is enabled for your library
- [ ] All entry points have `@mlflow.trace` decorator
- [ ] Test trace shows complete hierarchy with both decorator and autolog spans
- [ ] Session ID is captured in traces (if agent supports sessions/conversations)

Run the validation script to verify:

```bash
uv run python scripts/validate_tracing_runtime.py
```

Expected output:

```
============================================================
Validation Report
============================================================

✓ ALL CHECKS PASSED!

Your agent is properly integrated with MLflow tracing.
You can proceed with evaluation.
============================================================
```

## Troubleshooting

### Autolog Not Working

**Symptoms**: No library spans in trace, only top-level function span

**Solutions**:

1. Verify autolog call is before agent imports
2. Check correct library is specified (`langchain`, not `langgraph` if using LangChain)
3. Ensure library is actually installed: `pip list | grep langchain`
4. Check MLflow version supports your library version

### Missing Top-Level Span

**Symptoms**: Library spans present but no function span

**Solutions**:

1. Verify `@mlflow.trace` decorator is present above function
2. Check decorator is `@mlflow.trace` not `@trace`
3. Ensure `mlflow` is imported in the file
4. Verify decorator is directly above function (no blank lines)

### No Traces Captured

**Symptoms**: `mlflow.get_last_active_trace_id()` returns None

**Solutions**:

1. Check `MLFLOW_TRACKING_URI` is set
2. Check `MLFLOW_EXPERIMENT_ID` is set
3. Verify MLflow server is running (if using local server)
4. Check Databricks authentication (if using Databricks)
5. Run `mlflow --version` to ensure MLflow is installed

### Session ID Not Captured

**Symptoms**: Trace exists but no session_id tag

**Solutions**:

1. Verify `mlflow.get_last_active_trace_id()` is called early in function
2. Ensure `mlflow.set_trace_tag()` is called immediately after
3. Check trace_id is not None before setting tag
4. Use validation code from Step 4.2.5 to test

### Import Errors

**Symptoms**: `ModuleNotFoundError` when importing agent

**Solutions**:

1. Install agent package: `pip install -e .` from project root
2. Verify you're in correct virtual environment
3. Check package is installed: `pip list | grep <package>`

---

**For additional troubleshooting**, see `references/troubleshooting.md`
