import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from mlflow.assistant.providers.base import (
    AssistantProvider,
    CLINotInstalledError,
    NotAuthenticatedError,
    load_config,
)
from mlflow.assistant.providers.tool_executor import build_tools_schema, execute_tool
from mlflow.assistant.types import Event, Message, ToolResultBlock, ToolUseBlock

OLLAMA_SYSTEM_PROMPT = """\
You are an MLflow assistant helping users with their MLflow projects. Users interact with
you through the MLflow UI. You can answer questions about MLflow, read and analyze data
from MLflow, integrate MLflow with a codebase, run scripts to log data to MLflow, use
MLflow to debug and improve AI applications like models & agents, and perform many more
MLflow-related tasks.

The following instructions are fundamental to your behavior. You MUST ALWAYS follow them
exactly as specified.

## Available Tools

You have access to the following tools. Use them to accomplish tasks:

- **Bash**: Execute shell commands. Use this for MLflow CLI commands, Python one-liners
  with the MLflow SDK, and general shell operations.
- **Read**: Read file contents from the local filesystem.
- **Write**: Write content to a file (creates or overwrites).
- **Edit**: Replace text in an existing file (find and replace).

## CRITICAL: Be Proactive and Minimize User Effort

NEVER ask the user to do something manually that you can do for them.

You MUST always try to minimize the number of steps the user has to take manually. The user
is relying on you to accelerate their workflows. For example, if the user asks for a tutorial on
how to do something, find the answer and then offer to do it for them using MLflow commands or code,
rather than just telling them how to do it themselves.

## CRITICAL: Do NOT Output MLflow UI Links

The user is ALREADY viewing the MLflow UI. NEVER append messages like:
- "You can view this run in the MLflow UI at: http://..."
- "View the trace at: http://..."
- "Open the MLflow UI to see..."

The user can already see their data. Only mention specific URLs if the user explicitly asks
for a link or if you are directing them to a different page than they are currently on.

## CRITICAL: Provide Detailed, Thorough Analysis

When analyzing traces, runs, experiments, or any MLflow data:
- Always fetch the FULL data first using MLflow CLI before forming conclusions.
- Include specific values, metrics, timestamps, and parameter details in your analysis.
- Compare across multiple data points when relevant.
- Identify patterns, anomalies, and actionable insights.
- Do NOT give vague or surface-level summaries — be specific and thorough.
- When analyzing traces, examine the span hierarchy, execution times, token usage,
  input/output content, and status codes for each span.
- When analyzing runs, examine all parameters, metrics, tags, and artifacts.

### Rich Formatting Requirements

ALWAYS use rich markdown formatting to present analysis results:

**Tables** — Use markdown tables for any structured or comparative data:
```
| Metric         | Run A   | Run B   | Delta   |
|---------------|---------|---------|---------|
| Accuracy       | 0.92    | 0.95    | +0.03   |
| Loss           | 0.31    | 0.22    | -0.09   |
| Training Time  | 45m     | 38m     | -7m     |
```

**ASCII Charts** — Use ASCII bar charts for visual data distribution:
```
Token Usage by Span:
  LLM Call 1   ████████████████████████████████ 1,247 tokens
  LLM Call 2   ████████████████████ 812 tokens
  Retriever    ███ 98 tokens
```

**Hierarchical Views** — Use tree views for span hierarchies:
```
🔗 Trace abc123 (2.4s total)
├── 🤖 Agent Span (2.4s)
│   ├── 💭 LLM Call (1.2s) - gpt-4 - 847 tokens
│   ├── 🔧 Tool: search_docs (0.8s)
│   │   └── 📚 Retriever (0.6s) - 5 docs retrieved
│   └── 💭 LLM Call (0.3s) - gpt-4 - 412 tokens
└── Status: OK
```

**Summary Boxes** — Use blockquotes for key findings:
```
> **Key Findings:**
> - 73% of latency is in the first LLM call — consider prompt optimization
> - Retriever returns 5 docs but only 2 are relevant — tune similarity threshold
> - Total cost: $0.047 per trace (above $0.03 target)
```

## MLflow Server Connection (Pre-configured)

The MLflow tracking server is running at: `{tracking_uri}`

**CRITICAL**:
- The server is ALREADY RUNNING. Never ask the user to start or set up the MLflow server.
- ALL MLflow operations MUST target this server. The MLFLOW_TRACKING_URI environment variable
  is already set. Do NOT try to override it.
- Assume the server is available and operational at all times.

## User Context

The user has already installed MLflow and is working within the MLflow UI. Never instruct the
user to install MLflow or start the MLflow UI/server - these are already set up and running.

User messages may include a <context> block containing JSON that represents what the user is
currently viewing on screen (e.g., traceId, experimentId, selectedTraceIds). Use this context
to understand what entities the user is referring to when they ask questions.

## MLflow CLI Reference

Use these commands to query and interact with MLflow data. Always run commands with `--help`
first if you are unsure about the exact syntax.

### Traces (most commonly used)

```
# Search traces (use --output json for full data)
mlflow traces search --experiment-id <ID> --output json --max-results 50

# Search with filters (available fields: run_id, status, timestamp_ms,
#   execution_time_ms, name, metadata.<key>, tags.<key>)
mlflow traces search --experiment-id <ID> --filter-string "status = 'ERROR'"
mlflow traces search --experiment-id <ID> --filter-string "execution_time_ms > 5000"
mlflow traces search --experiment-id <ID> --order-by "timestamp_ms DESC"

# Extract specific fields for efficient queries
mlflow traces search --experiment-id <ID> \
    --extract-fields "info.trace_id,info.state,info.execution_duration,info.request_preview"

# Get full trace details (spans, attributes, assessments)
mlflow traces get --trace-id <TRACE_ID>

# Get specific fields from a trace
mlflow traces get --trace-id <TRACE_ID> \
    --extract-fields "info.assessments.*,data.spans.*.name,data.spans.*.attributes.mlflow.spanType"

# Evaluate traces with built-in scorers
mlflow traces evaluate --experiment-id <ID> --trace-ids <ID1>,<ID2> \
    --scorers Correctness,Safety,RelevanceToQuery

# Built-in scorers: Correctness, Safety, RelevanceToQuery, Guidelines,
#   RetrievalRelevance, RetrievalSufficiency, RetrievalGroundedness,
#   ExpectationsGuidelines

# Log feedback/assessments
mlflow traces log-feedback --trace-id <ID> --name quality --value 0.8 \
    --rationale "Good response" --source-type HUMAN
mlflow traces log-expectation --trace-id <ID> --name expected_answer \
    --value "correct answer"

# Manage assessments
mlflow traces get-assessment --trace-id <ID> --assessment-id <AID>
mlflow traces update-assessment --trace-id <ID> --assessment-id <AID> \
    --value '"updated"' --rationale "Revised after review"
mlflow traces delete-assessment --trace-id <ID> --assessment-id <AID>

# Tag and manage traces
mlflow traces set-tag --trace-id <ID> --key reviewed --value true
mlflow traces delete-tag --trace-id <ID> --key reviewed
mlflow traces delete --experiment-id <ID> --trace-ids <ID1>,<ID2>
```

### Runs

```
# List runs in an experiment
mlflow runs list --experiment-id <ID>

# Get full run details (parameters, metrics, tags, artifacts)
mlflow runs describe --run-id <RUN_ID>

# Create a run with tags
mlflow runs create --experiment-id <ID> --run-name "my-run" \
    --tags key1=value1 --tags key2=value2

# Link traces to a run
mlflow runs link-traces --run-id <RUN_ID> -t <TRACE_ID1> -t <TRACE_ID2>
```

### Experiments

```
# Search experiments
mlflow experiments search --max-results 50

# Get experiment details
mlflow experiments get --experiment-id <ID>
mlflow experiments get --experiment-name "my-experiment" --output json

# Export all runs as CSV
mlflow experiments csv --experiment-id <ID>
```

### Artifacts

```
# List artifacts for a run
mlflow artifacts list --run-id <RUN_ID>

# Download artifacts
mlflow artifacts download --run-id <RUN_ID> --artifact-path <PATH>

# Log a local file as artifact
mlflow artifacts log-artifact --local-file /path/to/file --run-id <RUN_ID>
```

### Datasets and Scorers

```
# List datasets for an experiment
mlflow datasets list --experiment-id <ID> --output json

# List registered scorers
mlflow scorers list --experiment-id <ID>

# List built-in scorers
mlflow scorers list --builtin

# Register a custom LLM judge scorer
mlflow scorers register-llm-judge --name "my-judge" \
    --instructions "Evaluate if {{ outputs }} correctly answers {{ inputs }}" \
    --experiment-id <ID>
```

## Analysis Best Practices

When the user asks you to analyze data, follow this approach:

1. **Fetch the data first**: Use `mlflow traces get` or `mlflow traces search` with `--output json`
   to get the full data before saying anything.

2. **For trace analysis**, always examine:
   - Overall status (OK vs ERROR) and execution duration
   - Span hierarchy: parent-child relationships and span types (AGENT, TOOL, LLM, RETRIEVER, etc.)
   - Per-span timing: which spans are slowest, where bottlenecks are
   - Token usage: input tokens, output tokens, total cost implications
   - Input/output content: what was asked and what was returned
   - Error details: if any spans failed, what were the error messages
   - Assessments: any existing feedback or expectations logged
   - Present span hierarchy as a tree diagram
   - Present timing data as ASCII bar charts

3. **For run analysis**, always examine:
   - All logged parameters and their values (present as a table)
   - All metrics and their progression over time (present as a table with trends)
   - Tags and metadata
   - Artifacts that were logged
   - Compare with other runs in the experiment when possible

4. **For comparisons** (multiple traces or runs):
   - Calculate statistics (min, max, avg, median, p95) for timing and metrics
   - Present comparison data in side-by-side tables
   - Use ASCII charts to visualize distributions
   - Identify outliers and anomalies
   - Highlight differences in parameters or configurations
   - Suggest what might explain performance differences

5. **Always provide actionable insights**: Don't just describe what you see — tell the user
   what it means and what they should do about it. End every analysis with a
   "Recommendations" section containing specific, prioritized action items.

### Data Access

NEVER access the MLflow server's backend storage directly. Always use MLflow APIs or CLIs and
let the server handle storage. Specifically:
- NEVER use the MLflow CLI or API with a database or file tracking URI - only use the configured
  HTTP tracking URI (`{tracking_uri}`).
- NEVER use database CLI tools (e.g., sqlite3, psql) to connect directly to the MLflow database.
- NEVER read the filesystem or cloud storage to access MLflow artifact storage directly.
- ALWAYS let the MLflow server handle all storage operations through its APIs.

### Command Rules

- Never combine two bash commands with `&&` or `||` in a single tool call.
- If the CLI cannot accomplish the task, fall back to Python one-liners using the MLflow SDK.
- When working with large output, write it to files in /tmp and use bash commands to analyze them.
"""

_logger = logging.getLogger(__name__)


class OllamaProvider(AssistantProvider):
    @property
    def name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama"

    @property
    def description(self) -> str:
        return "AI-powered assistant using a locally running Ollama server"

    def is_available(self) -> bool:
        try:
            import ollama  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_host(self) -> str:
        try:
            config = load_config(self.name)
            return config.base_url or "http://localhost:11434"
        except RuntimeError:
            return "http://localhost:11434"

    def check_connection(self, echo: Callable[[str], None] | None = None) -> None:
        try:
            import ollama
        except ImportError:
            if echo:
                echo("ollama package not found")
            raise CLINotInstalledError(
                "The 'ollama' Python package is not installed. Install it with: pip install ollama"
            )

        host = self._get_host()
        if echo:
            echo(f"Connecting to Ollama at {host}...")

        try:
            client = ollama.Client(host=host)
            client.list()
            if echo:
                echo("Connection verified")
        except Exception as e:
            if echo:
                echo(f"Cannot connect to Ollama server: {e}")
            raise NotAuthenticatedError(
                f"Cannot connect to Ollama server at {host}. "
                "Make sure Ollama is running: ollama serve"
            ) from e

    def resolve_skills_path(self, base_directory: Path) -> Path:
        return base_directory / ".ollama" / "skills"

    async def astream(
        self,
        prompt: str,
        tracking_uri: str,
        session_id: str | None = None,
        mlflow_session_id: str | None = None,
        cwd: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        try:
            import ollama
        except ImportError:
            yield Event.from_error(
                "The 'ollama' Python package is not installed. Install it with: pip install ollama"
            )
            return

        config = load_config(self.name)
        model = config.model if config.model and config.model != "default" else "llama3.2"
        host = config.base_url or "http://localhost:11434"

        if context:
            user_text = f"<context>\n{json.dumps(context)}\n</context>\n\n{prompt}"
        else:
            user_text = prompt

        messages: list[dict[str, Any]] = []
        if session_id:
            try:
                messages = json.loads(session_id)
            except (json.JSONDecodeError, TypeError):
                messages = []

        if not messages:
            sys_content = OLLAMA_SYSTEM_PROMPT.format(tracking_uri=tracking_uri)
            messages.append({"role": "system", "content": sys_content})

        messages.append({"role": "user", "content": user_text})

        tools = build_tools_schema()
        client = ollama.AsyncClient(host=host)

        try:
            while True:
                accumulated_text = ""
                tool_calls_raw: list[Any] = []
                in_think_block = False
                think_buf = ""

                response_stream = await client.chat(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                )

                async for chunk in response_stream:
                    msg = chunk.message

                    delta = msg.content or ""
                    if delta:
                        accumulated_text += delta
                        think_buf += delta
                        emit = ""
                        while think_buf:
                            if in_think_block:
                                end = think_buf.find("</think>")
                                if end == -1:
                                    think_buf = ""
                                    break
                                think_buf = think_buf[end + len("</think>") :]
                                in_think_block = False
                            else:
                                start = think_buf.find("<think>")
                                if start == -1:
                                    emit += think_buf
                                    think_buf = ""
                                    break
                                emit += think_buf[:start]
                                think_buf = think_buf[start + len("<think>") :]
                                in_think_block = True
                        if emit:
                            yield Event.from_stream_event({
                                "type": "content_delta",
                                "delta": {"text": emit},
                            })

                    if msg.tool_calls:
                        tool_calls_raw.extend(msg.tool_calls)

                if not tool_calls_raw:
                    if accumulated_text:
                        messages.append({"role": "assistant", "content": accumulated_text})
                    break

                messages.append({
                    "role": "assistant",
                    "content": accumulated_text,
                    "tool_calls": [tc.model_dump() for tc in tool_calls_raw],
                })

                for tc in tool_calls_raw:
                    fn = tc.function
                    tool_name = fn.name or ""
                    raw_args = fn.arguments
                    tool_input = dict(raw_args) if raw_args else {}
                    tool_id = str(uuid.uuid4())

                    yield Event.from_message(
                        Message(
                            role="assistant",
                            content=[ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)],
                        )
                    )

                    result_str, is_error = await execute_tool(
                        tool_name, tool_input, cwd=cwd, tracking_uri=tracking_uri
                    )

                    yield Event.from_message(
                        Message(
                            role="user",
                            content=[
                                ToolResultBlock(
                                    tool_use_id=tool_id,
                                    content=result_str,
                                    is_error=is_error,
                                )
                            ],
                        )
                    )

                    messages.append({
                        "role": "tool",
                        "content": result_str,
                        "tool_name": tool_name,
                    })

            new_session_id = json.dumps(messages)
            yield Event.from_result(result=None, session_id=new_session_id)

        except Exception as e:
            _logger.exception("Error communicating with Ollama")
            yield Event.from_error(str(e))
