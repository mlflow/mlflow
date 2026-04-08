---
name: analyzing-mlflow-trace
description: Analyzes a single MLflow trace or creates trace views to filter and focus on specific parts. Use when the user provides a trace ID and asks to debug, investigate, find issues, root-cause errors, understand behavior, analyze quality, or create a filtered view. Triggers on "analyze this trace", "what went wrong with this trace", "debug trace", "investigate trace", "why did this trace fail", "root cause this trace", "show me only the tool calls", "focus on the LLM spans", "create a view for", "filter to just the retriever".
---

# Analyzing a Single MLflow Trace

## Trace Structure

A trace captures the full execution of an AI/ML application as a tree of **spans**. Each span represents one operation (LLM call, tool invocation, retrieval step, etc.) and records its inputs, outputs, timing, and status. Traces also carry **assessments** — feedback from humans or LLM judges about quality.

It is recommended to read [references/trace-structure.md](references/trace-structure.md) before analyzing a trace — it covers the complete data model, all fields and types, analysis guidance, and OpenTelemetry compatibility notes.

## Handling CLI Output

Traces can be 100KB+ for complex agent executions. **Always redirect output to a file** — do not pipe `mlflow traces get` directly to `jq`, `head`, or other commands, as piping can silently produce no output.

```bash
# Fetch full trace to a file (traces get always outputs JSON, no --output flag needed)
mlflow traces get --trace-id <ID> > /tmp/trace.json

# Then process the file
jq '.info.state' /tmp/trace.json
jq '.data.spans | length' /tmp/trace.json
```

**Prefer fetching the full trace and parsing the JSON directly** rather than using `--extract-fields`. The `--extract-fields` flag has limited support for nested span data (e.g., span inputs/outputs may return empty objects). Fetch the complete trace once and parse it as needed.

## JSON Structure

The trace JSON has two top-level keys: `info` (metadata, assessments) and `data` (spans).

```
{
  "info": { "trace_id", "state", "request_time", "assessments", ... },
  "data": { "spans": [ { "span_id", "name", "status", "attributes", ... } ] }
}
```

**Key paths** (verified against actual CLI output):

| What | jq path |
|---|---|
| Trace state | `.info.state` |
| All spans | `.data.spans` |
| Root span | `.data.spans[] \| select(.parent_span_id == null)` |
| Span status code | `.data.spans[].status.code` (values: `STATUS_CODE_OK`, `STATUS_CODE_ERROR`, `STATUS_CODE_UNSET`) |
| Span status message | `.data.spans[].status.message` |
| Span inputs | `.data.spans[].attributes["mlflow.spanInputs"]` |
| Span outputs | `.data.spans[].attributes["mlflow.spanOutputs"]` |
| Assessments | `.info.assessments` |
| Assessment name | `.info.assessments[].assessment_name` |
| Feedback value | `.info.assessments[].feedback.value` |
| Feedback error | `.info.assessments[].feedback.error` |
| Assessment rationale | `.info.assessments[].rationale` |

**Important**: Span inputs and outputs are stored as serialized JSON strings inside `attributes`, not as top-level span fields. Traces from third-party OpenTelemetry clients may use different attribute names (e.g., GenAI Semantic Conventions, OpenInference, or custom keys) — check the raw `attributes` dict to find the equivalent fields.

**If paths don't match** (structure may vary by MLflow version), discover them:

```bash
# Top-level keys
jq 'keys' /tmp/trace.json

# Span keys
jq '.data.spans[0] | keys' /tmp/trace.json

# Status structure
jq '.data.spans[0].status' /tmp/trace.json
```

## Quick Health Check

After fetching a trace to a file, run this to get a summary:

```bash
jq '{
  state: .info.state,
  span_count: (.data.spans | length),
  error_spans: [.data.spans[] | select(.status.code == "STATUS_CODE_ERROR") | .name],
  assessment_errors: [.info.assessments[] | select(.feedback.error) | .assessment_name]
}' /tmp/trace.json
```

## Analysis Insights

- **`state: OK` does not mean correct output.** It only means no unhandled exception. Check assessments for quality signals, and if none exist, analyze the trace's inputs, outputs, and intermediate span data directly for issues.
- **Always consult the `rationale` when interpreting assessment values.** The `value` alone can be misleading — for example, a `user_frustration` assessment with `value: "no"` could mean "no frustration detected" or "the frustration check did not pass" (i.e., frustration *is* present), depending on how the scorer was configured. The `.rationale` field (a top-level assessment field, **not** nested under `.feedback`) explains what the value means in context and often describes the issue in plain language before you need to examine any spans.
- **Assessments tell you *what* went wrong; spans tell you *where*.** If assessments exist, use feedback/expectations to form a hypothesis, then confirm it in the span tree. If no assessments exist, examine span inputs/outputs to identify where the execution diverged from expected behavior.
- **Assessment errors are not trace errors.** If an assessment has an `error` field, it means the scorer or judge that evaluated the trace failed — not that the trace itself has a problem. The trace may be perfectly fine; the assessment's `value` is just unreliable. This can happen when a scorer crashes (e.g., timed out, returned unparseable output) or when a scorer was applied to a trace type it wasn't designed for (e.g., a retrieval relevance scorer applied to a trace with no retrieval steps). The latter is a scorer configuration issue, not a trace issue.
- **Span timing reveals performance issues.** Gaps between parent and child spans indicate overhead; repeated span names suggest retries; compare individual span durations to find bottlenecks.
- **Token usage explains latency and cost.** Look for token usage in trace metadata (e.g., `mlflow.trace.tokenUsage`) or span attributes (e.g., `mlflow.chat.tokenUsage`). Not all clients set these — check the raw `attributes` dict for equivalent fields. Spikes in input tokens may indicate prompt injection or overly large context.

## Codebase Correlation

MLflow Tracing captures inputs, outputs, and metadata from different parts of an application's call stack. By correlating trace contents with the source code, issues can be root-caused more precisely than from the trace alone.

- **Span names map to functions.** Span names typically match the function decorated with `@mlflow.trace` or wrapped in `mlflow.start_span()`. For autologged spans (LangChain, OpenAI, etc.), names follow framework conventions instead (e.g., `ChatOpenAI`, `RetrievalQA`).
- **The span tree mirrors the call stack.** If span A is the parent of span B, then function A called function B.
- **Span inputs/outputs correspond to function parameters/return values.** Comparing them against the code logic reveals whether the function behaved as designed or produced an unexpected result.
- **The trace shows *what happened*; the code shows *why*.** A retriever returning irrelevant results might trace back to a faulty similarity threshold. Incorrect span inputs might reveal wrong model parameters or missing environment variables set in code.

## Example: Investigating a Wrong Answer

A user reports that their customer support agent gave an incorrect answer for the query "What is our refund policy?" There are no assessments on the trace.

**1. Fetch the trace and check high-level signals.**

The trace has `state: OK` — no crash occurred. No assessments are present, so examine the trace's inputs and outputs directly. The `response_preview` says *"Our shipping policy states that orders are delivered within 3-5 business days..."* — this answers a different question than what was asked.

**2. Examine spans to locate the problem.**

The span tree shows:

```
customer_support_agent (AGENT) — OK
├── plan_action (LLM) — OK
│   outputs: {"tool_call": "search_knowledge_base", "args": {"query": "refund policy"}}
├── search_knowledge_base (TOOL) — OK
│   inputs: {"query": "refund policy"}
│   outputs: [{"doc": "Shipping takes 3-5 business days...", "score": 0.82}]
├── generate_response (LLM) — OK
│   inputs: {"messages": [..., {"role": "user", "content": "Context: Shipping takes 3-5 business days..."}]}
│   outputs: {"content": "Our shipping policy states..."}
```

The agent correctly decided to search for "refund policy," but the `search_knowledge_base` tool returned a shipping document. The LLM then faithfully answered using the wrong context. The problem is in the tool's retrieval, not the agent's reasoning or the LLM's generation.

**3. Correlate with the codebase.**

The span `search_knowledge_base` maps to a function in the application code. Investigating reveals the vector index was built from only the shipping FAQ — the refund policy documents were never indexed.

**4. Recommendations.**

- Re-index the knowledge base to include refund policy documents.
- Add a retrieval relevance scorer to detect when retrieved context doesn't match the query topic.
- Consider adding expectation assessments with correct answers for common queries to enable regression testing.

## Creating Trace Views

Trace views highlight specific parts of a trace. There are two types:

1. **Simple filter views** — show all spans matching a single criterion (e.g., "all tool calls").
2. **Multi-range views** — break the trace into labeled, ordered steps (e.g., "milestones", "key decisions", "walkthrough").

Both are created via the `mlflow traces create-view` CLI command.

**Choose multi-range when** the user asks for milestones, key steps, a walkthrough,
decision points, or any chaptered breakdown of the trace. Multi-range views use
`SpanRange` objects with `from_selector` / `to_selector` to define contiguous
sections of the span tree — they cover ranges over steps, not individual spans.

### Simple Filter Views

For single-criterion filtering, use the span filter flags:

```bash
VIEW_JSON=$(mlflow traces create-view --trace-id TRACE_ID \
    --name "Tool Calls" --span-type TOOL --created-by assistant)
VIEW_ID=$(echo "$VIEW_JSON" | jq -r '.view_id')
TRACE_ID=$(echo "$VIEW_JSON" | jq -r '.trace_id')
echo "[trace_view_created: {\"view_id\": \"$VIEW_ID\", \"trace_id\": \"$TRACE_ID\"}]"
```

### Multi-Range Views

For milestones, walkthroughs, or chaptered analysis, use `--ranges-json` to pass
a JSON array of `SpanRange` objects. Each range selects a contiguous section of
the trace using `from_selector` (start span) and optionally `to_selector` (end
span). The `position` field orders the ranges (0 = summary).

**How ranges work:**
- `from_selector` only (no `to_selector`): selects the matching span and its entire subtree
- `from_selector` + `to_selector`: selects all spans from the start span through the end span (inclusive of subtrees) in DFS order

```bash
# Example: create a milestones view for a customer support trace
VIEW_JSON=$(mlflow traces create-view --trace-id TRACE_ID \
    --name "Key Milestones" --created-by assistant \
    --ranges-json '[
      {"from_selector": {"span_id": "ROOT_SPAN_ID"}, "label": "Summary",
       "description": "Customer support agent processed a refund inquiry", "position": 0},
      {"from_selector": {"span_name": "plan_action"}, "label": "Step 1: Agent Decision",
       "description": "LLM determined which tool to call based on the user query",
       "output_path": "$.choices[0].message.content", "position": 1},
      {"from_selector": {"span_name": "search_knowledge_base"}, "label": "Step 2: Knowledge Retrieval",
       "description": "Retrieved documents from the knowledge base",
       "output_path": "$.documents[*].text", "position": 2},
      {"from_selector": {"span_name": "generate_response"}, "label": "Step 3: Response Generation",
       "description": "Generated the final response using retrieved context",
       "output_path": "$.choices[0].message.content", "position": 3}
    ]')
VIEW_ID=$(echo "$VIEW_JSON" | jq -r '.view_id')
TRACE_ID=$(echo "$VIEW_JSON" | jq -r '.trace_id')
echo "[trace_view_created: {\"view_id\": \"$VIEW_ID\", \"trace_id\": \"$TRACE_ID\"}]"
```

**To create a multi-range view:**

1. Analyze the trace to identify the logical steps/phases of execution
2. For each step, find the span (by `span_id`, `span_name`, or `span_type`) that anchors that phase
3. Build a JSON array of range objects ordered by `position` — position 0 is the summary, subsequent positions are steps
4. Each range should have a descriptive `label` and `description` explaining what happened in that phase
5. Use `from_selector` + `to_selector` when a step spans multiple sibling spans; use `from_selector` alone when a step is a single span and its subtree
6. Pass the array via `--ranges-json` and output the `[trace_view_created: ...]` marker

**SpanRange JSON fields:**
- `from_selector` (required): object with `span_id`, `span_name`, `span_type`, `attribute_key`, `attribute_value`
- `to_selector` (optional): same fields as `from_selector`
- `label`: display name for this range
- `description`: explanation of what happened in this phase
- `input_path` / `output_path`: JSONPath expressions to extract specific data
- `position`: integer ordering the ranges (0 = first/summary)

**Selecting spans for ranges:**
- Use `span_id` when you know the exact span (most precise)
- Use `span_name` when the span has a unique name in the trace
- Use `span_type` when selecting the first span of a given type
- Combine `from_selector` + `to_selector` to capture a range between two spans

### Other View Commands

```bash
# List views for a trace
mlflow traces list-views --trace-id tr-abc123

# List views for an experiment
mlflow traces list-views --experiment-id 1 --output json

# Get a specific view
mlflow traces get-view --trace-id tr-abc123 --view-id tv-def456

# Delete a view
mlflow traces delete-view --trace-id tr-abc123 --view-id tv-def456
```

### Trigger Phrases

| User Request | View Type |
|---|---|
| "Show me only the tool calls" | Simple filter |
| "Focus on the LLM spans" | Simple filter |
| "Filter to just the retriever" | Simple filter |
| "Show key milestones / decision points" | Multi-range (`--ranges-json`) |
| "Walk me through this trace" | Multi-range (`--ranges-json`) |
| "Break this down into steps" | Multi-range (`--ranges-json`) |
| "Create a view for..." | Depends on request |

### SpanSelector CLI Mapping (Simple Filters)

| User Request | CLI flags |
|---|---|
| "tool calls" | `--span-type TOOL` |
| "LLM spans" / "model calls" | `--span-type LLM` or `--span-type CHAT_MODEL` |
| "retriever" / "search" | `--span-type RETRIEVER` |
| Specific span name | `--span-name "ExactName"` |
| By attribute | `--attribute-key "key" --attribute-value "value"` |

### JSONPath Examples

| Use Case | input_path / output_path |
|---|---|
| Chat message content | `$.messages[0].content` |
| Model response text | `$.choices[0].message.content` |
| Tool result | `$.result` |
| Retrieved documents | `$.documents[*].text` |

### Important Notes

- Always output the `[trace_view_created: {...}]` marker AFTER creating the view
- The marker must be on its own line for the UI to parse it
- The view is persisted in the MLflow tracking store — it will survive page refreshes
- Views can be trace-scoped (use `--trace-id`) or experiment-scoped (use `--experiment-id`)
- If creating a trace-scoped view fails with a "not found" error, fall back to creating
  an experiment-scoped view using `--experiment-id` instead of `--trace-id`. This can
  happen when traces were fetched from a remote backend and don't exist in the local store.
- For multi-range views, each range's `description` should explain what happened in that
  phase of execution — this is what the user sees in the UI alongside the span data
