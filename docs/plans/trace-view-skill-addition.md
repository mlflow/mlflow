# Trace View Skill Addition for `analyze-mlflow-trace/SKILL.md`

The following content should be added to the `analyze-mlflow-trace/SKILL.md` file
in the `mlflow/skills` repository.

---

## Creating Trace Views

When the user asks to filter, focus, or customize what they see in a trace
(e.g., "show me only the tool calls", "focus on the LLM spans"), create a
TraceView using the Python API.

### Steps

1. Determine the appropriate SpanFilter from the user's request
2. Create the view using the MLflow Python API
3. Output the marker so the UI can apply it automatically

### Example

```python
import mlflow
from mlflow.entities.trace_view import SpanFilter

trace = mlflow.get_trace("TRACE_ID_FROM_CONTEXT")
view = trace.create_view(
    name="Tool Calls",
    span_filter=SpanFilter(span_type="TOOL"),
    output_path="$.result",
    created_by="assistant",
)
print(f'[trace_view_created: {{"view_id": "{view.view_id}", "trace_id": "{trace.info.trace_id}"}}]')
```

### Trigger Phrases

- "Show me only the tool calls"
- "Focus on the LLM spans"
- "Filter to just the retriever"
- "Show key decision points"
- "Create a view for..."
- "What did the tools do?" (analyze first, then offer to create a view)

### SpanFilter Mapping

| User Request | SpanFilter |
|---|---|
| "tool calls" | `SpanFilter(span_type="TOOL")` |
| "LLM spans" / "model calls" | `SpanFilter(span_type="LLM")` or `SpanFilter(span_type="CHAT_MODEL")` |
| "retriever" / "search" | `SpanFilter(span_type="RETRIEVER")` |
| Specific span name | `SpanFilter(span_name="ExactName")` |
| By attribute | `SpanFilter(attribute_key="key", attribute_value="value")` |

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
- Views can be experiment-scoped (applies to all traces) or trace-scoped (applies to one trace)

### Summarization and Analysis

You can also use the Trace's built-in methods for natural language analysis:

```python
trace = mlflow.get_trace("TRACE_ID")

# Quick summary
summary = trace.summarize(model="openai:/gpt-4o-mini")

# Specific analysis
answer = trace.analyze("How many tool calls were made?", model="openai:/gpt-4o-mini")

# Analysis scoped to a view
view = trace.views[0]  # or create one
answer = trace.analyze("What went wrong?", model="openai:/gpt-4o-mini", view=view)
```
