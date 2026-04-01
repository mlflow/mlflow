# Trace View Ranges Design

> Redesign of TraceView to support span ranges with descriptions, replacing
> the single-filter model.

## Problem

Traces are information-dense. Understanding a trace means understanding what
happened across *ranges* of spans -- what went in, what came out, and what it
means. The current TraceView model supports a single `SpanFilter` per view
with top-level JSONPath extraction, which reduces to "show me spans of type X."

The MLflow assistant already produces rich trace analysis -- identifying
decision points, grouping related spans into logical steps, explaining
significance, and tailoring summaries to an audience. But it cannot persist
that analysis as a trace view because the data model is too narrow to capture
what the assistant outputs.

## Goal

Make the TraceView data model expressive enough to capture what the assistant
already produces: an ordered list of labeled span ranges, each with a
description and optional input/output extraction. The first range (covering
the root span) serves as the trace-level summary. This is analogous to
chaptering a video timeline -- each chapter covers a contiguous segment and
explains what happened.

## Approaches Considered

### Approach A: SpanSelector + SpanRange (chosen)

Replace `SpanFilter` with `SpanSelector` (selecting spans by type, name, ID,
or attribute). A `TraceView` becomes a list of `SpanRange` entries, each
defined by from/to selectors with label, description, and JSONPath extraction.

- Clean data model that directly expresses the concept
- SpanSelector is a typed dataclass -- easy to validate, store, and index
- JSONPath handles the complex extraction from nested JSON inputs/outputs

### Approach B: Unified query language (JSONPath/JMESPath for everything)

Use a single query language for both span selection and input/output
extraction. Selectors would be query strings like
`[?span_type == 'LLM' && span_name == 'ChatOpenAI']`.

- More expressive (compound conditions, nested attribute access)
- But: opaque strings are harder to validate and index in the database
- Span selection is simple enough (match on 3-4 fields) that a query language
  is overkill
- Would require maintaining a custom parser or taking a library dependency
  for minimal benefit

See `tmp/trace_view_prototype_jsonpath.py` for a working prototype of this
approach.

### Approach C: Annotations layer alongside existing filters

Add a new annotations mechanism while keeping the existing SpanFilter for
experiment-scoped template views.

- Non-breaking, incremental
- But: two different mechanisms in one entity creates conceptual weight
- Naming gets muddled -- what *is* a TraceView?

Both alternative prototypes are preserved in `tmp/` for reference:
- `tmp/trace_view_prototype.py` -- SpanSelector approach (chosen)
- `tmp/trace_view_prototype_jsonpath.py` -- unified query language approach

## Data Model

### SpanSelector

Renamed from `SpanFilter` to reflect its role as a selector (it selects spans
AND defines what to extract via JSONPath). Gains a `span_id` field for direct
span references.

```python
@dataclass
class SpanSelector:
    span_name: str | None = None
    span_type: str | None = None
    span_id: str | None = None
    attribute_key: str | None = None
    attribute_value: str | None = None
```

All non-None fields are AND-combined when matching. A selector with no fields
set matches all spans.

### SpanRange

New entity. Defines a contiguous range of spans in DFS order, bookended by
selectors.

```python
@dataclass
class SpanRange:
    from_selector: SpanSelector
    to_selector: SpanSelector | None = None
    label: str = ""
    description: str = ""
    input_path: str | None = None   # JSONPath for input extraction
    output_path: str | None = None  # JSONPath for output extraction
    position: int = 0               # ordering within the view
    range_id: str | None = None     # server-generated
```

When `to_selector` is None, the range covers only the from-span and its
subtree.

### TraceView

Updated to carry a list of ranges instead of a single filter.

```python
@dataclass
class TraceView:
    name: str
    view_id: str | None = None
    trace_id: str | None = None
    experiment_id: str | None = None
    ranges: list[SpanRange] = field(default_factory=list)
    created_by: str | None = None
    create_time_ms: int | None = None
    last_update_time_ms: int | None = None
```

Removed fields (moved to SpanRange): `span_filter`, `input_path`,
`output_path`, `description`.

Convention: the first range (position 0) is the summary. Its description
provides the trace-level narrative. No special flag -- just ordering.

## Database Schema

### `trace_views` table (simplified)

```sql
CREATE TABLE trace_views (
    view_id                VARCHAR(50) PRIMARY KEY,
    name                   VARCHAR(256) NOT NULL,
    trace_id               VARCHAR(50) REFERENCES trace_info(request_id) ON DELETE CASCADE,
    experiment_id          INTEGER REFERENCES experiments(experiment_id),
    created_by             VARCHAR(256),
    created_timestamp      BIGINT NOT NULL,
    last_updated_timestamp BIGINT NOT NULL,
    CHECK ((trace_id IS NOT NULL AND experiment_id IS NULL) OR
           (trace_id IS NULL AND experiment_id IS NOT NULL))
);

CREATE INDEX ix_trace_views_trace_id ON trace_views (trace_id, created_timestamp);
CREATE INDEX ix_trace_views_experiment_id ON trace_views (experiment_id, created_timestamp);
```

Dropped columns from current schema: `span_filter`, `input_path`,
`output_path`, `description`.

### `trace_view_ranges` table (new)

```sql
CREATE TABLE trace_view_ranges (
    range_id      VARCHAR(50) PRIMARY KEY,
    view_id       VARCHAR(50) NOT NULL REFERENCES trace_views(view_id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    label         VARCHAR(256) NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    from_selector TEXT NOT NULL,
    to_selector   TEXT,
    input_path    TEXT,
    output_path   TEXT,
    UNIQUE (view_id, position)
);

CREATE INDEX ix_trace_view_ranges_view_id ON trace_view_ranges (view_id, position);
```

`from_selector` and `to_selector` are JSON-serialized SpanSelector objects.
Cascade delete on `view_id` ensures cleanup.

### Migration

Since TraceView hasn't shipped, replace the existing migration with a clean
one that creates both tables.

## REST API

Endpoints are unchanged. Payloads include `ranges` instead of the old
filter/extraction fields.

### Create

`POST /mlflow/traces/{trace_id}/views`

```json
{
    "name": "Escalation Flow",
    "created_by": "assistant",
    "ranges": [
        {
            "label": "Agent Execution",
            "description": "Agent attempted to send welcome email but escalated...",
            "from_selector": {"span_name": "AgentExecutor"}
        },
        {
            "label": "Template Lookup",
            "description": "Searched for email template, found 8 matches.",
            "from_selector": {"span_id": "span-2"},
            "to_selector": {"span_name": "search_content"},
            "input_path": "$.reasoning",
            "output_path": "$.results"
        }
    ]
}
```

Position is inferred from array order (0-indexed).

### Get / List

Responses include `ranges` ordered by `position`.

### Update

`PATCH /mlflow/traces/{trace_id}/views/{view_id}`

When `ranges` is present in the request, all existing ranges are deleted and
replaced. No per-range CRUD -- keeps the API simple.

### Delete

Unchanged. Cascade handles range cleanup.

### Python Client

```python
client.create_trace_view(
    trace_id="abc123",
    name="Escalation Flow",
    ranges=[
        SpanRange(
            from_selector=SpanSelector(span_name="AgentExecutor"),
            label="Agent Execution",
            description="...",
        ),
        SpanRange(
            from_selector=SpanSelector(span_id="span-2"),
            to_selector=SpanSelector(span_name="search_content"),
            label="Template Lookup",
            input_path="$.reasoning",
            output_path="$.results",
        ),
    ],
    created_by="assistant",
)
```

## DFS Resolution

Runtime logic that takes a TraceView + trace spans and produces resolved
output. Replaces the current `apply_view` and `find_first_matching_span`
in `view_utils.py`.

### Algorithm

1. Flatten the span tree into DFS order
2. For each SpanRange (ordered by position):
   a. Walk the DFS list, find first span matching `from_selector`
   b. If `to_selector` is None: collect that span + its subtree
   c. Otherwise: find first span matching `to_selector` after the from span,
      collect everything between (inclusive of subtrees)
   d. Apply `input_path` JSONPath to the first matched span's inputs
   e. Apply `output_path` JSONPath to the last matched span's outputs
3. Return ordered list of resolved ranges with labels, descriptions, and
   extracted values

### Key behaviors

- Nested spans are included: a range from an LLM span to a TOOL span
  captures internal spans (tokenizer, sql_query) at any depth
- Empty matches: if a selector matches nothing, the range resolves to an
  empty span list. Label and description still render.
- First range (position 0) is the summary by convention.

See `tmp/trace_view_prototype.py` for a working implementation of this
algorithm.

## Frontend (deferred -- separate design pass)

The backend changes enable a richer frontend. Key areas to address:

- **Summary view rendering** -- render resolved ranges (label, description,
  extracted I/O) instead of current single-filter output
- **Span range selection UI** -- select from/to spans directly in the trace
  tree to create ranges interactively
- **Create view from UI** -- form/flow for manually creating trace views
  without the assistant
- **View-aware span highlighting** -- highlight which spans belong to which
  range when a view is active

## Testing

- Unit tests for SpanSelector matching (replaces SpanFilter tests)
- Unit tests for SpanRange resolution (DFS walk, subtree collection,
  from/to boundary cases)
- Unit tests for JSONPath extraction on resolved ranges
- Store CRUD tests for trace_view_ranges (create, list with ordering,
  update with replace-all semantics, cascade delete)
- Integration test: create a view with multiple ranges via API, retrieve it,
  verify ranges are ordered and complete
