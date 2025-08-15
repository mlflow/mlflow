"""
CLI for traces
"""

import json

import click

from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.tracing.assessment import (
    log_expectation as _log_expectation,
)
from mlflow.tracing.assessment import (
    log_feedback as _log_feedback,
)
from mlflow.tracing.client import TracingClient
from mlflow.utils.string_utils import _create_table, jsonpath_extract_values

# Define reusable options following mlflow/runs.py pattern
EXPERIMENT_IDS = click.option(
    "--experiment-ids",
    type=click.STRING,
    required=True,
    help="Comma-separated list of experiment IDs to search within.",
)
TRACE_ID = click.option("--trace-id", type=click.STRING, required=True)


def _filter_json_by_fields(data: dict, field_paths: list) -> dict:
    """
    Filter a JSON dict to only include fields specified by the field paths.
    Expands wildcards but preserves original JSON structure.

    Args:
        data: Original JSON dictionary
        field_paths: List of dot-notation paths like ['info.trace_id', 'info.assessments.*']

    Returns:
        Filtered dictionary with original structure preserved
    """
    result = {}

    # Collect all actual paths by expanding wildcards
    expanded_paths = set()
    for field_path in field_paths:
        if "*" in field_path:
            # Find all actual paths that match this wildcard pattern
            matching_paths = _find_matching_paths(data, field_path)
            expanded_paths.update(matching_paths)
        else:
            # Direct path
            expanded_paths.add(field_path)

    # Build the result by including only the specified paths
    for path in expanded_paths:
        parts = path.split(".")
        _set_nested_value(result, parts, _get_nested_value_safe(data, parts))

    return result


def _find_matching_paths(data: dict, wildcard_path: str) -> list:
    """Find all actual paths in data that match a wildcard pattern."""
    parts = wildcard_path.split(".")

    def find_paths(current_data, current_parts, current_path=""):
        if not current_parts:
            return [current_path.lstrip(".")]

        part = current_parts[0]
        remaining = current_parts[1:]

        if part == "*":
            paths = []
            if isinstance(current_data, dict):
                for key in current_data.keys():
                    new_path = f"{current_path}.{key}"
                    paths.extend(find_paths(current_data[key], remaining, new_path))
            elif isinstance(current_data, list):
                for i, item in enumerate(current_data):
                    new_path = f"{current_path}.{i}"
                    paths.extend(find_paths(item, remaining, new_path))
            return paths
        else:
            if isinstance(current_data, dict) and part in current_data:
                new_path = f"{current_path}.{part}"
                return find_paths(current_data[part], remaining, new_path)
            return []

    return find_paths(data, parts)


def _get_nested_value_safe(data: dict, parts: list):
    """Safely get nested value, returning None if path doesn't exist."""
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def _set_nested_value(data: dict, parts: list, value):
    """Set a nested value in a dictionary, creating intermediate dicts/lists as needed."""
    if value is None:
        return

    current = data
    for i, part in enumerate(parts[:-1]):
        if part.isdigit() and isinstance(current, list):
            # Handle array index
            idx = int(part)
            while len(current) <= idx:
                current.append({})
            current = current[idx]
        else:
            # Handle object key
            if not isinstance(current, dict):
                return  # Can't set object key on non-dict
            if part not in current:
                # Look ahead to see if next part is a number (array index)
                next_part = parts[i + 1] if i + 1 < len(parts) else None
                if next_part and next_part.isdigit():
                    current[part] = []
                else:
                    current[part] = {}
            current = current[part]

    if parts:
        final_part = parts[-1]
        if final_part.isdigit() and isinstance(current, list):
            # Extend list if needed
            idx = int(final_part)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        elif isinstance(current, dict):
            current[final_part] = value


def _validate_field_paths(field_paths: list, sample_trace: dict):
    """Validate that field paths exist in the trace data structure."""
    invalid_paths = []

    for path in field_paths:
        # Skip validation for paths with wildcards - they'll be expanded later
        if "*" in path:
            continue

        # Test if the path exists by trying to extract values
        values = jsonpath_extract_values(sample_trace, path)
        if not values:  # Empty list means path doesn't exist
            invalid_paths.append(path)

    if invalid_paths:
        available_fields = _get_available_field_suggestions(sample_trace)

        # Create a nice error message
        error_msg = "❌ Invalid field path(s):\n"
        for path in invalid_paths:
            error_msg += f"   • {path}\n"

        error_msg += "\n💡 Use dot notation to specify nested fields:"
        error_msg += "\n   Examples: info.trace_id, info.state, info.assessments.*"

        if available_fields:
            error_msg += "\n\n📋 Available fields in this trace:\n"
            # Group by top-level key for better readability
            info_fields = [f for f in available_fields if f.startswith("info.")]
            data_fields = [f for f in available_fields if f.startswith("data.")]

            if info_fields:
                error_msg += f"   info.*: {', '.join(info_fields[:8])}"
                if len(info_fields) > 8:
                    error_msg += f", ... (+{len(info_fields) - 8} more)"
                error_msg += "\n"

            if data_fields:
                error_msg += f"   data.*: {', '.join(data_fields[:5])}"
                if len(data_fields) > 5:
                    error_msg += f", ... (+{len(data_fields) - 5} more)"
                error_msg += "\n"

        raise click.UsageError(error_msg)


def _get_available_field_suggestions(data: dict, prefix: str = "") -> list:
    """Get a list of available field paths for suggestions."""
    paths = []

    def collect_paths(obj, current_path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{current_path}.{key}" if current_path else key
                paths.append(path)
                # Only go 2 levels deep for suggestions to keep it manageable
                if current_path.count(".") < 2:
                    collect_paths(value, path)
        elif isinstance(obj, list) and obj:
            # Show array notation but don't expand all indices
            path = f"{current_path}.*" if current_path else "*"
            if path not in paths:
                paths.append(path)
            # Sample first item if it's an object
            if isinstance(obj[0], dict):
                collect_paths(obj[0], f"{current_path}.*" if current_path else "*")

    collect_paths(data, prefix)
    return sorted(set(paths))


@click.group("traces")
def commands():
    """
    Manage traces. To manage traces associated with a tracking server, set the
    MLFLOW_TRACKING_URI environment variable to the URL of the desired server.

    \b
    TRACE SCHEMA:
    trace:
      info:                           # Trace metadata
        trace_id: str                 # "tr-xxxxx" identifier
        client_request_id: str        # Request correlation ID
        state: str                    # "OK", "ERROR", etc.
        request_time: str             # ISO timestamp
        execution_duration_ms: int    # Total execution time
        request_preview: str          # Input summary
        response_preview: str         # Output summary
        trace_location: {             # Where trace is stored
          type: str                   # "MLFLOW_EXPERIMENT"
          mlflow_experiment: {
            experiment_id: str
          }
        }
        trace_metadata: {             # Rich metadata
          mlflow.traceInputs: str     # JSON string of inputs
          mlflow.traceOutputs: str    # JSON string of outputs
          mlflow.source.type: str     # "JOB", "NOTEBOOK", etc.
          mlflow.source.name: str     # Source identifier
          mlflow.trace.user: str      # User ID
          mlflow.trace.session: str   # Session ID
          mlflow.trace.sizeBytes: str # Trace size
          mlflow.databricks.*: str    # Databricks-specific metadata
          <custom_keys>: str          # User-defined metadata
        }
        tags: {                       # User-defined tags
          mlflow.traceName: str       # Trace name
          mlflow.user: str            # User email
          <custom_tags>: str          # User-defined tags
        }
        assessments: [                # Feedback/evaluations array
          {
            assessment_id: str        # "a-xxxxx" identifier
            assessment_name: str      # Name/type of assessment
            trace_id: str             # Parent trace ID
            feedback: {               # For feedback assessments
              value: any              # Score/rating (number, string, object)
            }
            expectation: {            # For expectation assessments
              value: any              # Ground truth value
            }
            source: {                 # Assessment creator
              source_type: str        # "HUMAN", "LLM_JUDGE", "CODE"
              source_id: str          # email, model name, script path
            }
            rationale: str            # Explanation text
            metadata: dict            # Additional assessment metadata
            create_time: str          # ISO timestamp
            last_update_time: str     # ISO timestamp
            valid: bool               # Validity flag
          }
        ]
      data:                           # Execution details
        spans: [                      # Nested execution tree
          {
            trace_id: str             # Parent trace ID
            span_id: str              # Unique span identifier
            parent_span_id: str       # Parent in execution tree
            name: str                 # Operation name
            start_time_unix_nano: int # Start timestamp (nanoseconds)
            end_time_unix_nano: int   # End timestamp (nanoseconds)
            attributes: {             # Span metadata
              mlflow.spanInputs: str  # JSON string of inputs
              mlflow.spanOutputs: str # JSON string of outputs
              mlflow.spanType: str    # "AGENT", "CHAIN", "LLM", etc.
              mlflow.spanFunctionName: str
              mlflow.traceRequestId: str
              <custom_attributes>: str
            }
            events: [                 # Execution events/logs
              {
                time_unix_nano: int
                name: str
                attributes: dict
              }
            ]
            status: {                 # Execution status
              code: str               # "STATUS_CODE_OK", etc.
              message: str
            }
          }
        ]

    \b
    FIELD SELECTION:
    Use --fields with dot notation to select specific fields.

    \b
    Examples:
      info.trace_id                           # Single field
      info.assessments.*                      # All assessment data
      info.assessments.*.feedback.value       # Just feedback scores
      info.assessments.*.source.source_type   # Assessment sources
      info.trace_metadata.mlflow.traceInputs  # Original inputs
      info.trace_metadata.mlflow.source.type  # Source type
      info.tags.mlflow.traceName              # Trace name
      data.spans.*                            # All span data
      data.spans.*.name                       # Span operation names
      data.spans.*.attributes.mlflow.spanType # Span types
      data.spans.*.events.*.name              # Event names
      info.trace_id,info.state,info.execution_duration_ms  # Multiple fields
    """


@commands.command("search")
@EXPERIMENT_IDS
@click.option(
    "--filter-string",
    type=click.STRING,
    help="""Filter string for trace search.

Examples:
- Filter by run ID: "run_id = '123abc'"
- Filter by status: "status = 'OK'"
- Filter by timestamp: "timestamp_ms > 1700000000000"
- Filter by metadata: "metadata.`mlflow.modelId` = 'model123'"
- Filter by tags: "tags.environment = 'production'"
- Multiple conditions: "run_id = '123' AND status = 'OK'"

Available fields:
- run_id: Associated MLflow run ID
- status: Trace status (OK, ERROR, etc.)
- timestamp_ms: Trace timestamp in milliseconds
- execution_time_ms: Trace execution time in milliseconds
- name: Trace name
- metadata.<key>: Custom metadata fields (use backticks for keys with dots)
- tags.<key>: Custom tag fields""",
)
@click.option(
    "--max-results",
    type=click.INT,
    default=100,
    help="Maximum number of traces to return (default: 100)",
)
@click.option(
    "--order-by",
    type=click.STRING,
    help="Comma-separated list of fields to order by (e.g., 'timestamp_ms DESC, status')",
)
@click.option("--page-token", type=click.STRING, help="Token for pagination from previous search")
@click.option(
    "--run-id",
    type=click.STRING,
    help="Filter traces by run ID (convenience option, adds to filter-string)",
)
@click.option(
    "--include-spans/--no-include-spans",
    default=True,
    help="Include span data in results (default: include)",
)
@click.option("--model-id", type=click.STRING, help="Filter traces by model ID")
@click.option(
    "--sql-warehouse-id",
    type=click.STRING,
    help="SQL warehouse ID for searching inference tables (Databricks only)",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format: 'table' for formatted table (default) or 'json' for JSON format",
)
@click.option(
    "--fields",
    type=click.STRING,
    help="Filter and select specific fields using dot notation. "
    "Examples: 'info.trace_id', 'info.assessments.*', 'data.spans.*.name'. "
    "Comma-separated for multiple fields. "
    "Defaults to standard columns for table mode, all fields for JSON mode.",
)
def search_traces(
    experiment_ids,
    filter_string,
    max_results,
    order_by,
    page_token,
    run_id,
    include_spans,
    model_id,
    sql_warehouse_id,
    output,
    fields,
):
    """
    Search for traces in the specified experiments.

    Examples:

    \b
    # Search all traces in experiment 1
    mlflow traces search --experiment-ids 1

    \b
    # Search traces from multiple experiments
    mlflow traces search --experiment-ids 1,2,3 --max-results 50

    \b
    # Filter traces by run ID
    mlflow traces search --experiment-ids 1 --run-id abc123def

    \b
    # Use filter string for complex queries
    mlflow traces search --experiment-ids 1 \\
        --filter-string "attribute.run_id = 'abc123' AND timestamp_ms > 1700000000000"

    \b
    # Order results and use pagination
    mlflow traces search --experiment-ids 1 \\
        --order-by "timestamp_ms DESC" \\
        --max-results 10 \\
        --page-token <token_from_previous>

    \b
    # Search without span data (faster for metadata-only queries)
    mlflow traces search --experiment-ids 1 --no-include-spans
    """
    client = TracingClient()
    exp_ids = experiment_ids.split(",")
    order_by_list = order_by.split(",") if order_by else None

    traces = client.search_traces(
        experiment_ids=exp_ids,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by_list,
        page_token=page_token,
        run_id=run_id,
        include_spans=include_spans,
        model_id=model_id,
        sql_warehouse_id=sql_warehouse_id,
    )

    # Determine which fields to show
    if fields:
        field_list = [f.strip() for f in fields.split(",")]
        # Validate fields against actual trace data
        if traces:
            _validate_field_paths(field_list, traces[0].to_dict())
    elif output == "json":
        # JSON mode defaults to all fields (full trace data)
        field_list = None  # Will output full JSON
    else:
        # Table mode defaults to standard columns
        field_list = [
            "info.trace_id",
            "info.request_time",
            "info.state",
            "info.execution_duration_ms",
            "info.request_preview",
            "info.response_preview",
        ]

    if output == "json":
        if field_list is None:
            # Full JSON output
            result = {"traces": [trace.to_dict() for trace in traces], "token": traces.token}
        else:
            # Custom fields JSON output - filter original structure
            traces_data = []
            for trace in traces:
                trace_dict = trace.to_dict()
                filtered_trace = _filter_json_by_fields(trace_dict, field_list)
                traces_data.append(filtered_trace)
            result = {"traces": traces_data, "token": traces.token}
        click.echo(json.dumps(result, indent=2))
    else:
        # Table output format
        table = []
        for trace in traces:
            trace_dict = trace.to_dict()
            row = []

            for field in field_list:
                values = jsonpath_extract_values(trace_dict, field)

                if not values:
                    cell_value = "N/A"
                elif len(values) == 1:
                    cell_value = values[0]
                else:
                    # Multiple values - join them
                    cell_value = ", ".join(str(v) for v in values[:3])  # Limit to first 3
                    if len(values) > 3:
                        cell_value += f", ... (+{len(values) - 3} more)"

                # Format specific fields
                if field == "info.request_time" and cell_value != "N/A":
                    # Convert ISO timestamp to readable format
                    from datetime import datetime

                    try:
                        dt = datetime.fromisoformat(cell_value.replace("Z", "+00:00"))
                        cell_value = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                    except Exception:
                        pass  # Keep original if conversion fails
                elif (
                    field == "info.execution_duration_ms"
                    and cell_value != "N/A"
                    and cell_value is not None
                ):
                    if cell_value < 1000:
                        cell_value = f"{cell_value}ms"
                    else:
                        cell_value = f"{cell_value / 1000:.1f}s"
                elif field in ["info.request_preview", "info.response_preview"]:
                    # Truncate previews to keep table readable
                    if len(str(cell_value)) > 20:
                        cell_value = str(cell_value)[:17] + "..."

                row.append(str(cell_value))

            table.append(row)

        click.echo(_create_table(table, headers=field_list))

        if traces.token:
            click.echo(f"\nNext page token: {traces.token}")


@commands.command("get")
@TRACE_ID
def get_trace(trace_id):
    """
    All trace details will print to stdout as JSON format.

    \b
    Example:
    mlflow traces get --trace-id tr-1234567890abcdef
    """
    client = TracingClient()
    trace = client.get_trace(trace_id)
    json_trace = json.dumps(trace.to_dict(), indent=4)
    click.echo(json_trace)


@commands.command("delete")
@EXPERIMENT_IDS
@click.option("--trace-ids", type=click.STRING, help="Comma-separated list of trace IDs to delete")
@click.option(
    "--max-timestamp-millis",
    type=click.INT,
    help="Delete traces older than this timestamp (milliseconds since epoch)",
)
@click.option("--max-traces", type=click.INT, help="Maximum number of traces to delete")
def delete_traces(experiment_ids, trace_ids, max_timestamp_millis, max_traces):
    """
    Delete traces from experiments.

    Either --trace-ids or timestamp criteria can be specified, but not both.

    \b
    Examples:
    # Delete specific traces from one experiment
    mlflow traces delete --experiment-ids 1 --trace-ids tr-abc123,tr-def456

    \b
    # Delete traces older than a timestamp from multiple experiments
    mlflow traces delete --experiment-ids 1,2,3 --max-timestamp-millis 1700000000000

    \b
    # Delete up to 100 old traces per experiment
    mlflow traces delete --experiment-ids 1,2 --max-timestamp-millis 1700000000000 --max-traces 100
    """
    client = TracingClient()
    trace_id_list = trace_ids.split(",") if trace_ids else None
    exp_ids = experiment_ids.split(",")

    # Delete traces from each experiment
    total_count = 0
    for experiment_id in exp_ids:
        count = client.delete_traces(
            experiment_id=experiment_id,
            trace_ids=trace_id_list,
            max_timestamp_millis=max_timestamp_millis,
            max_traces=max_traces,
        )
        total_count += count
        if len(exp_ids) > 1:
            click.echo(f"Deleted {count} trace(s) from experiment {experiment_id}.")

    if len(exp_ids) == 1:
        click.echo(f"Deleted {total_count} trace(s) from experiment {exp_ids[0]}.")
    else:
        click.echo(f"Total: Deleted {total_count} trace(s) from {len(exp_ids)} experiment(s).")


@commands.command("set-tag")
@TRACE_ID
@click.option("--key", type=click.STRING, required=True, help="Tag key")
@click.option("--value", type=click.STRING, required=True, help="Tag value")
def set_tag(trace_id, key, value):
    """
    Set a tag on a trace.

    \b
    Example:
    mlflow traces set-tag --trace-id tr-abc123 --key environment --value production
    """
    client = TracingClient()
    client.set_trace_tag(trace_id, key, value)
    click.echo(f"Set tag '{key}' on trace {trace_id}.")


@commands.command("delete-tag")
@TRACE_ID
@click.option("--key", type=click.STRING, required=True, help="Tag key to delete")
def delete_tag(trace_id, key):
    """
    Delete a tag from a trace.

    \b
    Example:
    mlflow traces delete-tag --trace-id tr-abc123 --key environment
    """
    client = TracingClient()
    client.delete_trace_tag(trace_id, key)
    click.echo(f"Deleted tag '{key}' from trace {trace_id}.")


@commands.command("log-feedback")
@TRACE_ID
@click.option(
    "--name", type=click.STRING, default="feedback", help="Feedback name (default: 'feedback')"
)
@click.option(
    "--value",
    type=click.STRING,
    help="Feedback value (number, string, bool, or JSON for complex values)",
)
@click.option(
    "--source-type",
    type=click.Choice(["HUMAN", "LLM_JUDGE", "CODE"]),
    help="Source type of the feedback",
)
@click.option(
    "--source-id",
    type=click.STRING,
    help="Source identifier (e.g., email for HUMAN, model name for LLM)",
)
@click.option("--rationale", type=click.STRING, help="Explanation/justification for the feedback")
@click.option("--metadata", type=click.STRING, help="Additional metadata as JSON string")
@click.option("--span-id", type=click.STRING, help="Associate feedback with a specific span ID")
def log_feedback(trace_id, name, value, source_type, source_id, rationale, metadata, span_id):
    """
    Log feedback (evaluation score) to a trace.

    \b
    Examples:
    # Simple numeric feedback
    mlflow traces log-feedback --trace-id tr-abc123 \\
        --name relevance --value 0.9 \\
        --rationale "Highly relevant response"

    \b
    # Human feedback with source
    mlflow traces log-feedback --trace-id tr-abc123 \\
        --name quality --value good \\
        --source-type HUMAN --source-id reviewer@example.com

    \b
    # Complex feedback with JSON value and metadata
    mlflow traces log-feedback --trace-id tr-abc123 \\
        --name metrics \\
        --value '{"accuracy": 0.95, "f1": 0.88}' \\
        --metadata '{"model": "gpt-4", "temperature": 0.7}'

    \b
    # LLM judge feedback
    mlflow traces log-feedback --trace-id tr-abc123 \\
        --name faithfulness --value 0.85 \\
        --source-type LLM_JUDGE --source-id gpt-4 \\
        --rationale "Response is faithful to context"
    """
    # Parse value if it's JSON
    if value:
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # Keep as string

    # Parse metadata
    metadata_dict = json.loads(metadata) if metadata else None

    # Create source if provided
    source = None
    if source_type and source_id:
        # Map CLI choices to AssessmentSourceType constants
        source_type_value = getattr(AssessmentSourceType, source_type)
        source = AssessmentSource(
            source_type=source_type_value,
            source_id=source_id,
        )

    assessment = _log_feedback(
        trace_id=trace_id,
        name=name,
        value=value,
        source=source,
        rationale=rationale,
        metadata=metadata_dict,
        span_id=span_id,
    )
    click.echo(
        f"Logged feedback '{name}' to trace {trace_id}. Assessment ID: {assessment.assessment_id}"
    )


@commands.command("log-expectation")
@TRACE_ID
@click.option(
    "--name",
    type=click.STRING,
    required=True,
    help="Expectation name (e.g., 'expected_answer', 'ground_truth')",
)
@click.option(
    "--value",
    type=click.STRING,
    required=True,
    help="Expected value (string or JSON for complex values)",
)
@click.option(
    "--source-type",
    type=click.Choice(["HUMAN", "LLM_JUDGE", "CODE"]),
    help="Source type of the expectation",
)
@click.option("--source-id", type=click.STRING, help="Source identifier")
@click.option("--metadata", type=click.STRING, help="Additional metadata as JSON string")
@click.option("--span-id", type=click.STRING, help="Associate expectation with a specific span ID")
def log_expectation(trace_id, name, value, source_type, source_id, metadata, span_id):
    """
    Log an expectation (ground truth label) to a trace.

    \b
    Examples:
    # Simple expected answer
    mlflow traces log-expectation --trace-id tr-abc123 \\
        --name expected_answer --value "Paris"

    \b
    # Human-annotated ground truth
    mlflow traces log-expectation --trace-id tr-abc123 \\
        --name ground_truth --value "positive" \\
        --source-type HUMAN --source-id annotator@example.com

    \b
    # Complex expected output with metadata
    mlflow traces log-expectation --trace-id tr-abc123 \\
        --name expected_response \\
        --value '{"answer": "42", "confidence": 0.95}' \\
        --metadata '{"dataset": "test_set_v1", "difficulty": "hard"}'
    """
    # Parse value if it's JSON
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass  # Keep as string

    # Parse metadata
    metadata_dict = json.loads(metadata) if metadata else None

    # Create source if provided
    source = None
    if source_type and source_id:
        # Map CLI choices to AssessmentSourceType constants
        source_type_value = getattr(AssessmentSourceType, source_type)
        source = AssessmentSource(
            source_type=source_type_value,
            source_id=source_id,
        )

    assessment = _log_expectation(
        trace_id=trace_id,
        name=name,
        value=value,
        source=source,
        metadata=metadata_dict,
        span_id=span_id,
    )
    click.echo(
        f"Logged expectation '{name}' to trace {trace_id}. "
        f"Assessment ID: {assessment.assessment_id}"
    )


@commands.command("get-assessment")
@TRACE_ID
@click.option("--assessment-id", type=click.STRING, required=True, help="Assessment ID")
def get_assessment(trace_id, assessment_id):
    """
    Get assessment details as JSON.

    \b
    Example:
    mlflow traces get-assessment --trace-id tr-abc123 --assessment-id asmt-def456
    """
    client = TracingClient()
    assessment = client.get_assessment(trace_id, assessment_id)
    json_assessment = json.dumps(assessment.to_dictionary(), indent=4)
    click.echo(json_assessment)


@commands.command("update-assessment")
@TRACE_ID
@click.option("--assessment-id", type=click.STRING, required=True, help="Assessment ID to update")
@click.option("--name", type=click.STRING, help="Updated assessment name")
@click.option("--value", type=click.STRING, help="Updated assessment value (JSON)")
@click.option("--rationale", type=click.STRING, help="Updated rationale")
@click.option("--metadata", type=click.STRING, help="Updated metadata as JSON")
def update_assessment(trace_id, assessment_id, name, value, rationale, metadata):
    """
    Update an existing assessment.

    \b
    Example:
    mlflow traces update-assessment --trace-id tr-abc123 --assessment-id asmt-def456 \\
        --value '{"accuracy": 0.98}' --rationale "Updated after review"
    """
    client = TracingClient()

    # Get the existing assessment first
    existing = client.get_assessment(trace_id, assessment_id)

    # Parse value if provided
    parsed_value = value
    if value:
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            pass  # Keep as string

    # Parse metadata if provided
    parsed_metadata = metadata
    if metadata:
        parsed_metadata = json.loads(metadata)

    # Create updated assessment - determine if it's feedback or expectation
    if hasattr(existing, "rationale"):
        # It's feedback
        from mlflow.entities import Feedback

        updated_assessment = Feedback(
            name=name or existing.name,
            value=parsed_value if value else existing.value,
            rationale=rationale or existing.rationale,
            metadata=parsed_metadata if metadata else existing.metadata,
        )
    else:
        # It's expectation
        from mlflow.entities import Expectation

        updated_assessment = Expectation(
            name=name or existing.name,
            value=parsed_value if value else existing.value,
            metadata=parsed_metadata if metadata else existing.metadata,
        )

    client.update_assessment(trace_id, assessment_id, updated_assessment)
    click.echo(f"Updated assessment {assessment_id} in trace {trace_id}.")


@commands.command("delete-assessment")
@TRACE_ID
@click.option("--assessment-id", type=click.STRING, required=True, help="Assessment ID to delete")
def delete_assessment(trace_id, assessment_id):
    """
    Delete an assessment from a trace.

    \b
    Example:
    mlflow traces delete-assessment --trace-id tr-abc123 --assessment-id asmt-def456
    """
    client = TracingClient()
    client.delete_assessment(trace_id, assessment_id)
    click.echo(f"Deleted assessment {assessment_id} from trace {trace_id}.")
