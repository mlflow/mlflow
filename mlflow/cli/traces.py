"""
Comprehensive MLflow Traces CLI for managing trace data, assessments, and metadata.

This module provides a complete command-line interface for working with MLflow traces,
including search, retrieval, deletion, tagging, and assessment management. It supports
both table and JSON output formats with flexible field selection capabilities.

AVAILABLE COMMANDS:
    search              Search traces with filtering, sorting, and field selection
    get                 Retrieve detailed trace information as JSON
    delete              Delete traces by ID or timestamp criteria
    set-tag             Add tags to traces
    delete-tag          Remove tags from traces
    log-feedback        Log evaluation feedback/scores to traces
    log-expectation     Log ground truth expectations to traces
    get-assessment      Retrieve assessment details
    update-assessment   Modify existing assessments
    delete-assessment   Remove assessments from traces

EXAMPLE USAGE:
    # Search traces across multiple experiments
    mlflow traces search --experiment-ids 1,2,3 --max-results 50

    # Filter traces by status and timestamp
    mlflow traces search --experiment-ids 1 \
        --filter-string "status = 'OK' AND timestamp_ms > 1700000000000"

    # Get specific fields in JSON format
    mlflow traces search --experiment-ids 1 \
        --extract-fields "info.trace_id,info.assessments.*,data.spans.*.name" \
        --output json

    # Extract trace names (using backticks for dots in field names)
    mlflow traces search --experiment-ids 1 \
        --extract-fields "info.trace_id,info.tags.`mlflow.traceName`" \
        --output json

    # Get full trace details
    mlflow traces get --trace-id tr-1234567890abcdef

    # Log feedback to a trace
    mlflow traces log-feedback --trace-id tr-abc123 \
        --name relevance --value 0.9 \
        --source-type HUMAN --source-id reviewer@example.com \
        --rationale "Highly relevant response"

    # Delete old traces
    mlflow traces delete --experiment-ids 1 \
        --max-timestamp-millis 1700000000000 --max-traces 100

    # Add custom tags
    mlflow traces set-tag --trace-id tr-abc123 \
        --key environment --value production

ASSESSMENT TYPES:
    • Feedback: Evaluation scores, ratings, or judgments
    • Expectations: Ground truth labels or expected outputs
    • Sources: HUMAN, LLM_JUDGE, or CODE with source identification

For detailed help on any command, use:
    mlflow traces COMMAND --help
"""

import json
import os
import warnings

import click

from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.tracing.assessment import (
    log_expectation as _log_expectation,
)
from mlflow.tracing.assessment import (
    log_feedback as _log_feedback,
)
from mlflow.tracing.client import TracingClient
from mlflow.utils.jsonpath_utils import (
    filter_json_by_fields,
    jsonpath_extract_values,
    validate_field_paths,
)
from mlflow.utils.string_utils import _create_table, format_table_cell_value

# Define reusable options following mlflow/runs.py pattern
EXPERIMENT_ID = click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    required=True,
    help="Experiment ID to search within. Can be set via MLFLOW_EXPERIMENT_ID env var.",
)
TRACE_ID = click.option("--trace-id", type=click.STRING, required=True)


@click.group("traces")
def commands():
    """
    Manage traces. To manage traces associated with a tracking server, set the
    MLFLOW_TRACKING_URI environment variable to the URL of the desired server.

    TRACE SCHEMA:
    info.trace_id                           # Unique trace identifier
    info.experiment_id                      # MLflow experiment ID
    info.request_time                       # Request timestamp (milliseconds)
    info.execution_duration                 # Total execution time (milliseconds)
    info.state                              # Trace status: OK, ERROR, etc.
    info.client_request_id                  # Optional client-provided request ID
    info.request_preview                    # Truncated request preview
    info.response_preview                   # Truncated response preview
    info.trace_metadata.mlflow.*           # MLflow-specific metadata
    info.trace_metadata.*                  # Custom metadata fields
    info.tags.mlflow.traceName             # Trace name tag
    info.tags.<key>                         # Custom tags
    info.assessments.*.assessment_id        # Assessment identifiers
    info.assessments.*.feedback.name        # Feedback names
    info.assessments.*.feedback.value       # Feedback scores/values
    info.assessments.*.feedback.rationale   # Feedback explanations
    info.assessments.*.expectation.name     # Ground truth names
    info.assessments.*.expectation.value    # Expected values
    info.assessments.*.source.source_type   # HUMAN, LLM_JUDGE, CODE
    info.assessments.*.source.source_id     # Source identifier
    info.token_usage                        # Token usage (property, not searchable via fields)
    data.spans.*.span_id                    # Individual span IDs
    data.spans.*.name                       # Span operation names
    data.spans.*.parent_id                  # Parent span relationships
    data.spans.*.start_time                 # Span start timestamps
    data.spans.*.end_time                   # Span end timestamps
    data.spans.*.status_code                # Span status codes
    data.spans.*.attributes.mlflow.spanType # AGENT, TOOL, LLM, etc.
    data.spans.*.attributes.<key>           # Custom span attributes
    data.spans.*.events.*.name              # Event names
    data.spans.*.events.*.timestamp         # Event timestamps
    data.spans.*.events.*.attributes.<key>  # Event attributes

    For additional details, see:
    https://mlflow.org/docs/latest/genai/tracing/concepts/trace/#traceinfo-metadata-and-context

    \b
    FIELD SELECTION:
    Use --extract-fields with dot notation to select specific fields.

    \b
    Examples:
      info.trace_id                           # Single field
      info.assessments.*                      # All assessment data
      info.assessments.*.feedback.value       # Just feedback scores
      info.assessments.*.source.source_type   # Assessment sources
      info.trace_metadata.mlflow.traceInputs  # Original inputs
      info.trace_metadata.mlflow.source.type  # Source type
      info.tags.`mlflow.traceName`            # Trace name (backticks for dots)
      data.spans.*                            # All span data
      data.spans.*.name                       # Span operation names
      data.spans.*.attributes.mlflow.spanType # Span types
      data.spans.*.events.*.name              # Event names
      info.trace_id,info.state,info.execution_duration  # Multiple fields
    """


@commands.command("search")
@EXPERIMENT_ID
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
    help=(
        "DEPRECATED. Use the `MLFLOW_TRACING_SQL_WAREHOUSE_ID` environment variable instead."
        "SQL warehouse ID (only needed when searching for traces by model "
        "stored in Databricks Unity Catalog)"
    ),
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format: 'table' for formatted table (default) or 'json' for JSON format",
)
@click.option(
    "--extract-fields",
    type=click.STRING,
    help="Filter and select specific fields using dot notation. "
    'Examples: "info.trace_id", "info.assessments.*", "data.spans.*.name". '
    'For field names with dots, use backticks: "info.tags.`mlflow.traceName`". '
    "Comma-separated for multiple fields. "
    "Defaults to standard columns for table mode, all fields for JSON mode.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show all available fields in error messages when invalid fields are specified.",
)
def search_traces(
    experiment_id: str,
    filter_string: str | None = None,
    max_results: int = 100,
    order_by: str | None = None,
    page_token: str | None = None,
    run_id: str | None = None,
    include_spans: bool = True,
    model_id: str | None = None,
    sql_warehouse_id: str | None = None,
    output: str = "table",
    extract_fields: str | None = None,
    verbose: bool = False,
) -> None:
    """
    Search for traces in the specified experiment.

    Examples:

    \b
    # Search all traces in experiment 1
    mlflow traces search --experiment-id 1

    \b
    # Using environment variable
    export MLFLOW_EXPERIMENT_ID=1
    mlflow traces search --max-results 50

    \b
    # Filter traces by run ID
    mlflow traces search --experiment-id 1 --run-id abc123def

    \b
    # Use filter string for complex queries
    mlflow traces search --experiment-id 1 \\
        --filter-string "run_id = 'abc123' AND timestamp_ms > 1700000000000"

    \b
    # Order results and use pagination
    mlflow traces search --experiment-id 1 \\
        --order-by "timestamp_ms DESC" \\
        --max-results 10 \\
        --page-token <token_from_previous>

    \b
    # Search without span data (faster for metadata-only queries)
    mlflow traces search --experiment-id 1 --no-include-spans
    """
    client = TracingClient()
    order_by_list = order_by.split(",") if order_by else None

    # Set the sql_warehouse_id in the environment variable
    if sql_warehouse_id is not None:
        warnings.warn(
            "The `sql_warehouse_id` parameter is deprecated. Please use the "
            "`MLFLOW_TRACING_SQL_WAREHOUSE_ID` environment variable instead.",
            category=FutureWarning,
        )
        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = sql_warehouse_id

    traces = client.search_traces(
        locations=[experiment_id],
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by_list,
        page_token=page_token,
        run_id=run_id,
        include_spans=include_spans,
        model_id=model_id,
    )

    # Determine which fields to show
    if extract_fields:
        field_list = [f.strip() for f in extract_fields.split(",")]
        # Validate fields against actual trace data
        if traces:
            try:
                validate_field_paths(field_list, traces[0].to_dict(), verbose=verbose)
            except ValueError as e:
                raise click.UsageError(str(e))
    elif output == "json":
        # JSON mode defaults to all fields (full trace data)
        field_list = None  # Will output full JSON
    else:
        # Table mode defaults to standard columns
        field_list = [
            "info.trace_id",
            "info.request_time",
            "info.state",
            "info.execution_duration",
            "info.request_preview",
            "info.response_preview",
        ]

    if output == "json":
        if field_list is None:
            # Full JSON output
            result = {
                "traces": [trace.to_dict() for trace in traces],
                "next_page_token": traces.token,
            }
        else:
            # Custom fields JSON output - filter original structure
            traces_data = []
            for trace in traces:
                trace_dict = trace.to_dict()
                filtered_trace = filter_json_by_fields(trace_dict, field_list)
                traces_data.append(filtered_trace)
            result = {"traces": traces_data, "next_page_token": traces.token}
        click.echo(json.dumps(result, indent=2))
    else:
        # Table output format
        table = []
        for trace in traces:
            trace_dict = trace.to_dict()
            row = []

            for field in field_list:
                values = jsonpath_extract_values(trace_dict, field)
                cell_value = format_table_cell_value(field, None, values)
                row.append(cell_value)

            table.append(row)

        click.echo(_create_table(table, headers=field_list))

        if traces.token:
            click.echo(f"\nNext page token: {traces.token}")


@commands.command("get")
@TRACE_ID
@click.option(
    "--extract-fields",
    type=click.STRING,
    help="Filter and select specific fields using dot notation. "
    "Examples: 'info.trace_id', 'info.assessments.*', 'data.spans.*.name'. "
    "Comma-separated for multiple fields. "
    "If not specified, returns all trace data.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show all available fields in error messages when invalid fields are specified.",
)
def get_trace(
    trace_id: str,
    extract_fields: str | None = None,
    verbose: bool = False,
) -> None:
    """
    All trace details will print to stdout as JSON format.

    \b
    Examples:
    # Get full trace
    mlflow traces get --trace-id tr-1234567890abcdef

    \b
    # Get specific fields only
    mlflow traces get --trace-id tr-1234567890abcdef \\
        --extract-fields "info.trace_id,info.assessments.*,data.spans.*.name"
    """
    client = TracingClient()
    trace = client.get_trace(trace_id)
    trace_dict = trace.to_dict()

    if extract_fields:
        field_list = [f.strip() for f in extract_fields.split(",")]
        # Validate fields against trace data
        try:
            validate_field_paths(field_list, trace_dict, verbose=verbose)
        except ValueError as e:
            raise click.UsageError(str(e))
        # Filter to selected fields only
        filtered_trace = filter_json_by_fields(trace_dict, field_list)
        json_trace = json.dumps(filtered_trace, indent=2)
    else:
        # Return full trace
        json_trace = json.dumps(trace_dict, indent=2)

    click.echo(json_trace)


@commands.command("delete")
@EXPERIMENT_ID
@click.option("--trace-ids", type=click.STRING, help="Comma-separated list of trace IDs to delete")
@click.option(
    "--max-timestamp-millis",
    type=click.INT,
    help="Delete traces older than this timestamp (milliseconds since epoch)",
)
@click.option("--max-traces", type=click.INT, help="Maximum number of traces to delete")
def delete_traces(
    experiment_id: str,
    trace_ids: str | None = None,
    max_timestamp_millis: int | None = None,
    max_traces: int | None = None,
) -> None:
    """
    Delete traces from an experiment.

    Either --trace-ids or timestamp criteria can be specified, but not both.

    \b
    Examples:
    # Delete specific traces
    mlflow traces delete --experiment-id 1 --trace-ids tr-abc123,tr-def456

    \b
    # Delete traces older than a timestamp
    mlflow traces delete --experiment-id 1 --max-timestamp-millis 1700000000000

    \b
    # Delete up to 100 old traces
    mlflow traces delete --experiment-id 1 --max-timestamp-millis 1700000000000 --max-traces 100
    """
    client = TracingClient()
    trace_id_list = trace_ids.split(",") if trace_ids else None

    count = client.delete_traces(
        experiment_id=experiment_id,
        trace_ids=trace_id_list,
        max_timestamp_millis=max_timestamp_millis,
        max_traces=max_traces,
    )
    click.echo(f"Deleted {count} trace(s) from experiment {experiment_id}.")


@commands.command("set-tag")
@TRACE_ID
@click.option("--key", type=click.STRING, required=True, help="Tag key")
@click.option("--value", type=click.STRING, required=True, help="Tag value")
def set_trace_tag(trace_id: str, key: str, value: str) -> None:
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
def delete_trace_tag(trace_id: str, key: str) -> None:
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
@click.option("--name", type=click.STRING, required=True, help="Feedback name")
@click.option(
    "--value",
    type=click.STRING,
    help="Feedback value (number, string, bool, or JSON for complex values)",
)
@click.option(
    "--source-type",
    type=click.Choice(
        [AssessmentSourceType.HUMAN, AssessmentSourceType.LLM_JUDGE, AssessmentSourceType.CODE]
    ),
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
def log_feedback(
    trace_id: str,
    name: str,
    value: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    rationale: str | None = None,
    metadata: str | None = None,
    span_id: str | None = None,
) -> None:
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
    type=click.Choice(
        [AssessmentSourceType.HUMAN, AssessmentSourceType.LLM_JUDGE, AssessmentSourceType.CODE]
    ),
    help="Source type of the expectation",
)
@click.option("--source-id", type=click.STRING, help="Source identifier")
@click.option("--metadata", type=click.STRING, help="Additional metadata as JSON string")
@click.option("--span-id", type=click.STRING, help="Associate expectation with a specific span ID")
def log_expectation(
    trace_id: str,
    name: str,
    value: str,
    source_type: str | None = None,
    source_id: str | None = None,
    metadata: str | None = None,
    span_id: str | None = None,
) -> None:
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
def get_assessment(trace_id: str, assessment_id: str) -> None:
    """
    Get assessment details as JSON.

    \b
    Example:
    mlflow traces get-assessment --trace-id tr-abc123 --assessment-id asmt-def456
    """
    client = TracingClient()
    assessment = client.get_assessment(trace_id, assessment_id)
    json_assessment = json.dumps(assessment.to_dictionary(), indent=2)
    click.echo(json_assessment)


@commands.command("update-assessment")
@TRACE_ID
@click.option("--assessment-id", type=click.STRING, required=True, help="Assessment ID to update")
@click.option("--value", type=click.STRING, help="Updated assessment value (JSON)")
@click.option("--rationale", type=click.STRING, help="Updated rationale")
@click.option("--metadata", type=click.STRING, help="Updated metadata as JSON")
def update_assessment(
    trace_id: str,
    assessment_id: str,
    value: str | None = None,
    rationale: str | None = None,
    metadata: str | None = None,
) -> None:
    """
    Update an existing assessment.

    NOTE: Assessment names cannot be changed once set. Only value, rationale,
    and metadata can be updated.

    \b
    Examples:
    # Update feedback value and rationale
    mlflow traces update-assessment --trace-id tr-abc123 --assessment-id asmt-def456 \\
        --value '{"accuracy": 0.98}' --rationale "Updated after review"

    \b
    # Update only the rationale
    mlflow traces update-assessment --trace-id tr-abc123 --assessment-id asmt-def456 \\
        --rationale "Revised evaluation"
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
    if hasattr(existing, "feedback"):
        # It's feedback
        from mlflow.entities import Feedback

        updated_assessment = Feedback(
            name=existing.name,  # Always use existing name (cannot be changed)
            value=parsed_value if value else existing.value,
            rationale=rationale if rationale is not None else existing.rationale,
            metadata=parsed_metadata if metadata else existing.metadata,
        )
    else:
        # It's expectation
        from mlflow.entities import Expectation

        updated_assessment = Expectation(
            name=existing.name,  # Always use existing name (cannot be changed)
            value=parsed_value if value else existing.value,
            metadata=parsed_metadata if metadata else existing.metadata,
        )

    client.update_assessment(trace_id, assessment_id, updated_assessment)
    click.echo(f"Updated assessment {assessment_id} in trace {trace_id}.")


@commands.command("delete-assessment")
@TRACE_ID
@click.option("--assessment-id", type=click.STRING, required=True, help="Assessment ID to delete")
def delete_assessment(trace_id: str, assessment_id: str) -> None:
    """
    Delete an assessment from a trace.

    \b
    Example:
    mlflow traces delete-assessment --trace-id tr-abc123 --assessment-id asmt-def456
    """
    client = TracingClient()
    client.delete_assessment(trace_id, assessment_id)
    click.echo(f"Deleted assessment {assessment_id} from trace {trace_id}.")
