"""
Get span timing report tool for MLflow traces.

This tool generates a detailed timing report showing span latencies, execution order,
and concurrency.
"""

from collections import defaultdict

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class GetSpanTimingReportTool(JudgeTool):
    """
    A tool that generates a span timing report for a trace.

    The report includes:
    - Span timing table with hierarchy
    - Summary statistics by span type
    - Top 10 longest-running spans
    - Concurrent operations detection
    """

    @property
    def name(self) -> str:
        """Return the name of this tool."""
        return ToolNames.GET_SPAN_TIMING_REPORT

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for LiteLLM/OpenAI function calling."""
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.GET_SPAN_TIMING_REPORT,
                description=(
                    "Generate a comprehensive span timing report for the trace, showing "
                    "latencies, execution order, hierarchy, duration statistics, longest "
                    "spans, and concurrent operations. Useful for analyzing system "
                    "performance and identifying bottlenecks."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={},
                    required=[],
                ),
            ),
            type="function",
        )

    def invoke(self, trace: Trace) -> str:
        """
        Generate span timing report for the trace.

        Args:
            trace: The trace to analyze

        Returns:
            Formatted timing report as a string
        """
        if not trace or not trace.data or not trace.data.spans:
            return "No spans found in trace"

        spans = trace.data.spans
        trace_info = trace.info

        # Build parent-child relationships
        children_by_parent = defaultdict(list)
        span_to_number = {}  # Map span_id to s1, s2, etc.
        span_self_duration = {}  # Map span_id to self duration in seconds
        span_ancestors = {}  # Map span_id to list of ancestor span numbers

        for span in spans:
            children_by_parent[span.parent_id].append(span)

        # Sort children by start time
        for parent_spans in children_by_parent.values():
            parent_spans.sort(key=lambda s: s.start_time_ns)

        # Calculate self duration for each span
        for span in spans:
            total_dur_s = (span.end_time_ns - span.start_time_ns) / 1_000_000_000

            # Calculate actual time covered by children (accounting for overlaps)
            children = children_by_parent.get(span.span_id, [])
            if children:
                # Create intervals for all children
                intervals = []
                for child in children:
                    intervals.append((child.start_time_ns, child.end_time_ns))

                # Merge overlapping intervals
                intervals.sort()
                merged = []
                for start, end in intervals:
                    if merged and start <= merged[-1][1]:
                        # Overlapping interval, merge
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        # Non-overlapping interval
                        merged.append((start, end))

                # Calculate total time covered by children
                children_dur_s = 0
                for start, end in merged:
                    children_dur_s += (end - start) / 1_000_000_000
            else:
                children_dur_s = 0

            # Self duration is total minus actual time covered by children
            self_dur_s = total_dur_s - children_dur_s
            span_self_duration[span.span_id] = self_dur_s

        # Build output
        lines = []

        # Header
        lines.append(f"SPAN TIMING REPORT FOR TRACE: {trace_info.trace_id}")
        lines.append(f"Total Duration: {trace_info.execution_duration / 1000:.2f}s")
        lines.append(f"Total Spans: {len(spans)}")
        lines.append("")

        # Column definitions
        lines.append("COLUMN DEFINITIONS:")
        lines.append("  self_dur:  Time spent in this span excluding its children (actual work)")
        lines.append(
            "  total_dur: Total time from span start to end (includes waiting for children)"
        )
        lines.append("  child_dur: Time spent waiting for child spans to complete")
        lines.append("  parent:    The immediate parent span number (e.g., s5 is parent of s10)")
        lines.append("  ancestors: Complete chain from root to parent (e.g., s1→s2→s4→s5 for s10)")
        lines.append("")

        # Span table
        lines.append("SPAN TABLE:")
        lines.append("-" * 200)
        lines.append(
            f"{'span_num':<8} {'span_id':<20} {'name':<30} "
            f"{'type':<12} {'self_dur':>9} {'total_dur':>10} {'child_dur':>10} "
            f"{'parent':<8} {'ancestors':<60}"
        )
        lines.append("-" * 200)

        span_counter = [0]

        def traverse_span(span_id, ancestors=None):
            """Recursively traverse span tree."""
            if ancestors is None:
                ancestors = []

            child_spans = children_by_parent.get(span_id, [])

            for span in child_spans:
                span_counter[0] += 1
                span_num = f"s{span_counter[0]}"
                span_to_number[span.span_id] = span_num

                # Store ancestors for this span
                span_ancestors[span.span_id] = ancestors.copy()

                # Get durations
                total_dur_s = (span.end_time_ns - span.start_time_ns) / 1_000_000_000
                self_dur_s = span_self_duration[span.span_id]
                child_dur_s = total_dur_s - self_dur_s

                # Get parent number
                parent_num = span_to_number.get(span.parent_id, "-") if span.parent_id else "-"

                # Format ancestors string
                ancestors_str = "→".join(ancestors) if ancestors else "root"

                # Format name - truncate if too long
                name = span.name[:27] + "..." if len(span.name) > 30 else span.name

                # Write row
                span_type = span.span_type or "UNKNOWN"
                lines.append(
                    f"{span_num:<8} {span.span_id:<20} {name:<30} "
                    f"{span_type:<12} {self_dur_s:>9.3f} {total_dur_s:>10.3f} "
                    f"{child_dur_s:>10.3f} {parent_num:<8} {ancestors_str:<60}"
                )

                # Traverse children with updated ancestors
                traverse_span(span.span_id, ancestors + [span_num])

        # Start from root spans
        traverse_span(None)

        # Summary by type
        lines.append("")
        lines.append("SUMMARY BY TYPE:")
        lines.append("-" * 80)
        lines.append(f"{'type':<20} {'count':>8} {'total_dur':>12} {'avg_dur':>12}")
        lines.append("-" * 80)

        span_types = defaultdict(int)
        total_duration_by_type = defaultdict(float)

        for span in spans:
            span_type = span.span_type or "UNKNOWN"
            span_types[span_type] += 1
            total_duration_by_type[span_type] += (
                span.end_time_ns - span.start_time_ns
            ) / 1_000_000_000

        for span_type in sorted(span_types.keys()):
            count = span_types[span_type]
            total_dur = total_duration_by_type[span_type]
            avg_dur = total_dur / count
            lines.append(f"{span_type:<20} {count:>8} {total_dur:>12.3f}s {avg_dur:>12.3f}s")

        # Top 10 spans by self duration
        lines.append("")
        lines.append("TOP 10 SPANS BY SELF DURATION (actual work, not including children):")
        lines.append("-" * 110)
        lines.append(
            f"{'rank':<6} {'span_num':<10} {'span_id':<20} {'name':<30} "
            f"{'type':<12} {'self_dur':>12}"
        )
        lines.append("-" * 110)

        # Sort by self duration
        sorted_spans = sorted(spans, key=lambda s: span_self_duration[s.span_id], reverse=True)
        for i, span in enumerate(sorted_spans[:10]):
            span_num = span_to_number.get(span.span_id, "?")
            name = span.name[:27] + "..." if len(span.name) > 30 else span.name
            self_dur_s = span_self_duration[span.span_id]
            span_type = span.span_type or "UNKNOWN"
            lines.append(
                f"{i + 1:<6} {span_num:<10} {span.span_id:<20} {name:<30} "
                f"{span_type:<12} {self_dur_s:>12.3f}s"
            )

        # Detect concurrent operations
        lines.append("")
        lines.append("CONCURRENT OPERATIONS:")
        lines.append("-" * 100)

        concurrent_pairs = []

        for i, span1 in enumerate(spans):
            for span2 in spans[i + 1 :]:
                # Check if spans overlap and are siblings (same parent)
                if (
                    span1.start_time_ns < span2.end_time_ns
                    and span2.start_time_ns < span1.end_time_ns
                    and span1.parent_id == span2.parent_id
                ):
                    overlap_start = max(span1.start_time_ns, span2.start_time_ns)
                    overlap_end = min(span1.end_time_ns, span2.end_time_ns)
                    overlap_s = (overlap_end - overlap_start) / 1_000_000_000

                    if overlap_s > 0.01:  # Only show significant overlaps (>10ms)
                        concurrent_pairs.append((span1, span2, overlap_s))
                        break  # Only show first concurrent pair for each span

        if concurrent_pairs:
            lines.append(f"{'span1':<10} {'span2':<10} {'name1':<30} {'name2':<30} {'overlap':>10}")
            lines.append("-" * 100)

            for span1, span2, overlap_s in concurrent_pairs[:20]:  # Limit to 20 pairs
                num1 = span_to_number.get(span1.span_id, "?")
                num2 = span_to_number.get(span2.span_id, "?")
                name1 = span1.name[:27] + "..." if len(span1.name) > 30 else span1.name
                name2 = span2.name[:27] + "..." if len(span2.name) > 30 else span2.name
                lines.append(f"{num1:<10} {num2:<10} {name1:<30} {name2:<30} {overlap_s:>10.3f}s")
        else:
            lines.append("No significant concurrent operations detected.")

        return "\n".join(lines)
