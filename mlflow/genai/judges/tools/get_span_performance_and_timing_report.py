"""
Get span timing report tool for MLflow traces.

This tool generates a timing report showing span latencies, execution order,
and concurrency patterns for performance analysis.
"""

from collections import defaultdict
from dataclasses import dataclass

from mlflow.entities.span import Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema
from mlflow.utils.annotations import experimental


@dataclass
class SpanTimingData:
    """Timing data for a single span."""

    span_id: str
    name: str
    span_type: str
    total_duration_s: float
    self_duration_s: float
    child_duration_s: float
    span_number: str
    parent_number: str | None
    ancestors: list[str]
    depth: int


@dataclass
class ConcurrentPair:
    """Information about concurrent span execution."""

    span1_num: str
    span2_num: str
    span1_name: str
    span2_name: str
    overlap_s: float


@experimental(version="3.5.0")
class GetSpanPerformanceAndTimingReportTool(JudgeTool):
    """
    A tool that generates a span timing report for a trace.

    The report includes span timing hierarchy, summary statistics,
    longest-running spans, and concurrent operations detection.
    """

    MAX_NAME_LENGTH = 30
    MIN_OVERLAP_THRESHOLD_S = 0.01
    TOP_SPANS_COUNT = 10
    MAX_CONCURRENT_PAIRS = 20

    @property
    def name(self) -> str:
        """Return the name of this tool.

        Returns:
            The tool name constant for the span timing report tool.
        """
        return ToolNames.GET_SPAN_PERFORMANCE_AND_TIMING_REPORT

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for LiteLLM/OpenAI function calling.

        Returns:
            ToolDefinition object containing the tool specification.
        """
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.GET_SPAN_PERFORMANCE_AND_TIMING_REPORT,
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
        """Generate span timing report for the trace.

        Args:
            trace: The MLflow trace object to analyze.

        Returns:
            Formatted timing report as a string.
        """
        if not trace or not trace.data or not trace.data.spans:
            return "No spans found in trace"

        spans = trace.data.spans
        trace_info = trace.info

        timing_data = self._calculate_timing_data(spans)
        concurrent_pairs = self._find_concurrent_operations(spans)
        type_summary = self._calculate_type_summary(spans)

        return self._format_report(
            trace_info=trace_info,
            timing_data=timing_data,
            concurrent_pairs=concurrent_pairs,
            type_summary=type_summary,
        )

    def _calculate_timing_data(self, spans: list[Span]) -> dict[str, SpanTimingData]:
        """Calculate timing data for all spans.

        Args:
            spans: List of spans from the trace.

        Returns:
            Dictionary mapping span IDs to their timing data.
        """
        children_by_parent = defaultdict(list)
        for span in spans:
            children_by_parent[span.parent_id].append(span)

        for parent_spans in children_by_parent.values():
            parent_spans.sort(key=lambda s: s.start_time_ns)

        self_durations = self._calculate_self_durations(spans, children_by_parent)

        timing_data = {}
        span_counter = [0]

        def process_span_tree(
            span_id: str | None, ancestors: list[str] | None = None, depth: int = 0
        ) -> None:
            """Recursively traverse and process the span tree.

            Args:
                span_id: ID of the current span being processed.
                ancestors: List of ancestor span numbers for hierarchy tracking.
                depth: Current depth in the span tree.
            """
            ancestors = ancestors or []

            for span in children_by_parent.get(span_id, []):
                span_counter[0] += 1
                span_num = f"s{span_counter[0]}"

                total_dur_s = (span.end_time_ns - span.start_time_ns) / 1_000_000_000
                self_dur_s = self_durations[span.span_id]
                child_dur_s = total_dur_s - self_dur_s

                parent_num = (
                    timing_data.get(
                        span.parent_id,
                        SpanTimingData(
                            span_id="",
                            name="",
                            span_type="",
                            total_duration_s=0,
                            self_duration_s=0,
                            child_duration_s=0,
                            span_number="",
                            parent_number=None,
                            ancestors=[],
                            depth=0,
                        ),
                    ).span_number
                    or None
                )

                timing_data[span.span_id] = SpanTimingData(
                    span_id=span.span_id,
                    name=span.name,
                    span_type=span.span_type or "UNKNOWN",
                    total_duration_s=total_dur_s,
                    self_duration_s=self_dur_s,
                    child_duration_s=child_dur_s,
                    span_number=span_num,
                    parent_number=parent_num,
                    ancestors=ancestors.copy(),
                    depth=depth,
                )

                process_span_tree(span.span_id, ancestors + [span_num], depth + 1)

        process_span_tree(None)
        return timing_data

    def _calculate_self_durations(
        self, spans: list[Span], children_by_parent: dict[str | None, list[Span]]
    ) -> dict[str, float]:
        """Calculate self duration for each span (total minus children).

        Args:
            spans: List of all spans in the trace.
            children_by_parent: Dictionary mapping parent IDs to their child spans.

        Returns:
            Dictionary mapping span IDs to their self durations in seconds.
        """
        self_durations = {}

        for span in spans:
            total_dur_ns = span.end_time_ns - span.start_time_ns
            children = children_by_parent.get(span.span_id, [])

            if not children:
                self_durations[span.span_id] = total_dur_ns / 1_000_000_000
                continue

            intervals = [(child.start_time_ns, child.end_time_ns) for child in children]
            merged_intervals = self._merge_intervals(intervals)

            children_dur_ns = sum(end - start for start, end in merged_intervals)
            self_durations[span.span_id] = (total_dur_ns - children_dur_ns) / 1_000_000_000

        return self_durations

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge overlapping time intervals.

        Args:
            intervals: List of (start, end) time intervals in nanoseconds.

        Returns:
            List of merged non-overlapping intervals.
        """
        if not intervals:
            return []

        intervals.sort()
        merged = [intervals[0]]

        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        return merged

    def _find_concurrent_operations(self, spans: list[Span]) -> list[ConcurrentPair]:
        """Find spans that execute concurrently.

        Args:
            spans: List of all spans to analyze for concurrency.

        Returns:
            List of concurrent span pairs with overlap information.
        """
        concurrent_pairs = []

        for i, span1 in enumerate(spans):
            for span2 in spans[i + 1 :]:
                if span1.parent_id != span2.parent_id:
                    continue

                overlap_start = max(span1.start_time_ns, span2.start_time_ns)
                overlap_end = min(span1.end_time_ns, span2.end_time_ns)

                if overlap_start >= overlap_end:
                    continue

                overlap_s = (overlap_end - overlap_start) / 1_000_000_000

                if overlap_s > self.MIN_OVERLAP_THRESHOLD_S:
                    concurrent_pairs.append(
                        ConcurrentPair(
                            span1_num="",
                            span2_num="",
                            span1_name=self._truncate_name(span1.name),
                            span2_name=self._truncate_name(span2.name),
                            overlap_s=overlap_s,
                        )
                    )

                    if len(concurrent_pairs) >= self.MAX_CONCURRENT_PAIRS:
                        return concurrent_pairs

        return concurrent_pairs

    def _calculate_type_summary(self, spans: list[Span]) -> dict[str, tuple[int, float]]:
        """Calculate summary statistics by span type.

        Args:
            spans: List of spans to summarize.

        Returns:
            Dictionary mapping span types to (count, total_duration) tuples.
        """
        type_stats = defaultdict(lambda: [0, 0.0])

        for span in spans:
            span_type = span.span_type or "UNKNOWN"
            duration_s = (span.end_time_ns - span.start_time_ns) / 1_000_000_000
            type_stats[span_type][0] += 1
            type_stats[span_type][1] += duration_s

        return {k: tuple(v) for k, v in type_stats.items()}

    def _truncate_name(self, name: str) -> str:
        """Truncate long names for display.

        Args:
            name: The span name to potentially truncate.

        Returns:
            Truncated name if it exceeds MAX_NAME_LENGTH, otherwise original name.
        """
        if len(name) <= self.MAX_NAME_LENGTH:
            return name
        return name[: self.MAX_NAME_LENGTH - 3] + "..."

    def _format_report(
        self,
        trace_info: TraceInfo,
        timing_data: dict[str, SpanTimingData],
        concurrent_pairs: list[ConcurrentPair],
        type_summary: dict[str, tuple[int, float]],
    ) -> str:
        """Format the complete timing report.

        Args:
            trace_info: Trace metadata information.
            timing_data: Calculated timing data for all spans.
            concurrent_pairs: List of concurrent span pairs.
            type_summary: Summary statistics by span type.

        Returns:
            Formatted report as a string.
        """
        lines = []

        self._add_header(lines, trace_info, len(timing_data))
        self._add_column_definitions(lines)
        self._add_span_table(lines, timing_data)
        self._add_type_summary(lines, type_summary)
        self._add_top_spans(lines, timing_data)
        self._add_concurrent_operations(lines, concurrent_pairs, timing_data)

        return "\n".join(lines)

    def _add_header(self, lines: list[str], trace_info: TraceInfo, span_count: int) -> None:
        """Add report header.

        Args:
            lines: List to append header lines to.
            trace_info: Trace metadata for header information.
            span_count: Total number of spans in the trace.
        """
        lines.extend(
            [
                f"SPAN TIMING REPORT FOR TRACE: {trace_info.trace_id}",
                f"Total Duration: {trace_info.execution_duration / 1000:.2f}s",
                f"Total Spans: {span_count}",
                "",
            ]
        )

    def _add_column_definitions(self, lines: list[str]) -> None:
        """Add column definitions section.

        Args:
            lines: List to append column definition lines to.
        """
        lines.extend(
            [
                "COLUMN DEFINITIONS:",
                "  self_dur:  Time spent in this span excluding its children (actual work)",
                "  total_dur: Total time from span start to end (includes waiting for children)",
                "  child_dur: Time spent waiting for child spans to complete",
                "  parent:    The immediate parent span number",
                "  ancestors: Complete chain from root to parent",
                "",
            ]
        )

    def _add_span_table(self, lines: list[str], timing_data: dict[str, SpanTimingData]) -> None:
        """Add the main span timing table.

        Args:
            lines: List to append table lines to.
            timing_data: Timing data for all spans to display.
        """
        lines.extend(
            [
                "SPAN TABLE:",
                "-" * 200,
                f"{'span_num':<8} {'span_id':<20} {'name':<30} "
                f"{'type':<12} {'self_dur':>9} {'total_dur':>10} {'child_dur':>10} "
                f"{'parent':<8} {'ancestors':<60}",
                "-" * 200,
            ]
        )

        sorted_data = sorted(
            timing_data.values(), key=lambda x: int(x.span_number[1:]) if x.span_number else 0
        )

        for data in sorted_data:
            if not data.span_number:
                continue

            name = self._truncate_name(data.name)
            parent = data.parent_number or "-"
            ancestors_str = "→".join(data.ancestors) if data.ancestors else "root"

            lines.append(
                f"{data.span_number:<8} {data.span_id:<20} {name:<30} "
                f"{data.span_type:<12} {data.self_duration_s:>9.3f} "
                f"{data.total_duration_s:>10.3f} {data.child_duration_s:>10.3f} "
                f"{parent:<8} {ancestors_str:<60}"
            )

    def _add_type_summary(
        self, lines: list[str], type_summary: dict[str, tuple[int, float]]
    ) -> None:
        """Add summary by span type.

        Args:
            lines: List to append summary lines to.
            type_summary: Summary statistics organized by span type.
        """
        lines.extend(
            [
                "",
                "SUMMARY BY TYPE:",
                "-" * 80,
                f"{'type':<20} {'count':>8} {'total_dur':>12} {'avg_dur':>12}",
                "-" * 80,
            ]
        )

        for span_type in sorted(type_summary.keys()):
            count, total_dur = type_summary[span_type]
            avg_dur = total_dur / count
            lines.append(f"{span_type:<20} {count:>8} {total_dur:>12.3f}s {avg_dur:>12.3f}s")

    def _add_top_spans(self, lines: list[str], timing_data: dict[str, SpanTimingData]) -> None:
        """Add top spans by self duration.

        Args:
            lines: List to append top spans section to.
            timing_data: Timing data for all spans to rank.
        """
        lines.extend(
            [
                "",
                "TOP 10 SPANS BY SELF DURATION (actual work, not including children):",
                "-" * 110,
                f"{'rank':<6} {'span_num':<10} {'span_id':<20} {'name':<30} "
                f"{'type':<12} {'self_dur':>12}",
                "-" * 110,
            ]
        )

        sorted_spans = sorted(timing_data.values(), key=lambda x: x.self_duration_s, reverse=True)[
            : self.TOP_SPANS_COUNT
        ]

        for i, data in enumerate(sorted_spans):
            name = self._truncate_name(data.name)
            lines.append(
                f"{i + 1:<6} {data.span_number:<10} {data.span_id:<20} {name:<30} "
                f"{data.span_type:<12} {data.self_duration_s:>12.3f}s"
            )

    def _add_concurrent_operations(
        self,
        lines: list[str],
        concurrent_pairs: list[ConcurrentPair],
        timing_data: dict[str, SpanTimingData],
    ) -> None:
        """Add concurrent operations section.

        Args:
            lines: List to append concurrent operations section to.
            concurrent_pairs: List of detected concurrent span pairs.
            timing_data: Timing data (currently unused but kept for consistency).
        """
        lines.extend(
            [
                "",
                "CONCURRENT OPERATIONS:",
                "-" * 100,
            ]
        )

        if not concurrent_pairs:
            lines.append("No significant concurrent operations detected.")
            return

        lines.extend(
            [
                f"{'span1':<10} {'span2':<10} {'name1':<30} {'name2':<30} {'overlap':>10}",
                "-" * 100,
            ]
        )

        lines.extend(
            f"{pair.span1_num:<10} {pair.span2_num:<10} "
            f"{pair.span1_name:<30} {pair.span2_name:<30} "
            f"{pair.overlap_s:>10.3f}s"
            for pair in concurrent_pairs
        )
