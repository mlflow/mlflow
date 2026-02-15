"""Quick script to test Claude Code tracing on a real transcript file.

Usage:
    uv run python dev/test_claude_code_trace.py <path_to_transcript.jsonl>
    uv run python dev/test_claude_code_trace.py --debug <path_to_transcript.jsonl>

Example:
    uv run python dev/test_claude_code_trace.py ~/.claude/projects/.../transcript.jsonl
"""

import json
import sys

import mlflow

from mlflow.claude_code.tracing import process_transcript, read_transcript, setup_mlflow

debug = "--debug" in sys.argv
args = [a for a in sys.argv[1:] if a != "--debug"]

if not args:
    print(__doc__)
    sys.exit(1)

transcript_path = args[0]

if debug:
    transcript = read_transcript(transcript_path)
    print(f"Total entries: {len(transcript)}")
    for i, entry in enumerate(transcript):
        entry_type = entry.get("type", "?")
        msg = entry.get("message", {})
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [f"{p.get('type', '?')}" for p in content if isinstance(p, dict)]
            content_summary = f"[{', '.join(parts)}]"
        elif isinstance(content, str):
            content_summary = content[:80]
        else:
            content_summary = str(content)[:80]
        print(f"  [{i}] type={entry_type} role={role} content={content_summary}")
    print()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("claude-code-tracing-test")
setup_mlflow()

print(f"Processing: {transcript_path}")
trace = process_transcript(transcript_path, "test-session")

if trace:
    print(f"Trace ID: {trace.info.trace_id}")
    print(f"Spans: {len(trace.data.spans)}")

    # Build children map for proper tree traversal
    children_map: dict[str | None, list] = {}
    for span in trace.data.spans:
        children_map.setdefault(span.parent_id, []).append(span)

    def print_tree(parent_id, depth=0):
        for span in children_map.get(parent_id, []):
            indent = "  " * (depth + 1)
            input_msgs = span.inputs.get("messages", []) if span.inputs else []
            print(f"{indent}{span.name} ({span.span_type}) input_messages={len(input_msgs)}")
            print_tree(span.span_id, depth + 1)

    # Find root spans (no parent or parent not in trace)
    span_ids = {s.span_id for s in trace.data.spans}
    root_parents = {s.parent_id for s in trace.data.spans if s.parent_id not in span_ids}
    for root_parent in root_parents:
        print_tree(root_parent)
else:
    print("Failed to create trace")
