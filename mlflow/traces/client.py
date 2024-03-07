from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
from mlflow.traces.types import Trace

class TraceClient(ABC):
    @abstractmethod
    def log_trace(self, trace: Trace):
        pass


class DummyTraceClient(TraceClient):


    @dataclass
    class _Node:
        id: str
        name: str
        start_sec: float
        end_sec: float
        children: List["_Node"] = field(default_factory=list)

    def log_trace(self, trace: Trace):
        # Print Trace and spans nicely
        print(f"## Trace Info ##")
        print(f"Trace ID: {trace.trace_info.trace_id}")
        print(f"Trace Name: {trace.trace_info.trace_name}")
        print(f"Start Time: {trace.trace_info.start_time}")
        print(f"End Time: {trace.trace_info.end_time}")
        print()
        print(f"## Trace Data ##")
        root_node = self._recover_tree(trace)
        self._print_tree(root_node)

        print()
        print(f"## Spans ##")
        for span in trace.trace_data.spans:
            print("-----------------------------------")
            print(f"Span ID: {span.span_id}")
            print(f"Name: {span.name}")
            print(f"Start Time: {span.start_time}")
            print(f"End Time: {span.end_time}")
            print(f"Inputs: {span.inputs}")
            print(f"Outputs: {span.outputs}")
            print(f"Attributes: {span.attributes}")
        print("-----------------------------------")

    def _recover_tree(self, trace: Trace):
        # Recover the tree from the trace
        id_to_node = {}

        root_start_time = trace.trace_info.start_time
        for span in trace.trace_data.spans:
            id_to_node[span.span_id] = self._Node(
                span.span_id,
                span.name,
                start_sec=(span.start_time - root_start_time) / 1e9,
                end_sec=(span.end_time - root_start_time) / 1e9,
            )

        root_node = None
        for span in trace.trace_data.spans:
            if span.context.parent_span_id is None:
                root_node = id_to_node[span.span_id]
            else:
                id_to_node[span.context.parent_span_id].children.append(id_to_node[span.span_id])
        return root_node

    def _print_tree(self, node, prefix=""):
        start_line = f"[{node.start_sec: .2f} s] START {node.name} (id={node.id})"
        end_line = f"[{node.end_sec: .2f} s] END {node.name}"

        print(prefix + start_line)
        for child in node.children:
            print(prefix + "  |  ")
            self._print_tree(child, prefix + "  |  ")
            print(prefix + "  |  ")

        if not node.children:
            print(prefix + "  |  ")

        print(prefix + end_line)
