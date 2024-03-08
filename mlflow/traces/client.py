from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
from mlflow.traces.types import Span, Trace

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

class DummyTraceClientWithHTMLDisplay(TraceClient):
    @dataclass
    class _Node:
        span: Span
        children: List["_Node"] = field(default_factory=list)

    def log_trace(self, trace: Trace):
        from IPython.display import display, HTML

        root_node = self._recover_tree(trace)

        html = self._generate_html_with_interaction_and_style(root_node)
        display(HTML(html))


    def _recover_tree(self, trace: Trace):
        # Recover the tree from the trace
        id_to_node = {}

        root_start_time = trace.trace_info.start_time
        for span in trace.trace_data.spans:
            _span = Span(**span.__dict__)
            _span.start_time = (span.start_time - root_start_time) / 1e9
            _span.end_time = (span.end_time - root_start_time) / 1e9
            id_to_node[_span.span_id] = self._Node(span=_span) 

        root_node = None
        for span in trace.trace_data.spans:
            if span.context.parent_span_id is None:
                root_node = id_to_node[span.span_id]
            else:
                id_to_node[span.context.parent_span_id].children.append(id_to_node[span.span_id])
        return root_node

    def _generate_html_with_interaction_and_style(self, root_node: _Node):
        # Initial CSS and JavaScript for styling and functionality
        html_content = """
        <style>
            .span-event {
                cursor: pointer;
                padding: 10px;
                margin: 5px 0;
                border: 1px solid #ddd;
                border-radius: 8px;
                transition: background-color 0.3s ease;
            }

            .metadata {
                display: none;
                padding: 5px;
                margin-top: 5px;
                background-color: #F9F9F9;
                border-left: 2px solid #ccc;
            }
            .expanded .metadata {
                display: block;
            }
        </style>
        <script>
            function toggleExpand(event) {
                event.stopPropagation();
                event.currentTarget.classList.toggle('expanded');
            }
        </script>
        """
        # Function to recursively generate HTML for each span event with depth-based indentation
        def _generate_span_html(node, depth=0):
            # Calculate indentation based on depth
            indent = depth * 40  # Adjust multiplier as needed for visual effect
            span = node.span
            color = self._generate_color(node.span.name)
            span_html = f"""
            <div class="span-event" onclick="toggleExpand(event)" style="margin-left: {indent}px; background-color: rgba({color}, 0.2);">
                <b>[{span.name}]</b>  Start Time: {span.start_time:.2f} s, End Time: {span.end_time:.2f} s
                <div class="metadata">
                    <p><b>Trace ID</b>: {span.context.trace_id}</p>
                    <p><b>Span ID</b>: {span.span_id}</p>
                    <p><b>Inputs</b>: {span.inputs}</p>
                    <p><b>Outputs</b>: {span.outputs}</p>
                    <p><b>Attributes</b>: {span.attributes}</p>
                </div>
            </div>
            """
            for child in node.children:
                span_html += _generate_span_html(child, depth + 1)
            return span_html

        html_content += _generate_span_html(root_node)
        return html_content

    def _generate_color(self, span_name: str):
        import colorsys
        import hashlib
        import random
        seed = sum(ord(char) for char in span_name)
        random.seed(seed)

        # Generate a random hue
        hue = random.randint(0, 359)

        r, g, b = colorsys.hsv_to_rgb(hue/360.0, 0.8, 1.0)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return f"{r}, {g}, {b}"
