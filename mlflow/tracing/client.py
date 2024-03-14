import colorsys
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from mlflow.tracing.types.model import Span, Trace


def get_trace_client():
    # TODO: There will be a real implementation of the trace client
    #  E.g. https://github.com/B-Step62/mlflow/blob/trace-api-poc/mlflow/traces/client.py
    return DummyTraceClientWithHTMLDisplay()


class TraceClient(ABC):
    @abstractmethod
    def log_trace(self, trace: Trace):
        pass


class NoOpClient(TraceClient):
    def log_trace(self, trace: Trace):
        pass


class DummyTraceClientWithHTMLDisplay(TraceClient):
    """
    This is a toy client implementation purely for demonstration purposes. It simply
    logs the trace to the notebook cell output as an HTML representation. This should
    only be used for the very first iteration of the API/schema design, and should not
    be carried over to the UI feedback phase. At that point, we will have a real client
    that has much better UI and other features e.g. queuing traces.
    """

    @dataclass
    class _Node:
        span: Span
        children: List = field(default_factory=list)

    def log_trace(self, trace: Trace):
        from IPython.display import HTML, display

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
            id_to_node[_span.context.span_id] = self._Node(span=_span)

        root_node = None
        for span in trace.trace_data.spans:
            if span.parent_span_id is None:
                root_node = id_to_node[span.context.span_id]
            else:
                id_to_node[span.parent_span_id].children.append(id_to_node[span.context.span_id])
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

            pre {
                white-space: pre-wrap;
            }
        </style>
        <script>
            function toggleExpand(event) {
                event.stopPropagation();
                event.currentTarget.classList.toggle('expanded');
            }
        </script>
        """

        def _pretty_print_dict(dict):
            return f"<pre>{json.dumps(dict, default=str, indent=2)}</pre>"

        # Function to recursively generate HTML for each span event with depth-based indentation
        def _generate_span_html(node, depth=0):
            # Calculate indentation based on depth
            indent = depth * 40  # Adjust multiplier as needed for visual effect
            span = node.span
            color = self._generate_color(node.span.name)
            span_html = f"""
            <div class="span-event" onclick="toggleExpand(event)" style="margin-left:
                {indent}px; background-color: rgba({color}, 0.2);">
                <b>[{span.name}]</b>
                Start Time: {span.start_time:.2f} s,
                End Time: {span.end_time:.2f} s
                <div class="metadata">
                    <p><b>Trace ID</b>: {span.context.trace_id}</p>
                    <p><b>Span ID</b>: {span.context.span_id}</p>
                    <p><b>Inputs</b>: {_pretty_print_dict(span.inputs)}</p>
                    <p><b>Outputs</b>: {_pretty_print_dict(span.outputs)}</p>
                    <p><b>Attributes</b>: {_pretty_print_dict(span.attributes)}</p>
                </div>
            </div>
            """
            for child in node.children:
                span_html += _generate_span_html(child, depth + 1)
            return span_html

        html_content += _generate_span_html(root_node)
        return html_content

    def _generate_color(self, span_name: str):
        seed = sum(ord(char) for char in span_name)
        random.seed(seed)

        # Generate a random hue
        hue = random.randint(0, 359)

        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 1.0)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return f"{r}, {g}, {b}"
