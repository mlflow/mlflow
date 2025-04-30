from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_info_v3 import TraceInfoV3
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import Trace as ProtoTrace

_logger = logging.getLogger(__name__)


@dataclass
class Trace(_MlflowObject):
    """A trace object.

    Args:
        info: A lightweight object that contains the metadata of a trace.
        data: A container object that holds the spans data of a trace.
    """

    info: TraceInfoV3
    data: TraceData

    def __post_init__(self):
        if isinstance(self.info, TraceInfo):
            self.info = self.info.to_v3(request=self.data.request, response=self.data.response)

    def __repr__(self) -> str:
        return f"Trace(trace_id={self.info.trace_id})"

    def to_dict(self) -> dict[str, Any]:
        return {"info": self.info.to_dict(), "data": self.data.to_dict()}

    def to_json(self, pretty=False) -> str:
        from mlflow.tracing.utils import TraceJSONEncoder

        return json.dumps(self.to_dict(), cls=TraceJSONEncoder, indent=2 if pretty else None)

    @classmethod
    def from_dict(cls, trace_dict: dict[str, Any]) -> Trace:
        info = trace_dict.get("info")
        data = trace_dict.get("data")
        if info is None or data is None:
            raise MlflowException(
                "Unable to parse Trace from dictionary. Expected keys: 'info' and 'data'. "
                f"Received keys: {list(trace_dict.keys())}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return cls(
            info=TraceInfoV3.from_dict(info),
            data=TraceData.from_dict(data),
        )

    @classmethod
    def from_json(cls, trace_json: str) -> Trace:
        try:
            trace_dict = json.loads(trace_json)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Unable to parse trace JSON: {trace_json}. Error: {e}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return cls.from_dict(trace_dict)

    def _serialize_for_mimebundle(self):
        # databricks notebooks will use the request ID to
        # fetch the trace from the backend. including the
        # full JSON can cause notebooks to exceed size limits
        return json.dumps(self.info.request_id)

    def _repr_mimebundle_(self, include=None, exclude=None):
        """
        This method is used to trigger custom display logic in IPython notebooks.
        See https://ipython.readthedocs.io/en/stable/config/integrating.html#MyObject
        for more details.

        At the moment, the only supported MIME type is "application/databricks.mlflow.trace",
        which contains a JSON representation of the Trace object. This object is deserialized
        in Databricks notebooks to display the Trace object in a nicer UI.
        """
        from mlflow.tracing.display import (
            get_display_handler,
            get_notebook_iframe_html,
            is_using_tracking_server,
        )
        from mlflow.utils.databricks_utils import is_in_databricks_runtime

        bundle = {"text/plain": repr(self)}

        if not get_display_handler().disabled:
            if is_in_databricks_runtime():
                bundle["application/databricks.mlflow.trace"] = self._serialize_for_mimebundle()
            elif is_using_tracking_server():
                bundle["text/html"] = get_notebook_iframe_html([self])

        return bundle

    def to_pandas_dataframe_row(self) -> dict[str, Any]:
        print(f"### DEBUG Trace.to_pandas_dataframe_row - trace_id: {self.info.trace_id}")
        print(f"### DEBUG Trace.to_pandas_dataframe_row - assessments count: {len(self.info.assessments)}")
        
        if hasattr(self.info, 'tags'):
            assessment_tags = {k: v for k, v in self.info.tags.items() if k.startswith("mlflow.assessment.")}
            print(f"### DEBUG Trace.to_pandas_dataframe_row - found {len(assessment_tags)} assessment tags")
            if assessment_tags and not self.info.assessments:
                print("### DEBUG Trace.to_pandas_dataframe_row - WARNING: Has assessment tags but no assessments!")
        
        # Check if any assessments would be missed
        if hasattr(self.info, 'tags') and hasattr(self.info, 'assessments'):
            tag_assessment_count = len([k for k in self.info.tags.keys() if k.startswith("mlflow.assessment.")])
            if tag_assessment_count > 0 and len(self.info.assessments) == 0:
                print(f"### DEBUG CRITICAL: {tag_assessment_count} assessment tags found but assessments array is empty!")
                
                # Try to extract assessments from tags
                print("### DEBUG Attempting to extract assessments from tags...")
                try:
                    import json
                    from mlflow.entities.assessment import Assessment
                    
                    extracted_assessments = []
                    for key, value in self.info.tags.items():
                        if key.startswith('mlflow.assessment.'):
                            try:
                                assessment_data = json.loads(value)
                                print(f"### DEBUG Assessment data parsed from tag: {assessment_data.get('assessment_name')}")
                                # Just report, don't modify for now
                            except json.JSONDecodeError as e:
                                print(f"### DEBUG Error parsing assessment from tag {key}: {e}")
                except Exception as e:
                    print(f"### DEBUG Error extracting assessments: {str(e)}")
        
        row_data = {
            "trace_id": self.info.trace_id,
            "trace": self,
            "timestamp_ms": self.info.timestamp_ms,
            "status": self.info.status,
            "execution_time_ms": self.info.execution_time_ms,
            "request": self._deserialize_json_attr(self.data.request),
            "response": self._deserialize_json_attr(self.data.response),
            "request_metadata": self.info.request_metadata,
            "spans": [span.to_dict() for span in self.data.spans],
            "tags": self.info.tags,
            "assessments": self.info.assessments,
            # For backward compatibility, we need to keep the old "request_id" field
            # Ref: https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/evaluation-schema
            "request_id": self.info.request_id,
        }
        
        print(f"### DEBUG Trace.to_pandas_dataframe_row - returned row with assessments length: {len(row_data['assessments'])}")
        return row_data

    def _deserialize_json_attr(self, value: str):
        try:
            return json.loads(value)
        except Exception:
            _logger.debug(f"Failed to deserialize JSON attribute: {value}", exc_info=True)
            return value

    def search_spans(
        self, span_type: Optional[SpanType] = None, name: Optional[Union[str, re.Pattern]] = None
    ) -> list[Span]:
        """
        Search for spans that match the given criteria within the trace.

        Args:
            span_type: The type of the span to search for.
            name: The name of the span to search for. This can be a string or a regular expression.

        Returns:
            A list of spans that match the given criteria.
            If there is no match, an empty list is returned.

        .. code-block:: python

            import mlflow
            import re
            from mlflow.entities import SpanType


            @mlflow.trace(span_type=SpanType.CHAIN)
            def run(x: int) -> int:
                x = add_one(x)
                x = add_two(x)
                x = multiply_by_two(x)
                return x


            @mlflow.trace(span_type=SpanType.TOOL)
            def add_one(x: int) -> int:
                return x + 1


            @mlflow.trace(span_type=SpanType.TOOL)
            def add_two(x: int) -> int:
                return x + 2


            @mlflow.trace(span_type=SpanType.TOOL)
            def multiply_by_two(x: int) -> int:
                return x * 2


            # Run the function and get the trace
            y = run(2)
            trace_id = mlflow.get_last_active_trace_id()
            trace = mlflow.get_trace(trace_id)

            # 1. Search spans by name (exact match)
            spans = trace.search_spans(name="add_one")
            print(spans)
            # Output: [Span(name='add_one', ...)]

            # 2. Search spans by name (regular expression)
            pattern = re.compile(r"add.*")
            spans = trace.search_spans(name=pattern)
            print(spans)
            # Output: [Span(name='add_one', ...), Span(name='add_two', ...)]

            # 3. Search spans by type
            spans = trace.search_spans(span_type=SpanType.LLM)
            print(spans)
            # Output: [Span(name='run', ...)]

            # 4. Search spans by name and type
            spans = trace.search_spans(name="add_one", span_type=SpanType.TOOL)
            print(spans)
            # Output: [Span(name='add_one', ...)]
        """

        def _match_name(span: Span) -> bool:
            if isinstance(name, str):
                return span.name == name
            elif isinstance(name, re.Pattern):
                return name.search(span.name) is not None
            elif name is None:
                return True
            else:
                raise MlflowException(
                    f"Invalid type for 'name'. Expected str or re.Pattern. Got: {type(name)}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        def _match_type(span: Span) -> bool:
            if isinstance(span_type, str):
                return span.span_type == span_type
            elif span_type is None:
                return True
            else:
                raise MlflowException(
                    "Invalid type for 'span_type'. Expected str or mlflow.entities.SpanType. "
                    f"Got: {type(span_type)}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        return [span for span in self.data.spans if _match_name(span) and _match_type(span)]

    @staticmethod
    def pandas_dataframe_columns() -> list[str]:
        return [
            "trace_id",
            "trace",
            "timestamp_ms",
            "status",
            "execution_time_ms",
            "request",
            "response",
            "request_metadata",
            "spans",
            "tags",
            "assessments",
            "request_id",
        ]

    def to_proto(self):
        """
        Convert into a proto object to sent to the MLflow backend.

        NB: The Trace definition in MLflow backend doesn't include the `data` field,
            but rather only contains TraceInfoV3.
        """

        return ProtoTrace(trace_info=self.info.to_proto())

    def extract_assessments_from_tags(self):
        """Extract assessment objects from tags that start with 'mlflow.assessment.'"""
        if not hasattr(self.info, 'tags'):
            print("### DEBUG extract_assessments_from_tags - no tags field found")
            return []
            
        try:
            import json
            from mlflow.entities.assessment import Assessment
            
            assessment_tags = {k: v for k, v in self.info.tags.items() if k.startswith('mlflow.assessment.')}
            print(f"### DEBUG extract_assessments_from_tags - found {len(assessment_tags)} assessment tags")
            
            extracted_assessments = []
            for key, value in assessment_tags.items():
                try:
                    assessment_data = json.loads(value)
                    print(f"### DEBUG extract_assessments_from_tags - extracting assessment: {assessment_data.get('assessment_name')}")
                    assessment = Assessment.from_dictionary(assessment_data)
                    extracted_assessments.append(assessment)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"### DEBUG extract_assessments_from_tags - Error parsing assessment from tag {key}: {e}")
            
            print(f"### DEBUG extract_assessments_from_tags - extracted {len(extracted_assessments)} assessments")
            # Update the assessments field with what we've extracted
            self.info.assessments = extracted_assessments
            return extracted_assessments
        except Exception as e:
            print(f"### DEBUG extract_assessments_from_tags - Error: {str(e)}")
            return []
