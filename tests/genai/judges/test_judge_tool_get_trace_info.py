"""
Tests for GetTraceInfoTool.
"""

import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.get_trace_info import GetTraceInfoTool
from mlflow.types.llm import ToolDefinition


def test_get_trace_info_tool_name():
    """Test GetTraceInfoTool name property."""
    tool = GetTraceInfoTool()
    assert tool.name == "get_trace_info"


def test_get_trace_info_tool_get_definition():
    """Test GetTraceInfoTool get_definition returns proper ToolDefinition."""
    tool = GetTraceInfoTool()
    definition = tool.get_definition()
    
    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "get_trace_info"
    assert "metadata about the trace" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert len(definition.function.parameters.required) == 0
    assert definition.type == "function"


def test_get_trace_info_tool_invoke_success():
    """Test GetTraceInfoTool returns trace.info when trace exists."""
    tool = GetTraceInfoTool()
    
    # Create trace with info
    trace_info = TraceInfo(
        trace_id="test-trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250
    )
    trace = Trace(info=trace_info, data=None)
    
    # Invoke tool
    result = tool.invoke(trace)
    
    # Should return the TraceInfo object
    assert result is trace_info
    assert result.trace_id == "test-trace-123"
    assert result.timestamp_ms == 1234567890
    assert result.execution_time_ms == 250
    assert result.status == "OK"


def test_get_trace_info_tool_invoke_none_trace():
    """Test GetTraceInfoTool returns None when trace is None."""
    tool = GetTraceInfoTool()
    
    result = tool.invoke(None)
    assert result is None


def test_get_trace_info_tool_invoke_trace_without_info():
    """Test GetTraceInfoTool returns None when trace has no info."""
    tool = GetTraceInfoTool()
    
    # Create trace without info
    trace = Trace(info=None, data=None)
    
    result = tool.invoke(trace)
    assert result is None


def test_get_trace_info_tool_invoke_different_states():
    """Test GetTraceInfoTool with different trace states."""
    tool = GetTraceInfoTool()
    
    trace_info = TraceInfo(
        trace_id="test-trace-456",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=9876543210,
        state=TraceState.ERROR,
        execution_duration=500
    )
    trace = Trace(info=trace_info, data=None)
    
    # Invoke normally
    result = tool.invoke(trace)
    
    # Should return the TraceInfo object
    assert result is trace_info
    assert result.trace_id == "test-trace-456"
    assert result.state == TraceState.ERROR