import json
import time
import uuid
from dataclasses import dataclass
from unittest import mock

import pytest

import mlflow
from mlflow.entities import TraceInfo, TraceData, Trace
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.destination import TraceDestination
from mlflow.tracing.trace_manager import InMemoryTraceManager


@pytest.mark.timeout(30)  # Add timeout to prevent test hanging
def test_export():
    """Test the export functionality of the DatabricksAgentSpanExporter with direct mocking.
    
    This test uses direct mocking instead of mlflow.start_span to avoid potential 
    deadlocks that can occur when using the OpenTelemetry instrumentation in
    certain environments, especially CI environments where resource constraints
    might make threading and locking issues more likely to occur.
    
    By creating mock objects and calling the exporter directly, we isolate the
    test from any underlying threading issues in the trace collection system.
    """
    # Define the custom destination
    @dataclass
    class DatabricksAgentMonitoring(TraceDestination):
        databricks_monitor_id: str

        @property
        def type(self):
            return "databricks_agent_monitoring"
    
    # Clear any previous trace state
    InMemoryTraceManager.reset()
    
    # Create mock objects
    mock_deploy_client = mock.MagicMock()
    
    # Import and mock necessary components
    with mock.patch("mlflow.tracing.export.databricks_agent_legacy.get_deploy_client", 
                    return_value=mock_deploy_client):
        
        # Set up the destination
        mlflow.tracing.set_destination(destination=DatabricksAgentMonitoring("dummy-model-endpoint"))
        
        # Create test data directly
        from opentelemetry.sdk.trace import ReadableSpan
        
        # Create a random trace ID
        mock_trace_id = uuid.uuid4().int
        
        # Create a mock span with this trace ID
        mock_span = mock.MagicMock(spec=ReadableSpan)
        mock_context = mock.MagicMock()
        mock_context.trace_id = mock_trace_id
        mock_span.context = mock_context
        mock_span._parent = None  # Ensure it's a root span
        
        # Create a trace to return
        trace_info = TraceInfo(
            request_id=str(mock_trace_id),
            experiment_id="0",
            timestamp_ms=int(time.time() * 1000),
            execution_time_ms=100,
            status=TraceStatus.OK,
            request_metadata={},
            tags={}
        )
        
        trace_data = TraceData()
        trace = Trace(trace_info, trace_data)
        
        # Set up the trace manager
        trace_manager = InMemoryTraceManager.get_instance()
        trace_manager._trace_id_to_request_id[mock_trace_id] = str(mock_trace_id)
        
        # Mock the pop_trace method to return our trace
        with mock.patch.object(
            InMemoryTraceManager, 
            "pop_trace", 
            return_value=trace
        ):
            # Get the exporter from the provider
            from mlflow.tracing.provider import _get_trace_exporter
            exporter = _get_trace_exporter()
            
            # Call export directly
            exporter.export([mock_span])
    
    # Verify predict was called
    mock_deploy_client.predict.assert_called_once()
    
    # Verify the call arguments
    call_args = mock_deploy_client.predict.call_args
    assert call_args.kwargs["endpoint"] == "dummy-model-endpoint"
    
    # Verify the trace data was sent
    trace_json = json.loads(call_args.kwargs["inputs"]["inputs"][0])
    assert trace_json["info"]["trace_id"] == str(mock_trace_id)
