import pytest

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.fluent import get_trace

def test_trace_with_classmethod():
    class TestModel:
        @mlflow.trace
        @classmethod
        def predict(cls, x, y):
            return x + y

    # Call the classmethod
    result = TestModel.predict(1, 2)
    assert result == 3
    
    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.tracing.fluent.get_last_active_trace_id()
    assert trace_id is not None
    
    trace = get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0
    
    # The first span should be our traced function
    span = trace.data.spans[0]
    assert span.name == "predict"
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == {"output": 3}

def test_trace_with_classmethod_order_reversed():
    class TestModel:
        @classmethod
        @mlflow.trace
        def predict(cls, x, y):
            return x + y

    # Call the classmethod
    result = TestModel.predict(1, 2)
    assert result == 3
    
    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.tracing.fluent.get_last_active_trace_id()
    assert trace_id is not None
    
    trace = get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0
    
    # The first span should be our traced function
    span = trace.data.spans[0]
    assert span.name == "predict"
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == {"output": 3}

def test_trace_with_classmethod_with_params():
    class TestModel:
        @mlflow.trace(name="custom_predict", span_type=SpanType.MODEL_INFERENCE)
        @classmethod
        def predict(cls, x, y):
            return x + y

    # Call the classmethod
    result = TestModel.predict(1, 2)
    assert result == 3
    
    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.tracing.fluent.get_last_active_trace_id()
    assert trace_id is not None
    
    trace = get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0
    
    # The first span should be our traced function with custom name
    span = trace.data.spans[0]
    assert span.name == "custom_predict"
    assert span.span_type == SpanType.MODEL_INFERENCE
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == {"output": 3}

def test_trace_with_staticmethod():
    class TestModel:
        @mlflow.trace
        @staticmethod
        def predict(x, y):
            return x + y

    # Call the staticmethod
    result = TestModel.predict(1, 2)
    assert result == 3
    
    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.tracing.fluent.get_last_active_trace_id()
    assert trace_id is not None
    
    trace = get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0
    
    # The first span should be our traced function
    span = trace.data.spans[0]
    assert span.name == "predict"
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == {"output": 3}

def test_trace_with_staticmethod_order_reversed():
    class TestModel:
        @staticmethod
        @mlflow.trace
        def predict(x, y):
            return x + y

    # Call the staticmethod
    result = TestModel.predict(1, 2)
    assert result == 3
    
    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.tracing.fluent.get_last_active_trace_id()
    assert trace_id is not None
    
    trace = get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0
    
    # The first span should be our traced function
    span = trace.data.spans[0]
    assert span.name == "predict"
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == {"output": 3}

def test_trace_with_staticmethod_with_params():
    class TestModel:
        @mlflow.trace(name="custom_predict", span_type=SpanType.MODEL_INFERENCE)
        @staticmethod
        def predict(x, y):
            return x + y

    # Call the staticmethod
    result = TestModel.predict(1, 2)
    assert result == 3
    
    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.tracing.fluent.get_last_active_trace_id()
    assert trace_id is not None
    
    trace = get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0
    
    # The first span should be our traced function with custom name
    span = trace.data.spans[0]
    assert span.name == "custom_predict"
    assert span.span_type == SpanType.MODEL_INFERENCE
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == {"output": 3}