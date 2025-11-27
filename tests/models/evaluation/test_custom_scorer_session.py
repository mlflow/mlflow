
import pandas as pd
import pytest
from unittest import mock

import mlflow
from mlflow.entities import Trace, TraceInfo, TraceData
from mlflow.entities.trace_location import TraceLocation, TraceLocationType
from mlflow.entities.trace_state import TraceState
from mlflow.metrics import make_metric, MetricValue
from mlflow.models.evaluation.evaluators.default import DefaultEvaluator

def create_trace(trace_id, timestamp):
    info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation(type=TraceLocationType.MLFLOW_EXPERIMENT, mlflow_experiment=None),
        request_time=timestamp,
        state=TraceState.OK,
        execution_duration=100,
        trace_metadata={},
        tags={"mlflow.traceName": "test_trace"}
    )
    data = TraceData(spans=[])
    return Trace(info, data)

def test_custom_scorer_with_trace_and_session():
    trace1 = create_trace("tr-1", 1000)
    trace2 = create_trace("tr-2", 2000)
    
    # Session 1 has trace1 and trace2
    data = pd.DataFrame({
        "trace": [trace1, trace2],
        "input": ["input1", "input2"],
        "output": ["output1", "output2"],
        "session_id": ["s1", "s1"]
    })
    
    def eval_fn(predictions, metrics, trace=None, session=None):
        assert isinstance(trace, pd.Series)
        assert isinstance(session, pd.Series)
        assert len(trace) == len(predictions)
        assert len(session) == len(predictions)
        
        t1 = trace.iloc[0]
        s1 = session.iloc[0]
        
        assert t1.info.trace_id == "tr-1"
        # _test_first_row passes a single row, so session grouping only finds 1 trace.
        # We only assert full session length if we are running on the full dataset (len > 1).
        if len(predictions) > 1:
            assert len(s1) == 2
            assert s1[0].info.trace_id == "tr-1"
            assert s1[1].info.trace_id == "tr-2"
        
        return MetricValue(scores=[1.0] * len(predictions), justifications=["OK"] * len(predictions))

    custom_metric = make_metric(
        eval_fn=eval_fn,
        greater_is_better=True,
        name="session_metric"
    )
    
    results = mlflow.evaluate(
        data=data,
        predictions="output",
        model_type=None,
        extra_metrics=[custom_metric]
    )
        
    assert results.metrics["session_metric/mean"] == 1.0

def test_custom_scorer_missing_session_id():
    trace1 = create_trace("tr-1", 1000)
    data = pd.DataFrame({
        "trace": [trace1],
        "input": ["input1"],
        "output": ["output1"]
        # Missing session_id
    })
    
    def eval_fn(predictions, metrics, session):
        return MetricValue(scores=[0.0], justifications=["Fail"])
        
    custom_metric = make_metric(
        eval_fn=eval_fn,
        greater_is_better=True,
        name="fail_metric"
    )
    
    # Expect failure because session cannot be constructed
    with pytest.raises(Exception, match=r"missing columns.*session"):
        mlflow.evaluate(
            data=data,
            predictions="output",
            model_type=None,
            extra_metrics=[custom_metric]
        )

def test_custom_scorer_inferred_trace_column():
    trace1 = create_trace("tr-1", 1000)
    # Column name is NOT "trace"
    data = pd.DataFrame({
        "my_trace_col": [trace1],
        "input": ["input1"],
        "output": ["output1"],
        "session_id": ["s1"]
    })
    
    def eval_fn(predictions, metrics, trace=None, session=None):
        assert trace is not None
        assert session is not None
        assert trace.iloc[0].info.trace_id == "tr-1"
        return MetricValue(scores=[1.0], justifications=["OK"])
        
    custom_metric = make_metric(
        eval_fn=eval_fn,
        greater_is_better=True,
        name="inferred_metric"
    )
    
    results = mlflow.evaluate(
        data=data,
        predictions="output",
        model_type=None,
        extra_metrics=[custom_metric]
    )
        
    assert results.metrics["inferred_metric/mean"] == 1.0
