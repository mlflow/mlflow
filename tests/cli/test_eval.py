import re
from unittest import mock

import click
import pandas as pd
import pytest

import mlflow
from mlflow.cli.eval import evaluate_traces
from mlflow.entities import Trace, TraceInfo
from mlflow.genai.scorers.base import scorer


def test_evaluate_traces_with_single_trace_table_output():
    experiment_id = mlflow.create_experiment("test_experiment")

    mock_trace = mock.Mock(spec=Trace)
    mock_trace.info = mock.Mock(spec=TraceInfo)
    mock_trace.info.trace_id = "tr-test-123"
    mock_trace.info.experiment_id = experiment_id

    mock_results = mock.Mock()
    mock_results.run_id = "run-eval-456"
    mock_results.result_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-test-123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "feedback": {"value": "yes"},
                        "rationale": "The answer is relevant",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-eval-456"},
                    }
                ],
            }
        ]
    )

    with (
        mock.patch(
            "mlflow.cli.eval.MlflowClient.get_trace", return_value=mock_trace
        ) as mock_get_trace,
        mock.patch("mlflow.cli.eval.evaluate", return_value=mock_results) as mock_evaluate,
    ):
        evaluate_traces(
            experiment_id=experiment_id,
            trace_ids="tr-test-123",
            scorers="RelevanceToQuery",
            output_format="table",
        )

        mock_get_trace.assert_called_once_with("tr-test-123", display=False)

        assert mock_evaluate.call_count == 1
        call_args = mock_evaluate.call_args
        assert "data" in call_args.kwargs

        expected_df = pd.DataFrame([{"trace_id": "tr-test-123", "trace": mock_trace}])
        pd.testing.assert_frame_equal(call_args.kwargs["data"], expected_df)

        assert "scorers" in call_args.kwargs
        assert len(call_args.kwargs["scorers"]) == 1
        assert call_args.kwargs["scorers"][0].__class__.__name__ == "RelevanceToQuery"


def test_evaluate_traces_with_multiple_traces_json_output():
    experiment = mlflow.create_experiment("test_experiment_multi")

    mock_trace1 = mock.Mock(spec=Trace)
    mock_trace1.info = mock.Mock(spec=TraceInfo)
    mock_trace1.info.trace_id = "tr-test-1"
    mock_trace1.info.experiment_id = experiment

    mock_trace2 = mock.Mock(spec=Trace)
    mock_trace2.info = mock.Mock(spec=TraceInfo)
    mock_trace2.info.trace_id = "tr-test-2"
    mock_trace2.info.experiment_id = experiment

    mock_results = mock.Mock()
    mock_results.run_id = "run-eval-789"
    mock_results.result_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-test-1",
                "assessments": [
                    {
                        "assessment_name": "Correctness",
                        "feedback": {"value": "correct"},
                        "rationale": "Content is correct",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-eval-789"},
                    }
                ],
            },
            {
                "trace_id": "tr-test-2",
                "assessments": [
                    {
                        "assessment_name": "Correctness",
                        "feedback": {"value": "correct"},
                        "rationale": "Also correct",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-eval-789"},
                    }
                ],
            },
        ]
    )

    with (
        mock.patch(
            "mlflow.cli.eval.MlflowClient.get_trace",
            side_effect=[mock_trace1, mock_trace2],
        ) as mock_get_trace,
        mock.patch("mlflow.cli.eval.evaluate", return_value=mock_results) as mock_evaluate,
    ):
        evaluate_traces(
            experiment_id=experiment,
            trace_ids="tr-test-1,tr-test-2",
            scorers="Correctness",
            output_format="json",
        )

        assert mock_get_trace.call_count == 2
        mock_get_trace.assert_any_call("tr-test-1", display=False)
        mock_get_trace.assert_any_call("tr-test-2", display=False)

        assert mock_evaluate.call_count == 1
        call_args = mock_evaluate.call_args
        expected_df = pd.DataFrame(
            [
                {"trace_id": "tr-test-1", "trace": mock_trace1},
                {"trace_id": "tr-test-2", "trace": mock_trace2},
            ]
        )
        pd.testing.assert_frame_equal(call_args.kwargs["data"], expected_df)


def test_evaluate_traces_with_nonexistent_trace():
    experiment = mlflow.create_experiment("test_experiment_error")

    with mock.patch("mlflow.cli.eval.MlflowClient.get_trace", return_value=None) as mock_get_trace:
        with pytest.raises(click.UsageError, match="Trace with ID 'tr-nonexistent' not found"):
            evaluate_traces(
                experiment_id=experiment,
                trace_ids="tr-nonexistent",
                scorers="RelevanceToQuery",
                output_format="table",
            )

        mock_get_trace.assert_called_once_with("tr-nonexistent", display=False)


def test_evaluate_traces_with_trace_from_wrong_experiment():
    experiment1 = mlflow.create_experiment("test_experiment_1")
    experiment2 = mlflow.create_experiment("test_experiment_2")

    mock_trace = mock.Mock(spec=Trace)
    mock_trace.info = mock.Mock(spec=TraceInfo)
    mock_trace.info.trace_id = "tr-test-123"
    mock_trace.info.experiment_id = experiment2

    with mock.patch(
        "mlflow.cli.eval.MlflowClient.get_trace", return_value=mock_trace
    ) as mock_get_trace:
        with pytest.raises(click.UsageError, match="belongs to experiment"):
            evaluate_traces(
                experiment_id=experiment1,
                trace_ids="tr-test-123",
                scorers="RelevanceToQuery",
                output_format="table",
            )

        mock_get_trace.assert_called_once_with("tr-test-123", display=False)


def test_evaluate_traces_integration():
    experiment_id = mlflow.create_experiment("test_experiment_integration")
    mlflow.set_experiment(experiment_id=experiment_id)

    # Create a few real traces with inputs and outputs
    trace_ids = []
    for i in range(3):
        with mlflow.start_span(name=f"test_span_{i}") as span:
            span.set_inputs({"question": f"What is test {i}?"})
            span.set_outputs(f"This is answer {i}")
            trace_ids.append(span.trace_id)

    # Define a simple code-based scorer inline
    @scorer
    def simple_scorer(outputs):
        """Extract the digit from the output string and return it as the score"""
        match = re.search(r"\d+", outputs)
        if match:
            return float(match.group())
        return 0.0

    with mock.patch(
        "mlflow.cli.eval.resolve_scorers", return_value=[simple_scorer]
    ) as mock_resolve:
        evaluate_traces(
            experiment_id=experiment_id,
            trace_ids=",".join(trace_ids),
            scorers="simple_scorer",  # This will be intercepted by our mock
            output_format="table",
        )
        mock_resolve.assert_called_once()

    # Verify that the evaluation results are as expected
    traces = mlflow.search_traces(locations=[experiment_id], return_type="list")
    assert len(traces) == 3

    # Sort traces by their outputs to get consistent ordering
    traces = sorted(traces, key=lambda t: t.data.spans[0].outputs)

    for i, trace in enumerate(traces):
        assessments = trace.info.assessments
        assert len(assessments) > 0

        scorer_assessments = [a for a in assessments if a.name == "simple_scorer"]
        assert len(scorer_assessments) == 1

        assessment = scorer_assessments[0]
        # Each trace should have a score equal to its index (0, 1, 2)
        assert assessment.value == float(i)
