"""Tests for mlflow.cli.eval module."""

from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.cli.eval import evaluate_traces
from mlflow.entities import Trace, TraceInfo


def test_evaluate_traces_with_single_trace_table_output():
    """Test evaluate_traces with a single trace and table output."""
    # Create a test experiment
    experiment = mlflow.create_experiment("test_experiment")

    # Create realistic mock trace
    mock_trace = mock.Mock(spec=Trace)
    mock_trace.info = mock.Mock(spec=TraceInfo)
    mock_trace.info.trace_id = "tr-test-123"
    mock_trace.info.experiment_id = experiment

    # Mock MlflowClient.get_trace()
    with mock.patch("mlflow.cli.eval.MlflowClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.get_trace.return_value = mock_trace

        # Mock resolve_scorers
        mock_scorer = mock.Mock()
        with mock.patch("mlflow.cli.eval.resolve_scorers") as mock_resolve:
            mock_resolve.return_value = [mock_scorer]

            # Mock evaluate() with realistic return value
            mock_results = mock.Mock()
            mock_results.run_id = "run-eval-456"
            mock_results.tables = {
                "eval_results": pd.DataFrame(
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
            }

            with mock.patch("mlflow.cli.eval.evaluate") as mock_evaluate:
                mock_evaluate.return_value = mock_results

                # Mock click.echo to capture output
                with mock.patch("mlflow.cli.eval.click.echo") as mock_echo:
                    # Call the function
                    evaluate_traces(
                        experiment_id=experiment,
                        trace_ids="tr-test-123",
                        scorers="RelevanceToQuery",
                        output="table",
                    )

                    # Verify mocks were called with expected arguments
                    mock_client.get_trace.assert_called_once_with("tr-test-123", display=False)
                    mock_resolve.assert_called_once_with(["RelevanceToQuery"], experiment)

                    # Verify evaluate() was called with DataFrame
                    assert mock_evaluate.call_count == 1
                    call_args = mock_evaluate.call_args
                    assert "data" in call_args.kwargs
                    assert isinstance(call_args.kwargs["data"], pd.DataFrame)
                    assert "scorers" in call_args.kwargs
                    assert call_args.kwargs["scorers"] == [mock_scorer]

                    # Verify output was displayed
                    assert mock_echo.call_count >= 2  # At least progress message and table


def test_evaluate_traces_with_multiple_traces_json_output():
    """Test evaluate_traces with multiple traces and JSON output."""
    # Create a test experiment
    experiment = mlflow.create_experiment("test_experiment_multi")

    # Create realistic mock traces
    mock_trace1 = mock.Mock(spec=Trace)
    mock_trace1.info = mock.Mock(spec=TraceInfo)
    mock_trace1.info.trace_id = "tr-test-1"
    mock_trace1.info.experiment_id = experiment

    mock_trace2 = mock.Mock(spec=Trace)
    mock_trace2.info = mock.Mock(spec=TraceInfo)
    mock_trace2.info.trace_id = "tr-test-2"
    mock_trace2.info.experiment_id = experiment

    # Mock MlflowClient.get_trace()
    with mock.patch("mlflow.cli.eval.MlflowClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.get_trace.side_effect = [mock_trace1, mock_trace2]

        # Mock resolve_scorers
        mock_scorer = mock.Mock()
        with mock.patch("mlflow.cli.eval.resolve_scorers") as mock_resolve:
            mock_resolve.return_value = [mock_scorer]

            # Mock evaluate() with realistic return value for multiple traces
            mock_results = mock.Mock()
            mock_results.run_id = "run-eval-789"
            mock_results.tables = {
                "eval_results": pd.DataFrame(
                    [
                        {
                            "trace_id": "tr-test-1",
                            "assessments": [
                                {
                                    "assessment_name": "Safety",
                                    "feedback": {"value": "safe"},
                                    "rationale": "Content is safe",
                                    "metadata": {"mlflow.assessment.sourceRunId": "run-eval-789"},
                                }
                            ],
                        },
                        {
                            "trace_id": "tr-test-2",
                            "assessments": [
                                {
                                    "assessment_name": "Safety",
                                    "feedback": {"value": "safe"},
                                    "rationale": "Also safe",
                                    "metadata": {"mlflow.assessment.sourceRunId": "run-eval-789"},
                                }
                            ],
                        },
                    ]
                )
            }

            with mock.patch("mlflow.cli.eval.evaluate") as mock_evaluate:
                mock_evaluate.return_value = mock_results

                # Mock click.echo to capture output
                with mock.patch("mlflow.cli.eval.click.echo"):
                    # Call the function
                    evaluate_traces(
                        experiment_id=experiment,
                        trace_ids="tr-test-1,tr-test-2",
                        scorers="Safety",
                        output="json",
                    )

                # Verify get_trace was called for both traces
                assert mock_client.get_trace.call_count == 2
                mock_client.get_trace.assert_any_call("tr-test-1", display=False)
                mock_client.get_trace.assert_any_call("tr-test-2", display=False)

                # Verify evaluate() was called with DataFrame containing both traces
                assert mock_evaluate.call_count == 1
                call_args = mock_evaluate.call_args
                traces_df = call_args.kwargs["data"]
                assert len(traces_df) == 2
                assert "tr-test-1" in traces_df["trace_id"].values
                assert "tr-test-2" in traces_df["trace_id"].values


def test_evaluate_traces_with_nonexistent_trace():
    """Test evaluate_traces raises error when trace doesn't exist."""
    experiment = mlflow.create_experiment("test_experiment_error")

    # Mock MlflowClient.get_trace() to return None
    with mock.patch("mlflow.cli.eval.MlflowClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.get_trace.return_value = None

        # Should raise UsageError
        with pytest.raises(Exception, match="Trace with ID 'tr-nonexistent' not found"):
            evaluate_traces(
                experiment_id=experiment,
                trace_ids="tr-nonexistent",
                scorers="RelevanceToQuery",
                output="table",
            )


def test_evaluate_traces_with_trace_from_wrong_experiment():
    """Test evaluate_traces raises error when trace belongs to different experiment."""
    experiment1 = mlflow.create_experiment("test_experiment_1")
    experiment2 = mlflow.create_experiment("test_experiment_2")

    # Create mock trace belonging to experiment2
    mock_trace = mock.Mock(spec=Trace)
    mock_trace.info = mock.Mock(spec=TraceInfo)
    mock_trace.info.trace_id = "tr-test-123"
    mock_trace.info.experiment_id = experiment2

    # Mock MlflowClient.get_trace()
    with mock.patch("mlflow.cli.eval.MlflowClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.get_trace.return_value = mock_trace

        # Should raise UsageError
        with pytest.raises(Exception, match="belongs to experiment"):
            evaluate_traces(
                experiment_id=experiment1,
                trace_ids="tr-test-123",
                scorers="RelevanceToQuery",
                output="table",
            )
