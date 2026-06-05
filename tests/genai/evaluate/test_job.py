import os
from unittest import mock

import pytest

from mlflow.entities.run_status import RunStatus
from mlflow.genai.evaluation.job import invoke_genai_evaluate_job


def _serialized_scorer(name: str = "scorer") -> str:
    return f'{{"name": "{name}"}}'


def test_invoke_genai_evaluate_job_has_metadata():
    """Without ``_job_fn_metadata`` the function isn't registered with the
    job runner and POSTs to the endpoint will silently fail to submit.
    """
    assert hasattr(invoke_genai_evaluate_job, "_job_fn_metadata")
    assert invoke_genai_evaluate_job._job_fn_metadata.name == "invoke_genai_evaluate"


def test_invoke_genai_evaluate_job_success():
    mock_client = mock.MagicMock()
    mock_trace = mock.MagicMock()
    mock_client._tracing_client.batch_get_traces.return_value = [mock_trace, mock_trace]
    mock_scorer = mock.MagicMock()

    with (
        mock.patch("mlflow.genai.evaluation.job.MlflowClient", return_value=mock_client),
        mock.patch(
            "mlflow.genai.evaluation.job.Scorer.model_validate_json", return_value=mock_scorer
        ) as mock_validate,
        mock.patch("mlflow.start_run") as mock_start_run,
        mock.patch("mlflow.genai.evaluate") as mock_evaluate,
    ):
        result = invoke_genai_evaluate_job(
            experiment_id="exp-123",
            trace_ids=["trace-1", "trace-2"],
            serialized_scorers=[_serialized_scorer("a"), _serialized_scorer("b")],
            run_id="run-123",
        )

        mock_client.link_traces_to_run.assert_called_once_with(["trace-1", "trace-2"], "run-123")
        mock_client._tracing_client.batch_get_traces.assert_called_once_with(["trace-1", "trace-2"])
        assert mock_validate.call_count == 2

        # Run reuse is the whole reason the handler creates the run upfront —
        # if we forget ``run_id=...`` mlflow.genai.evaluate will start a fresh
        # nested run and the UI's navigate-to-run-page link will be wrong.
        mock_start_run.assert_called_once_with(run_id="run-123")

        mock_evaluate.assert_called_once()
        evaluate_kwargs = mock_evaluate.call_args.kwargs
        assert evaluate_kwargs["data"] == [mock_trace, mock_trace]
        assert evaluate_kwargs["scorers"] == [mock_scorer, mock_scorer]

        # On the happy path the ActiveRun context manager's __exit__ writes
        # FINISHED for us; the job must NOT also call set_terminated explicitly
        # (that would be a duplicate write to the same run). End-to-end FINISHED
        # transition is verified in tests/server/jobs/test_genai_evaluate_invocation.py.
        mock_client.set_terminated.assert_not_called()
        assert result == {"run_id": "run-123", "total_traces": 2, "scorer_count": 2}


def test_invoke_genai_evaluate_job_batches_large_trace_list():
    """``link_traces_to_run`` has a per-call cap of ``MAX_TRACE_LINKS_PER_REQUEST``
    (100), so we must batch large trace lists or the call will be rejected.
    """
    mock_client = mock.MagicMock()
    mock_traces = [mock.MagicMock() for _ in range(250)]
    mock_client._tracing_client.batch_get_traces.return_value = mock_traces
    trace_ids = [f"trace-{i}" for i in range(250)]

    with (
        mock.patch("mlflow.genai.evaluation.job.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.genai.evaluation.job.Scorer.model_validate_json"),
        mock.patch("mlflow.start_run"),
        mock.patch("mlflow.genai.evaluate"),
    ):
        invoke_genai_evaluate_job(
            experiment_id="exp-123",
            trace_ids=trace_ids,
            serialized_scorers=[_serialized_scorer()],
            run_id="run-123",
        )

        # 250 traces with a cap of 100 = three batches of (100, 100, 50).
        assert mock_client.link_traces_to_run.call_count == 3
        batch_sizes = [len(call.args[0]) for call in mock_client.link_traces_to_run.call_args_list]
        assert batch_sizes == [100, 100, 50]


def test_invoke_genai_evaluate_job_setup_failure_marks_run_failed():
    """If setup fails BEFORE the active-run context manager is entered (here:
    ``batch_get_traces`` raising), the context manager's __exit__ never runs,
    so the job itself must flip the run to FAILED — otherwise it'd stay
    RUNNING forever in /evaluation-runs.
    """
    mock_client = mock.MagicMock()
    mock_client._tracing_client.batch_get_traces.side_effect = Exception("trace fetch failed")

    with (
        mock.patch("mlflow.genai.evaluation.job.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.start_run") as mock_start_run,
        mock.patch("mlflow.genai.evaluate") as mock_evaluate,
    ):
        with pytest.raises(Exception, match="trace fetch failed"):
            invoke_genai_evaluate_job(
                experiment_id="exp-123",
                trace_ids=["trace-1"],
                serialized_scorers=[_serialized_scorer()],
                run_id="run-123",
            )

        # Setup blew up *before* we reached the with block, so neither
        # mlflow.start_run nor mlflow.genai.evaluate should have been touched.
        mock_start_run.assert_not_called()
        mock_evaluate.assert_not_called()

        # This is the only path where the job code itself writes the terminal
        # status — the context manager can't help here because we never entered it.
        mock_client.set_terminated.assert_called_once_with(
            "run-123", RunStatus.to_string(RunStatus.FAILED)
        )


def test_invoke_genai_evaluate_job_evaluate_failure_defers_to_context_manager():
    """When ``mlflow.genai.evaluate`` raises INSIDE the active-run context
    manager, terminal status is owned by ``ActiveRun.__exit__`` (see
    mlflow/tracking/fluent.py:380), so the job code itself must NOT also call
    set_terminated. The exception must still propagate so huey records the
    job as FAILED with the original traceback. End-to-end FAILED transition
    is covered by tests/server/jobs/test_genai_evaluate_invocation.py
    (``test_invoke_genai_evaluate_missing_trace_marks_run_failed``).
    """
    mock_client = mock.MagicMock()
    mock_client._tracing_client.batch_get_traces.return_value = [mock.MagicMock()]

    with (
        mock.patch("mlflow.genai.evaluation.job.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.genai.evaluation.job.Scorer.model_validate_json"),
        mock.patch("mlflow.start_run"),
        mock.patch("mlflow.genai.evaluate", side_effect=Exception("harness boom")),
    ):
        with pytest.raises(Exception, match="harness boom"):
            invoke_genai_evaluate_job(
                experiment_id="exp-123",
                trace_ids=["trace-1"],
                serialized_scorers=[_serialized_scorer()],
                run_id="run-123",
            )

        # Critical: must NOT call set_terminated explicitly here. Doing so
        # would be a duplicate write on top of the context manager's __exit__.
        mock_client.set_terminated.assert_not_called()


def test_invoke_genai_evaluate_job_propagates_username_via_env(monkeypatch):
    """``username`` is propagated to subprocess env vars so judge LLM calls
    are authorised as the original caller, not the server admin. Same
    pattern as ``invoke_scorer_job``.
    """
    mock_client = mock.MagicMock()
    mock_client._tracing_client.batch_get_traces.return_value = [mock.MagicMock()]
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    captured_env: dict[str, str] = {}

    def _capture_env(*args, **kwargs):
        captured_env["MLFLOW_TRACKING_USERNAME"] = os.environ.get("MLFLOW_TRACKING_USERNAME", "")

    with (
        mock.patch("mlflow.genai.evaluation.job.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.genai.evaluation.job.Scorer.model_validate_json"),
        mock.patch("mlflow.start_run"),
        mock.patch("mlflow.genai.evaluate", side_effect=_capture_env),
    ):
        invoke_genai_evaluate_job(
            experiment_id="exp-123",
            trace_ids=["trace-1"],
            serialized_scorers=[_serialized_scorer()],
            run_id="run-123",
            username="alice",
        )

        assert captured_env["MLFLOW_TRACKING_USERNAME"] == "alice"


def test_invoke_genai_evaluate_job_skips_username_propagation_when_none(monkeypatch):
    mock_client = mock.MagicMock()
    mock_client._tracing_client.batch_get_traces.return_value = [mock.MagicMock()]
    # Pre-populate the env so we can detect a write where we shouldn't see one.
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "unchanged")

    with (
        mock.patch("mlflow.genai.evaluation.job.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.genai.evaluation.job.Scorer.model_validate_json"),
        mock.patch("mlflow.start_run"),
        mock.patch("mlflow.genai.evaluate"),
    ):
        invoke_genai_evaluate_job(
            experiment_id="exp-123",
            trace_ids=["trace-1"],
            serialized_scorers=[_serialized_scorer()],
            run_id="run-123",
            username=None,
        )

        assert os.environ["MLFLOW_TRACKING_USERNAME"] == "unchanged"
