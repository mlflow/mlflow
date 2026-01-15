"""
Unit tests for the optimize_prompts_job wrapper.

These tests focus on the helper functions and job function logic without
requiring a full job execution infrastructure.
"""

import json
from unittest import mock

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.job import (
    _build_predict_fn,
    _create_optimizer,
    _load_scorers,
    optimize_prompts_job,
)
from mlflow.genai.optimize.optimizers import GepaPromptOptimizer, MetaPromptOptimizer
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.builtin_scorers import Correctness, Safety


def test_create_gepa_optimizer_success():
    config = {"reflection_model": "openai:/gpt-4o", "max_metric_calls": 50}
    optimizer = _create_optimizer("gepa", json.dumps(config))
    assert isinstance(optimizer, GepaPromptOptimizer)
    assert optimizer.reflection_model == "openai:/gpt-4o"
    assert optimizer.max_metric_calls == 50


def test_create_gepa_optimizer_case_insensitive():
    config = {"reflection_model": "openai:/gpt-4o"}
    optimizer = _create_optimizer("GEPA", json.dumps(config))
    assert isinstance(optimizer, GepaPromptOptimizer)


def test_create_gepa_optimizer_missing_reflection_model():
    config = {"max_metric_calls": 50}
    with pytest.raises(MlflowException, match="'reflection_model' must be specified"):
        _create_optimizer("gepa", json.dumps(config))


def test_create_metaprompt_optimizer_success():
    config = {"reflection_model": "openai:/gpt-4o", "guidelines": "Be concise"}
    optimizer = _create_optimizer("metaprompt", json.dumps(config))
    assert isinstance(optimizer, MetaPromptOptimizer)


def test_create_metaprompt_optimizer_missing_reflection_model():
    config = {"guidelines": "Be concise"}
    with pytest.raises(MlflowException, match="'reflection_model' must be specified"):
        _create_optimizer("metaprompt", json.dumps(config))


def test_create_optimizer_unsupported_type():
    with pytest.raises(MlflowException, match="Unsupported optimizer type: 'invalid'"):
        _create_optimizer("invalid", None)


def test_load_builtin_scorers():
    scorers = _load_scorers(["Correctness", "Safety"], "exp-123")

    assert len(scorers) == 2
    assert isinstance(scorers[0], Correctness)
    assert isinstance(scorers[1], Safety)


def test_load_custom_scorers():
    with (
        mock.patch("mlflow.genai.scorers.base.is_in_databricks_runtime", return_value=True),
        mock.patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True),
    ):
        experiment_id = mlflow.create_experiment("test_load_custom_scorers")

        @scorer
        def custom_scorer_1(outputs) -> bool:
            return len(outputs) > 0

        @scorer
        def custom_scorer_2(outputs) -> bool:
            return len(outputs) > 0

        custom_scorer_1.register(experiment_id=experiment_id, name="custom_scorer_1")
        custom_scorer_2.register(experiment_id=experiment_id, name="custom_scorer_2")

        scorers = _load_scorers(["custom_scorer_1", "custom_scorer_2"], experiment_id)

        assert len(scorers) == 2
        assert scorers[0].name == "custom_scorer_1"
        assert scorers[1].name == "custom_scorer_2"

        mlflow.delete_experiment(experiment_id)


def test_load_scorer_not_found_raises_error():
    experiment_id = mlflow.create_experiment("test_load_scorer_not_found")

    with pytest.raises(MlflowException, match="Scorer 'unknown_scorer' not found"):
        _load_scorers(["unknown_scorer"], experiment_id)

    mlflow.delete_experiment(experiment_id)


def test_build_predict_fn_success():
    mock_prompt = mock.MagicMock()
    mock_prompt.model_config = {"provider": "openai", "model_name": "gpt-4o"}
    mock_prompt.format.return_value = "formatted prompt"

    mock_litellm = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.choices = [mock.MagicMock()]
    mock_response.choices[0].message.content = "response text"
    mock_litellm.completion.return_value = mock_response

    with (
        mock.patch("mlflow.genai.optimize.job.load_prompt", return_value=mock_prompt),
        mock.patch.dict("sys.modules", {"litellm": mock_litellm}),
    ):
        predict_fn = _build_predict_fn("prompts:/test/1")
        result = predict_fn(question="What is AI?")

        assert result == "response text"
        mock_litellm.completion.assert_called_once()
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["model"] == "openai/gpt-4o"
        mock_prompt.format.assert_called_with(question="What is AI?")


def test_build_predict_fn_missing_model_config():
    mock_prompt = mock.MagicMock()
    mock_prompt.model_config = None

    mock_litellm = mock.MagicMock()

    with (
        mock.patch("mlflow.genai.optimize.job.load_prompt", return_value=mock_prompt),
        mock.patch.dict("sys.modules", {"litellm": mock_litellm}),
    ):
        with pytest.raises(MlflowException, match="doesn't have a model configuration"):
            _build_predict_fn("prompts:/test/1")


def test_build_predict_fn_missing_provider():
    mock_prompt = mock.MagicMock()
    mock_prompt.model_config = {"model_name": "gpt-4o"}

    mock_litellm = mock.MagicMock()

    with (
        mock.patch("mlflow.genai.optimize.job.load_prompt", return_value=mock_prompt),
        mock.patch.dict("sys.modules", {"litellm": mock_litellm}),
    ):
        with pytest.raises(MlflowException, match="doesn't have a model configuration"):
            _build_predict_fn("prompts:/test/1")


def test_optimize_prompts_job_has_metadata():
    assert hasattr(optimize_prompts_job, "_job_fn_metadata")
    metadata = optimize_prompts_job._job_fn_metadata
    assert metadata.name == "optimize_prompts"
    assert metadata.max_workers == 2


def test_optimize_prompts_job_calls():
    mock_dataset = mock.MagicMock()
    mock_df = mock.MagicMock()
    mock_df.__len__ = mock.MagicMock(return_value=10)
    mock_dataset.to_df.return_value = mock_df

    mock_prompt = mock.MagicMock()
    mock_prompt.model_config = {"provider": "openai", "model_name": "gpt-4o"}

    mock_optimizer = mock.MagicMock()
    mock_loaded_scorers = [mock.MagicMock(), mock.MagicMock()]
    mock_predict_fn = mock.MagicMock()

    mock_result = mock.MagicMock()
    mock_result.optimized_prompts = [mock.MagicMock()]
    mock_result.optimized_prompts[0].uri = "prompts:/test/2"
    mock_result.optimizer_name = "GepaPromptOptimizer"
    mock_result.initial_eval_score = 0.5
    mock_result.final_eval_score = 0.9

    with (
        mock.patch("mlflow.genai.optimize.job._record_event"),
        mock.patch("mlflow.genai.optimize.job.get_dataset", return_value=mock_dataset),
        mock.patch("mlflow.genai.optimize.job.load_prompt", return_value=mock_prompt),
        mock.patch(
            "mlflow.genai.optimize.job._create_optimizer", return_value=mock_optimizer
        ) as mock_create_optimizer,
        mock.patch(
            "mlflow.genai.optimize.job._load_scorers", return_value=mock_loaded_scorers
        ) as mock_load_scorers,
        mock.patch(
            "mlflow.genai.optimize.job._build_predict_fn", return_value=mock_predict_fn
        ) as mock_build_predict_fn,
        mock.patch("mlflow.genai.optimize.job.set_experiment"),
        mock.patch("mlflow.genai.optimize.job.start_run"),
        mock.patch("mlflow.genai.optimize.job.MlflowClient"),
        mock.patch(
            "mlflow.genai.optimize.job.optimize_prompts", return_value=mock_result
        ) as mock_optimize_prompts,
    ):
        optimize_prompts_job(
            run_id="run-123",
            experiment_id="exp-123",
            prompt_uri="prompts:/test/1",
            dataset_id="dataset-123",
            optimizer_type="gepa",
            optimizer_config_json='{"reflection_model": "openai:/gpt-4o"}',
            scorers=["Correctness", "Safety"],
        )

        # Verify _create_optimizer was called with correct args
        mock_create_optimizer.assert_called_once_with(
            "gepa", '{"reflection_model": "openai:/gpt-4o"}'
        )

        # Verify _load_scorers was called with correct args
        mock_load_scorers.assert_called_once_with(["Correctness", "Safety"], "exp-123")

        # Verify _build_predict_fn was called with correct args
        mock_build_predict_fn.assert_called_once_with("prompts:/test/1")

        # Verify optimize_prompts was called with correct args
        mock_optimize_prompts.assert_called_once_with(
            predict_fn=mock_predict_fn,
            train_data=mock_df,
            prompt_uris=["prompts:/test/1"],
            optimizer=mock_optimizer,
            scorers=mock_loaded_scorers,
            enable_tracking=True,
        )


def test_optimize_prompts_job_records_telemetry_event():
    mock_dataset = mock.MagicMock()
    mock_df = mock.MagicMock()
    mock_df.__len__ = mock.MagicMock(return_value=10)
    mock_dataset.to_df.return_value = mock_df

    mock_prompt = mock.MagicMock()
    mock_prompt.model_config = {"provider": "openai", "model_name": "gpt-4o"}

    mock_optimizer = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.optimized_prompts = [mock.MagicMock()]
    mock_result.optimized_prompts[0].uri = "prompts:/test/2"
    mock_result.optimizer_name = "GepaPromptOptimizer"
    mock_result.initial_eval_score = 0.5
    mock_result.final_eval_score = 0.9

    with (
        mock.patch("mlflow.genai.optimize.job._record_event") as mock_record_event,
        mock.patch("mlflow.genai.optimize.job.get_dataset", return_value=mock_dataset),
        mock.patch("mlflow.genai.optimize.job.load_prompt", return_value=mock_prompt),
        mock.patch("mlflow.genai.optimize.job._create_optimizer", return_value=mock_optimizer),
        mock.patch("mlflow.genai.optimize.job._load_scorers", return_value=[mock.MagicMock()]),
        mock.patch("mlflow.genai.optimize.job._build_predict_fn", return_value=lambda **k: "r"),
        mock.patch("mlflow.genai.optimize.job.set_experiment"),
        mock.patch("mlflow.genai.optimize.job.start_run"),
        mock.patch("mlflow.genai.optimize.job.MlflowClient"),
        mock.patch("mlflow.genai.optimize.job.optimize_prompts", return_value=mock_result),
    ):
        result = optimize_prompts_job(
            run_id="run-123",
            experiment_id="exp-123",
            prompt_uri="prompts:/test/1",
            dataset_id="dataset-123",
            optimizer_type="gepa",
            optimizer_config_json='{"reflection_model": "openai:/gpt-4o"}',
            scorers=["Correctness", "Safety"],
        )

        # Verify telemetry event was recorded
        mock_record_event.assert_called_once()
        call_args = mock_record_event.call_args
        from mlflow.telemetry.events import OptimizePromptsJobEvent

        assert call_args[0][0] == OptimizePromptsJobEvent
        assert call_args[0][1]["optimizer_type"] == "gepa"
        assert call_args[0][1]["scorer_count"] == 2

        # Verify result structure
        assert result["run_id"] == "run-123"
        assert result["source_prompt_uri"] == "prompts:/test/1"
        assert result["optimized_prompt_uri"] == "prompts:/test/2"
        assert result["optimizer_name"] == "GepaPromptOptimizer"
        assert result["initial_eval_score"] == 0.5
        assert result["final_eval_score"] == 0.9
        assert result["dataset_id"] == "dataset-123"
        assert result["scorers"] == ["Correctness", "Safety"]
