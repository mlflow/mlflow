import math
from pathlib import Path
from unittest import mock

import pytest
from sqlalchemy import create_engine, text

from mlflow.entities import Experiment, Run, ViewType
from mlflow.store.fs2db import migrate
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import local_file_uri_to_path

Clients = tuple[MlflowClient, MlflowClient]


def _get_all_experiments(client: MlflowClient) -> list[Experiment]:
    return client.search_experiments(view_type=ViewType.ALL)


def _get_all_runs(client: MlflowClient, experiment_ids: list[str]) -> list[Run]:
    runs = []
    for exp_id in experiment_ids:
        runs.extend(client.search_runs(experiment_ids=[exp_id], run_view_type=ViewType.ALL))
    return runs


def test_experiments(clients: Clients) -> None:
    src, dst = clients
    src_exps = _get_all_experiments(src)
    dst_exps = _get_all_experiments(dst)

    # DB auto-creates a Default experiment (id=0) during _initialize_tables,
    # so filter to only source experiment IDs for comparison.
    src_by_id = {e.experiment_id: e for e in src_exps}
    dst_by_id = {e.experiment_id: e for e in dst_exps if e.experiment_id in src_by_id}

    assert any(e.lifecycle_stage == "deleted" for e in src_by_id.values())

    for exp_id, src_exp in src_by_id.items():
        dst_exp = dst_by_id[exp_id]
        assert dst_exp.name == src_exp.name
        assert dst_exp.lifecycle_stage == src_exp.lifecycle_stage
        assert dst_exp.creation_time == src_exp.creation_time
        assert dst_exp.last_update_time == src_exp.last_update_time
        assert dst_exp.artifact_location == src_exp.artifact_location

        src_tags = {k: v for k, v in src_exp.tags.items() if not k.startswith("mlflow.")}
        dst_tags = {k: v for k, v in dst_exp.tags.items() if not k.startswith("mlflow.")}
        assert dst_tags == src_tags


def test_runs(clients: Clients) -> None:
    src, dst = clients
    exp_ids = [e.experiment_id for e in _get_all_experiments(src)]
    src_runs = _get_all_runs(src, exp_ids)
    dst_runs = _get_all_runs(dst, exp_ids)
    assert len(dst_runs) == len(src_runs)

    src_by_id = {r.info.run_id: r for r in src_runs}
    dst_by_id = {r.info.run_id: r for r in dst_runs}
    assert set(dst_by_id) == set(src_by_id)

    assert any(r.info.lifecycle_stage == "deleted" for r in src_by_id.values())

    for run_id, src_run in src_by_id.items():
        dst_run = dst_by_id[run_id]
        assert dst_run.info.status == src_run.info.status
        assert dst_run.info.lifecycle_stage == src_run.info.lifecycle_stage
        assert dst_run.info.start_time == src_run.info.start_time
        assert dst_run.info.end_time == src_run.info.end_time
        assert dst_run.info.run_name == src_run.info.run_name
        assert dst_run.data.params == src_run.data.params
        assert set(dst_run.data.metrics) == set(src_run.data.metrics)
        for key, src_val in src_run.data.metrics.items():
            dst_val = dst_run.data.metrics[key]
            if math.isnan(src_val):
                assert math.isnan(dst_val)
            elif math.isinf(src_val):
                # DB stores Inf as ±1.7976931348623157e308
                assert math.copysign(1, dst_val) == math.copysign(1, src_val)
            else:
                assert dst_val == src_val

        src_tags = {k: v for k, v in src_run.data.tags.items() if not k.startswith("mlflow.")}
        dst_tags = {k: v for k, v in dst_run.data.tags.items() if not k.startswith("mlflow.")}
        assert dst_tags == src_tags


def test_dataset_inputs(clients: Clients) -> None:
    src, dst = clients
    exp_ids = [e.experiment_id for e in _get_all_experiments(src)]
    src_runs = _get_all_runs(src, exp_ids)
    dst_by_id = {r.info.run_id: r for r in _get_all_runs(dst, exp_ids)}

    for src_run in src_runs:
        dst_run = dst_by_id[src_run.info.run_id]
        src_ds = src_run.inputs.dataset_inputs if src_run.inputs else []
        dst_ds = dst_run.inputs.dataset_inputs if dst_run.inputs else []
        assert len(dst_ds) == len(src_ds)

        src_by_name = {d.dataset.name: d for d in src_ds}
        dst_by_name = {d.dataset.name: d for d in dst_ds}
        for name, src_di in src_by_name.items():
            dst_di = dst_by_name[name]
            assert dst_di.dataset.digest == src_di.dataset.digest
            assert dst_di.dataset.source_type == src_di.dataset.source_type
            assert dst_di.dataset.source == src_di.dataset.source
            assert dst_di.dataset.schema == src_di.dataset.schema
            assert dst_di.dataset.profile == src_di.dataset.profile
            assert {t.key: t.value for t in dst_di.tags} == {t.key: t.value for t in src_di.tags}


def test_model_inputs(clients: Clients) -> None:
    # search_runs doesn't populate model_inputs; use get_run instead
    src, dst = clients
    exp_ids = [e.experiment_id for e in _get_all_experiments(src)]
    run_ids = [r.info.run_id for r in _get_all_runs(src, exp_ids)]

    all_src_model_inputs = []
    for run_id in run_ids:
        src_run = src.get_run(run_id)
        dst_run = dst.get_run(run_id)
        src_mi = src_run.inputs.model_inputs if src_run.inputs else []
        dst_mi = dst_run.inputs.model_inputs if dst_run.inputs else []
        assert len(dst_mi) == len(src_mi)
        assert sorted(m.model_id for m in dst_mi) == sorted(m.model_id for m in src_mi)
        all_src_model_inputs.extend(src_mi)

    assert len(all_src_model_inputs) > 0


def test_traces(clients: Clients) -> None:
    src, dst = clients
    exp_ids = [e.experiment_id for e in _get_all_experiments(src)]

    src_traces = src.search_traces(locations=exp_ids)
    dst_traces = dst.search_traces(locations=exp_ids)
    assert len(dst_traces) == len(src_traces)

    dst_by_id = {t.info.request_id: t for t in dst_traces}
    for src_trace in src_traces:
        dst_trace = dst_by_id[src_trace.info.request_id]
        assert dst_trace.info.status == src_trace.info.status
        assert dst_trace.info.request_time == src_trace.info.request_time
        assert dst_trace.info.execution_duration == src_trace.info.execution_duration
        assert set(dst_trace.info.tags) >= set(src_trace.info.tags)


def test_assessments(clients: Clients) -> None:
    src, dst = clients
    exp_ids = [e.experiment_id for e in _get_all_experiments(src)]

    src_traces = src.search_traces(locations=exp_ids)
    dst_traces = dst.search_traces(locations=exp_ids)
    dst_by_id = {t.info.request_id: t for t in dst_traces}

    for src_trace in src_traces:
        dst_trace = dst_by_id[src_trace.info.request_id]
        src_assessments = src_trace.search_assessments(all=True)
        dst_assessments = dst_trace.search_assessments(all=True)
        assert len(dst_assessments) == len(src_assessments)

        src_by_name = {a.name: a for a in src_assessments}
        dst_by_name = {a.name: a for a in dst_assessments}
        for name, src_a in src_by_name.items():
            dst_a = dst_by_name[name]
            assert dst_a.source.source_type == src_a.source.source_type
            assert dst_a.source.source_id == src_a.source.source_id
            assert dst_a.rationale == src_a.rationale
            assert dst_a.metadata == src_a.metadata
            if src_a.feedback is not None:
                assert dst_a.feedback is not None
                assert dst_a.feedback.value == src_a.feedback.value
            if src_a.expectation is not None:
                assert dst_a.expectation is not None
                assert dst_a.expectation.value == src_a.expectation.value


def test_logged_models(clients: Clients) -> None:
    src, dst = clients
    exp_ids = [e.experiment_id for e in _get_all_experiments(src)]

    src_models = src.search_logged_models(experiment_ids=exp_ids)
    dst_models = dst.search_logged_models(experiment_ids=exp_ids)
    assert len(dst_models) == len(src_models)

    dst_by_id = {m.model_id: m for m in dst_models}
    for src_model in src_models:
        dst_model = dst_by_id[src_model.model_id]
        assert dst_model.name == src_model.name
        assert dst_model.creation_timestamp == src_model.creation_timestamp
        assert dst_model.last_updated_timestamp == src_model.last_updated_timestamp
        assert dst_model.status == src_model.status
        assert dst_model.model_type == src_model.model_type
        assert dst_model.source_run_id == src_model.source_run_id
        assert set(dst_model.tags) >= set(src_model.tags)


def test_run_outputs(clients: Clients) -> None:
    src, dst = clients
    exp_ids = [e.experiment_id for e in _get_all_experiments(src)]
    src_runs = _get_all_runs(src, exp_ids)
    dst_by_id = {r.info.run_id: r for r in _get_all_runs(dst, exp_ids)}

    for src_run in src_runs:
        dst_run = dst_by_id[src_run.info.run_id]
        src_outputs = src_run.outputs.model_outputs if src_run.outputs else []
        dst_outputs = dst_run.outputs.model_outputs if dst_run.outputs else []
        assert len(dst_outputs) == len(src_outputs)
        assert sorted(o.model_id for o in dst_outputs) == sorted(o.model_id for o in src_outputs)


def test_registered_models(clients: Clients) -> None:
    src, dst = clients

    src_models = src.search_registered_models()
    dst_models = dst.search_registered_models()
    assert len(dst_models) == len(src_models)

    dst_by_name = {m.name: m for m in dst_models}
    for src_model in src_models:
        dst_model = dst_by_name[src_model.name]
        assert dst_model.description == src_model.description
        assert dst_model.creation_timestamp == src_model.creation_timestamp
        assert dst_model.last_updated_timestamp == src_model.last_updated_timestamp
        assert set(dst_model.tags) >= set(src_model.tags)

        src_versions = src.search_model_versions(f"name='{src_model.name}'")
        dst_versions = dst.search_model_versions(f"name='{dst_model.name}'")
        assert len(dst_versions) == len(src_versions)


def test_model_versions(clients: Clients) -> None:
    src, dst = clients

    src_models = src.search_registered_models()

    for src_rm in src_models:
        src_versions = src.search_model_versions(f"name='{src_rm.name}'")
        dst_versions = dst.search_model_versions(f"name='{src_rm.name}'")
        assert len(dst_versions) == len(src_versions)

        dst_by_ver = {v.version: v for v in dst_versions}
        for src_mv in src_versions:
            dst_mv = dst_by_ver[src_mv.version]
            assert dst_mv.description == src_mv.description
            assert dst_mv.creation_timestamp == src_mv.creation_timestamp
            assert dst_mv.status == src_mv.status
            assert dst_mv.source == src_mv.source
            assert dst_mv.run_id == src_mv.run_id
            assert set(dst_mv.tags) >= set(src_mv.tags)


def test_prompts(clients: Clients) -> None:
    src, dst = clients

    # search_registered_models excludes prompts, so use search_prompts instead.
    src_prompts = src.search_prompts()
    dst_prompts = dst.search_prompts()
    assert len(dst_prompts) == len(src_prompts)
    assert len(src_prompts) > 0

    for src_prompt in src_prompts:
        name = src_prompt.name
        dst_prompt = next(p for p in dst_prompts if p.name == name)
        assert dst_prompt is not None

        src_pv = src.get_prompt_version(name, 1)
        dst_pv = dst.get_prompt_version(name, 1)
        assert src_pv is not None
        assert dst_pv is not None
        assert dst_pv.template == src_pv.template


def test_rollback_on_failure(clients: Clients, tmp_path: Path) -> None:
    src, _ = clients
    source = Path(local_file_uri_to_path(src.tracking_uri))
    target_uri = f"sqlite:///{tmp_path / 'rollback.db'}"

    with mock.patch(
        "mlflow.store.fs2db._tracking._migrate_runs_in_dir",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            migrate(source, target_uri, progress=False)

    engine = create_engine(target_uri)
    with engine.connect() as conn:
        for table in ("experiments", "runs", "registered_models"):
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            assert count == 0
