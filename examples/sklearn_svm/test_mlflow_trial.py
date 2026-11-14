import mlflow
import pandas as pd
import pytest
from types import SimpleNamespace

import train


@pytest.fixture(autouse=True)
def _mlflow_tmp(tmp_path, monkeypatch):
    uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)

def test_one_trial(tmp_path):
    args = SimpleNamespace(
        max_ngram=1,
        max_features=20000,
        min_df=1,
        C=1.0,
        top_k=20,
        experiment_name="one-trial-exp",
        run_name="real",
        seed=0)

    train.run(args)

    exp = mlflow.get_experiment_by_name(args.experiment_name)
    assert exp is not None
    
    runs = mlflow.search_runs([exp.experiment_id], max_results=1, order_by=["attributes.start_time DESC"])
    assert len(runs) == 1

    run_id = runs.iloc[0]["run_id"]
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)
    assert run.data.params == {}
    assert run.data.metrics == {}


