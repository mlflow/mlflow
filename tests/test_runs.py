from click.testing import CliRunner
from unittest import mock
import pytest

import os
import shutil
import tempfile
import textwrap

from mlflow import experiments
from mlflow.runs import list_run

import mlflow


def test_list_run():
    with mlflow.start_run(run_name="apple"):
        pass
    result = CliRunner().invoke(list_run, ["--experiment-id", "0"])
    assert "apple" in result.output


def test_list_run_experiment_id_required():
    result = CliRunner().invoke(list_run, [])
    assert "Missing option '--experiment-id'" in result.output


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny Client does not support predict due to the pandas dependency",
)
def test_csv_generation():
    import numpy as np
    import pandas as pd

    with mock.patch("mlflow.experiments.fluent.search_runs") as mock_search_runs:
        mock_search_runs.return_value = pd.DataFrame(
            {
                "run_id": np.array(["all_set", "with_none", "with_nan"]),
                "experiment_id": np.array([1, 1, 1]),
                "param_optimizer": np.array(["Adam", None, "Adam"]),
                "avg_loss": np.array([42.0, None, np.nan], dtype=np.float32),
            },
            columns=["run_id", "experiment_id", "param_optimizer", "avg_loss"],
        )
        expected_csv = textwrap.dedent(
            """\
        run_id,experiment_id,param_optimizer,avg_loss
        all_set,1,Adam,42.0
        with_none,1,,
        with_nan,1,Adam,
        """
        )
        tempdir = tempfile.mkdtemp()
        try:
            result_filename = os.path.join(tempdir, "result.csv")
            CliRunner().invoke(
                experiments.generate_csv_with_runs,
                ["--experiment-id", "1", "--filename", result_filename],
            )
            with open(result_filename, "r") as fd:
                assert expected_csv == fd.read()
        finally:
            shutil.rmtree(tempdir)
