import os

import pandas as pd

from mlflow.pipelines.utils.execution import _MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR
from mlflow.pipelines.steps.preprocessing import PreprocessingStep

from unittest import mock


def test_preprocessing_step_run(tmp_path):
    ingest_output_dir = tmp_path / "steps" / "ingest" / "outputs"
    ingest_output_dir.mkdir(parents=True)
    preprocessing_output_dir = tmp_path / "steps" / "preprocessing" / "outputs"
    preprocessing_output_dir.mkdir(parents=True)

    num_ingested_rows = 1000
    # Since there is no cleanup method, it is a passthrough and it should be the same
    num_written_rows = 1000
    input_dataframe = pd.DataFrame(
        {
            "a": list(range(num_ingested_rows)),
            "b": [str(i) for i in range(num_ingested_rows)],
            "y": [float(i % 2) for i in range(num_ingested_rows)],
        }
    )
    input_dataframe.to_parquet(str(ingest_output_dir / "dataset.parquet"))

    with mock.patch.dict(
        os.environ, {_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_path)}
    ), mock.patch("mlflow.pipelines.step.get_pipeline_name", return_value="fake_name"):
        preprocessing_step = PreprocessingStep({}, "fake_root")
        preprocessing_step._run(str(preprocessing_output_dir))

    (preprocessing_output_dir / "summary.html").exists()
    (preprocessing_output_dir / "card.html").exists()

    output_df = pd.read_parquet(str(preprocessing_output_dir / "preprocessed.parquet"))
    assert len(output_df) == 1000

    assert output_df.columns.tolist() == ["a", "b", "y"]
    assert set(output_df.a.tolist()) == set(range(num_written_rows))
    assert set(output_df.b.tolist()) == set(str(i) for i in range(num_written_rows))
    assert set(output_df.y.tolist()) == {0.0, 1.0}
