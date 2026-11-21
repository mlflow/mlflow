import json
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import mlflow.johnsnowlabs
from mlflow.environment_variables import MLFLOW_ALLOW_MODEL_CODE_EXECUTION
from mlflow.exceptions import MlflowException


@pytest.fixture
def nlp():
    nlp = mock.MagicMock()
    johnsnowlabs = types.ModuleType("johnsnowlabs")
    johnsnowlabs.nlp = nlp
    with (
        mock.patch.dict(sys.modules, {"johnsnowlabs": johnsnowlabs}),
        mock.patch.dict(os.environ, {"JOHNSNOWLABS_LICENSE_JSON": json.dumps({"SECRET": "s"})}),
        mock.patch(
            "mlflow.utils._spark_utils._get_active_spark_session", return_value=None
        ) as get_active_spark_session,
    ):
        yield nlp
        get_active_spark_session.assert_called_once_with()


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    (tmp_path / "MLmodel").write_text("flavors:\n  johnsnowlabs:\n    model_data: jsl-model\n")
    jars_dir = tmp_path / "jsl-model" / "jars.jsl"
    jars_dir.mkdir(parents=True)
    (jars_dir / "model.jar").write_bytes(b"")
    (jars_dir / "license.json").write_text(
        json.dumps({"HC_LICENSE": "license", "MODEL_ENV_VAR": "value"})
    )
    return tmp_path


def test_load_model_refuses_model_jars_and_license_when_model_code_is_disallowed(
    monkeypatch, nlp, model_path
):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, "false")
    monkeypatch.setenv("PYSPARK_PYTHON", "/custom/python")

    with pytest.raises(MlflowException, match=MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name):
        mlflow.johnsnowlabs.load_model(str(model_path))

    assert "MODEL_ENV_VAR" not in os.environ
    assert os.environ["PYSPARK_PYTHON"] == "/custom/python"
    nlp.start.assert_not_called()
    nlp.load.assert_not_called()


def test_load_model_starts_session_with_model_jars_when_model_code_is_allowed(
    monkeypatch, nlp, model_path
):
    monkeypatch.setenv(MLFLOW_ALLOW_MODEL_CODE_EXECUTION.name, "true")

    mlflow.johnsnowlabs.load_model(str(model_path))

    assert os.environ["MODEL_ENV_VAR"] == "value"
    jars_dir = model_path / "jsl-model" / "jars.jsl"
    nlp.start.assert_called_once()
    assert nlp.start.call_args.kwargs["jar_paths"] == [str(jars_dir / "model.jar")]
    assert nlp.start.call_args.kwargs["json_license_path"] == str(jars_dir / "license.json")
    nlp.load.assert_called_once_with(path=str(model_path / "jsl-model"))
